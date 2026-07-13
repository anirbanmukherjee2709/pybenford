"""Data preparation and frequency-counting utilities for Benford analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pybenford.constants import SUMMATION_EXPECTED
from pybenford.digits import extract_first_two_digits

__all__ = [
    "CleaningReport",
    "DataProfile",
    "DigitFrequencies",
    "DuplicationResult",
    "Stratum",
    "SummationFrequencies",
    "clean_numeric_array",
    "data_profile",
    "digit_counts",
    "number_duplication",
    "second_order_differences",
    "summation_by_digits",
    "to_numeric_array",
]


# ---------------------------------------------------------------------------
# 1. Type coercion
# ---------------------------------------------------------------------------


def to_numeric_array(data: object) -> NDArray[np.float64]:
    """Coerce input to a 1-D float64 NumPy array.

    Accepts lists, tuples, NumPy arrays, and duck-typed objects that
    expose ``.to_numpy()`` (pandas) or ``.to_list()`` (polars).
    Does NOT clean the data — only coerces type.

    Raises
    ------
    TypeError
        If *data* is not a supported type.
    ValueError
        If *data* is empty or not 1-D.
    """
    arr: NDArray[np.float64]
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float64, copy=False)
    elif isinstance(data, (list, tuple)):
        arr = np.asarray(data, dtype=np.float64)
    elif hasattr(data, "to_numpy"):
        arr = np.asarray(data.to_numpy(), dtype=np.float64)
    elif hasattr(data, "to_list"):
        arr = np.asarray(data.to_list(), dtype=np.float64)
    else:
        raise TypeError(
            f"unsupported type {type(data).__name__}; "
            "expected list, tuple, ndarray, or object with .to_numpy()/.to_list()"
        )

    if arr.size == 0:
        raise ValueError("input data is empty")
    if arr.ndim != 1:
        raise ValueError(f"input must be 1-D, got {arr.ndim}-D")
    return arr


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleaningReport:
    """Summary of what ``clean_numeric_array`` removed."""

    original_count: int
    clean_count: int
    nan_inf_removed: int
    zeros_removed: int
    sign_filtered: int
    below_threshold_removed: int


def clean_numeric_array(
    data: ArrayLike,
    *,
    sign_filter: str = "all",
    min_abs_value: float | None = None,
    drop_zero: bool = True,
) -> tuple[NDArray[np.float64], CleaningReport]:
    """Clean an array for Benford analysis (Nigrini Ch. 4).

    Steps applied in order:
    (a) remove NaN / inf,
    (b) remove zeros when *drop_zero* is True,
    (c) apply *sign_filter*,
    (d) remove values with ``|x| < min_abs_value`` when set.

    Parameters
    ----------
    data
        Array-like of numeric values.
    sign_filter
        ``"all"`` — take absolute value of everything,
        ``"positive"`` — keep only values > 0,
        ``"negative"`` — keep only values < 0, then take absolute value.
    min_abs_value
        Drop values whose absolute value is below this threshold.
    drop_zero
        If True (default), remove exact zeros.

    Returns
    -------
    tuple of (cleaned array, CleaningReport)
        The cleaned array contains only positive finite values.
    """
    if sign_filter not in ("all", "positive", "negative"):
        raise ValueError(
            f"sign_filter must be 'all', 'positive', or 'negative', got '{sign_filter}'"
        )

    arr = np.asarray(data, dtype=np.float64).ravel()
    original_count = len(arr)

    # (a) remove NaN / inf
    finite_mask = np.isfinite(arr)
    nan_inf_removed = int(np.sum(~finite_mask))
    arr = arr[finite_mask]

    # (b) remove zeros
    if drop_zero:
        nonzero_mask = arr != 0.0
        zeros_removed = int(np.sum(~nonzero_mask))
        arr = arr[nonzero_mask]
    else:
        zeros_removed = 0

    # (c) apply sign filter
    if sign_filter == "all":
        sign_filtered = 0
        arr = np.abs(arr)
    elif sign_filter == "positive":
        pos_mask = arr > 0.0
        sign_filtered = int(np.sum(~pos_mask))
        arr = arr[pos_mask]
    else:  # "negative"
        neg_mask = arr < 0.0
        sign_filtered = int(np.sum(~neg_mask))
        arr = np.abs(arr[neg_mask])

    # (d) remove values below threshold
    if min_abs_value is not None:
        above_mask = arr >= min_abs_value
        below_threshold_removed = int(np.sum(~above_mask))
        arr = arr[above_mask]
    else:
        below_threshold_removed = 0

    report = CleaningReport(
        original_count=original_count,
        clean_count=len(arr),
        nan_inf_removed=nan_inf_removed,
        zeros_removed=zeros_removed,
        sign_filtered=sign_filtered,
        below_threshold_removed=below_threshold_removed,
    )
    return arr, report


# ---------------------------------------------------------------------------
# 3. Data profiling (Nigrini Ch. 2 Fig 2.1, Ch. 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    """One row of a Nigrini data profile."""

    name: str
    count: int
    total_sum: float
    percentage: float


@dataclass(frozen=True)
class DataProfile:
    """Nigrini's five-stratum data profile plus an invalid stratum."""

    total_count: int
    strata: tuple[Stratum, ...]
    invalid_count: int


def _make_stratum(
    name: str,
    subset: NDArray[np.float64],
    total: int,
) -> Stratum:
    count = len(subset)
    s = float(np.sum(subset)) if count > 0 else 0.0
    pct = count / total * 100.0 if total > 0 else 0.0
    return Stratum(name=name, count=count, total_sum=s, percentage=pct)


def data_profile(
    data: ArrayLike,
    *,
    threshold: float = 10.0,
    extra_thresholds: Sequence[float] | None = None,
) -> DataProfile:
    """Build Nigrini's five-stratum data profile.

    Standard strata: large positive (>= *threshold*), small positive
    (0 < x < *threshold*), zeros, small negative
    (-*threshold* < x < 0), large negative (<= -*threshold*).
    NaN / inf values are tallied as an "Invalid" stratum.

    Parameters
    ----------
    data
        Array-like of raw numeric values.
    threshold
        Boundary between "large" and "small" strata (default 10.0).
    extra_thresholds
        Additional boundaries that subdivide the large-positive
        stratum into finer ranges.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    total = len(arr)

    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(np.sum(invalid_mask))
    valid = arr[~invalid_mask]

    strata: list[Stratum] = []

    # Large positive — optionally subdivided
    large_pos = valid[valid >= threshold]
    if extra_thresholds is not None:
        bounds = sorted(b for b in extra_thresholds if b > threshold)
        if bounds:
            all_bounds = [threshold, *bounds]
            for i, low in enumerate(all_bounds):
                if i + 1 < len(all_bounds):
                    high = all_bounds[i + 1]
                    subset = large_pos[(large_pos >= low) & (large_pos < high)]
                    name = f">= {low} and < {high}"
                else:
                    subset = large_pos[large_pos >= low]
                    name = f">= {low}"
                strata.append(_make_stratum(name, subset, total))
        else:
            strata.append(_make_stratum("Large positive", large_pos, total))
    else:
        strata.append(_make_stratum("Large positive", large_pos, total))

    strata.append(_make_stratum("Small positive", valid[(valid > 0) & (valid < threshold)], total))
    strata.append(_make_stratum("Zero", valid[valid == 0.0], total))
    strata.append(_make_stratum("Small negative", valid[(valid < 0) & (valid > -threshold)], total))
    strata.append(_make_stratum("Large negative", valid[valid <= -threshold], total))

    if invalid_count > 0:
        pct = invalid_count / total * 100.0 if total > 0 else 0.0
        strata.append(Stratum(name="Invalid", count=invalid_count, total_sum=0.0, percentage=pct))

    return DataProfile(
        total_count=total,
        strata=tuple(strata),
        invalid_count=invalid_count,
    )


# ---------------------------------------------------------------------------
# 4. Digit frequency counting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DigitFrequencies:
    """Observed digit counts and proportions."""

    digits: NDArray[np.int64]
    counts: NDArray[np.int64]
    proportions: NDArray[np.float64]
    total: int


def digit_counts(
    data: ArrayLike,
    digit_extractor: Callable[[ArrayLike], NDArray[np.float64]],
    possible_digits: ArrayLike,
) -> DigitFrequencies:
    """Count occurrences of each digit value in *possible_digits*.

    Parameters
    ----------
    data
        Array-like of numeric values.
    digit_extractor
        A callable from ``pybenford.digits`` (e.g.
        ``extract_first_digit``).  Applied to *data* to obtain the
        digit for each element.
    possible_digits
        Array-like of integer digit values that define the histogram
        bins (e.g. ``range(1, 10)`` for first digits).

    Raises
    ------
    ValueError
        If no valid digits can be extracted from *data*.
    """
    arr = np.asarray(data, dtype=np.float64)
    extracted = digit_extractor(arr)
    valid_mask = ~np.isnan(extracted)
    valid_digits = extracted[valid_mask].astype(np.int64)

    if len(valid_digits) == 0:
        raise ValueError("no valid digits extracted from data")

    possible = np.asarray(possible_digits, dtype=np.int64)
    counts = np.array(
        [np.count_nonzero(valid_digits == d) for d in possible],
        dtype=np.int64,
    )
    total = int(np.sum(counts))
    proportions = counts.astype(np.float64) / total if total > 0 else np.zeros(len(possible))

    return DigitFrequencies(
        digits=possible,
        counts=counts,
        proportions=proportions,
        total=total,
    )


# ---------------------------------------------------------------------------
# 5. Summation test (Nigrini Ch. 3 & 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummationFrequencies:
    """Value sums grouped by first-two digits."""

    digits: NDArray[np.int64]
    sums: NDArray[np.float64]
    proportions: NDArray[np.float64]
    expected_proportions: NDArray[np.float64]
    grand_sum: float


def summation_by_digits(data: ArrayLike) -> SummationFrequencies:
    """Group values by first-two digits and sum the original values.

    Expected proportions are uniform at ``1/90`` per bin
    (Nigrini Ch. 5).

    Raises
    ------
    ValueError
        If no valid first-two digits can be extracted.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    ft = extract_first_two_digits(arr)
    valid_mask = ~np.isnan(ft)
    valid_arr = arr[valid_mask]
    valid_ft = ft[valid_mask].astype(np.int64)

    if len(valid_ft) == 0:
        raise ValueError("no valid first-two digits extracted")

    possible = np.arange(10, 100, dtype=np.int64)
    sums = np.array(
        [float(np.sum(valid_arr[valid_ft == d])) for d in possible],
        dtype=np.float64,
    )
    grand_sum = float(np.sum(valid_arr))
    proportions = sums / grand_sum if grand_sum != 0.0 else np.zeros(90, dtype=np.float64)
    expected: NDArray[np.float64] = np.full(90, SUMMATION_EXPECTED, dtype=np.float64)

    return SummationFrequencies(
        digits=possible,
        sums=sums,
        proportions=proportions,
        expected_proportions=expected,
        grand_sum=grand_sum,
    )


# ---------------------------------------------------------------------------
# 6. Second-order differences (Nigrini Ch. 5)
# ---------------------------------------------------------------------------


def second_order_differences(data: ArrayLike) -> NDArray[np.float64]:
    """Sort, diff, scale by 10, round, and take absolute value.

    Matches the notebook formula ``abs(round((x * 10).diff(), 0))``.
    Returns ``N - 1`` elements where ``N`` is the number of finite
    input values.

    Raises
    ------
    ValueError
        If fewer than 2 finite values are available.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]

    if len(finite) < 2:
        raise ValueError("need at least 2 finite values for second-order differences")

    sorted_arr = np.sort(finite)
    diffs = np.diff(sorted_arr)
    return np.abs(np.round(diffs * 10.0, 0))


# ---------------------------------------------------------------------------
# 7. Number duplication test (Nigrini Ch. 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicationResult:
    """Most frequently occurring values and their first-two digits."""

    values: NDArray[np.float64]
    counts: NDArray[np.int64]
    first_two_digits: NDArray[np.float64]
    total_unique: int
    total_records: int

    def __str__(self) -> str:
        w = 55
        border = "=" * w
        lines: list[str] = [border, "  Number Duplication Test", border]
        lines.append(
            f" Total Records: {self.total_records:,}  |  Unique Values: {self.total_unique:,}"
        )
        lines.append("")

        if len(self.counts) > 0 and int(self.counts[0]) > 1:
            lines.append(" Top Duplicated Values:")
            lines.append("  Value       Count   First-Two")
            for i in range(len(self.values)):
                lines.append(
                    f"  {self.values[i]:>10,.0f}"
                    f"    {self.counts[i]:>7,}"
                    f"    {int(self.first_two_digits[i]):>9}"
                )
        else:
            lines.append(" No duplicated values found.")

        lines.append(border)
        return "\n".join(lines)


def number_duplication(
    data: ArrayLike,
    *,
    top_n: int = 10,
) -> DuplicationResult:
    """Find the most frequently duplicated numbers.

    Returns the *top_n* values sorted by count descending, then by
    value descending for ties.

    Raises
    ------
    ValueError
        If no finite values are present.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]

    if len(finite) == 0:
        raise ValueError("no valid values for duplication analysis")

    unique_vals, raw_counts = np.unique(finite, return_counts=True)

    sort_idx = np.lexsort((-unique_vals, -raw_counts))

    sorted_vals = unique_vals[sort_idx]
    sorted_counts = raw_counts[sort_idx]

    n = min(top_n, len(sorted_vals))
    top_vals = sorted_vals[:n]
    top_counts = sorted_counts[:n].astype(np.int64)

    return DuplicationResult(
        values=top_vals,
        counts=top_counts,
        first_two_digits=extract_first_two_digits(top_vals),
        total_unique=len(unique_vals),
        total_records=len(finite),
    )
