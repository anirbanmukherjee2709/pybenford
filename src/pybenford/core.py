"""Public API for Benford's Law analysis.

Users interact with the :class:`BenfordAnalysis` class, which
orchestrates cleaning, profiling, and every statistical test defined in
the lower-level modules (digits, distributions, statistics, utils).

Example
-------
>>> from pybenford import BenfordAnalysis
>>> ba = BenfordAnalysis(data, sign_filter="positive", min_abs_value=10.0)
>>> result = ba.first_digit()
>>> print(result.mad_conformity)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2 as _chi2_dist

from pybenford.constants import LAST_TWO_DIGITS_EXPECTED
from pybenford.digits import (
    extract_first_digit,
    extract_first_three_digits,
    extract_first_two_digits,
    extract_last_two_digits,
    extract_second_digit,
    extract_third_digit,
)
from pybenford.distributions import (
    first_digit_distribution,
    first_three_digits_distribution,
    first_two_digits_distribution,
    second_digit_distribution,
    third_digit_distribution,
)
from pybenford.statistics import (
    DigitTest,
    DistortionResult,
    MantissaArcResult,
    chi_square_test,
    distortion_factor_test,
    ks_test,
    mad_test,
    mantissa_arc_test,
    z_significant,
    z_statistic,
)
from pybenford.utils import (
    CleaningReport,
    DataProfile,
    DuplicationResult,
    SummationFrequencies,
    clean_numeric_array,
    data_profile,
    digit_counts,
    number_duplication,
    second_order_differences,
    summation_by_digits,
    to_numeric_array,
)

__all__ = [
    "BenfordAnalysis",
    "SummationResult",
    "TestResult",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


_TEST_NAME_MAP: dict[str, str] = {
    "first_digit": "First Digit Test",
    "second_digit": "Second Digit Test",
    "third_digit": "Third Digit Test",
    "first_two_digits": "First Two Digits Test",
    "first_three_digits": "First Three Digits Test",
    "last_two_digits": "Last Two Digits Test",
    "second_order": "Second Order Test",
    "summation": "Summation Test",
}

_MAD_CONFORMITY_MAP: dict[str, str] = {
    "close_conformity": "Close Conformity",
    "acceptable_conformity": "Acceptable Conformity",
    "marginally_acceptable_conformity": "Marginally Acceptable",
    "nonconformity": "Nonconformity",
    "not_applicable": "N/A (no threshold defined)",
}

_WIDTH = 55


@dataclass(frozen=True)
class TestResult:
    """Complete result of a Benford digit test."""

    test_name: str
    digits: NDArray[np.int64]
    counts: NDArray[np.int64]
    observed: NDArray[np.float64]
    expected: NDArray[np.float64]
    z_scores: NDArray[np.float64]
    significant_flags: NDArray[np.bool_]
    chi_square: float
    chi_square_critical: float
    chi_square_significant: bool
    ks_statistic: float
    ks_critical: float
    ks_significant: bool
    mad: float
    mad_conformity: str
    n: int
    alpha: float

    def __str__(self) -> str:
        w = _WIDTH
        border = "=" * w
        sep = "-" * w
        title = _TEST_NAME_MAP.get(self.test_name, self.test_name)
        header = f"  {title}  (n={self.n:,}  alpha={self.alpha})"

        lines: list[str] = [border, header, border]

        if len(self.digits) <= 10:
            lines.append(" Digit   Count   Observed   Expected   Z-Score   Sig")
            for i in range(len(self.digits)):
                flag = "  *" if self.significant_flags[i] else ""
                lines.append(
                    f" {self.digits[i]:>5}{self.counts[i]:>7,}"
                    f"   {self.observed[i] * 100:>5.2f}%"
                    f"    {self.expected[i] * 100:>5.2f}%"
                    f"     {self.z_scores[i]:>5.2f}{flag}"
                )
        else:
            flagged_idx = [i for i in range(len(self.digits)) if self.significant_flags[i]]
            if flagged_idx:
                lines.append(f" Flagged Digits ({len(flagged_idx)} of {len(self.digits)}):")
                lines.append(" Digit   Count   Observed   Expected   Z-Score")
                for i in flagged_idx:
                    lines.append(
                        f" {self.digits[i]:>5}{self.counts[i]:>7,}"
                        f"   {self.observed[i] * 100:>5.2f}%"
                        f"    {self.expected[i] * 100:>5.2f}%"
                        f"     {self.z_scores[i]:>5.2f}  *"
                    )
            else:
                lines.append(f" No individual digits flagged at alpha={self.alpha}.")

        lines.append(sep)
        mad_label = _MAD_CONFORMITY_MAP.get(self.mad_conformity, self.mad_conformity)
        lines.append(f" MAD:        {self.mad:.4f} — {mad_label}")
        chi_verdict = "Pass" if not self.chi_square_significant else "FAIL"
        lines.append(
            f" Chi-Square: {self.chi_square:.4f}"
            f"  (critical: {self.chi_square_critical:.4f})"
            f" — {chi_verdict}"
        )
        ks_verdict = "Pass" if not self.ks_significant else "FAIL"
        lines.append(
            f" KS:         {self.ks_statistic:.4f}"
            f"  (critical: {self.ks_critical:.4f})"
            f"  — {ks_verdict}"
        )
        lines.append(border)
        return "\n".join(lines)


@dataclass(frozen=True)
class SummationResult:
    """Complete result of the summation test."""

    test_name: str
    digits: NDArray[np.int64]
    sums: NDArray[np.float64]
    observed: NDArray[np.float64]
    expected: NDArray[np.float64]
    z_scores: NDArray[np.float64]
    significant_flags: NDArray[np.bool_]
    chi_square: float
    chi_square_critical: float
    chi_square_significant: bool
    grand_sum: float
    n: int
    alpha: float

    def __str__(self) -> str:
        w = _WIDTH
        border = "=" * w
        sep = "-" * w
        title = _TEST_NAME_MAP.get(self.test_name, self.test_name)
        header = f"  {title}  (n={self.n:,}  alpha={self.alpha})"

        lines: list[str] = [border, header, border]
        lines.append(f" Grand Sum: {self.grand_sum:,.0f}")
        lines.append("")

        flagged_idx = [i for i in range(len(self.digits)) if self.significant_flags[i]]
        if flagged_idx:
            lines.append(f" Flagged Digits ({len(flagged_idx)} of {len(self.digits)}):")
            lines.append(" Digit         Sum   Observed   Expected   Z-Score")
            for i in flagged_idx:
                lines.append(
                    f" {self.digits[i]:>5}{self.sums[i]:>12,.0f}"
                    f"   {self.observed[i] * 100:>5.2f}%"
                    f"    {self.expected[i] * 100:>5.2f}%"
                    f"     {self.z_scores[i]:>5.2f}  *"
                )
        else:
            lines.append(f" No individual digits flagged at alpha={self.alpha}.")

        lines.append(sep)
        chi_verdict = "Pass" if not self.chi_square_significant else "FAIL"
        lines.append(
            f" Chi-Square: {self.chi_square:.4f}"
            f"  (critical: {self.chi_square_critical:.4f})"
            f" — {chi_verdict}"
        )
        lines.append(border)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------


class BenfordAnalysis:
    """One-stop Benford's Law analysis on a numeric dataset.

    Parameters
    ----------
    data
        Raw numeric input (list, tuple, ndarray, pandas/polars Series).
    sign_filter
        ``"all"`` keeps everything (absolute value),
        ``"positive"`` keeps only positive values,
        ``"negative"`` keeps only negative values (then abs).
    min_abs_value
        Drop values whose absolute value is below this threshold.
    drop_zero
        If True (default), remove exact zeros before analysis.

    Raises
    ------
    ValueError
        If no data remains after cleaning.
    """

    def __init__(
        self,
        data: object,
        *,
        sign_filter: Literal["all", "positive", "negative"] = "all",
        min_abs_value: float | None = None,
        drop_zero: bool = True,
    ) -> None:
        raw = to_numeric_array(data)
        self.profile: DataProfile = data_profile(raw)

        cleaned, report = clean_numeric_array(
            raw,
            sign_filter=sign_filter,
            min_abs_value=min_abs_value,
            drop_zero=drop_zero,
        )
        if len(cleaned) == 0:
            raise ValueError("no data remains after cleaning")

        self.clean_data: NDArray[np.float64] = cleaned
        self.n: int = len(cleaned)
        self.cleaning_report: CleaningReport = report

    # ── private helper ─────────────────────────────────────────────────

    def _run_digit_test(
        self,
        *,
        test_name: str,
        data: NDArray[np.float64],
        extractor: Callable[[ArrayLike], NDArray[np.float64]],
        digit_range: NDArray[np.int64],
        expected: NDArray[np.float64],
        digit_test: DigitTest | None,
        alpha: float,
    ) -> TestResult:
        """Shared implementation for every digit-level test."""
        freq = digit_counts(data, extractor, digit_range)
        n = freq.total
        obs = freq.proportions

        z = z_statistic(obs, expected, n)
        sig = z_significant(z, alpha=alpha)

        chi = chi_square_test(freq.counts, expected, n, alpha=alpha)
        ks = ks_test(obs, expected, n, alpha=alpha)

        if digit_test is not None:
            mad_res = mad_test(obs, expected, digit_test)
            mad_val = mad_res.mad
            mad_conf = mad_res.conformity.value
        else:
            k = len(expected)
            mad_val = float(np.sum(np.abs(obs - expected)) / k)
            mad_conf = "not_applicable"

        return TestResult(
            test_name=test_name,
            digits=freq.digits,
            counts=freq.counts,
            observed=obs,
            expected=expected,
            z_scores=np.asarray(z, dtype=np.float64),
            significant_flags=np.asarray(sig, dtype=np.bool_),
            chi_square=chi.statistic,
            chi_square_critical=chi.critical_value,
            chi_square_significant=bool(chi.significant),
            ks_statistic=ks.statistic,
            ks_critical=ks.critical_value,
            ks_significant=bool(ks.significant),
            mad=mad_val,
            mad_conformity=mad_conf,
            n=n,
            alpha=alpha,
        )

    # ── public digit tests ─────────────────────────────────────────────

    def first_digit(self, *, alpha: float = 0.05) -> TestResult:
        """First-digit test (K=9, Nigrini Eq. 1.1)."""
        return self._run_digit_test(
            test_name="first_digit",
            data=self.clean_data,
            extractor=extract_first_digit,
            digit_range=np.arange(1, 10, dtype=np.int64),
            expected=first_digit_distribution(),
            digit_test=DigitTest.FIRST,
            alpha=alpha,
        )

    def second_digit(self, *, alpha: float = 0.05) -> TestResult:
        """Second-digit test (K=10, Nigrini Eq. 1.2)."""
        return self._run_digit_test(
            test_name="second_digit",
            data=self.clean_data,
            extractor=extract_second_digit,
            digit_range=np.arange(0, 10, dtype=np.int64),
            expected=second_digit_distribution(),
            digit_test=DigitTest.SECOND,
            alpha=alpha,
        )

    def third_digit(self, *, alpha: float = 0.05) -> TestResult:
        """Third-digit test (K=10, Nigrini Table 1.2)."""
        return self._run_digit_test(
            test_name="third_digit",
            data=self.clean_data,
            extractor=extract_third_digit,
            digit_range=np.arange(0, 10, dtype=np.int64),
            expected=third_digit_distribution(),
            digit_test=None,
            alpha=alpha,
        )

    def first_two_digits(self, *, alpha: float = 0.05) -> TestResult:
        """First-two-digits test (K=90, Nigrini Eq. 1.3)."""
        return self._run_digit_test(
            test_name="first_two_digits",
            data=self.clean_data,
            extractor=extract_first_two_digits,
            digit_range=np.arange(10, 100, dtype=np.int64),
            expected=first_two_digits_distribution(),
            digit_test=DigitTest.FIRST_TWO,
            alpha=alpha,
        )

    def first_three_digits(self, *, alpha: float = 0.05) -> TestResult:
        """First-three-digits test (K=900)."""
        return self._run_digit_test(
            test_name="first_three_digits",
            data=self.clean_data,
            extractor=extract_first_three_digits,
            digit_range=np.arange(100, 1000, dtype=np.int64),
            expected=first_three_digits_distribution(),
            digit_test=DigitTest.FIRST_THREE,
            alpha=alpha,
        )

    def last_two_digits(self, *, alpha: float = 0.05) -> TestResult:
        """Last-two-digits test (K=100, uniform expectation)."""
        return self._run_digit_test(
            test_name="last_two_digits",
            data=self.clean_data,
            extractor=extract_last_two_digits,
            digit_range=np.arange(0, 100, dtype=np.int64),
            expected=np.full(100, LAST_TWO_DIGITS_EXPECTED, dtype=np.float64),
            digit_test=None,
            alpha=alpha,
        )

    def second_order(
        self,
        *,
        alpha: float = 0.05,
        digits: Literal["first_two", "first_three"] = "first_two",
    ) -> TestResult:
        """Second-order test on successive differences."""
        diffs = second_order_differences(self.clean_data)

        if digits == "first_two":
            return self._run_digit_test(
                test_name="second_order",
                data=diffs,
                extractor=extract_first_two_digits,
                digit_range=np.arange(10, 100, dtype=np.int64),
                expected=first_two_digits_distribution(),
                digit_test=DigitTest.FIRST_TWO,
                alpha=alpha,
            )
        return self._run_digit_test(
            test_name="second_order",
            data=diffs,
            extractor=extract_first_three_digits,
            digit_range=np.arange(100, 1000, dtype=np.int64),
            expected=first_three_digits_distribution(),
            digit_test=DigitTest.FIRST_THREE,
            alpha=alpha,
        )

    # ── summation test ─────────────────────────────────────────────────

    def summation(self, *, alpha: float = 0.05) -> SummationResult:
        """Summation test — sums grouped by first-two digits (Nigrini Ch. 5)."""
        sf: SummationFrequencies = summation_by_digits(self.clean_data)

        z = z_statistic(sf.proportions, sf.expected_proportions, self.n)
        sig = z_significant(z, alpha=alpha)

        # Chi-square from proportions (not integer counts) because the
        # summation test works with value sums, not observation counts.
        chi_sq = float(
            self.n
            * np.sum((sf.proportions - sf.expected_proportions) ** 2 / sf.expected_proportions)
        )
        dof = len(sf.proportions) - 1
        chi_crit = float(_chi2_dist.ppf(1.0 - alpha, dof))

        return SummationResult(
            test_name="summation",
            digits=sf.digits,
            sums=sf.sums,
            observed=sf.proportions,
            expected=sf.expected_proportions,
            z_scores=np.asarray(z, dtype=np.float64),
            significant_flags=np.asarray(sig, dtype=np.bool_),
            chi_square=chi_sq,
            chi_square_critical=chi_crit,
            chi_square_significant=bool(chi_sq > chi_crit),
            grand_sum=sf.grand_sum,
            n=self.n,
            alpha=alpha,
        )

    # ── pass-through tests ─────────────────────────────────────────────

    def distortion_factor(self, *, alpha: float = 0.05) -> DistortionResult:
        """Distortion Factor Model (Nigrini Eq. 6.1-6.10)."""
        return distortion_factor_test(self.clean_data, alpha=alpha)

    def mantissa_arc(self) -> MantissaArcResult:
        """Mantissa Arc Test (Alexander 2009)."""
        return mantissa_arc_test(self.clean_data)

    def number_duplication(self, *, top_n: int = 10) -> DuplicationResult:
        """Number Duplication Test (Nigrini Ch. 6)."""
        return number_duplication(self.clean_data, top_n=top_n)
