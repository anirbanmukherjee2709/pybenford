"""Statistical tests for Benford's Law conformity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, norm

from pybenford.constants import (
    CRIT_KS,
    DF_EXPECTED_MEAN,
    DF_SD_CONSTANT,
    MAD_CONFORM,
    NUM_BINS,
)
from pybenford.digits import collapse_numbers, extract_mantissa

__all__ = [
    "ChiSquareResult",
    "ConformityLevel",
    "DigitTest",
    "DistortionResult",
    "KSResult",
    "MADResult",
    "MantissaArcResult",
    "chi_square_test",
    "distortion_factor_test",
    "ks_test",
    "mad_test",
    "mantissa_arc_test",
    "sum_squared_differences",
    "z_pvalue",
    "z_significant",
    "z_statistic",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConformityLevel(Enum):
    """MAD conformity classification (Nigrini Table 7.1)."""

    CLOSE = "close_conformity"
    ACCEPTABLE = "acceptable_conformity"
    MARGINAL = "marginally_acceptable_conformity"
    NONCONFORMITY = "nonconformity"


class DigitTest(Enum):
    """Digit-level test identifiers."""

    FIRST = "first"
    SECOND = "second"
    FIRST_TWO = "first_two"
    FIRST_THREE = "first_three"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChiSquareResult:
    """Result of a Pearson chi-square goodness-of-fit test."""

    statistic: float
    critical_value: float
    degrees_of_freedom: int
    p_value: float
    significant: bool


@dataclass(frozen=True)
class KSResult:
    """Result of a Kolmogorov-Smirnov conformity test."""

    statistic: float
    critical_value: float
    significant: bool


@dataclass(frozen=True)
class MADResult:
    """Result of the Mean Absolute Deviation conformity test."""

    mad: float
    conformity: ConformityLevel
    digit_test: DigitTest
    thresholds: dict[str, float]


@dataclass(frozen=True)
class DistortionResult:
    """Result of Nigrini's Distortion Factor Model."""

    distortion_factor: float
    actual_mean: float
    expected_mean: float
    z_statistic: float
    p_value: float
    significant: bool
    direction: str
    percentage: float

    def __str__(self) -> str:
        w = 55
        border = "=" * w
        lines: list[str] = [border, "  Distortion Factor Test", border]

        direction_label = ""
        if self.direction == "understated":
            direction_label = " (Understated)"
        elif self.direction == "overstated":
            direction_label = " (Overstated)"

        lines.append(
            f" Distortion Factor:  {self.distortion_factor:.4f}{direction_label}"
        )
        lines.append(f" Actual Mean:         {self.actual_mean:.4f}")
        lines.append(f" Expected Mean:       {self.expected_mean:.4f}")
        lines.append(
            f" Z-Statistic:         {self.z_statistic:.4f}"
            f"  (p={self.p_value:.4f})"
        )
        sig_label = "Yes" if self.significant else "No"
        lines.append(f" Significant:         {sig_label}")
        lines.append(border)
        return "\n".join(lines)


@dataclass(frozen=True)
class MantissaArcResult:
    """Result of the Mantissa Arc Test."""

    mean_angle: float
    mean_x: float
    mean_y: float
    L2: float
    p_value: float

    def __str__(self) -> str:
        import math

        w = 55
        border = "=" * w
        lines: list[str] = [border, "  Mantissa Arc Test", border]
        lines.append(
            f" Gravity Center:  ({self.mean_x:.4f}, {self.mean_y:.4f})"
        )
        lines.append(f" L2 Statistic:     {self.L2:.4f}")
        deg = self.mean_angle * 180.0 / math.pi
        lines.append(
            f" Mean Angle:        {self.mean_angle:.4f} rad"
            f"  ({deg:.2f} deg)"
        )
        lines.append(f" P-Value:           {self.p_value:.4f}")
        lines.append(border)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_proportions(
    observed: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    n: int,
) -> None:
    """Shared input checks for proportion-based tests."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(observed) != len(expected):
        raise ValueError(
            f"observed and expected must have same length, "
            f"got {len(observed)} and {len(expected)}"
        )
    if np.any(np.asarray(observed) < 0) or np.any(np.asarray(expected) < 0):
        raise ValueError("proportions must be non-negative")


# ---------------------------------------------------------------------------
# 1. Z-Statistic (per-digit test) — Nigrini Eq. 7.1
# ---------------------------------------------------------------------------


def z_statistic(
    observed: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    n: int,
    *,
    continuity_correction: bool = True,
) -> npt.NDArray[np.floating]:
    """Compute per-digit Z-statistics.

    Parameters
    ----------
    observed : array of observed proportions (must sum to ~1.0)
    expected : array of expected (Benford) proportions (must sum to ~1.0)
    n : total number of records in the dataset
    continuity_correction : if True (default), apply Fleiss 1/(2N) correction

    Returns
    -------
    Array of Z-statistics, one per digit/bin.
    """
    _validate_proportions(observed, expected, n)

    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)

    diff = np.abs(obs - exp)
    correction = 1.0 / (2 * n) if continuity_correction else 0.0
    numerator = np.where(diff > correction, diff - correction, 0.0)

    denom_sq = exp * (1.0 - exp) / n
    denom = np.sqrt(np.maximum(denom_sq, 0.0))

    safe_denom = np.where(denom > 0, denom, 1.0)
    raw = numerator / safe_denom

    result: npt.NDArray[np.floating] = np.where(
        exp == 0.0,
        np.inf,
        np.where(denom > 0, raw, 0.0),
    )
    return result


def z_pvalue(
    z: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Two-tailed p-value from Z-statistics.

    Formula: p = 2 * (1 - Phi(|z|))
    where Phi is the standard normal CDF.
    Uses scipy.stats.norm.sf for numerical stability.
    """
    p: npt.NDArray[np.floating] = 2.0 * norm.sf(np.abs(z))
    return p


def z_significant(
    z: npt.NDArray[np.floating],
    alpha: float = 0.05,
) -> npt.NDArray[np.bool_]:
    """Boolean mask: True where |Z| exceeds the critical value for alpha.

    Common alpha values and their critical Z:
    - 0.10 -> 1.645
    - 0.05 -> 1.960  (default)
    - 0.01 -> 2.576
    """
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    sig: npt.NDArray[np.bool_] = np.abs(z) > critical
    return sig


# ---------------------------------------------------------------------------
# 2. Chi-Square Test — Nigrini Ch. 7
# ---------------------------------------------------------------------------


def chi_square_test(
    observed_counts: npt.NDArray[np.integer],
    expected_proportions: npt.NDArray[np.floating],
    n: int,
    *,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Pearson chi-square goodness-of-fit test.

    Parameters
    ----------
    observed_counts : array of actual counts per digit/bin
    expected_proportions : array of Benford expected proportions
    n : total records (sum of observed_counts)
    alpha : significance level (default 0.05)

    Notes
    -----
    Nigrini warns: chi-square suffers from "excess power" for large N.
    With N > 25,000, even trivial deviations are flagged as significant.
    Prefer MAD for large datasets.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(observed_counts) != len(expected_proportions):
        raise ValueError(
            "observed_counts and expected_proportions must have same length"
        )

    obs = np.asarray(observed_counts, dtype=np.float64)
    exp_prop = np.asarray(expected_proportions, dtype=np.float64)

    obs_sum = int(np.sum(observed_counts))
    if obs_sum != n:
        raise ValueError(f"sum(observed_counts)={obs_sum} must equal n={n}")

    expected_counts = exp_prop * n

    valid = expected_counts > 0
    chi_sq = float(
        np.sum((obs[valid] - expected_counts[valid]) ** 2 / expected_counts[valid])
    )

    dof = len(observed_counts) - 1
    p_value = float(chi2.sf(chi_sq, dof))
    critical_value = float(chi2.ppf(1.0 - alpha, dof))

    return ChiSquareResult(
        statistic=chi_sq,
        critical_value=critical_value,
        degrees_of_freedom=dof,
        p_value=p_value,
        significant=chi_sq > critical_value,
    )


# ---------------------------------------------------------------------------
# 3. Kolmogorov-Smirnov Test — Nigrini Ch. 7
# ---------------------------------------------------------------------------


def ks_test(
    observed: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    n: int,
    *,
    alpha: float = 0.05,
) -> KSResult:
    """Kolmogorov-Smirnov test for Benford conformity.

    Notes
    -----
    Like chi-square, K-S suffers from excess power for large N.
    Prefer MAD for large datasets.
    """
    _validate_proportions(observed, expected, n)

    cdf_obs = np.cumsum(np.asarray(observed, dtype=np.float64))
    cdf_exp = np.cumsum(np.asarray(expected, dtype=np.float64))
    d_stat = float(np.max(np.abs(cdf_obs - cdf_exp)))

    conf = (1.0 - alpha) * 100.0
    if conf not in CRIT_KS:
        raise ValueError(
            f"alpha={alpha} (confidence={conf}%) not supported; "
            f"available: {sorted(CRIT_KS.keys())}"
        )
    c_alpha = CRIT_KS[conf]
    crit = c_alpha / np.sqrt(n)

    return KSResult(
        statistic=d_stat,
        critical_value=float(crit),
        significant=bool(d_stat > crit),
    )


# ---------------------------------------------------------------------------
# 4. MAD Test — Nigrini Table 7.1
# ---------------------------------------------------------------------------


def mad_test(
    observed: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    digit_test: DigitTest,
) -> MADResult:
    """Mean Absolute Deviation conformity test.

    This is the PREFERRED conformity measure per Nigrini.
    Unlike chi-square and K-S, MAD does NOT penalize large N.

    Parameters
    ----------
    observed : array of observed proportions
    expected : array of expected proportions
    digit_test : which test type (determines critical value thresholds)
    """
    if len(observed) != len(expected):
        raise ValueError(
            f"observed and expected must have same length, "
            f"got {len(observed)} and {len(expected)}"
        )

    expected_bins = NUM_BINS[digit_test.value]
    if len(observed) != expected_bins:
        raise ValueError(
            f"{digit_test.value} test expects {expected_bins} bins, "
            f"got {len(observed)}"
        )

    k = len(observed)
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    mad_value = float(np.sum(np.abs(obs - exp)) / k)

    close, acceptable, marginal = MAD_CONFORM[digit_test.value]

    if mad_value <= close:
        conformity = ConformityLevel.CLOSE
    elif mad_value <= acceptable:
        conformity = ConformityLevel.ACCEPTABLE
    elif mad_value <= marginal:
        conformity = ConformityLevel.MARGINAL
    else:
        conformity = ConformityLevel.NONCONFORMITY

    thresholds = {
        "close": close,
        "acceptable": acceptable,
        "marginal": marginal,
    }

    return MADResult(
        mad=mad_value,
        conformity=conformity,
        digit_test=digit_test,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# 5. Distortion Factor Model — Nigrini Eq. 6.1–6.10
# ---------------------------------------------------------------------------


def distortion_factor_test(
    data: npt.NDArray[np.floating],
    *,
    alpha: float = 0.05,
) -> DistortionResult:
    """Nigrini's Distortion Factor Model.

    Tests whether data appears systematically overstated or understated.

    Parameters
    ----------
    data : raw numeric data (NOT proportions). Values < 10 are excluded.
          Negative values use absolute value.

    Notes
    -----
    Assumes manipulation occurs within the same order of magnitude
    (Uniform Percentage Error model). Most affected by first-two digits.
    The SD approximation (0.638253/sqrt(N)) is from Nigrini (1996).
    """
    abs_data = np.abs(np.asarray(data, dtype=np.float64))
    valid = np.isfinite(abs_data) & (abs_data >= 10.0)
    filtered = abs_data[valid]

    if len(filtered) == 0:
        raise ValueError("no valid values remain after filtering (|x| >= 10, finite)")

    collapsed = collapse_numbers(filtered)
    am = float(np.mean(collapsed))
    em = DF_EXPECTED_MEAN
    df_val = (am - em) / em

    n_valid = len(filtered)
    sd = DF_SD_CONSTANT / np.sqrt(n_valid)
    z = df_val / sd
    p_val = float(2.0 * norm.sf(np.abs(z)))

    critical_z = float(norm.ppf(1.0 - alpha / 2.0))
    significant = bool(np.abs(z) > critical_z)

    if df_val > 0:
        direction = "overstated"
    elif df_val < 0:
        direction = "understated"
    else:
        direction = "neutral"

    return DistortionResult(
        distortion_factor=df_val,
        actual_mean=am,
        expected_mean=em,
        z_statistic=float(z),
        p_value=p_val,
        significant=significant,
        direction=direction,
        percentage=df_val * 100.0,
    )


# ---------------------------------------------------------------------------
# 6. Sum of Squared Differences
# ---------------------------------------------------------------------------


def sum_squared_differences(
    observed: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
) -> float:
    """Sum of squared differences between observed and expected proportions."""
    if len(observed) != len(expected):
        raise ValueError(
            f"observed and expected must have same length, "
            f"got {len(observed)} and {len(expected)}"
        )
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    return float(np.sum((obs - exp) ** 2))


# ---------------------------------------------------------------------------
# 7. Mantissa Arc Test — Alexander (2009), Nigrini Ch. 7
# ---------------------------------------------------------------------------


def mantissa_arc_test(
    data: npt.NDArray[np.floating],
) -> MantissaArcResult:
    """Mantissa arc test for Benford conformity.

    Notes
    -----
    Nigrini warns this test is "very sensitive" -- even data that passes
    all other tests may fail this one. Consider using sqrt(N) or
    cubed root of N adjustments for large datasets.
    """
    mantissas = extract_mantissa(np.asarray(data, dtype=np.float64))
    valid_mask: npt.NDArray[np.bool_] = ~np.isnan(mantissas)
    m = mantissas[valid_mask]

    if len(m) == 0:
        raise ValueError("no valid values for mantissa computation")

    n = len(m)
    angles = 2.0 * np.pi * m
    mean_x = float(np.mean(np.cos(angles)))
    mean_y = float(np.mean(np.sin(angles)))
    mean_angle = float(np.mean(angles))
    l2 = float(np.sqrt(mean_x**2 + mean_y**2))
    p_value = float(np.exp(-n * l2**2))

    return MantissaArcResult(
        mean_angle=mean_angle,
        mean_x=mean_x,
        mean_y=mean_y,
        L2=l2,
        p_value=p_value,
    )
