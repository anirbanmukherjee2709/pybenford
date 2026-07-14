"""Statistical constants for Benford's Law analysis.

Every fixed value used across ``pybenford`` lives here so that numeric
literals do not drift between modules. All thresholds, critical values,
and expected distributions are sourced from:

- Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic
  Accounting, Auditing, and Fraud Detection*. Wiley.
- Miller, S.J. (2015). *Benford's Law: Theory and Applications*.
  Princeton University Press.
- Kossovsky, A.E. (2014). *Benford's Law: Theory, The General Law of
  Relative Quantities, and Forensic Fraud Detection Applications*.
  World Scientific.
- Fleiss, J.L. (1981). *Statistical Methods for Rates and Proportions*,
  2nd ed. Wiley (continuity correction for the per-digit Z test).

Cross-referenced against ``docs/reference.md`` and the reference
implementation in ``examples/Benford's Law Analysis_v4.ipynb``.

Naming conventions
------------------
Test identifiers used as dict keys throughout the package:

- ``"first"``        — first-digit test (K=9,  df=8)
- ``"second"``       — second-digit test (K=10, df=9)
- ``"third"``        — third-digit test (K=10, df=9)
- ``"first_two"``    — first-two-digits test (K=90,  df=89)
- ``"first_three"``  — first-three-digits test (K=900, df=899)
- ``"last_two"``     — last-two-digits test (K=100, df=99)

Confidence-level keys are percentage points (e.g. ``95`` for 95%,
``99.9`` for 99.9%). Python hashes ``95 == 95.0`` to the same slot, so
callers may use either numeric form.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "COLORS",
    "CONFS",
    "CRIT_CHI2",
    "CRIT_KS",
    "CRIT_Z",
    "DF_EXPECTED_MEAN",
    "DF_SD_CONSTANT",
    "DOF",
    "FIRST_DIGIT_PROBS",
    "FOURTH_DIGIT_PROBS",
    "LARGE_SAMPLE_THRESHOLD",
    "LAST_TWO_DIGITS_EXPECTED",
    "MAD_CONFORM",
    "MAD_CONFORM_MY_LAW",
    "MIN_SAMPLE_SIZE_FIRST_TWO",
    "MIN_SAMPLE_SIZE_RECOMMENDED",
    "NUM_BINS",
    "P_VALUES",
    "SECOND_DIGIT_PROBS",
    "SUMMATION_EXPECTED",
    "TEST_NAMES",
    "THIRD_DIGIT_PROBS",
]


# ---------------------------------------------------------------------------
# Test identifiers, bin counts, and degrees of freedom
# ---------------------------------------------------------------------------

TEST_NAMES: Final[tuple[str, ...]] = (
    "first",
    "second",
    "third",
    "first_two",
    "first_three",
    "last_two",
)
"""Canonical identifiers for every digit-level test supported."""

NUM_BINS: Final[dict[str, int]] = {
    "first": 9,
    "second": 10,
    "third": 10,
    "first_two": 90,
    "first_three": 900,
    "last_two": 100,
}
"""Number of histogram bins (``K``) for each test.

Used as the divisor in the MAD formula ``MAD = (1/K) * sum|AP - EP|``
(Nigrini Eq. 2.4).
"""

DOF: Final[dict[str, int]] = {
    "first": 8,
    "second": 9,
    "third": 9,
    "first_two": 89,
    "first_three": 899,
    "last_two": 99,
}
"""Degrees of freedom (``K - 1``) for the chi-square test of each test."""


# ---------------------------------------------------------------------------
# Expected digit proportions (Nigrini Table 1.2)
# ---------------------------------------------------------------------------

FIRST_DIGIT_PROBS: Final[tuple[float, ...]] = (
    0.30103,  # d=1
    0.17609,  # d=2
    0.12494,  # d=3
    0.09691,  # d=4
    0.07918,  # d=5
    0.06695,  # d=6
    0.05799,  # d=7
    0.05115,  # d=8
    0.04576,  # d=9
)
"""First-digit expected proportions for ``d = 1..9`` (Nigrini Eq. 1.1).

Computed as ``log10(1 + 1/d)``. Index ``i`` corresponds to digit
``i + 1`` (there is no leading-zero case).
"""

SECOND_DIGIT_PROBS: Final[tuple[float, ...]] = (
    0.11968,  # d=0
    0.11389,  # d=1
    0.10882,  # d=2
    0.10433,  # d=3
    0.10031,  # d=4
    0.09668,  # d=5
    0.09337,  # d=6
    0.09035,  # d=7
    0.08757,  # d=8
    0.08500,  # d=9
)
"""Unconditional second-digit proportions for ``d = 0..9`` (Nigrini Eq. 1.2).

Computed as ``sum over d1=1..9 of log10(1 + 1/(10*d1 + d2))``.
"""

THIRD_DIGIT_PROBS: Final[tuple[float, ...]] = (
    0.10178,  # d=0
    0.10138,  # d=1
    0.10097,  # d=2
    0.10057,  # d=3
    0.10018,  # d=4
    0.09979,  # d=5
    0.09940,  # d=6
    0.09902,  # d=7
    0.09864,  # d=8
    0.09827,  # d=9
)
"""Unconditional third-digit proportions for ``d = 0..9`` (Nigrini Table 1.2)."""

FOURTH_DIGIT_PROBS: Final[tuple[float, ...]] = (
    0.10018,  # d=0
    0.10014,  # d=1
    0.10010,  # d=2
    0.10006,  # d=3
    0.10002,  # d=4
    0.09998,  # d=5
    0.09994,  # d=6
    0.09990,  # d=7
    0.09986,  # d=8
    0.09982,  # d=9
)
"""Unconditional fourth-digit proportions for ``d = 0..9`` (Nigrini Table 1.2).

From the fourth position onward, digits are essentially uniform at 0.1,
so no separate tables are stored for later positions.
"""

SUMMATION_EXPECTED: Final[float] = 1.0 / 90.0
"""Expected proportion per first-two-digit bin in the Summation Test.

For a perfect Benford set starting at an integer power of 10, the sum of
values within each first-two-digit bin is uniform at ``1/90 ≈ 0.01111``
(Nigrini Ch. 3 & 5).
"""

LAST_TWO_DIGITS_EXPECTED: Final[float] = 0.01
"""Expected uniform proportion for each last-two-digit pair (``1/100``).

The last-two-digits test is not a Benford distribution test but a
complementary forensic check for rounding, fabrication, or threshold
effects (Nigrini §3.5).
"""


# ---------------------------------------------------------------------------
# Chi-square critical values, keyed by degrees of freedom then confidence
# ---------------------------------------------------------------------------

CRIT_CHI2: Final[dict[int, dict[float, float]]] = {
    8: {
        80: 11.030,
        85: 12.027,
        90: 13.362,
        95: 15.507,
        99: 20.090,
        99.9: 26.124,
        99.99: 31.827,
        99.999: 37.332,
        99.9999: 42.701,
        99.99999: 47.972,
    },
    9: {
        80: 12.242,
        85: 13.288,
        90: 14.684,
        95: 16.919,
        99: 21.666,
        99.9: 27.877,
        99.99: 33.720,
        99.999: 39.341,
        99.9999: 44.811,
        99.99999: 50.172,
    },
    89: {
        80: 99.991,
        85: 102.826,
        90: 106.469,
        95: 112.022,
        99: 122.942,
        99.9: 135.978,
        99.99: 147.350,
        99.999: 157.702,
        99.9999: 167.348,
        99.99999: 176.471,
    },
    99: {
        80: 110.607,
        85: 113.585,
        90: 117.407,
        95: 123.225,
        99: 134.642,
        99.9: 148.230,
        99.99: 160.056,
        99.999: 170.798,
        99.9999: 180.792,
        99.99999: 190.230,
    },
    899: {
        80: 934.479,
        85: 942.981,
        90: 953.752,
        95: 969.865,
        99: 1000.575,
        99.9: 1035.753,
        99.99: 1065.314,
        99.999: 1091.422,
        99.9999: 1115.141,
        99.99999: 1137.082,
    },
}
"""Chi-square critical values: ``CRIT_CHI2[dof][confidence]``.

Reject Benford conformity when the observed chi-square statistic exceeds
the critical value at the chosen confidence level. For large ``N`` the
chi-square test suffers from "excess power" (Nigrini §7) — prefer MAD as
the primary conformity measure and use chi-square for sanity checks only.
"""


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov critical multipliers: K-S critical = C / sqrt(N)
# ---------------------------------------------------------------------------

CRIT_KS: Final[dict[float, float]] = {
    80: 1.073,
    85: 1.138,
    90: 1.224,
    95: 1.358,
    99: 1.628,
    99.9: 1.949,
    99.99: 2.225,
    99.999: 2.470,
    99.9999: 2.693,
    99.99999: 2.899,
}
"""Numerators ``C`` for the K-S critical value ``D_crit = C / sqrt(N)``.

Reject conformity when the observed supremum ``D = max|F_obs - F_exp|``
exceeds ``D_crit``. Same large-``N`` excess-power caveat as chi-square.
"""


# ---------------------------------------------------------------------------
# Per-digit Z-test critical values and corresponding p-values
# ---------------------------------------------------------------------------

CRIT_Z: Final[dict[float, float]] = {
    80: 1.285,
    85: 1.435,
    90: 1.645,
    95: 1.960,
    99: 2.576,
    99.9: 3.290,
    99.99: 3.890,
    99.999: 4.417,
    99.9999: 4.892,
    99.99999: 5.327,
}
"""Two-sided Z critical values keyed by confidence percentage.

Applied to the Nigrini Eq. 7.1 per-digit Z statistic

    Z = (|AP - EP| - 1/(2N)) / sqrt(EP*(1-EP)/N)

where the ``1/(2N)`` continuity correction (Fleiss 1981) is included
only when ``1/(2N) < |AP - EP|``; otherwise the numerator is set to 0.
Z suffers from excess power for large ``N`` — use it for per-digit
drill-down, not overall conformity.
"""

CONFS: Final[dict[float, float]] = CRIT_Z
"""Backwards-compatible alias for :data:`CRIT_Z` (notebook nomenclature)."""

P_VALUES: Final[dict[float, float]] = {
    80: 0.20,
    85: 0.15,
    90: 0.10,
    95: 0.05,
    99: 0.01,
    99.9: 0.001,
    99.99: 0.0001,
    99.999: 0.00001,
    99.9999: 0.000001,
    99.99999: 0.0000001,
}
"""Two-sided p-values corresponding to each confidence level in :data:`CRIT_Z`."""


# ---------------------------------------------------------------------------
# MAD conformity thresholds (Nigrini Table 7.1)
# ---------------------------------------------------------------------------

MAD_CONFORM: Final[dict[str, tuple[float, float, float]]] = {
    "first": (0.006, 0.012, 0.015),
    "second": (0.008, 0.010, 0.012),
    "first_two": (0.0012, 0.0018, 0.0022),
    "first_three": (0.00036, 0.00044, 0.00050),
}
"""Upper bounds of MAD conformity zones per test (Nigrini Table 7.1).

Each tuple is ``(close, acceptable, marginal)`` — the inclusive upper
bound of the Close, Acceptable, and Marginally Acceptable zones. MAD
strictly greater than the third value indicates Nonconformity:

- ``MAD <= close``                → Close Conformity
- ``close   < MAD <= acceptable`` → Acceptable Conformity
- ``acceptable < MAD <= marginal`` → Marginally Acceptable Conformity
- ``MAD > marginal``              → Nonconformity

MAD is the preferred conformity measure because it does not scale with
``N``, so it avoids the excess-power problem that chi-square and K-S
exhibit at large sample sizes (Nigrini §7, particularly for ``N >
25,000``). No MAD thresholds exist for the last-two-digits test, which
is a uniformity test rather than a Benford conformity test.
"""

MAD_CONFORM_MY_LAW: Final[dict[str, tuple[float, float, float]]] = {
    "first_two": (0.0024, 0.0036, 0.0044),
}
"""MAD thresholds for period-over-period "My Law" comparison (Nigrini Table 13.1).

Compares a period's first-two-digit distribution against the same
entity's historical distribution (rather than the Benford distribution),
which is useful when a series is structurally non-Benford but stable
over time.
"""


# ---------------------------------------------------------------------------
# Distortion Factor Model (Nigrini Ch. 6)
# ---------------------------------------------------------------------------

DF_EXPECTED_MEAN: Final[float] = 39.0865
"""Expected mean of collapsed values under Benford (Nigrini Eq. 6.7).

Equal to ``90 / ln(10)``. Used as the baseline in the distortion factor
``DF = (AM - 39.0865) / 39.0865`` (Eq. 6.8), where ``AM`` is the mean of
numbers collapsed to two digits left of the decimal. Positive DF
indicates overstatement, negative indicates understatement.
"""

DF_SD_CONSTANT: Final[float] = 0.638253
"""Numerator of the distortion-factor standard deviation (Nigrini Eq. 6.9).

Scales to ``SD_DF = 0.638253 / sqrt(N)``; the corresponding Z statistic
is ``Z = DF / SD_DF`` (Eq. 6.10).
"""


# ---------------------------------------------------------------------------
# Sample size guidance (Nigrini §4.2)
# ---------------------------------------------------------------------------

MIN_SAMPLE_SIZE_RECOMMENDED: Final[int] = 1000
"""Recommended minimum ``N`` for a reliable Benford analysis."""

MIN_SAMPLE_SIZE_FIRST_TWO: Final[int] = 300
"""Minimum ``N`` below which the first-two-digits test is unreliable.

When the effective sample analyzed by the first-two-digits,
first-three-digits, second-order, or summation test falls below 300,
the package emits a :class:`~pybenford.core.SmallSampleWarning`, per
Nigrini's guidance.
"""

LARGE_SAMPLE_THRESHOLD: Final[int] = 25_000
"""Sample size above which chi-square and K-S lose usefulness.

For ``N > 25,000`` both tests exhibit excess power and reject
near-perfect conformity. The MAD-based conformity assessment in
:data:`MAD_CONFORM` should drive conclusions beyond this threshold.
"""


# ---------------------------------------------------------------------------
# Plot color palette (matches the reference notebook)
# ---------------------------------------------------------------------------

COLORS: Final[dict[str, str]] = {
    "m": "#00798c",  # main / observed
    "b": "#E2DCD8",  # background
    "s": "#9c3848",  # secondary / expected line
    "af": "#edae49",  # alert fill (non-conforming digit)
    "ab": "#33658a",  # alert border
    "h": "#d1495b",  # highlight
    "h2": "#f64740",  # highlight alt
    "t": "#16DB93",  # tertiary / success
}
"""Hex color palette shared by every plotting utility.

Keys match the notebook reference implementation so existing styling
code transfers without remapping.
"""
