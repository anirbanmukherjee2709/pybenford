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
    "CRIT_KS",
    "DF_EXPECTED_MEAN",
    "DF_SD_CONSTANT",
    "FIRST_DIGIT_PROBS",
    "FOURTH_DIGIT_PROBS",
    "LARGE_SAMPLE_THRESHOLD",
    "LAST_TWO_DIGITS_EXPECTED",
    "MAD_CONFORM",
    "MIN_SAMPLE_SIZE_FIRST_TWO",
    "MIN_SAMPLE_SIZE_RECOMMENDED",
    "NUM_BINS",
    "SECOND_DIGIT_PROBS",
    "SUMMATION_EXPECTED",
    "THIRD_DIGIT_PROBS",
]


# ---------------------------------------------------------------------------
# Bin counts
# ---------------------------------------------------------------------------

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
