# statistics.py — Full Implementation Specification

## Overview

This module contains all statistical test functions for Benford's Law analysis. Every function is a pure, stateless computation — no side effects, no plotting, no I/O. Functions take NumPy arrays of observed and expected proportions (or raw data) and return results as typed dataclasses or named tuples.

**Dependencies:** numpy, scipy.stats (for norm.sf only), and internal imports from `pybenford.constants`.

---

## 1. Z-Statistic (per-digit test)

**Source:** Nigrini (2012), Equation 7.1, adapted from Fleiss (1981).

### Formula

```
Z = (|AP - EP| - 1/(2N)) / sqrt(EP * (1 - EP) / N)
```

Where:
- `AP` = actual (observed) proportion for a digit
- `EP` = expected (Benford) proportion for that digit  
- `N` = total number of records
- `1/(2N)` = continuity correction term (Fleiss). **Only applied when `|AP - EP| > 1/(2N)`**. If the correction would make the numerator negative, set numerator to 0 (Z = 0).

### Function Signature

```python
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
```

### P-value computation (separate helper)

```python
def z_pvalue(z: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Two-tailed p-value from Z-statistics.
    
    Formula: p = 2 * (1 - Phi(|z|))
    where Phi is the standard normal CDF.
    Uses scipy.stats.norm.sf for numerical stability.
    """
```

### Significance classification

```python
def z_significant(
    z: npt.NDArray[np.floating],
    alpha: float = 0.05,
) -> npt.NDArray[np.bool_]:
    """Boolean mask: True where |Z| exceeds the critical value for alpha.
    
    Common alpha values and their critical Z:
    - 0.10 → 1.645
    - 0.05 → 1.960  (default)
    - 0.01 → 2.576
    """
```

### Edge Cases
- If `n <= 0`, raise `ValueError`.
- If `expected` contains zeros, the Z-stat is undefined → return `np.inf` for those bins.
- Both arrays must be same length; raise `ValueError` if not.
- If any proportion is negative, raise `ValueError`.

---

## 2. Chi-Square Test (all-digits-at-once)

**Source:** Standard Pearson chi-square, used in Nigrini Chapter 7.

### Formula

```
chi_sq = sum((observed_count_i - expected_count_i)^2 / expected_count_i)
```

Where:
- `observed_count_i = AP_i * N`
- `expected_count_i = EP_i * N`
- Degrees of freedom = K - 1 (where K = number of bins: 9 for first digit, 10 for second digit, 90 for first-two, 900 for first-three)

### Function Signature

```python
@dataclass(frozen=True)
class ChiSquareResult:
    statistic: float
    critical_value: float
    degrees_of_freedom: int
    p_value: float
    significant: bool  # True if statistic > critical_value

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
```

### Edge Cases
- Expected counts of zero → exclude those bins from calculation or raise warning.
- `n` must equal `sum(observed_counts)`.

---

## 3. Kolmogorov-Smirnov Test

**Source:** Nigrini Chapter 7, standard one-sample K-S test.

### Formula

```
D = max(|CDF_observed_i - CDF_expected_i|)
critical = c_alpha / sqrt(N)
```

Where:
- CDF = cumulative sum of proportions
- `c_alpha` values: 1.2238 (α=0.10), 1.3581 (α=0.05), 1.6276 (α=0.01)

### Function Signature

```python
@dataclass(frozen=True)
class KSResult:
    statistic: float       # D = max absolute CDF difference
    critical_value: float  # c_alpha / sqrt(N)
    significant: bool      # True if statistic > critical_value

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
```

### Edge Cases
- Same validation as Z-statistic (array lengths, non-negative, n > 0).
- The critical values should be stored in `constants.py` (verify they're there from last session).

---

## 4. Mean Absolute Deviation (MAD) — THE KEY TEST

**Source:** Nigrini (2012), Table 7.1. Drake and Nigrini (2000).

### Formula

```
MAD = (1/K) * sum(|AP_i - EP_i|)
```

Where K = number of bins (9, 10, 90, or 900).

### Conformity Classification (Table 7.1 — these are THE critical values)

| Test Type       | Close      | Acceptable | Marginal   | Nonconformity |
|-----------------|------------|------------|------------|---------------|
| First digit     | ≤ 0.006    | ≤ 0.012    | ≤ 0.015    | > 0.015       |
| Second digit    | ≤ 0.008    | ≤ 0.010    | ≤ 0.012    | > 0.012       |
| First-two digit | ≤ 0.0012   | ≤ 0.0018   | ≤ 0.0022   | > 0.0022      |
| First-three digit | ≤ 0.00036 | ≤ 0.00044 | ≤ 0.00050  | > 0.00050     |

### Function Signatures

```python
class ConformityLevel(Enum):
    CLOSE = "close_conformity"
    ACCEPTABLE = "acceptable_conformity"
    MARGINAL = "marginally_acceptable_conformity"
    NONCONFORMITY = "nonconformity"

class DigitTest(Enum):
    FIRST = "first"
    SECOND = "second"
    FIRST_TWO = "first_two"
    FIRST_THREE = "first_three"

@dataclass(frozen=True)
class MADResult:
    mad: float
    conformity: ConformityLevel
    digit_test: DigitTest
    thresholds: dict[str, float]  # the specific thresholds used

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
```

### Edge Cases
- Arrays must be same length.
- For `digit_test=FIRST`, length must be 9. For `SECOND`, 10. For `FIRST_TWO`, 90. For `FIRST_THREE`, 900.
- Validate array lengths match expected bin counts.

---

## 5. Distortion Factor Model

**Source:** Nigrini (2012), Equations 6.1, 6.3, 6.5–6.10.

### Collapsing Numbers (Equation 6.1)

Numbers are "collapsed" to have exactly 2 digits left of the decimal point:
- 6,340 → 63.40
- 0.0245 → 24.5
- 100 → 10.0

Formula: `collapsed = x / 10^(floor(log10(|x|)) - 1)`

**This function already exists in `digits.py` as `collapse_numbers()`.** Use it.

### Expected Mean (Equation 6.7)

```
EM = 90 / ln(10) ≈ 39.0865
```

(The limit as N → ∞. Use this constant rather than computing per-dataset.)

### Distortion Factor (Equation 6.8)

```
DF = (AM - 39.0865) / 39.0865
```

Where AM = actual mean of the collapsed numbers.

- DF > 0 → data appears **overstated**
- DF < 0 → data appears **understated**
- DF ≈ 0 → no apparent distortion

### Statistical Significance (Equations 6.9, 6.10)

```
SD_DF = 0.638253 / sqrt(N)
Z = DF / SD_DF
```

- |Z| > 1.96 → significant at 0.05 level
- |Z| > 2.57 → significant at 0.01 level

### Function Signature

```python
@dataclass(frozen=True)
class DistortionResult:
    distortion_factor: float    # DF value
    actual_mean: float          # AM of collapsed numbers
    expected_mean: float        # 39.0865
    z_statistic: float          # Z = DF / SD
    p_value: float              # two-tailed p-value
    significant: bool           # at alpha=0.05
    direction: str              # "overstated", "understated", or "neutral"
    percentage: float           # DF * 100 (extent of distortion as %)

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
```

### Edge Cases
- Exclude values where |x| < 10 (no valid first-two digits).
- Exclude zeros, NaN, inf.
- If no valid values remain after filtering, raise `ValueError`.
- The constant 39.0865 should be in `constants.py` as `EXPECTED_MEAN_COLLAPSED`.
- The constant 0.638253 should be in `constants.py` as `DISTORTION_FACTOR_SD_COEFF`.

---

## 6. Sum of Squared Differences (SSD) — optional extra

Not in the original notebook, but useful and simple:

```
SSD = sum((AP_i - EP_i)^2)
```

```python
def sum_squared_differences(
    observed: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
) -> float:
    """Sum of squared differences between observed and expected proportions."""
```

---

## 7. Mantissa Arc Test Statistics

**Source:** Nigrini Chapter 7, referencing Alexander (2009).

The mantissa of log10(x) should be uniformly distributed on [0, 1). Each number maps to a point on the unit circle: angle = 2π × mantissa.

The test checks whether the center of mass of these points is close to the origin (0, 0).

```python
@dataclass(frozen=True)
class MantissaArcResult:
    mean_angle: float          # atan2(mean_y, mean_x) — direction of the gravity center (amended in Pass 2; original definition was the arithmetic mean)
    mean_x: float              # mean of cos(2*pi*mantissa)
    mean_y: float              # mean of sin(2*pi*mantissa)
    L2: float                  # sqrt(mean_x^2 + mean_y^2) — distance from origin
    p_value: float             # significance (based on N and L2)

def mantissa_arc_test(
    data: npt.NDArray[np.floating],
) -> MantissaArcResult:
    """Mantissa arc test for Benford conformity.
    
    Notes
    -----
    Nigrini warns this test is "very sensitive" — even data that passes
    all other tests may fail this one. Consider using sqrt(N) or
    cubed root of N adjustments for large datasets.
    """
```

---

## Constants to verify/add in constants.py

Make sure these exist (they may already be there from last session):

```python
# K-S critical values
KS_CRITICAL = {0.10: 1.2238, 0.05: 1.3581, 0.01: 1.6276}

# Distortion factor constants  
EXPECTED_MEAN_COLLAPSED = 39.0865  # 90 / ln(10)
DISTORTION_FACTOR_SD_COEFF = 0.638253  # from Nigrini (1996)

# MAD thresholds (Table 7.1)
MAD_THRESHOLDS = {
    "first": {"close": 0.006, "acceptable": 0.012, "marginal": 0.015},
    "second": {"close": 0.008, "acceptable": 0.010, "marginal": 0.012},
    "first_two": {"close": 0.0012, "acceptable": 0.0018, "marginal": 0.0022},
    "first_three": {"close": 0.00036, "acceptable": 0.00044, "marginal": 0.00050},
}
```

---

## Testing Requirements

For each function, tests should cover:

1. **Known-good Benford data** — generate a perfect Benford set, verify Z ≈ 0, MAD = "close conformity", chi-sq not significant
2. **Known-bad data** — uniform distribution should fail all tests
3. **Edge cases** — N=1, N=10, N=1_000_000, all-same-digit data
4. **Numerical verification** — replicate Nigrini's worked examples:
   - Census data: first-two digit MAD = 0.0006 (close conformity)
   - Census data: Z-stat for digit 32 = 2.260 (N=19,482, AP=0.0152, EP=0.0134)
   - Census data distortion: DF = 0.0074, Z = 1.6185 (not significant)
5. **Continuity correction** — verify correction is skipped when |AP-EP| < 1/(2N)
6. **Type safety** — mypy clean, proper input validation

---

## Module Structure

```python
"""Statistical tests for Benford's Law conformity analysis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm

from pybenford.constants import (
    MAD_THRESHOLDS,
    KS_CRITICAL,
    EXPECTED_MEAN_COLLAPSED,
    DISTORTION_FACTOR_SD_COEFF,
)

# Enums and dataclasses first
# Then individual test functions
# Then composite/convenience functions if any
```
