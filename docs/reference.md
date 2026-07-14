# Technical Reference: Benford's Law — Complete Implementation Specification

> Comprehensive reference for implementing `pybenford`. Every formula, threshold, test
> specification, data requirement, and edge case documented below is extracted from:
>
> - **Nigrini (2012)** — *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection* (Wiley)
> - **Miller (2015)** — *Benford's Law: Theory and Applications* (Princeton University Press)
> - **Kossovsky (2014)** — *Benford's Law: Theory, The General Law of Relative Quantities, and Forensic Fraud Detection Applications* (World Scientific)
> - Research papers in project knowledge base
>
> This file lives at `docs/reference.md` and is read by Claude Code during implementation.

---

## 1. Expected Digit Distributions

### 1.1 General Significant Digit Law (Nigrini Eq. 1.21, Hill 1995)

The master formula for ANY digit combination:

```
P(D1=d1, D2=d2, ..., Dk=dk) = log10(1 + 1 / (sum of di * 10^(k-i) for i=1..k))
```

Where `d1 ∈ {1,...,9}` and `dj ∈ {0,...,9}` for j = 2,...,k.

This single formula covers all tests below. The denominator is the integer formed by
concatenating the digits.

### 1.2 First Digit (d = 1..9) — Nigrini Eq. 1.1

```
P(D1 = d) = log10(1 + 1/d)
```

| Digit | Probability |
|-------|------------|
| 1 | 0.30103 |
| 2 | 0.17609 |
| 3 | 0.12494 |
| 4 | 0.09691 |
| 5 | 0.07918 |
| 6 | 0.06695 |
| 7 | 0.05799 |
| 8 | 0.05115 |
| 9 | 0.04576 |

K (number of bins) = 9. Degrees of freedom for chi-square = 8.

### 1.3 Second Digit (d2 = 0..9) — Nigrini Eq. 1.2

```
P(D2 = d2) = sum over d1=1..9 of log10(1 + 1/(d1*10 + d2))
```

| Digit | Probability |
|-------|------------|
| 0 | 0.11968 |
| 1 | 0.11389 |
| 2 | 0.10882 |
| 3 | 0.10433 |
| 4 | 0.10031 |
| 5 | 0.09668 |
| 6 | 0.09337 |
| 7 | 0.09035 |
| 8 | 0.08757 |
| 9 | 0.08500 |

K = 10. Degrees of freedom for chi-square = 9.

**Conditional second digit** (Nigrini Eq. 4.1): P(D2=d2 | D1=d1) = log10(1 + 1/(10*d1 + d2)) / log10(1 + 1/d1).
The largest bias toward low digits occurs when d1=1; smallest bias when d1=9.

### 1.4 First-Two Digits (d = 10..99) — Nigrini Eq. 1.3

```
P(D1D2 = d) = log10(1 + 1/d)
```

K = 90. Degrees of freedom for chi-square = 89.
This is the primary test recommended by Nigrini — more informative than first-digit alone.

### 1.5 First-Three Digits (d = 100..999)

```
P(D1D2D3 = d) = log10(1 + 1/d)
```

K = 900. Degrees of freedom for chi-square = 899.

### 1.6 Third and Fourth Digit Probabilities (Nigrini Table 1.2)

Third digit proportions range from 0.10178 (digit 0) to 0.09827 (digit 9).
Fourth digit proportions range from 0.10018 (digit 0) to 0.09982 (digit 9).
From the fourth digit onward, digits are essentially uniformly distributed for practical purposes.

### 1.7 Last-Two Digits (d = 00..99)

Expected proportion: **uniform at 1/100 = 0.01** for each combination.
K = 100.

This is NOT a Benford distribution test but a complementary forensic test.

### 1.8 Summation Test Expected Proportion

For a perfect Benford Set starting at an integer power of 10:
```
EP = 1/90 ≈ 0.01111  (uniform across all 90 first-two digit bins)
```

---

## 2. Conformity Assessment Tests

### 2.1 MAD — Mean Absolute Deviation (PREFERRED measure)

**Nigrini Eq. 2.4:**
```
MAD = (1/K) * sum(|AP_i - EP_i|) for i = 1..K
```

Where AP = actual proportion, EP = expected proportion, K = number of bins.

**Critical property**: MAD does NOT use N. This makes it superior to chi-square and K-S
for large datasets (N > 25,000) where traditional tests become overly sensitive, rejecting
near-perfect conformity.

**Critical Values (Nigrini Table 7.1, derived from 25 diverse real-world datasets):**

| Test | Close Conformity | Acceptable | Marginally Acceptable | Nonconformity |
|------|-----------------|------------|----------------------|---------------|
| First digits (K=9) | 0.000–0.006 | 0.006–0.012 | 0.012–0.015 | >0.015 |
| Second digits (K=10) | 0.000–0.008 | 0.008–0.010 | 0.010–0.012 | >0.012 |
| First-two digits (K=90) | 0.0000–0.0012 | 0.0012–0.0018 | 0.0018–0.0022 | >0.0022 |
| First-three digits (K=900) | 0.00000–0.00036 | 0.00036–0.00044 | 0.00044–0.00050 | >0.00050 |

**My Law MAD Critical Values** (for period-over-period comparison, Nigrini Table 13.1):

| Test | Close | Acceptable | Marginal | Nonconformity |
|------|-------|------------|----------|---------------|
| First-two digits | 0.0000–0.0024 | 0.0024–0.0036 | 0.0036–0.0044 | >0.0044 |

When different tests give conflicting conformity conclusions, Nigrini recommends using
the most pessimistic conclusion.

### 2.2 Z-Statistic — Per-Digit Significance Test

**Nigrini Eq. 7.1 (adapted from Fleiss 1981):**
```
Z = (|AP - EP| - 1/(2N)) / sqrt(EP * (1 - EP) / N)
```

**Continuity correction rule**: The term `1/(2N)` is a Yates correction and is used ONLY
when `1/(2N) < |AP - EP|`. If `1/(2N) >= |AP - EP|`, set the entire numerator to 0.

**Critical Z-values:**

| Confidence Level | Z critical | P-value |
|-----------------|-----------|---------|
| 80% | 1.285 | 0.20 |
| 85% | 1.435 | 0.15 |
| 90% | 1.645 | 0.10 |
| 95% | 1.96 | 0.05 |
| 99% | 2.576 | 0.01 |
| 99.9% | 3.29 | 0.001 |
| 99.99% | 3.89 | 0.0001 |
| 99.999% | 4.417 | 0.00001 |
| 99.9999% | 4.892 | 0.000001 |
| 99.99999% | 5.327 | 0.0000001 |

**Significance calculation** (Nigrini Eq. 7.2):
```
Significance = 2 * (1 - NORMSDIST(Z))
```

**Important caveat**: Z-statistic suffers from "excess power" for large N. Use MAD as the
PRIMARY conformity measure; use Z for per-digit drill-down only.

### 2.3 Chi-Square Test

**Formula:**
```
chi2 = sum((observed_count_i - expected_count_i)^2 / expected_count_i) for i = 1..K
```

Where `expected_count_i = N * EP_i`.

**Critical values for first digits (df = 8):**

| Confidence | Critical value |
|-----------|---------------|
| 80% | 11.030 |
| 85% | 12.027 |
| 90% | 13.362 |
| 95% | 15.507 |
| 99% | 20.090 |
| 99.9% | 26.124 |
| 99.99% | 31.827 |
| 99.999% | 37.332 |
| 99.9999% | 42.701 |
| 99.99999% | 47.972 |

**Critical values for second digits (df = 9):**

| Confidence | Critical value |
|-----------|---------------|
| 80% | 12.242 |
| 85% | 13.288 |
| 90% | 14.684 |
| 95% | 16.919 |
| 99% | 21.666 |
| 99.9% | 27.877 |

Same excess power caveat as Z-statistic for large N.

### 2.4 Kolmogorov-Smirnov Test

**Formula:**
```
D = max(|F_observed(x) - F_expected(x)|)   (supremum over cumulative distribution)
```

**Critical value:**
```
KS_critical = C / sqrt(N)
```

| Confidence | C |
|-----------|------|
| 80% | 1.073 |
| 85% | 1.138 |
| 90% | 1.224 |
| 95% | 1.358 |
| 99% | 1.628 |
| 99.9% | 1.949 |
| 99.99% | 2.225 |
| 99.999% | 2.470 |
| 99.9999% | 2.693 |
| 99.99999% | 2.899 |

Reject conformity if `D > KS_critical`.

---

## 3. Advanced Tests

### 3.1 Distortion Factor Model (Nigrini Ch. 6)

Detects systematic over- or understatement in data.

**Step 1: Filter** — Delete all numbers < 10 (including negatives).

**Step 2: Collapse** (Nigrini Eq. 6.1) — Reduce each number to two digits left of decimal:
```
collapsed = number / 10^(floor(log10(number)) - 1)
```
Examples: 6,340 → 63.40; 0.0529 → 52.9; 110,364 → 11.0364

**Step 3: Actual Mean (AM)** — Nigrini Eq. 6.3:
```
AM = mean(collapsed_values)
```

**Step 4: Expected Mean (EM)** — Nigrini Eq. 6.5 & 6.7:
```
EM_exact = 90 / (N * (10^(1/N) - 1))
EM_approx = 90 / ln(10) = 39.0865
```
Use 39.0865 in practice.

**Step 5: Distortion Factor (DF)** — Nigrini Eq. 6.8:
```
DF = (AM - 39.0865) / 39.0865
```
DF > 0 → overstated; DF < 0 → understated. Multiply by 100 for percentage.

**Step 6: Statistical Significance** — Nigrini Eq. 6.9 & 6.10:
```
SD_DF = 0.638253 / sqrt(N)
Z = DF / SD_DF
```

### 3.2 Second-Order Test (Nigrini & Miller, Ch. 5)

1. Sort ascending: y_1 ≤ y_2 ≤ ... ≤ y_N
2. Compute N-1 differences: diff_i = y_(i+1) - y_i
3. Scale small differences so all have ≥ 2 digits left of decimal
4. Compute first-two digits, compare to Benford

**Expected**: "Almost Benford" for continuous data. Prime spikes at 10, 20, ..., 90 for discrete data.
Use MAD with liberal bounds for conformity assessment.

### 3.3 Summation Test (Nigrini Ch. 3 & 5)

Group by first-two digits, sum values, compare proportions to EP = 1/90.
**Caveat (Kossovsky)**: real-world data typically shows unequal sums favoring lower digits.
Use for detecting outlier spikes, not strict conformity.

### 3.4 Number Duplication Test

Frequency count of exact values, sorted descending. Drill-down for spike investigation.

### 3.5 Last-Two Digits Test

Expected: uniform 1/100 for each of 00-99. Detects rounding, fabrication, thresholds.

### 3.6 Mantissa Arc Test (Alexander 2009)

Map mantissas to unit circle, compute center of gravity, test distance from (0,0).
Very sensitive for large N — consider using sqrt(N) instead of N.

### 3.7 Ordered Mantissa Plot

Sort mantissas ascending, plot vs expected uniform line. Straight line = conformity.

---

## 4. Data Prerequisites

### 4.1 When Benford Applies
- Sizes of facts/events (populations, revenues, flow rates)
- No built-in min/max (except min of 0 or 10 acceptable)
- Not identification numbers
- Data spans several orders of magnitude
- Numbers have ≥ 4 digits for good fit

### 4.2 Sample Size
- ≥ 1,000: recommended minimum
- ≥ 300: minimum for first-two digits test
- < 300: first-digit test only
- > 25,000: use MAD, not chi-square/K-S

### 4.3 Data Cleaning
- Exclude totals/subtotals
- Analyze income and expense separately
- Standard strata: large positive (≥10), small positive, zeros, small negative, large negative
- Delete numbers < 10 for standard tests

---

## 5. Full Digit Proportion Table (Nigrini Table 1.2)

| Digit | 1st | 2nd | 3rd | 4th |
|-------|-----|-----|-----|-----|
| 0 | — | 0.11968 | 0.10178 | 0.10018 |
| 1 | 0.30103 | 0.11389 | 0.10138 | 0.10014 |
| 2 | 0.17609 | 0.10882 | 0.10097 | 0.10010 |
| 3 | 0.12494 | 0.10433 | 0.10057 | 0.10006 |
| 4 | 0.09691 | 0.10031 | 0.10018 | 0.10002 |
| 5 | 0.07918 | 0.09668 | 0.09979 | 0.09998 |
| 6 | 0.06695 | 0.09337 | 0.09940 | 0.09994 |
| 7 | 0.05799 | 0.09035 | 0.09902 | 0.09990 |
| 8 | 0.05115 | 0.08757 | 0.09864 | 0.09986 |
| 9 | 0.04576 | 0.08500 | 0.09827 | 0.09982 |

---

## 6. Implementation Constants (verified against notebook code)

`CRIT_CHI2` and `CONFS` (along with six other unreferenced constants) were
removed from `pybenford.constants` in Pass 3a.

```python
CRIT_KS = {80: 1.073, 85: 1.138, 90: 1.224, 95: 1.358, 99: 1.628,
           99.9: 1.949, 99.99: 2.225, 99.999: 2.47,
           99.9999: 2.693, 99.99999: 2.899}

MAD_CONFORM = {
    1:  [0.006, 0.012, 0.015],      # first digits
    2:  [0.008, 0.010, 0.012],      # second digits
    22: [0.0012, 0.0018, 0.0022],   # first-two digits
    3:  [0.00036, 0.00044, 0.00050] # first-three digits
}

DF_EXPECTED_MEAN = 39.0865     # 90 / ln(10)
DF_SD_CONSTANT = 0.638253      # SD = 0.638253 / sqrt(N)
```

---

## 7. Scale Invariance & Base Invariance

- Multiplying by any constant preserves first-order Benford conformity
- Second-order test CAN detect non-power-of-10 scaling
- Base B formula: P(d) = log10(1 + 1/d) / log10(B)

---

## 8. References

1. Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
2. Miller, S.J. (2015). *Benford's Law: Theory and Applications*. Princeton University Press.
3. Kossovsky, A.E. (2014). *Benford's Law: Theory, The General Law of Relative Quantities, and Forensic Fraud Detection Applications*. World Scientific.
4. Hill, T.P. (1995). "A Statistical Derivation of the Significant-Digit Law." *Statistical Science* 10(4).
5. Fleiss, J.L. (1981). *Statistical Methods for Rates and Proportions*. 2nd ed. Wiley.
6. Drake, P.D. & Nigrini, M.J. (2000). "Computer Assisted Analytical Procedures Using Benford's Law." *Journal of Accounting Education* 18(2).
7. Miller, S.J. & Nigrini, M.J. (2008). "The Modulo 1 Central Limit Theorem and Benford's Law for Products."
