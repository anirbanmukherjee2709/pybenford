# pybenford

Professional-grade Benford's Law analysis toolkit for forensic accounting, auditing, and fraud detection.

[![PyPI version](https://img.shields.io/pypi/v/pybenford)](https://pypi.org/project/pybenford/)
[![Python versions](https://img.shields.io/pypi/pyversions/pybenford)](https://pypi.org/project/pybenford/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-271%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)]()

## Why pybenford?

Existing Benford's Law packages on PyPI cover first-digit chi-square and not much else. Most are unmaintained. `pybenford` implements the complete Nigrini forensic accounting workflow from *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection* (Nigrini, 2012):

- MAD conformity classification with Nigrini's empirical thresholds (close, acceptable, marginally acceptable, nonconformity)
- Distortion factor model for detecting overstatement vs. understatement
- Second-order test on differences of sorted values
- Summation test with uniform 1/90 expectation
- Mantissa arc test (Alexander, 2009) with L-squared statistic
- Number duplication analysis
- All standard digit tests: first, second, third, first-two, first-three, last-two
- Z-statistic with Fleiss continuity correction, chi-square, Kolmogorov-Smirnov
- Publication-quality matplotlib visualizations
- Pure NumPy internals, no pandas dependency

## Installation

```bash
pip install pybenford             # analysis (numpy + scipy)
pip install "pybenford[plot]"     # analysis + matplotlib visualizations
```

Requires Python >= 3.10.

**Breaking change in 0.2.0:** matplotlib is no longer a required dependency. If you use the six `plot_*` functions, install the `[plot]` extra. On an analysis-only install, accessing a plot function — including via `from pybenford import *`, which resolves every exported name — raises an `ImportError` pointing at `pip install "pybenford[plot]"`.

## Quick Start

```python
from pybenford import BenfordAnalysis

analysis = BenfordAnalysis(data)  # list, numpy array, or pandas/polars Series
result = analysis.first_digit()
print(result)
```

Every result object has a formatted `print()` output. The two reports below are produced from the 3,144 county-level records (`SUMLEV == "050"`) of the US Census 2025 county population estimates, by this exact code:

```python
import csv

from pybenford import BenfordAnalysis

with open("co-est2025-alldata.csv", encoding="latin-1", newline="") as f:
    data = [
        float(row["POPESTIMATE2025"])
        for row in csv.DictReader(f)
        if row["SUMLEV"].strip() == "050"
    ]

print(BenfordAnalysis(data).first_digit())
```

> **Reproducibility note:** the census dataset ships in the GitHub repository, not in the PyPI package. Download [`examples/data/co-est2025-alldata.csv`](https://raw.githubusercontent.com/anirbanmukherjee2709/pybenford/main/examples/data/co-est2025-alldata.csv) (raw file) or clone the repo.

```
=======================================================
  First Digit Test  (n=3,144  alpha=0.05)
=======================================================
 Digit   Count   Observed   Expected   Z-Score   Sig
     1    943   29.99%    30.10%      0.11
     2    588   18.70%    17.61%      1.59
     3    382   12.15%    12.49%      0.56
     4    295    9.38%     9.69%      0.55
     5    250    7.95%     7.92%      0.04
     6    191    6.08%     6.69%      1.35
     7    175    5.57%     5.80%      0.52
     8    169    5.38%     5.12%      0.62
     9    151    4.80%     4.58%      0.57
-------------------------------------------------------
 MAD:        0.0036 — Close Conformity
 Chi-Square: 5.6231  (critical: 15.5073) — Pass
 KS:         0.0098  (critical: 0.0242)  — Pass
=======================================================
```

For tests with many digit bins (first-two, first-three), the display shows only flagged digits instead of all 90 or 900 rows. On the same `data`:

```python
print(BenfordAnalysis(data).first_two_digits())
```

```
=======================================================
  First Two Digits Test  (n=3,144  alpha=0.05)
=======================================================
 Flagged Digits (6 of 90):
 Digit   Count   Observed   Expected   Z-Score
    35     23    0.73%     1.22%      2.43  *
    49     16    0.51%     0.88%      2.12  *
    66     33    1.05%     0.65%      2.65  *
    70     29    0.92%     0.62%      2.08  *
    75     28    0.89%     0.58%      2.22  *
    76      8    0.25%     0.57%      2.22  *
-------------------------------------------------------
 MAD:        0.0016 — Acceptable Conformity
 Chi-Square: 111.1969  (critical: 112.0220) — Pass
 KS:         0.0118  (critical: 0.0242)  — Pass
=======================================================
```

All results are also accessible programmatically:

```python
result = analysis.first_digit()

result.mad                     # 0.0034
result.mad_conformity          # "close"
result.chi_square              # 4.6922
result.chi_square_significant  # False
result.ks_statistic            # 0.0083
result.ks_critical             # 0.0240
result.z_scores                # array of per-digit Z-scores
result.significant_flags       # bool array of flagged digits
result.observed                # array of observed proportions
result.expected                # array of expected Benford proportions
result.digits                  # array of digit labels
result.counts                  # array of raw counts
result.n                       # number of records analyzed
result.alpha                   # significance level used
result.test_name               # e.g. "First Digit Test"
```

## Demo Notebook

A complete walkthrough of every test and visualization is available in [`examples/demo.ipynb`](https://github.com/anirbanmukherjee2709/pybenford/blob/main/examples/demo.ipynb). It runs against US Census county population data and shows the output of all 11 tests, 6 plot functions, and programmatic result access.


## Data Preparation

```python
analysis = BenfordAnalysis(
    data,                        # list, array, or Series of numbers
    sign_filter="positive",      # "all", "positive", or "negative"
    min_abs_value=10.0,          # exclude small values (optional)
    drop_zero=True,              # exclude zeros (default: True)
)

print(analysis.profile)          # data profile per Nigrini Ch. 4
```

`sign_filter` separates income from expense items for independent analysis. `min_abs_value` excludes values below a minimum magnitude, since very small numbers distort digit distributions.

## Visualization

Plot functions return `(Figure, Axes)` with no side effects.

```python
from pybenford.visualization import plot_digit_test, plot_mantissa_arc

result = analysis.first_two_digits()
fig, ax = plot_digit_test(result, show_confidence=True)
fig.savefig("first_two_digits.png", dpi=150)

arc = analysis.mantissa_arc()
fig, ax = plot_mantissa_arc(arc)

from pybenford.visualization import plot_z_scores
fig, ax = plot_z_scores(result, critical_value=1.96)
```

## Available Tests

| Method | Description | Reference |
|--------|-------------|-----------|
| `first_digit()` | First significant digit (1-9) | Nigrini Ch. 5 |
| `second_digit()` | Second significant digit (0-9) | Nigrini Ch. 5 |
| `third_digit()` | Third significant digit (0-9) | Nigrini Ch. 5 |
| `first_two_digits()` | First two digits (10-99) | Nigrini Ch. 5 |
| `first_three_digits()` | First three digits (100-999) | Nigrini Ch. 5 |
| `last_two_digits()` | Last two digits (00-99), uniform expected | Nigrini Ch. 5 |
| `second_order()` | Digit test on sorted differences | Nigrini Ch. 6 |
| `summation()` | Sum proportions vs. uniform 1/90 | Nigrini Ch. 5 |
| `distortion_factor()` | Overstatement/understatement detection | Nigrini Ch. 6 |
| `mantissa_arc()` | Uniformity of mantissas on unit circle | Nigrini Ch. 7 |
| `number_duplication()` | Most frequently duplicated values | Nigrini Ch. 5 |

## Statistical Measures

Each digit test result includes:

- Z-statistic per digit bin (Fleiss continuity correction)
- Chi-square goodness-of-fit with critical value
- Kolmogorov-Smirnov statistic with critical value
- MAD (Mean Absolute Deviation) with Nigrini's conformity classification
- Per-bin significance flags at configurable alpha

### MAD Conformity Thresholds

MAD is the preferred conformity measure because chi-square and KS become overly sensitive with large datasets (N > 25,000), rejecting near-perfect conformity. MAD is sample-size independent.

| Test | Close | Acceptable | Marginal | Nonconformity |
|------|-------|------------|----------|---------------|
| First digit | < 0.006 | < 0.012 | < 0.015 | >= 0.015 |
| Second digit | < 0.008 | < 0.010 | < 0.012 | >= 0.012 |
| First two digits | < 0.0012 | < 0.0018 | < 0.0022 | >= 0.0022 |
| First three digits | < 0.00036 | < 0.00044 | < 0.00050 | >= 0.00050 |

## References

- Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
- Miller, S.J. (2015). *Benford's Law: Theory and Applications*. Princeton University Press.
- Kossovsky, A.E. (2014). *Benford's Law: Theory, the General Law of Relative Quantities, and Forensic Fraud Detection Applications*. World Scientific.

## License

MIT License. See [LICENSE](LICENSE) for details.
