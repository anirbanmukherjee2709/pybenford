# pybenford

Professional-grade Benford's Law analysis toolkit for forensic accounting, auditing, and fraud detection.

[![PyPI version](https://img.shields.io/pypi/v/pybenford.svg)](https://pypi.org/project/pybenford/)
[![Python versions](https://img.shields.io/pypi/pyversions/pybenford.svg)](https://pypi.org/project/pybenford/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-185%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)]()

## Why pybenford?

Existing Benford's Law packages on PyPI are either basic (first-digit chi-square only), outdated, or unmaintained. **pybenford** implements the complete Nigrini forensic accounting workflow as described in *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection* (Nigrini, 2012):

- **MAD conformity classification** with Nigrini's thresholds: close, acceptable, marginally acceptable, nonconformity
- **Distortion factor model** detecting overstatement vs. understatement
- **Second-order test** on differences of sorted values
- **Summation test** with uniform 1/90 expectation
- **Mantissa arc test** (Alexander, 2009) with L-squared statistic
- **Number duplication analysis**
- All standard digit tests: first, second, third, first-two, first-three, last-two
- Z-statistic with Fleiss continuity correction, chi-square, Kolmogorov-Smirnov
- Publication-quality matplotlib visualizations
- Pure NumPy internals — no pandas dependency, fast on large datasets

## Installation

```bash
pip install pybenford
```

## Quick Start

```python
from pybenford import BenfordAnalysis

# Load your data (list, numpy array, or pandas/polars Series)
data = [...]  # e.g., invoice amounts, population figures, financial values

# Create analysis object — cleaning happens automatically
analysis = BenfordAnalysis(data, sign_filter="positive", min_abs_value=10.0)

# View data profile (Nigrini Ch. 4)
print(analysis.profile)

# Run the first-digit test
result = analysis.first_digit()
print(f"MAD: {result.mad:.6f} ({result.mad_conformity})")
print(f"Chi-square: {result.chi_square:.2f} (significant: {result.chi_square_significant})")

# Run the first-two digits test (most useful for forensic work)
result = analysis.first_two_digits()

# Run all advanced tests
summation = analysis.summation()
second_order = analysis.second_order()
distortion = analysis.distortion_factor()
mantissa = analysis.mantissa_arc()
duplicates = analysis.number_duplication(top_n=20)
```

## Visualization

Every plot function returns `(Figure, Axes)` — no side effects, full control.

```python
from pybenford.visualization import plot_digit_test, plot_mantissa_arc

# Digit distribution with confidence bands
result = analysis.first_two_digits()
fig, ax = plot_digit_test(result, show_confidence=True)
fig.savefig("first_two_digits.png", dpi=150)

# Mantissa arc test
arc = analysis.mantissa_arc()
fig, ax = plot_mantissa_arc(arc, analysis.clean_data)

# Z-scores with significance thresholds
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

- **Z-statistic** per digit bin (Fleiss continuity correction)
- **Chi-square** goodness-of-fit with critical value
- **Kolmogorov-Smirnov** statistic with critical value
- **MAD** (Mean Absolute Deviation) with Nigrini's conformity classification
- Per-bin significance flags at configurable alpha

## References

- Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
- Miller, S.J. (2015). *Benford's Law: Theory and Applications*. Princeton University Press.
- Kossovsky, A.E. (2014). *Benford's Law: Theory, the General Law of Relative Quantities, and Forensic Fraud Detection Applications*. World Scientific.

## License

MIT License. See [LICENSE](LICENSE) for details.
