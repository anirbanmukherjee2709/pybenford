# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-07

### Changed

- Store computed mantissas on `MantissaArcResult` during `mantissa_arc_test()`
- `plot_mantissa_arc(result)` no longer requires a separate `data` argument
- `plot_ordered_mantissas(result)` accepts `MantissaArcResult` instead of raw array
- Old call signatures still work with `DeprecationWarning`
- Updated demo notebook and README to new API

## [0.1.0] - 2026-04-23

### Added

- `BenfordAnalysis` class — single entry point for all Benford's Law tests
- First digit, second digit, third digit tests
- First-two digits, first-three digits tests
- Last-two digits test (uniform expectation)
- Second-order test (differences of sorted values)
- Summation test with uniform 1/90 expectation
- Distortion factor model (Nigrini) with overstatement/understatement detection
- Mantissa arc test (Alexander) with L-squared and p-value
- Number duplication analysis
- Z-statistic with Fleiss continuity correction
- Chi-square goodness-of-fit test
- Kolmogorov-Smirnov test
- MAD conformity classification: close, acceptable, marginal, nonconformity
- Sum of squared differences (SSD) measure
- Data profiling with configurable strata (Nigrini Ch. 4)
- Input cleaning: NaN/inf removal, sign filtering, minimum absolute value
- 6 publication-quality matplotlib plots: digit test, summation, mantissa arc, ordered mantissas, Z-scores, distortion factor
- All plots return (Figure, Axes) for composability
- Full type annotations (PEP 561 py.typed)
- 185 tests, 99% code coverage
