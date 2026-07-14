# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SmallSampleWarning` — emitted when the constructor sample is below 1,000
  or when the effective sample of the first-two, first-three, second-order,
  or summation test is below 300 (Nigrini §4.2)
- `SummationFrequencies.n_valid` — count of values with valid first-two
  digits, the effective sample of the summation test
- Census regression suite: locked end-to-end outputs on the bundled county
  population estimates (`tests/test_regression.py`)
- Distribution reference-table cross-checks: Nigrini's published digit
  probability tables pinned against the first-principles computations
  (`tests/test_distributions.py`)

### Removed

- `CRIT_CHI2` — unreferenced chi-square critical-value table
- `CRIT_Z` — unreferenced per-digit Z critical values
- `CONFS` — unreferenced alias of `CRIT_Z`
- `P_VALUES` — unreferenced p-value table
- `DOF` — unreferenced degrees-of-freedom table
- `TEST_NAMES` — unreferenced test-identifier tuple
- `MAD_CONFORM_MY_LAW` — unreferenced "My Law" MAD thresholds
- `COLORS` — unreferenced palette (plotting uses its own local palette)

### Fixed

- Digit-boundary snap: ULP-relative correction in the positional digit
  extractors so decimal-entered values a few ULPs below a digit boundary
  extract correctly (e.g. `0.29` no longer yields first-two digits `28`)
- Second-order test: `second_order_differences` returns raw sorted diffs;
  the legacy `×10 + round` formula destroyed sub-unit spacings
- `MantissaArcResult.mean_angle` is now the direction of the gravity center
  (`arctan2(mean_y, mean_x)`), not the arithmetic mean of the angles
- `z_statistic` returns 0 (not inf) when expected and observed are both 0

### Changed

- Supported Python floor raised from 3.9 to 3.10 (from Pass 1)
- Summation test inferential output labeled heuristic: the z-score and
  chi-square variance models assume count proportions, not sum shares
- Vectorized digit counting (identical counts) and summation grouping
  (order-of-magnitude speedup; grouped sums may differ from the previous
  implementation within float summation-order error, ~1e-15 relative)

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
