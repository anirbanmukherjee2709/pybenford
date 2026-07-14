# Spec: Vectorization, Test Debt, Dead Code (Pass 3a of 3)

**Status:** Rev 4 — span computation bounds both valid_digits and
possible_digits (data-driven wide spans previously wrapped or exploded the
fast path); absent-wide-digit regression test added
**Author:** Claude.ai planning session, 2026-07-14
**Implementer:** Claude Code (sole author — Codex is suggest-only, never writes)
**Baseline:** main at `111b927` with CI green (Passes 1–2 complete)
**Scope:** Performance (bin-loop vectorization), test debt (empty
test_distributions.py, census regression, coverage gaps), dead-code pruning.
NO packaging/pyproject/CI changes — those are Pass 3b. No version bump, no
publish.

## Read-first instructions (mandatory, in order)

1. Read this spec completely.
2. Read `src/pybenford/utils.py` (`digit_counts`, `summation_by_digits`),
   `src/pybenford/constants.py`, `src/pybenford/distributions.py`,
   `tests/test_distributions.py` (currently empty), `.gitignore`,
   `CHANGELOG.md`, and the current coverage report
   (`pytest -q` term-missing output — 5 missed lines).
3. Baseline gate from `pybenford-dev`: `pip install -e ".[dev]"`, checkout
   assert, seven-pin heredoc, `pytest -q` (expect 234 passed, coverage ≥ 99).
   STOP if red.

## Anti-hallucination guardrails

- Implement exactly the tasks below. Other bugs found: report, don't fix.
- Do NOT touch `pyproject.toml`, `.github/`, or `src/pybenford/visualization.py`.
- Do NOT upgrade numpy in the local env (must stay 2.4.6 — 2.5.x stubs break
  mypy under the 3.10 target). No `pip install -U numpy`.
- Every locked numeric expectation below was computed from the code at
  `111b927`; if your implementation produces a different value, STOP and
  report — never adjust an expectation to make a test pass.
- No existing-test modifications in this pass. New tests only.

## Task A — vectorize the two bin-counting loops

Audit benchmarks (n=5M): first-three counting 529ms → 67ms, summation
274ms → 70ms. Counts are exactly equal to the loop's; weighted sums are the
same additions in a different order and drift up to ~1.4e-9 absolute on 50k
positive rows (gate-1 measurement) — do NOT claim identical results anywhere.

In `src/pybenford/utils.py`:

1. `digit_counts` — replace the per-digit list comprehension with an
   OFFSET bincount that preserves current semantics exactly, including
   negative bins from custom extractors and duplicate/unsorted
   `possible_digits`:

   ```python
   possible = np.asarray(possible_digits, dtype=np.int64)
   if possible.size == 0:
       # preserve current behavior: empty counts, total 0, zero proportions
       return DigitFrequencies(
           digits=possible,
           counts=np.zeros(0, dtype=np.int64),
           proportions=np.zeros(0, dtype=np.float64),
           total=0,
       )
   lo = min(int(valid_digits.min()), int(possible.min()))
   hi = max(int(valid_digits.max()), int(possible.max()))
   span = hi - lo + 1  # Python ints: no int64 overflow; bounds BOTH arrays
   if 0 < span <= 1_000_000:
       # fast path: offset bincount (all internal callers land here; span <= 1000)
       bc = np.bincount(valid_digits - lo, minlength=span)
       counts = bc[possible - lo].astype(np.int64)
   else:
       # exact sparse fallback: any int64 digits, widely separated or extreme
       # bins, O(n log n) time, O(unique) memory
       uniq, cnt = np.unique(valid_digits, return_counts=True)
       idx = np.minimum(np.searchsorted(uniq, possible), len(uniq) - 1)
       counts = np.where(uniq[idx] == possible, cnt[idx], 0).astype(np.int64)
   ```

   (`valid_digits` is non-empty here — the existing "no valid digits" guard
   raises first. The span check is computed in Python ints so extreme bins
   cannot overflow; the fallback preserves exact loop semantics for any
   int64 values with no O(span) memory. No new precondition, no contract
   change.) `total` and proportions as before.

2. `summation_by_digits` — replace the per-digit sum comprehension with:

   ```python
   sums = np.bincount(valid_ft, weights=valid_arr, minlength=100)[10:100].astype(np.float64)
   ```

   (`valid_ft` is guaranteed in [10, 99] by `extract_first_two_digits`.)

   CONTRACT NOTE (add to the docstring): the summation test is defined for
   non-negative amounts (Nigrini Ch. 5; `BenfordAnalysis` always supplies
   cleaned positive data). Signed inputs remain accepted, but group sums and
   proportions under magnitude cancellation are numerically order-sensitive
   in ANY implementation — the previous per-bin loop was equally unstable
   (e.g. tiled `[1e16, 1, -1e16]` sums to 4 under the old loop and 0 under
   bincount; the exact answer is 10). State that grouped sums may differ
   from naive per-bin summation only through float summation order (observed
   max relative drift 2.3e-15 on 50k positive rows).

New equivalence tests in `tests/test_utils.py` (the old loop implemented
INLINE in the test as the oracle):

- `digit_counts` vs loop-oracle on `10 ** rng.uniform(0, 4, 50_000)`
  (`default_rng(3)`) for first-digit, first-two, and first-three ranges —
  exact array equality of counts.
- `digit_counts` edge preservation: `possible_digits=[]` returns empty
  counts/total 0 (current behavior); `possible_digits=[-1, 1]` with data
  `[1.5]` returns `[0, 1]`.
- Fallback-path regression (widely separated bins): a custom extractor
  returning values from `{0.0, 2.0**40}` (exactly float64-representable —
  do NOT use int64.max, which is not) with `possible_digits=[0, 2**40]` and
  with `[-2**40, 2**40]` — counts equal the inline loop oracle exactly.
  Assert both paths are exercised: the wide-bin case must take the sparse
  branch (span > 1_000_000), the standard ranges the bincount branch.
- Absent-wide-DIGIT regression (span driven by the data, not the bins):
  extractor output `2.0**40` with `possible_digits=[0]` — must take the
  sparse branch (the data value, not the bin range, forces the wide span)
  and return count `[0]`, total 0, matching the loop oracle.
- `summation_by_digits` vs loop-oracle on the same 50k POSITIVE data —
  `np.allclose(rtol=1e-9, atol=0)` on sums (1e-9 is the TEST TOLERANCE;
  observed relative drift is ~2.3e-15); exact equality on `digits` and
  `n_valid`.
- Digits absent from the data yield count 0 / sum 0.0 (e.g. data `[111.0]`
  → first-two counts are 1 at digit 11, 0 elsewhere).

## Task B — populate `tests/test_distributions.py` (currently 0 bytes)

- `digit_range(k)` for k=1,2,3: correct start/stop/length; `k=0` and negative
  raise ValueError (this covers the currently-missed `distributions.py` line).
- `benford_distribution(k).sum() == pytest.approx(1.0)` for k=1..4.
- Cross-check the Nigrini reference tables in `constants.py` against the
  first-principles computations, all at `abs=5e-6` (tables are 5dp):
  - `first_digit_distribution()` vs `FIRST_DIGIT_PROBS`
  - `second_digit_distribution()` vs `SECOND_DIGIT_PROBS`
  - `third_digit_distribution()` vs `THIRD_DIGIT_PROBS`
  - fourth-digit marginal computed inline from `benford_distribution(4)`
    (reshape to (9,10,10,10), sum over the first three axes) vs
    `FOURTH_DIGIT_PROBS`
- Consistency: `second_digit_distribution()` equals the marginal of
  `benford_distribution(2)` reshaped (9,10) summed over axis 0 (atol 1e-12);
  same idea for third from `benford_distribution(3)`.

These tests give the reference tables a purpose (they cross-pin the formula
and Nigrini's published values), which is why Task D keeps them.

## Task C — census regression test with locked outputs

New file `tests/test_regression.py`. Data:
`examples/data/co-est2025-alldata.csv` read with the stdlib `csv` module
(NO pandas), `encoding="latin-1"`, filter rows where
`row["SUMLEV"].strip() == "050"` (county level only — the file also contains
51 state aggregates), take `float(row["POPESTIMATE2025"])`.

Guard: `pytest.mark.skipif` when the CSV path (resolved relative to the test
file: `Path(__file__).parent.parent / "examples" / "data" / ...`) does not
exist, so the suite still passes from an sdist without examples.

Locked expectations, computed at `111b927` (assert with
`pytest.approx(rel=1e-9)` for floats, exact for ints/strings):

```
record count after filter      3144
BenfordAnalysis(data).n        3144
first_digit():  mad            0.003586242795
                mad_conformity "close_conformity"
                chi_square     5.6231120182
first_two_digits(): mad        0.001613208349
                mad_conformity "acceptable_conformity"
second_order(): mad            0.005399086450
                n              3091
distortion_factor(): distortion_factor  -0.001881979793
                     z_statistic        -0.1653344519
```

No SmallSampleWarning fires (n=3144 ≥ 1000); assert the calls are
warning-free via `warnings.catch_warnings` with
`simplefilter("error", SmallSampleWarning)`.

## Task D — prune dead constants

Remove from `src/pybenford/constants.py` (and its `__all__`) these
verified-unreferenced names: `CRIT_CHI2`, `CRIT_Z`, `CONFS`, `P_VALUES`,
`DOF`, `TEST_NAMES`, `MAD_CONFORM_MY_LAW`, `COLORS`.

Notes:
- `constants.COLORS` is NOT the palette visualization.py uses (that module
  defines its own local `COLORS`); its "shared by every plotting utility"
  docstring is false. Do not touch visualization.py.
- KEEP: the sample-size trio (used since Pass 2), `CRIT_KS`, `MAD_CONFORM`,
  `NUM_BINS`, `SUMMATION_EXPECTED`, `LAST_TWO_DIGITS_EXPECTED`,
  `DF_EXPECTED_MEAN`, `DF_SD_CONSTANT`, and the four digit-probability
  tables (now referenced by Task B tests).
- Update the module docstring's naming-conventions section if it references
  removed names.
- `docs/reference.md` documents `CRIT_CHI2` and `CONFS` in its
  "Implementation Constants" section (gate-1 finding): remove the entries for
  the eight deleted names from that section and add a one-line note that
  they were removed in Pass 3a. No other reference.md edits.
- These are public-API removals: list each under `Removed` in the CHANGELOG
  `Unreleased` section (they justify the 0.2.0 minor bump).

## Task E — close the 5 uncovered lines

Current gate shows 5 missed statements. One (distributions.py, `k < 1` raise)
is covered by Task B. For the remainder (per the term-missing report — one
subnormal-recompute branch in `digits.py`, three validation raises in
`statistics.py`):

- Subnormal branch: `extract_first_digit(np.array([5e-324]))` — exercises the
  underflow-recompute path. Expected first digit 4 (5e-324 is stored as
  ≈4.94e-324); verify at implementation time and report the observed value if
  it differs.
- Validation raises: write the matching `pytest.raises(ValueError)` test for
  each of the three uncovered guard lines, exactly as the guards are written.
- Target: 0 missed lines. If any line proves unreachable, mark it
  `# pragma: no cover` with a one-line justification and report it — do not
  contort tests to hit dead branches. Leave the coverage floor at 99
  (config is Pass 3b territory).

## Task F — ignore the `.claude/` session directory

Append to `.gitignore`:

```
# Claude Code session settings
.claude/
```

(The directory is currently untracked; this just prevents accidents.)

## Task G — CHANGELOG

Under `Unreleased`: `Changed` — vectorized digit counting (identical counts)
and summation grouping (order-of-magnitude speedup; grouped sums may differ
from the previous implementation within float summation-order error,
~1e-15 relative); `Removed` — the eight Task D constants (enumerate);
`Added` — census regression suite, distribution reference-table cross-checks.

## Validation gate (unchanged: seven pins)

Post-edit `pip install -e ".[dev]"`, checkout assert, seven-pin heredoc, then
`ruff check src tests` · `ruff format --check src tests` · `mypy src` ·
`pytest -q`. Expect: >234 tests (report exact count), all passing, coverage
≥ 99 (report the missed-line count; target 0). Paste real output.

## Commit plan

1. `docs: add pass 3a spec (vectorization, test debt, dead code)` — this file.
2. `perf: vectorize bin counting; tests: distributions, census regression,
   coverage; chore: prune dead constants` — Tasks A–G, gate-validated.
   Do not push until the diff review (gate 2) approves.

## Done criteria

- Bin loops gone from `digit_counts`/`summation_by_digits`; equivalence tests
  pass; locked census values reproduce exactly.
- `tests/test_distributions.py` no longer empty; reference tables cross-pinned.
- Eight constants removed and CHANGELOG'd; `.claude/` ignored.
- `git status --porcelain` empty; exactly two new commits; not pushed.
- Summary reports: exact test count, missed-line count, any pragma added,
  and zero existing-test modifications.
