# Spec: Correctness Fixes (Pass 2 of 3)

**Status:** Rev 4 — guardrail license list corrected to "Tasks B and F",
removing the last contradiction with Task D and the done criteria
**Author:** Claude.ai planning session, 2026-07-13
**Implementer:** Claude Code (sole author — Codex is suggest-only, never writes)
**Baseline:** main at `9a35d11` with CI green (Pass 1 complete)
**Scope:** Statistical-correctness fixes found by the adversarial audit, with
tests reproducing each finding. Behavior changes → CHANGELOG under `Unreleased`.
No version bump, no publish (0.2.0 ships after Pass 3).

## Read-first instructions (mandatory, in order)

1. Read this spec completely.
2. Read `src/pybenford/digits.py`, `src/pybenford/utils.py`
   (`second_order_differences`, `SummationFrequencies`, `summation_by_digits`),
   `src/pybenford/statistics.py` (`z_statistic`, `mantissa_arc_test`),
   `src/pybenford/core.py` (`BenfordAnalysis.__init__`, `_run_digit_test`,
   `summation`, `SummationResult.__str__`), `src/pybenford/constants.py`
   (sample-size constants block), `tests/test_utils.py`,
   `tests/test_statistics.py`, `tests/test_display.py`, `tests/test_core.py`,
   `docs/statistics_py_spec.md`, `docs/result_display_spec.md`, `CHANGELOG.md`.
3. Baseline gate from the activated `pybenford-dev` env:
   `pip install -e ".[dev]"`, the checkout assert, then `pytest -q`
   (expect 212 passed, coverage ≥ 99). STOP if red.

## Anti-hallucination guardrails

- Implement exactly the tasks below. Any other bug found: report, don't fix.
- Every numeric expectation below was re-verified after the gate-1 review; if
  an implementation produces a different value, STOP and report — do not
  adjust the expectation to make it pass.
- Existing-test modifications are licensed ONLY where a task explicitly
  enumerates them (Tasks B and F). List every modified assertion in the summary.
- Unexpected gate output = stop-and-review. Never commit red.
- No changes to plotting code, packaging metadata (beyond the hypothesis dev
  pin), or CI in this pass.

## Task A — ULP-relative snap in digit extraction (fixes `0.29 → 28`)

Float representation places some decimal inputs a few ULPs below their digit
boundary: `extract_first_two_digits([0.29]) == 28`, `[0.64] == 63`.

A fixed absolute epsilon is WRONG (gate-1 blocker): 1e-8 in digit space
misclassifies ordinary currency values whose true significand legitimately
sits within the band — `1_999_999.99` (significand 1.99999999) would snap
1 → 2. The correction must be relative, sized to float drift (~2 ULPs), not
to digit space.

In `src/pybenford/digits.py`:

- Add `_REL_EPS: Final[float] = 4e-15` with a comment: relative boundary
  correction ≈ 18 ULPs at 1.0; absorbs the few-ULP drift from log/divide
  while only misclassifying true values within 4e-15 RELATIVE distance of a
  digit boundary — i.e. values carrying 15+ significant decimal digits, which
  decimal-entered data does not produce (verified: 0 mismatches on 500k
  two-decimal and 200k three-decimal currency values; `1_999_999.99`,
  `19_999_999.99`, `199_999_999.99` all correct).
- Apply snap-then-clip in all FIVE positional extractors (NOT in
  `extract_mantissa`, `extract_last_two_digits`, or `collapse_numbers`):
  - `extract_first_digit`: `np.minimum(np.floor(m * (1.0 + _REL_EPS)), 9.0)`
  - `extract_first_two_digits`: `np.minimum(np.floor(m * 10.0 * (1.0 + _REL_EPS)), 99.0)`
  - `extract_first_three_digits`: `np.minimum(np.floor(m * 100.0 * (1.0 + _REL_EPS)), 999.0)`
  - `extract_second_digit`: `np.minimum(np.floor(m * 10.0 * (1.0 + _REL_EPS)), 99.0) % 10.0`
  - `extract_third_digit`: `np.minimum(np.floor(m * 100.0 * (1.0 + _REL_EPS)), 999.0) % 10.0`

  The clip is mandatory: a true significand within the relative band of 10.0
  (e.g. `9.99999999999999`, 15 nines-class values) floors to 10/100/1000 after
  the bump; the clip returns it to the top in-range digit, which is the
  correct answer for such values. NaN propagation is unchanged.

New tests in `tests/test_digits.py` (file currently empty — this pass starts
populating it):

- Fixed cases, first digit: `[0.29, 0.64, 12.99, 1999999.99, 19999999.99,
  199999999.99, 9.99999999999999, 0.30]` → `[2, 6, 1, 1, 1, 1, 9, 3]`.
- Fixed cases, first-two: same inputs → `[29, 64, 12, 19, 19, 19, 99, 30]`.
- Fixed cases, first-three: `[199999999.99, 0.29, 12.99]` → `[199, 290, 129]`.
- Boundary/clip expectations for ALL FIVE extractors on
  `[9.99999999999999]`: first → 9, first-two → 99, first-three → 999,
  second → 9, third → 9.
- Powers of 10 from 1e-300 to 1e300 → first digit all 1 (snap must not break
  the existing clamp).
- Hypothesis property test: for integers `i` in `[1, 10**9)`, digits of
  `i / 100.0` equal the digits read from the RIGHT-PADDED string
  `str(i).ljust(3, "0")`: first digit = `int(s[0])`, second = `int(s[1])`,
  third = `int(s[2])`, first-two = `int(s[:2])`, first-three = `int(s[:3])`.
  (Padding defines the oracle for `i < 100`: significand of 1 is 1.000….)
  Add `hypothesis==6.156.6` to the dev extras — the only dependency change in
  this pass; it reaches CI via `pip install -e ".[dev]"`. Add it to the
  validation-gate pin assertion heredoc as the seventh pin.

## Task B — `second_order_differences`: remove the ×10+round distortion

The legacy notebook formula `abs(round(diffs * 10))` destroys the second-order
test for sub-unit spacings: on `np.sort(rng.uniform(0, 100, 5000))` (spacing
~0.02), 91% of diffs round to 0 and are dropped, and surviving digits have
total-variation distance ~0.956 from the true diff digits. Digit extraction is
scale-invariant, so the ×10 was always a no-op and the round is pure damage.

New body (signature and error behavior unchanged):

```python
arr = np.asarray(data, dtype=np.float64).ravel()
finite = arr[np.isfinite(arr)]
if len(finite) < 2:
    raise ValueError("need at least 2 finite values for second-order differences")
return np.diff(np.sort(finite))
```

Ties yield exact 0.0, which the extractors map to NaN and `digit_counts`
drops. Rewrite the docstring: what changed, why, and that for integer-valued
data the digit distribution of the result is identical to the old formula's.

LICENSED existing-test updates in `tests/test_utils.py` — exactly these four
assertions change because outputs are no longer scaled by 10 or rounded:

1. basic test: expected `[10, 30]` → `[1, 3]`
2. unsorted-input test: expected `[10, 30]` → `[1, 3]`
3. finite-filter test: expected `[10, 30]` → `[1, 3]`
4. rounding test: expected `[0]` → the exact raw diff (assert with
   `pytest.approx`); rename the test — it now verifies NO rounding occurs.

New tests in `tests/test_utils.py` (seeds stated, oracle on the input side —
a NaN-count-equals-zero-count assertion is tautological and forbidden):

- Exactness: output equals `np.diff(np.sort(finite))` for a mixed
  finite/NaN/inf input.
- Integer-data equivalence: `rng = np.random.default_rng(0)`, integers
  1..10**6, n=2000 — first-two-digit COUNTS of the new output equal those of
  the old formula `np.abs(np.round(np.diff(np.sort(x)) * 10.0, 0))`
  (old formula implemented inline in the test).
- Continuous-data repair: `rng = np.random.default_rng(7)`,
  `np.sort(rng.uniform(0, 100, 5000))` — assert the output contains ZERO
  zeros and `extract_first_two_digits(output)` contains ZERO NaNs (under the
  legacy formula ~91% were destroyed; this pins the repair).
- Tie semantics: input `[1.0, 1.0, 1.04]` → output `[0.0, 0.04]`
  (`pytest.approx`); `extract_first_two_digits` on it yields exactly one NaN
  (the tie) and one valid digit pair (40).

## Task C — small-sample warnings on the EFFECTIVE sample

`constants.py` claims sub-300 samples restrict analysis; nothing enforces
anything. Gate-1 correctly required warnings to key on the sample actually
analyzed (`freq.total`), not the cleaned input length — 999 ties + 1 value
gives a second-order test with an effective n of 1.

- In `core.py` add module-level `class SmallSampleWarning(UserWarning): ...`;
  export from `pybenford/__init__.py` (imports and `__all__`).
- Constructor (data-level): in `BenfordAnalysis.__init__`, if
  `self.n < MIN_SAMPLE_SIZE_RECOMMENDED` → warn (SmallSampleWarning,
  stacklevel=2) citing Nigrini §4.2.
- Per-test (effective-level): add keyword-only parameter
  `warn_min_n: int | None = None` to `_run_digit_test`; after `freq` is
  computed, if `warn_min_n is not None and freq.total < warn_min_n` → warn
  (stacklevel=3) naming the test and `freq.total`. Callers
  `first_two_digits`, `first_three_digits`, and `second_order` pass
  `warn_min_n=MIN_SAMPLE_SIZE_FIRST_TWO`. `first_digit`, `second_digit`,
  `third_digit`, `last_two_digits` pass nothing (9/10/100-bin tests have no
  300 threshold).
- Summation: add field `n_valid: int` to `SummationFrequencies`, populated
  with `len(valid_ft)` in `summation_by_digits`. In `BenfordAnalysis.summation`,
  warn when `sf.n_valid < MIN_SAMPLE_SIZE_FIRST_TWO`. (`SummationResult.n`
  semantics unchanged.)
- Fix the `MIN_SAMPLE_SIZE_FIRST_TWO` docstring in `constants.py`: replace the
  false "restricts analysis" sentence with the `SmallSampleWarning` behavior.

New tests in `tests/test_core.py`:

- Parameterized over ALL FOUR warning sites (`first_two_digits`,
  `first_three_digits`, `second_order`, `summation`): warns at effective
  n=299-class fixtures, silent at 300 (`pytest.warns` /
  `warnings.catch_warnings` with `simplefilter("error", SmallSampleWarning)`).
  Construct boundary fixtures from log-uniform data of exact length; for
  `second_order` remember effective n is len-1.
- Constructor boundary: warns at n=999, silent at n=1000.
- Effective-vs-input case from gate-1: 1000 values with 999 identical → the
  `second_order` call warns even though `self.n == 1000`.
- Existing tests must still pass — warnings are not errors. If any existing
  test breaks on the new warning, report it; do not weaken the fix.

## Task D — `mean_angle`: vector direction, not arithmetic mean

In `statistics.py`, `mantissa_arc_test`:
`mean_angle = float(np.arctan2(mean_y, mean_x))`.

Docstring: "direction of the gravity center in radians, in [-π, π] (the raw
`np.arctan2` range — `arctan2(-0.0, -1.0)` returns -π); unstable and of no
diagnostic value when the gravity center is at or near the origin (L2 ≈ 0,
i.e. conforming data)."

No existing test asserts a computed `mean_angle`, so Task D modifies no
existing assertions — it only adds tests.

New test: `rng = np.random.default_rng(42)` used FRESH (no prior draws), fixture
`np.concatenate([10 ** rng.uniform(0.0, 0.1, 900), 10 ** rng.uniform(0.9, 1.0, 100)])`
— assert `res.mean_angle == pytest.approx(np.arctan2(res.mean_y, res.mean_x))`
and that it differs from the arithmetic mean of the angles by more than 0.5.
Verified values for this exact fixture: atan2 = 0.2525148462, arithmetic
mean = 0.8778727267, difference = -0.6253578804.

## Task E — `z_statistic`: exp=0, obs=0 must give 0, not inf

Outer branch becomes
`np.where(exp == 0.0, np.where(obs > 0.0, np.inf, 0.0), ...)`.

New tests: `z_statistic(np.array([0.0]), np.array([0.0]), 100)[0] == 0.0`;
`z_statistic(np.array([0.1]), np.array([0.0]), 100)[0] == np.inf`.

## Task F — label the summation test's inferential output as heuristic

Gate-1 correctly extended the audit finding: BOTH the χ² and the per-digit
z-scores in the summation test apply count-proportion variance models
(`EP*(1-EP)/N`, multinomial χ²) to proportions of VALUE SUMS, which has no
sampling justification; Nigrini provides no critical values. Keep all
computations (descriptive/screening diagnostics; the plot remains primary)
but remove authoritative verdict language:

- `SummationResult.__str__`:
  - flagged block header → `" Flagged Digits (heuristic z-scores) (...):"`
  - no-flags branch → `" No digits flagged by heuristic z-scores at alpha=...."`
    (the current wording is an unqualified inferential verdict too)
  - chi-square line → `" Chi-Square (heuristic): ..."` with verdict wording
    `"— exceeds critical"` / `"— within critical"` (no Pass/FAIL).
- Docstrings (`BenfordAnalysis.summation`, `summation_by_digits`, and the
  inline comment in `summation`): state that z and χ² here are heuristic
  screens — the variance model assumes count proportions, not sum shares —
  and interpretation should rest on the plot and the magnitude of deviations.
- LICENSED: update the affected `tests/test_display.py` assertions
  (enumerate each in the summary).

New display tests: fixtures exercising BOTH verdict branches ("exceeds
critical" and "within critical") and asserting the heuristic labels appear —
the non-flagged fixture must assert the z-score heuristic wording
specifically ("No digits flagged by heuristic z-scores"), not merely the χ²
heuristic label. The current display fixture is always-significant; add a
non-significant one.

## Task G — repository docs consistency

Two design docs still prescribe the OLD behavior and would contradict the
code after this pass (gate-1 finding):

- `docs/statistics_py_spec.md` — the line defining `mean_angle` as the
  arithmetic mean of angles: update to the atan2 definition with a note
  "(amended in Pass 2; original definition was the arithmetic mean)".
- `docs/result_display_spec.md` — TWO lines: (a) the line prescribing `FAIL`
  wording for the summation chi-square, and (b) the line prescribing the old
  `Flagged Digits` header for the summation display — update both to the
  Task F wording with the same amendment note.

No other edits to either document.

## Task H — CHANGELOG

Add `## [Unreleased]` at the top: Python floor 3.10 (from Pass 1); Fixed —
digit-boundary snap, second-order distortion, mean_angle, z edge case;
Added — SmallSampleWarning, `SummationFrequencies.n_valid`; Changed —
summation inferential output labeled heuristic. No version number.

## Validation gate (identical to Pass 1, pinned toolchain)

Post-edit `pip install -e ".[dev]"`, checkout assert, the pin-assertion
heredoc extended to seven pins (`hypothesis==6.156.6` added), then:
`ruff check src tests` · `ruff format --check src tests` · `mypy src` ·
`pytest -q`. Expect: >212 tests (report the exact count), all passing,
coverage ≥ 99 enforced. Paste real output.

## Commit plan

1. `docs: add correctness fixes spec (pass 2)` — this file.
2. `fix: correctness fixes from adversarial audit (ULP-relative digit snap,
   second-order, effective-n warnings, mantissa angle, z edge case, summation
   heuristic labeling)` — all Tasks A–H, gate-validated. Do not push until
   the diff review (gate 2) approves.

## Done criteria

- All probe reproductions and boundary cases above pass as tests; full gate
  green with the seven pins.
- `git status --porcelain` empty; exactly two new commits; not pushed.
- Summary lists: new test count, every existing assertion modified (licenses:
  Task B's four and Task F's enumerated display assertions — and nothing
  else; Task D modifies none), and any discrepancy encountered.
