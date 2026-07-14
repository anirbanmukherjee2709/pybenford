# Result Display Spec — `__str__` Methods for All Result Dataclasses

## CRITICAL INSTRUCTIONS FOR CLAUDE CODE

1. **Read first, code second.** Before writing ANY code, read the following files in full:
   - `src/pybenford/core.py`
   - `src/pybenford/statistics.py`
   - `src/pybenford/utils.py`
   - Every file that defines a result dataclass (TestResult, SummationResult, DistortionResult, MantissaArcResult, DuplicationResult). Use `grep -rn "class TestResult" src/` and similar to find them.
2. **Do NOT change any existing logic, formulas, or field names.** This spec ONLY adds `__str__` methods and fixes one casting bug. No other changes.
3. **Do NOT move, rename, or restructure any dataclass.** Add the `__str__` method directly inside the existing class definition.
4. **Match existing code style** — look at how the codebase handles type hints, imports, and formatting. Follow it exactly.
5. **After implementation, run:** `python -m pytest tests/ -v` and `python examples/test_census_population.py` to verify output looks correct and nothing broke.
6. **Report what you changed before committing.** List every file modified, every method added, and the bug fix.

---

## Problem

Currently, `print(result)` dumps the raw dataclass `__repr__` — a single wall of text with full numpy arrays. Example:

```
TestResult(test_name='first_digit', digits=array([1, 2, 3, ...]), counts=array([943, 588, ...]), observed=array([0.29993639, ...]), ...)
```

This is unusable. We need human-readable `__str__` methods on every result dataclass.

---

## Bug Fix: `np.False_` Leaking into Output

The `ks_significant` field in `TestResult` currently stores `np.False_` / `np.True_` (numpy bool) instead of Python `bool`. Same may apply to `chi_square_significant`.

**Fix:** In whatever code constructs TestResult and SummationResult, wrap the KS and chi-square significance values in `bool()`. Example:

```python
# BEFORE (buggy)
ks_significant=suprem > ks_critical,

# AFTER (fixed)
ks_significant=bool(suprem > ks_critical),
```

Search for ALL places where TestResult(...) and SummationResult(...) are constructed and ensure every `bool`-typed field gets a Python `bool`, not `np.bool_`.

---

## Design Rules for ALL `__str__` Methods

1. **Use plain ASCII box-drawing.** `=` for top/bottom borders, `-` for internal separators. No Unicode box characters (they break in some terminals).
2. **Fixed width of 55 characters** for the box.
3. **Percentages displayed as `XX.XX%`** (2 decimal places, multiply proportion by 100).
4. **Z-scores displayed as `X.XX`** (2 decimal places).
5. **Floats displayed as `X.XXXX`** (4 decimal places) for MAD, KS, chi-square values.
6. **Counts and integers use comma separators** for thousands (e.g., `3,144`).
7. **Flagged digits marked with `*`** at end of row.
8. **Test name displayed as title case with spaces.** Map `test_name` field:
   - `"first_digit"` → `"First Digit Test"`
   - `"second_digit"` → `"Second Digit Test"`
   - `"third_digit"` → `"Third Digit Test"`
   - `"first_two_digits"` → `"First Two Digits Test"`
   - `"first_three_digits"` → `"First Three Digits Test"`
   - `"last_two_digits"` → `"Last Two Digits Test"`
   - `"second_order"` → `"Second Order Test"`
   - `"summation"` → `"Summation Test"`
9. **MAD conformity displayed as readable text.** Map `mad_conformity` field:
   - `"close_conformity"` → `"Close Conformity"`
   - `"acceptable_conformity"` → `"Acceptable Conformity"`
   - `"marginally_acceptable_conformity"` → `"Marginally Acceptable"`
   - `"nonconformity"` → `"Nonconformity"`
   - `"not_applicable"` → `"N/A (no threshold defined)"`
10. **Pass/Fail for chi-square and KS.** If NOT significant → `"Pass"`. If significant → `"FAIL"`.

---

## 1. TestResult.__str__

### Adaptive table logic

- If `len(self.digits) <= 10` → show FULL digit table (all rows).
- If `len(self.digits) > 10` → show ONLY flagged rows (where `significant_flags == True`). If no digits flagged, print `"No individual digits flagged."`.

### Full table format (≤10 digits)

```
=======================================================
  First Digit Test  (n=3,144  alpha=0.05)
=======================================================
 Digit   Count   Observed   Expected   Z-Score   Sig
     1     943    29.99%     30.10%      0.11
     2     588    18.70%     17.61%      1.59
     3     382    12.15%     12.49%      0.56
     4     295     9.38%      9.69%      0.55
     5     250     7.95%      7.92%      0.04
     6     191     6.08%      6.69%      1.35
     7     175     5.57%      5.80%      0.52
     8     169     5.38%      5.12%      0.62
     9     151     4.80%      4.58%      0.57
-------------------------------------------------------
 MAD:        0.0036 — Close Conformity
 Chi-Square: 5.6231  (critical: 15.5073) — Pass
 KS:         0.0098  (critical: 0.0242)  — Pass
=======================================================
```

### Flagged-only format (>10 digits)

```
=======================================================
  First Two Digits Test  (n=3,144  alpha=0.05)
=======================================================
 Flagged Digits (6 of 90):
 Digit   Count   Observed   Expected   Z-Score
    25      23     0.73%      1.22%      2.43  *
    49      16     0.51%      0.88%      2.12  *
    66       8     0.25%      0.57%      2.22  *
    67       9     0.29%      0.56%      1.94  *
    70      29     0.92%      0.62%      2.08  *
    56      33     1.05%      0.65%      2.65  *
-------------------------------------------------------
 MAD:        0.0016 — Acceptable Conformity
 Chi-Square: 111.1969 (critical: 112.0220) — Pass
 KS:         0.0118   (critical: 0.0242)   — Pass
=======================================================
```

### No-flags format

If `len(self.digits) > 10` and no digits are flagged:

```
 No individual digits flagged at alpha=0.05.
```

(replaces the flagged digits section)

### Column alignment notes

- `Digit` column: right-aligned, width 5
- `Count` column: right-aligned, width 7 (with comma formatting)
- `Observed` and `Expected`: right-aligned, width 8, format as `XX.XX%`
- `Z-Score`: right-aligned, width 8, format as `X.XX`
- `Sig` column: only shows `*` if flagged, otherwise blank

---

## 2. SummationResult.__str__

Same adaptive logic as TestResult but display `Sum` instead of `Count`.

### Full format (always use flagged-only since digits are always 90)

```
=======================================================
  Summation Test  (n=3,144  alpha=0.05)
=======================================================
 Grand Sum: 341,784,857

 Flagged Digits (heuristic z-scores) (57 of 90):    (amended in Pass 2; original prescribed "Flagged Digits (57 of 90):")
 Digit         Sum   Observed   Expected   Z-Score
    10  13,528,984     3.96%      1.11%     15.15  *
    11   7,665,693     2.24%      1.11%      5.97  *
    ...
-------------------------------------------------------
 Chi-Square (heuristic): 2013.5123 (critical: 112.0220) — exceeds critical    (amended in Pass 2; original prescribed FAIL wording)
=======================================================
```

Notes:
- SummationResult does NOT have `ks_statistic`, `ks_critical`, `ks_significant`, `mad`, or `mad_conformity` fields. Do NOT try to print them.
- Check the actual SummationResult fields before implementing. Only print fields that exist.
- `Sum` column: right-aligned, comma-formatted, width 12.

---

## 3. DistortionResult.__str__

```
=======================================================
  Distortion Factor Test
=======================================================
 Distortion Factor:  -0.0019 (Understated)
 Actual Mean:         39.0129
 Expected Mean:       39.0865
 Z-Statistic:         -0.1653  (p=0.8687)
 Significant:         No
=======================================================
```

Notes:
- `direction` field values: `"understated"` → `"Understated"`, `"overstated"` → `"Overstated"`, `"none"` or empty → omit the parenthetical.
- `significant` field: `True` → `"Yes"`, `False` → `"No"`.
- Distortion factor and percentage: 4 decimal places.
- Means: 4 decimal places.
- Z-statistic: 4 decimal places.
- P-value: 4 decimal places.

---

## 4. MantissaArcResult.__str__

```
=======================================================
  Mantissa Arc Test
=======================================================
 Gravity Center:  (-0.0093, 0.0220)
 L2 Statistic:     0.0239
 Mean Angle:        3.1387 rad  (179.87 deg)
 P-Value:           0.1663
=======================================================
```

Notes:
- Convert `mean_angle` from radians to degrees for the parenthetical: `degrees = mean_angle * 180 / pi`.
- Gravity center x and y: 4 decimal places.
- L2: 4 decimal places.
- P-value: 4 decimal places.
- MantissaArcResult may or may not have a `significant` field. Check the actual dataclass. If it has one, print it. If not, let the user interpret the p-value.

---

## 5. DuplicationResult.__str__

```
=======================================================
  Number Duplication Test
=======================================================
 Total Records: 3,144  |  Unique Values: 3,092

 Top Duplicated Values:
  Value       Count   First-Two
  74,967          2          74
  54,037          2          54
  41,853          2          41
  40,083          2          40
  33,894          2          33
  33,322          2          33
  32,515          2          32
  31,171          2          31
  27,983          2          27
  26,799          2          26
=======================================================
```

Notes:
- `values` and `counts` are parallel arrays. `first_two_digits` is also parallel.
- Value: right-aligned, comma-formatted, width 10.
- Count: right-aligned, width 7.
- First-Two: right-aligned, width 9.
- If all counts are 1 (no duplicates), print: `"No duplicated values found."`

---

## Testing Requirements

After implementing all `__str__` methods:

1. Run `python examples/test_census_population.py` and visually verify the output matches the formats above.
2. Run `python -m pytest tests/ -v` to ensure nothing is broken.
3. Write a quick test in `tests/test_display.py` that:
   - Creates a TestResult with known values and calls `str()` on it
   - Asserts the output contains expected substrings ("Close Conformity", "Pass", "FAIL", etc.)
   - Tests both the ≤10 digit path and the >10 digit path
   - Tests the no-flagged-digits case
   - Tests DistortionResult, MantissaArcResult, DuplicationResult str output

---

## Checklist Before Committing

- [ ] `__str__` added to TestResult
- [ ] `__str__` added to SummationResult
- [ ] `__str__` added to DistortionResult
- [ ] `__str__` added to MantissaArcResult
- [ ] `__str__` added to DuplicationResult
- [ ] `np.bool_` → `bool()` fix applied everywhere TestResult/SummationResult is constructed
- [ ] `python -m pytest tests/ -v` passes
- [ ] `python examples/test_census_population.py` produces clean formatted output
- [ ] `ruff check src/pybenford/` passes
- [ ] `mypy src/pybenford/` passes (or at least no new errors)
