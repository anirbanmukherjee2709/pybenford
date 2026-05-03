"""
Test pybenford against US Census county population estimates.

Population data is a textbook Benford-conforming dataset — it spans
multiple orders of magnitude and arises from a natural process. If
pybenford produces correct results here, the core engine is working.

Usage:
    cd ~/Anirban/projects/pybenford
    python examples/test_census_population.py
"""

import pandas as pd
import numpy as np
from pybenford import BenfordAnalysis

# ── 1. Load and prep data ──────────────────────────────────────────────
DATA_PATH = "examples/data/co-est2025-alldata.csv"

df = pd.read_csv(DATA_PATH, encoding="latin-1")

# SUMLEV 50 = county-level rows (excludes state-level summaries at SUMLEV 40)
df_counties = df[df["SUMLEV"] == 50].copy()
population = df_counties["POPESTIMATE2025"].dropna().values

print(f"Dataset: US Census County Population Estimates 2025")
print(f"Records: {len(population):,}")
print(f"Range: {population.min():,.0f} – {population.max():,.0f}")
print(f"Orders of magnitude: {np.log10(population.max()) - np.log10(population[population > 0].min()):.1f}")
print("=" * 70)


# ── 2. Initialize BenfordAnalysis ──────────────────────────────────────
ba = BenfordAnalysis(population, sign_filter="positive", drop_zero=True)


# ── 3. Run all digit tests ─────────────────────────────────────────────
tests = {
    "First Digit":       ba.first_digit,
    "Second Digit":      ba.second_digit,
    "Third Digit":       ba.third_digit,
    "First Two Digits":  ba.first_two_digits,
    "First Three Digits": ba.first_three_digits,
    "Last Two Digits":   ba.last_two_digits,
}

for name, test_fn in tests.items():
    print(f"\n{'─' * 70}")
    print(f"TEST: {name}")
    print(f"{'─' * 70}")
    try:
        result = test_fn(alpha=0.05)
        print(result)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


# ── 4. Second-order test ───────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("TEST: Second Order (first_two digits)")
print(f"{'─' * 70}")
try:
    result = ba.second_order(alpha=0.05, digits="first_two")
    print(result)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")


# ── 5. Summation test ─────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("TEST: Summation")
print(f"{'─' * 70}")
try:
    result = ba.summation(alpha=0.05)
    print(result)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")


# ── 6. Distortion factor ──────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("TEST: Distortion Factor")
print(f"{'─' * 70}")
try:
    result = ba.distortion_factor(alpha=0.05)
    print(result)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")


# ── 7. Mantissa arc test ──────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("TEST: Mantissa Arc")
print(f"{'─' * 70}")
try:
    result = ba.mantissa_arc()
    print(result)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")


# ── 8. Number duplication ─────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("TEST: Number Duplication (top 10)")
print(f"{'─' * 70}")
try:
    result = ba.number_duplication(top_n=10)
    print(result)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")


print(f"\n{'=' * 70}")
print("All tests completed.")
