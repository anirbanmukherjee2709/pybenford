"""Tests for pybenford.digits module."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pybenford.digits import (
    extract_first_digit,
    extract_first_three_digits,
    extract_first_two_digits,
    extract_second_digit,
    extract_third_digit,
)

# ═══════════════════════════════════════════════════════════════════════════
# Boundary snap — float drift must not shift decimal-entered values across
# digit boundaries (0.29 → 28 class of bugs)
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundarySnap:
    def test_first_digit_fixed_cases(self) -> None:
        values = [0.29, 0.64, 12.99, 1999999.99, 19999999.99, 199999999.99, 9.99999999999999, 0.30]
        result = extract_first_digit(values)
        np.testing.assert_array_equal(result, [2.0, 6.0, 1.0, 1.0, 1.0, 1.0, 9.0, 3.0])

    def test_first_two_digits_fixed_cases(self) -> None:
        values = [0.29, 0.64, 12.99, 1999999.99, 19999999.99, 199999999.99, 9.99999999999999, 0.30]
        result = extract_first_two_digits(values)
        np.testing.assert_array_equal(result, [29.0, 64.0, 12.0, 19.0, 19.0, 19.0, 99.0, 30.0])

    def test_first_three_digits_fixed_cases(self) -> None:
        result = extract_first_three_digits([199999999.99, 0.29, 12.99])
        np.testing.assert_array_equal(result, [199.0, 290.0, 129.0])

    def test_clip_at_top_of_range_all_extractors(self) -> None:
        values = [9.99999999999999]
        np.testing.assert_array_equal(extract_first_digit(values), [9.0])
        np.testing.assert_array_equal(extract_first_two_digits(values), [99.0])
        np.testing.assert_array_equal(extract_first_three_digits(values), [999.0])
        np.testing.assert_array_equal(extract_second_digit(values), [9.0])
        np.testing.assert_array_equal(extract_third_digit(values), [9.0])

    def test_powers_of_ten_first_digit_is_one(self) -> None:
        powers = [10.0**k for k in range(-300, 301)]
        result = extract_first_digit(powers)
        np.testing.assert_array_equal(result, np.ones(len(powers)))

    @given(st.integers(min_value=1, max_value=10**9 - 1))
    def test_digits_match_string_oracle(self, i: int) -> None:
        # Right-padding defines the oracle for i < 100: the significand of
        # 1 is 1.000…, so str(1) reads as "100" for digit purposes.
        s = str(i).ljust(3, "0")
        x = [i / 100.0]
        assert extract_first_digit(x)[0] == int(s[0])
        assert extract_second_digit(x)[0] == int(s[1])
        assert extract_third_digit(x)[0] == int(s[2])
        assert extract_first_two_digits(x)[0] == int(s[:2])
        assert extract_first_three_digits(x)[0] == int(s[:3])
