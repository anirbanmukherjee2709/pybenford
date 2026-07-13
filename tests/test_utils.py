"""Tests for pybenford.utils module."""

from __future__ import annotations

import numpy as np
import pytest

from pybenford.digits import extract_first_digit, extract_first_two_digits
from pybenford.utils import (
    DataProfile,
    DigitFrequencies,
    DuplicationResult,
    SummationFrequencies,
    clean_numeric_array,
    data_profile,
    digit_counts,
    number_duplication,
    second_order_differences,
    summation_by_digits,
    to_numeric_array,
)

# ── helpers for duck-typed inputs ──────────────────────────────────────────


class _PandasLike:
    """Mimics pandas Series via .to_numpy()."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        return np.array(self._values, dtype=np.float64)


class _PolarsLike:
    """Mimics polars Series via .to_list()."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_list(self) -> list[float]:
        return list(self._values)


# ═══════════════════════════════════════════════════════════════════════════
# 1. to_numeric_array
# ═══════════════════════════════════════════════════════════════════════════


class TestToNumericArray:
    def test_from_list(self) -> None:
        arr = to_numeric_array([1, 2, 3])
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
        assert arr.dtype == np.float64

    def test_from_tuple(self) -> None:
        arr = to_numeric_array((4.0, 5.0))
        np.testing.assert_array_equal(arr, [4.0, 5.0])

    def test_from_numpy_int(self) -> None:
        arr = to_numeric_array(np.array([1, 2], dtype=np.int32))
        assert arr.dtype == np.float64

    def test_from_numpy_float(self) -> None:
        src = np.array([1.5, 2.5], dtype=np.float64)
        arr = to_numeric_array(src)
        np.testing.assert_array_equal(arr, src)

    def test_from_pandas_like(self) -> None:
        arr = to_numeric_array(_PandasLike([10.0, 20.0]))
        np.testing.assert_array_equal(arr, [10.0, 20.0])

    def test_from_polars_like(self) -> None:
        arr = to_numeric_array(_PolarsLike([7.0, 8.0]))
        np.testing.assert_array_equal(arr, [7.0, 8.0])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            to_numeric_array([])

    def test_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            to_numeric_array(np.array([], dtype=np.float64))

    def test_2d_array_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            to_numeric_array(np.array([[1, 2], [3, 4]]))

    def test_unsupported_dict_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported type"):
            to_numeric_array({"a": 1})

    def test_unsupported_int_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported type"):
            to_numeric_array(42)

    def test_unsupported_set_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported type"):
            to_numeric_array({1, 2, 3})

    def test_preserves_nan(self) -> None:
        arr = to_numeric_array([1.0, np.nan, 3.0])
        assert np.isnan(arr[1])


# ═══════════════════════════════════════════════════════════════════════════
# 2. clean_numeric_array
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanNumericArray:
    def test_removes_nan_inf(self) -> None:
        arr, rpt = clean_numeric_array([1.0, np.nan, np.inf, -np.inf, 2.0])
        assert rpt.nan_inf_removed == 3
        assert len(arr) == 2

    def test_drops_zeros(self) -> None:
        arr, rpt = clean_numeric_array([0, 1, 0, 2], drop_zero=True)
        assert rpt.zeros_removed == 2
        np.testing.assert_array_equal(arr, [1.0, 2.0])

    def test_keeps_zeros(self) -> None:
        arr, rpt = clean_numeric_array([0, 1, 0, 2], drop_zero=False)
        assert rpt.zeros_removed == 0
        assert 0.0 in arr

    def test_sign_filter_all(self) -> None:
        arr, rpt = clean_numeric_array([-3, -1, 2, 5])
        np.testing.assert_array_equal(arr, [3.0, 1.0, 2.0, 5.0])
        assert rpt.sign_filtered == 0

    def test_sign_filter_positive(self) -> None:
        arr, rpt = clean_numeric_array([-3, -1, 2, 5], sign_filter="positive")
        np.testing.assert_array_equal(arr, [2.0, 5.0])
        assert rpt.sign_filtered == 2

    def test_sign_filter_negative(self) -> None:
        arr, rpt = clean_numeric_array([-3, -1, 2, 5], sign_filter="negative")
        np.testing.assert_array_equal(arr, [3.0, 1.0])
        assert rpt.sign_filtered == 2

    def test_min_abs_value(self) -> None:
        arr, rpt = clean_numeric_array([1, 5, 15, 100], min_abs_value=10.0)
        np.testing.assert_array_equal(arr, [15.0, 100.0])
        assert rpt.below_threshold_removed == 2

    def test_invalid_sign_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="sign_filter"):
            clean_numeric_array([1, 2], sign_filter="invalid")

    def test_all_nan_returns_empty(self) -> None:
        arr, rpt = clean_numeric_array([np.nan, np.nan])
        assert len(arr) == 0
        assert rpt.nan_inf_removed == 2
        assert rpt.clean_count == 0

    def test_all_zeros_dropped(self) -> None:
        arr, rpt = clean_numeric_array([0, 0, 0])
        assert len(arr) == 0
        assert rpt.zeros_removed == 3

    def test_report_counts_consistent(self) -> None:
        _, rpt = clean_numeric_array([1, -2, 0, np.nan, 5, np.inf, -0.5, 3])
        removed = (
            rpt.nan_inf_removed
            + rpt.zeros_removed
            + rpt.sign_filtered
            + rpt.below_threshold_removed
        )
        assert rpt.original_count == rpt.clean_count + removed

    def test_combined_pipeline(self) -> None:
        arr, rpt = clean_numeric_array(
            [np.nan, 0, -5, 3, 1, -100, np.inf],
            sign_filter="positive",
            min_abs_value=2.0,
            drop_zero=True,
        )
        np.testing.assert_array_equal(arr, [3.0])
        assert rpt.original_count == 7


# ═══════════════════════════════════════════════════════════════════════════
# 3. data_profile
# ═══════════════════════════════════════════════════════════════════════════


class TestDataProfile:
    def test_mixed_data(self) -> None:
        data = [100, 5, 0, -3, -50, np.nan]
        prof = data_profile(data)
        assert isinstance(prof, DataProfile)
        assert prof.total_count == 6
        assert prof.invalid_count == 1
        names = [s.name for s in prof.strata]
        assert "Large positive" in names
        assert "Small positive" in names
        assert "Zero" in names
        assert "Small negative" in names
        assert "Large negative" in names
        assert "Invalid" in names

    def test_all_positive(self) -> None:
        prof = data_profile([10, 20, 30])
        lp = next(s for s in prof.strata if s.name == "Large positive")
        assert lp.count == 3
        assert prof.invalid_count == 0

    def test_all_invalid(self) -> None:
        prof = data_profile([np.nan, np.inf, -np.inf])
        assert prof.invalid_count == 3
        lp = next(s for s in prof.strata if s.name == "Large positive")
        assert lp.count == 0

    def test_empty_data(self) -> None:
        prof = data_profile(np.array([], dtype=np.float64))
        assert prof.total_count == 0

    def test_extra_thresholds(self) -> None:
        data = [15, 50, 150, 500, 1500]
        prof = data_profile(data, threshold=10.0, extra_thresholds=[100, 1000])
        names = [s.name for s in prof.strata]
        assert ">= 10.0 and < 100" in names
        assert ">= 100 and < 1000" in names
        assert ">= 1000" in names
        sub = {s.name: s.count for s in prof.strata}
        assert sub[">= 10.0 and < 100"] == 2
        assert sub[">= 100 and < 1000"] == 2
        assert sub[">= 1000"] == 1

    def test_percentages_sum(self) -> None:
        data = [100, 5, 0, -3, -50]
        prof = data_profile(data)
        total_pct = sum(s.percentage for s in prof.strata)
        assert total_pct == pytest.approx(100.0)

    def test_zero_values(self) -> None:
        prof = data_profile([0, 0, 0])
        z = next(s for s in prof.strata if s.name == "Zero")
        assert z.count == 3

    def test_stratum_sum(self) -> None:
        prof = data_profile([10, 20])
        lp = next(s for s in prof.strata if s.name == "Large positive")
        assert lp.total_sum == pytest.approx(30.0)

    def test_extra_thresholds_below_threshold_ignored(self) -> None:
        prof = data_profile([15, 50], threshold=10.0, extra_thresholds=[5])
        names = [s.name for s in prof.strata]
        assert "Large positive" in names


# ═══════════════════════════════════════════════════════════════════════════
# 4. digit_counts
# ═══════════════════════════════════════════════════════════════════════════


class TestDigitCounts:
    def test_first_digit(self) -> None:
        data = [100, 200, 300, 100, 100]
        result = digit_counts(data, extract_first_digit, range(1, 10))
        assert isinstance(result, DigitFrequencies)
        assert result.total == 5
        assert result.counts[0] == 3  # digit 1
        assert result.counts[1] == 1  # digit 2
        assert result.counts[2] == 1  # digit 3

    def test_first_two_digits(self) -> None:
        data = [1050, 1150, 1050]
        result = digit_counts(data, extract_first_two_digits, range(10, 100))
        assert result.counts[0] == 2  # digit 10: two 1050s
        assert result.counts[1] == 1  # digit 11: one 1150
        assert result.total == 3

    def test_proportions_sum_to_one(self) -> None:
        result = digit_counts([10, 20, 30, 40, 50], extract_first_digit, range(1, 10))
        assert result.proportions.sum() == pytest.approx(1.0)

    def test_missing_digits_get_zero(self) -> None:
        result = digit_counts([100, 100], extract_first_digit, range(1, 10))
        assert result.counts[0] == 2
        assert np.all(result.counts[1:] == 0)

    def test_single_value(self) -> None:
        result = digit_counts([42], extract_first_digit, range(1, 10))
        assert result.total == 1
        assert result.counts[3] == 1  # digit 4

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid digits"):
            digit_counts([0, np.nan], extract_first_digit, range(1, 10))


# ═══════════════════════════════════════════════════════════════════════════
# 5. summation_by_digits
# ═══════════════════════════════════════════════════════════════════════════


class TestSummationByDigits:
    def test_basic(self) -> None:
        data = [10, 11, 20, 21, 10]
        result = summation_by_digits(data)
        assert isinstance(result, SummationFrequencies)
        assert len(result.digits) == 90
        assert result.grand_sum == pytest.approx(72.0)
        # Digit 10: values 10 + 10 = 20
        assert result.sums[0] == pytest.approx(20.0)
        # Digit 11: value 11
        assert result.sums[1] == pytest.approx(11.0)
        # Digit 20: value 20
        assert result.sums[10] == pytest.approx(20.0)

    def test_proportions_sum_to_one(self) -> None:
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        result = summation_by_digits(data)
        assert result.proportions.sum() == pytest.approx(1.0)

    def test_expected_proportions_uniform(self) -> None:
        result = summation_by_digits([10, 20, 30])
        np.testing.assert_allclose(result.expected_proportions, 1.0 / 90.0)

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid"):
            summation_by_digits([0, np.nan])


# ═══════════════════════════════════════════════════════════════════════════
# 6. second_order_differences
# ═══════════════════════════════════════════════════════════════════════════


class TestSecondOrderDifferences:
    def test_basic_sorted(self) -> None:
        result = second_order_differences([1, 2, 5])
        # diffs = [1, 3], * 10 = [10, 30], round, abs
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_unsorted_data(self) -> None:
        result = second_order_differences([5, 1, 2])
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_length_is_n_minus_1(self) -> None:
        result = second_order_differences([1, 2, 3, 4, 5])
        assert len(result) == 4

    def test_identical_values(self) -> None:
        result = second_order_differences([7, 7, 7])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_single_value_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            second_order_differences([42])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            second_order_differences(np.array([], dtype=np.float64))

    def test_filters_nan_inf(self) -> None:
        result = second_order_differences([1, np.nan, 2, np.inf, 5])
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            second_order_differences([np.nan, np.nan])

    def test_rounding(self) -> None:
        # 1.03 and 1.07 → diff = 0.04, *10 = 0.4, round = 0.0
        result = second_order_differences([1.03, 1.07])
        np.testing.assert_array_equal(result, [0.0])


# ═══════════════════════════════════════════════════════════════════════════
# 7. number_duplication
# ═══════════════════════════════════════════════════════════════════════════


class TestNumberDuplication:
    def test_basic(self) -> None:
        data = [100, 200, 100, 300, 200, 100]
        result = number_duplication(data, top_n=3)
        assert isinstance(result, DuplicationResult)
        assert result.total_records == 6
        assert result.total_unique == 3
        assert result.values[0] == 100.0
        assert result.counts[0] == 3

    def test_sort_count_desc_value_desc(self) -> None:
        data = [10, 20, 10, 20, 30]
        result = number_duplication(data, top_n=3)
        # 10 and 20 each appear 2 times; 20 > 10 so 20 first
        assert result.values[0] == 20.0
        assert result.values[1] == 10.0
        assert result.values[2] == 30.0

    def test_top_n_exceeds_unique(self) -> None:
        data = [1, 2, 3]
        result = number_duplication(data, top_n=100)
        assert len(result.values) == 3

    def test_all_same_value(self) -> None:
        result = number_duplication([42, 42, 42])
        assert result.total_unique == 1
        assert result.counts[0] == 3

    def test_first_two_digits_included(self) -> None:
        result = number_duplication([1234, 1234])
        assert result.first_two_digits[0] == 12.0

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid values"):
            number_duplication([np.nan, np.inf])

    def test_single_value(self) -> None:
        result = number_duplication([99])
        assert result.total_unique == 1
        assert result.total_records == 1
        assert result.counts[0] == 1

    def test_negative_values(self) -> None:
        result = number_duplication([-50, -50, 50])
        assert result.total_unique == 2
