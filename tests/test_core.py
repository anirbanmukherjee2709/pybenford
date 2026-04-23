"""Tests for pybenford.core — the BenfordAnalysis public API."""

from __future__ import annotations

import numpy as np
import pytest

from pybenford.core import BenfordAnalysis, SummationResult, TestResult
from pybenford.distributions import first_digit_distribution, first_two_digits_distribution
from pybenford.statistics import DistortionResult, MantissaArcResult
from pybenford.utils import DuplicationResult

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _benford_data(n: int = 5000, seed: int = 42) -> np.ndarray:  # type: ignore[type-arg]
    """Generate data whose first digits follow Benford's law."""
    rng = np.random.default_rng(seed)
    mantissas = rng.uniform(0, 1, n)
    return 10.0 ** (mantissas + rng.integers(1, 5, n))


# ═══════════════════════════════════════════════════════════════════════════
# Constructor
# ═══════════════════════════════════════════════════════════════════════════


class TestConstructor:
    def test_from_list(self) -> None:
        ba = BenfordAnalysis([10, 20, 30, 40, 50, 60, 70, 80, 90])
        assert ba.n == 9

    def test_from_numpy(self) -> None:
        ba = BenfordAnalysis(np.arange(10, 110, dtype=np.float64))
        assert ba.n == 100

    def test_sign_filter_positive(self) -> None:
        ba = BenfordAnalysis([-10, -20, 30, 40], sign_filter="positive")
        assert ba.n == 2
        np.testing.assert_array_equal(ba.clean_data, [30.0, 40.0])

    def test_sign_filter_negative(self) -> None:
        ba = BenfordAnalysis([-10, -20, 30, 40], sign_filter="negative")
        assert ba.n == 2
        np.testing.assert_array_equal(ba.clean_data, [10.0, 20.0])

    def test_min_abs_value(self) -> None:
        ba = BenfordAnalysis([1, 5, 10, 100], min_abs_value=10.0)
        assert ba.n == 2

    def test_drop_zero(self) -> None:
        ba = BenfordAnalysis([0, 10, 20], drop_zero=True)
        assert ba.n == 2

    def test_profile_on_raw_data(self) -> None:
        ba = BenfordAnalysis([100, 5, 0, -3, -50])
        assert ba.profile.total_count == 5
        assert ba.n == 4  # zero dropped by default

    def test_cleaning_report(self) -> None:
        ba = BenfordAnalysis([np.nan, 0, 10, 20, np.inf], drop_zero=True)
        assert ba.cleaning_report.nan_inf_removed == 2
        assert ba.cleaning_report.zeros_removed == 1
        assert ba.cleaning_report.clean_count == 2

    def test_empty_after_cleaning_raises(self) -> None:
        with pytest.raises(ValueError, match="no data remains"):
            BenfordAnalysis([0, 0, 0], drop_zero=True)

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="no data remains"):
            BenfordAnalysis([np.nan, np.nan])


# ═══════════════════════════════════════════════════════════════════════════
# first_digit
# ═══════════════════════════════════════════════════════════════════════════


class TestFirstDigit:
    def test_result_type_and_shape(self) -> None:
        ba = BenfordAnalysis(_benford_data())
        r = ba.first_digit()
        assert isinstance(r, TestResult)
        assert r.test_name == "first_digit"
        assert len(r.digits) == 9
        assert len(r.counts) == 9
        assert len(r.observed) == 9
        assert len(r.expected) == 9
        assert len(r.z_scores) == 9
        assert len(r.significant_flags) == 9

    def test_observed_sums_to_one(self) -> None:
        r = BenfordAnalysis(_benford_data()).first_digit()
        assert r.observed.sum() == pytest.approx(1.0)

    def test_expected_matches_distribution(self) -> None:
        r = BenfordAnalysis(_benford_data()).first_digit()
        np.testing.assert_allclose(r.expected, first_digit_distribution())

    def test_benford_data_shows_conformity(self) -> None:
        r = BenfordAnalysis(_benford_data(10_000)).first_digit()
        assert r.mad_conformity in ("close_conformity", "acceptable_conformity")
        assert not r.chi_square_significant

    def test_uniform_data_shows_nonconformity(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.integers(10, 100, 5000).astype(np.float64)
        r = BenfordAnalysis(data).first_digit()
        assert r.mad_conformity == "nonconformity"

    def test_alpha_parameter(self) -> None:
        ba = BenfordAnalysis(_benford_data())
        r = ba.first_digit(alpha=0.01)
        assert r.alpha == 0.01


# ═══════════════════════════════════════════════════════════════════════════
# second_digit
# ═══════════════════════════════════════════════════════════════════════════


class TestSecondDigit:
    def test_result_shape(self) -> None:
        r = BenfordAnalysis(_benford_data()).second_digit()
        assert r.test_name == "second_digit"
        assert len(r.digits) == 10

    def test_benford_data_conforms(self) -> None:
        r = BenfordAnalysis(_benford_data(10_000)).second_digit()
        assert r.mad_conformity in ("close_conformity", "acceptable_conformity")


# ═══════════════════════════════════════════════════════════════════════════
# third_digit
# ═══════════════════════════════════════════════════════════════════════════


class TestThirdDigit:
    def test_result_shape(self) -> None:
        r = BenfordAnalysis(_benford_data()).third_digit()
        assert r.test_name == "third_digit"
        assert len(r.digits) == 10

    def test_mad_not_applicable(self) -> None:
        r = BenfordAnalysis(_benford_data()).third_digit()
        assert r.mad_conformity == "not_applicable"


# ═══════════════════════════════════════════════════════════════════════════
# first_two_digits
# ═══════════════════════════════════════════════════════════════════════════


class TestFirstTwoDigits:
    def test_result_shape(self) -> None:
        r = BenfordAnalysis(_benford_data()).first_two_digits()
        assert r.test_name == "first_two_digits"
        assert len(r.digits) == 90

    def test_expected_matches(self) -> None:
        r = BenfordAnalysis(_benford_data()).first_two_digits()
        np.testing.assert_allclose(r.expected, first_two_digits_distribution())

    def test_benford_data_conforms(self) -> None:
        r = BenfordAnalysis(_benford_data(10_000)).first_two_digits()
        assert r.mad_conformity in (
            "close_conformity",
            "acceptable_conformity",
            "marginally_acceptable_conformity",
        )


# ═══════════════════════════════════════════════════════════════════════════
# first_three_digits
# ═══════════════════════════════════════════════════════════════════════════


class TestFirstThreeDigits:
    def test_result_shape(self) -> None:
        r = BenfordAnalysis(_benford_data()).first_three_digits()
        assert r.test_name == "first_three_digits"
        assert len(r.digits) == 900


# ═══════════════════════════════════════════════════════════════════════════
# last_two_digits
# ═══════════════════════════════════════════════════════════════════════════


class TestLastTwoDigits:
    def test_result_shape(self) -> None:
        r = BenfordAnalysis(_benford_data()).last_two_digits()
        assert r.test_name == "last_two_digits"
        assert len(r.digits) == 100

    def test_uniform_expected(self) -> None:
        r = BenfordAnalysis(_benford_data()).last_two_digits()
        np.testing.assert_allclose(r.expected, 0.01)

    def test_mad_not_applicable(self) -> None:
        r = BenfordAnalysis(_benford_data()).last_two_digits()
        assert r.mad_conformity == "not_applicable"


# ═══════════════════════════════════════════════════════════════════════════
# second_order
# ═══════════════════════════════════════════════════════════════════════════


class TestSecondOrder:
    def test_result_type(self) -> None:
        r = BenfordAnalysis(_benford_data()).second_order()
        assert isinstance(r, TestResult)
        assert r.test_name == "second_order"
        assert len(r.digits) == 90

    def test_first_three_variant(self) -> None:
        r = BenfordAnalysis(_benford_data()).second_order(digits="first_three")
        assert len(r.digits) == 900


# ═══════════════════════════════════════════════════════════════════════════
# summation
# ═══════════════════════════════════════════════════════════════════════════


class TestSummation:
    def test_result_type(self) -> None:
        r = BenfordAnalysis(_benford_data()).summation()
        assert isinstance(r, SummationResult)
        assert r.test_name == "summation"
        assert len(r.digits) == 90
        assert len(r.sums) == 90

    def test_expected_uniform(self) -> None:
        r = BenfordAnalysis(_benford_data()).summation()
        np.testing.assert_allclose(r.expected, 1.0 / 90.0)

    def test_grand_sum_positive(self) -> None:
        r = BenfordAnalysis(_benford_data()).summation()
        assert r.grand_sum > 0


# ═══════════════════════════════════════════════════════════════════════════
# distortion_factor
# ═══════════════════════════════════════════════════════════════════════════


class TestDistortionFactor:
    def test_returns_distortion_result(self) -> None:
        r = BenfordAnalysis(_benford_data()).distortion_factor()
        assert isinstance(r, DistortionResult)

    def test_expected_mean_value(self) -> None:
        r = BenfordAnalysis(_benford_data()).distortion_factor()
        assert r.expected_mean == pytest.approx(39.0865)


# ═══════════════════════════════════════════════════════════════════════════
# mantissa_arc
# ═══════════════════════════════════════════════════════════════════════════


class TestMantissaArc:
    def test_returns_mantissa_result(self) -> None:
        r = BenfordAnalysis(_benford_data()).mantissa_arc()
        assert isinstance(r, MantissaArcResult)
        assert 0.0 <= r.L2 <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# number_duplication
# ═══════════════════════════════════════════════════════════════════════════


class TestNumberDuplication:
    def test_returns_duplication_result(self) -> None:
        data = [100.0] * 5 + [200.0] * 3 + [300.0]
        r = BenfordAnalysis(data).number_duplication()
        assert isinstance(r, DuplicationResult)
        assert r.values[0] == 100.0
        assert r.counts[0] == 5

    def test_top_n(self) -> None:
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        r = BenfordAnalysis(data).number_duplication(top_n=3)
        assert len(r.values) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_small_dataset(self) -> None:
        ba = BenfordAnalysis([10, 20, 30, 40, 50])
        r = ba.first_digit()
        assert r.n == 5

    def test_all_same_value(self) -> None:
        ba = BenfordAnalysis([42.0] * 100)
        r = ba.first_digit()
        assert r.counts[3] == 100  # digit 4
        assert r.mad_conformity == "nonconformity"

    def test_single_value_first_digit(self) -> None:
        ba = BenfordAnalysis([999.0])
        r = ba.first_digit()
        assert r.n == 1

    def test_mixed_magnitudes(self) -> None:
        data = [10, 100, 1000, 10_000, 100_000]
        ba = BenfordAnalysis(data)
        r = ba.first_digit()
        assert r.counts[0] == 5  # all start with 1
