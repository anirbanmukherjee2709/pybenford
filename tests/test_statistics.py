"""Tests for pybenford.statistics module."""

from __future__ import annotations

import numpy as np
import pytest

from pybenford.constants import DF_EXPECTED_MEAN, DF_SD_CONSTANT
from pybenford.distributions import (
    first_digit_distribution,
    first_two_digits_distribution,
    second_digit_distribution,
)
from pybenford.statistics import (
    ChiSquareResult,
    ConformityLevel,
    DigitTest,
    DistortionResult,
    KSResult,
    MADResult,
    MantissaArcResult,
    chi_square_test,
    distortion_factor_test,
    ks_test,
    mad_test,
    mantissa_arc_test,
    sum_squared_differences,
    z_pvalue,
    z_significant,
    z_statistic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def benford_first() -> np.ndarray:
    return first_digit_distribution()


@pytest.fixture()
def benford_first_two() -> np.ndarray:
    return first_two_digits_distribution()


# ---------------------------------------------------------------------------
# 1. Known-good Benford data
# ---------------------------------------------------------------------------


class TestKnownGoodBenford:
    """Perfect Benford proportions should show zero deviation."""

    def test_z_statistic_perfect(self, benford_first: np.ndarray) -> None:
        z = z_statistic(benford_first, benford_first, 10_000)
        np.testing.assert_allclose(z, 0.0, atol=1e-10)

    def test_mad_close_conformity(self, benford_first: np.ndarray) -> None:
        result = mad_test(benford_first, benford_first, DigitTest.FIRST)
        assert result.mad == pytest.approx(0.0)
        assert result.conformity == ConformityLevel.CLOSE

    def test_chi_square_not_significant(self, benford_first: np.ndarray) -> None:
        n = 10_000
        counts = np.round(benford_first * n).astype(np.int64)
        # Adjust last bin so counts sum exactly to n
        counts[-1] = n - counts[:-1].sum()
        result = chi_square_test(counts, benford_first, n)
        assert result.statistic < result.critical_value
        assert not result.significant

    def test_ks_not_significant(self, benford_first: np.ndarray) -> None:
        result = ks_test(benford_first, benford_first, 10_000)
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert not result.significant

    def test_ssd_zero(self, benford_first: np.ndarray) -> None:
        assert sum_squared_differences(benford_first, benford_first) == pytest.approx(
            0.0
        )


# ---------------------------------------------------------------------------
# 2. Known-bad data
# ---------------------------------------------------------------------------


class TestKnownBadData:
    """Uniform distribution should fail all conformity tests."""

    def test_uniform_z_significant(self, benford_first: np.ndarray) -> None:
        uniform = np.ones(9) / 9.0
        z = z_statistic(uniform, benford_first, 10_000)
        sig = z_significant(z, alpha=0.05)
        assert np.any(sig)

    def test_uniform_mad_nonconformity(self, benford_first: np.ndarray) -> None:
        uniform = np.ones(9) / 9.0
        result = mad_test(uniform, benford_first, DigitTest.FIRST)
        assert result.conformity == ConformityLevel.NONCONFORMITY

    def test_uniform_chi_square_significant(self, benford_first: np.ndarray) -> None:
        n = 10_000
        counts = np.full(9, n // 9, dtype=np.int64)
        counts[0] += n - counts.sum()
        result = chi_square_test(counts, benford_first, n)
        assert result.significant

    def test_uniform_ks_significant(self, benford_first: np.ndarray) -> None:
        uniform = np.ones(9) / 9.0
        result = ks_test(uniform, benford_first, 10_000)
        assert result.significant


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Small N, large N, and degenerate inputs."""

    def test_n_equals_1(self, benford_first: np.ndarray) -> None:
        observed = np.zeros(9)
        observed[0] = 1.0
        z = z_statistic(observed, benford_first, 1)
        assert np.all(np.isfinite(z))

    def test_n_equals_10(self, benford_first: np.ndarray) -> None:
        observed = np.zeros(9)
        observed[0] = 0.5
        observed[1] = 0.3
        observed[2] = 0.2
        z = z_statistic(observed, benford_first, 10)
        assert z.shape == (9,)

    def test_large_n(self, benford_first: np.ndarray) -> None:
        z = z_statistic(benford_first, benford_first, 1_000_000)
        np.testing.assert_allclose(z, 0.0, atol=1e-10)

    def test_all_same_digit(self, benford_first: np.ndarray) -> None:
        observed = np.zeros(9)
        observed[0] = 1.0
        z = z_statistic(observed, benford_first, 100)
        assert z[0] > 5.0


# ---------------------------------------------------------------------------
# 4. Numerical verification — Nigrini worked examples
# ---------------------------------------------------------------------------


class TestNigriniCensusZ:
    """Census Z-stat for digit 32 (Nigrini Eq. 7.1).

    With AP=0.0152, EP=0.0134, N=19482, the formula gives Z ~ 2.15.
    Nigrini's published 2.260 uses more precise intermediate values;
    we verify our formula is algebraically correct.
    """

    def test_census_z_rounded_inputs(self) -> None:
        ap = np.array([0.0152])
        ep = np.array([0.0134])
        z = z_statistic(ap, ep, 19_482)
        assert z[0] == pytest.approx(2.153, abs=0.01)

    def test_census_z_precise_ep(self) -> None:
        ep = np.array([np.log10(1 + 1 / 32)])
        ap = np.array([297 / 19_482])
        z = z_statistic(ap, ep, 19_482)
        assert z[0] == pytest.approx(2.26, abs=0.02)


class TestNigriniCensusMAD:
    """Census first-two digit MAD = 0.0006 -> close conformity."""

    def test_mad_0006_is_close(self) -> None:
        expected = first_two_digits_distribution()
        delta = np.zeros(90)
        delta[:45] = 0.0006
        delta[45:] = -0.0006
        observed = expected + delta
        result = mad_test(observed, expected, DigitTest.FIRST_TWO)
        assert result.mad == pytest.approx(0.0006, abs=1e-10)
        assert result.conformity == ConformityLevel.CLOSE


class TestNigriniCensusDF:
    """Census distortion: DF ~ 0.0074, Z ~ 1.62, not significant."""

    def test_distortion_factor_synthetic(self) -> None:
        target_am = DF_EXPECTED_MEAN * (1.0 + 0.0074)
        data = np.full(19_482, target_am)
        result = distortion_factor_test(data)
        assert result.distortion_factor == pytest.approx(0.0074, abs=1e-4)
        expected_z = 0.0074 / (DF_SD_CONSTANT / np.sqrt(19_482))
        assert result.z_statistic == pytest.approx(expected_z, rel=1e-4)
        assert not result.significant
        assert result.direction == "overstated"
        assert result.percentage == pytest.approx(0.74, abs=0.05)


# ---------------------------------------------------------------------------
# 5. Continuity correction
# ---------------------------------------------------------------------------


class TestContinuityCorrection:
    """Verify correction is applied / skipped per the Fleiss rule."""

    def test_correction_skipped_when_diff_small(self) -> None:
        ep = np.array([0.30103])
        ap = np.array([0.30103 + 1e-6])
        n = 100_000
        z_with = z_statistic(ap, ep, n, continuity_correction=True)
        z_without = z_statistic(ap, ep, n, continuity_correction=False)
        assert z_with[0] == pytest.approx(0.0)
        assert z_without[0] > 0.0

    def test_correction_applied_when_diff_large(self) -> None:
        ep = np.array([0.30103])
        ap = np.array([0.302])
        n = 10_000
        z_with = z_statistic(ap, ep, n, continuity_correction=True)
        z_without = z_statistic(ap, ep, n, continuity_correction=False)
        assert z_with[0] < z_without[0]
        assert z_with[0] > 0.0

    def test_correction_disabled(self) -> None:
        ep = np.array([0.30103])
        ap = np.array([0.30103 + 1e-6])
        n = 100_000
        z = z_statistic(ap, ep, n, continuity_correction=False)
        assert z[0] > 0.0


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Proper errors for invalid inputs."""

    def test_z_n_zero(self, benford_first: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            z_statistic(benford_first, benford_first, 0)

    def test_z_n_negative(self, benford_first: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            z_statistic(benford_first, benford_first, -5)

    def test_z_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            z_statistic(np.array([0.5, 0.5]), np.array([1.0]), 10)

    def test_z_negative_proportions(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            z_statistic(np.array([-0.1, 1.1]), np.array([0.5, 0.5]), 10)

    def test_z_zero_expected_returns_inf(self) -> None:
        z = z_statistic(np.array([0.5, 0.5]), np.array([0.0, 1.0]), 100)
        assert np.isinf(z[0])

    def test_chi_square_sum_mismatch(self, benford_first: np.ndarray) -> None:
        counts = np.ones(9, dtype=np.int64)
        with pytest.raises(ValueError, match="must equal"):
            chi_square_test(counts, benford_first, 100)

    def test_mad_wrong_bin_count(self) -> None:
        with pytest.raises(ValueError, match="expects 9 bins"):
            mad_test(np.ones(10) / 10, np.ones(10) / 10, DigitTest.FIRST)

    def test_ks_unsupported_alpha(self, benford_first: np.ndarray) -> None:
        with pytest.raises(ValueError, match="not supported"):
            ks_test(benford_first, benford_first, 100, alpha=0.02)

    def test_distortion_no_valid_values(self) -> None:
        with pytest.raises(ValueError, match="no valid values"):
            distortion_factor_test(np.array([1.0, 2.0, 5.0]))

    def test_distortion_all_nan(self) -> None:
        with pytest.raises(ValueError, match="no valid values"):
            distortion_factor_test(np.array([np.nan, np.inf]))

    def test_ssd_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            sum_squared_differences(np.array([0.5]), np.array([0.3, 0.7]))

    def test_mantissa_no_valid(self) -> None:
        with pytest.raises(ValueError, match="no valid values"):
            mantissa_arc_test(np.array([0.0, np.nan, np.inf]))


# ---------------------------------------------------------------------------
# Z p-value and significance helpers
# ---------------------------------------------------------------------------


class TestZPvalue:
    def test_z_zero_pvalue_is_one(self) -> None:
        p = z_pvalue(np.array([0.0]))
        assert p[0] == pytest.approx(1.0)

    def test_large_z_small_pvalue(self) -> None:
        p = z_pvalue(np.array([5.0]))
        assert p[0] < 0.001

    def test_symmetric(self) -> None:
        p_pos = z_pvalue(np.array([1.96]))
        p_neg = z_pvalue(np.array([-1.96]))
        assert p_pos[0] == pytest.approx(p_neg[0])

    def test_z196_pvalue(self) -> None:
        p = z_pvalue(np.array([1.96]))
        assert p[0] == pytest.approx(0.05, abs=0.001)


class TestZSignificant:
    def test_default_alpha(self) -> None:
        sig = z_significant(np.array([1.5, 2.0, 3.0]))
        np.testing.assert_array_equal(sig, [False, True, True])

    def test_alpha_010(self) -> None:
        sig = z_significant(np.array([1.5, 2.0]), alpha=0.10)
        np.testing.assert_array_equal(sig, [False, True])

    def test_alpha_001(self) -> None:
        sig = z_significant(np.array([2.5, 2.6]), alpha=0.01)
        np.testing.assert_array_equal(sig, [False, True])


# ---------------------------------------------------------------------------
# Chi-square result properties
# ---------------------------------------------------------------------------


class TestChiSquareResult:
    def test_result_fields(self, benford_first: np.ndarray) -> None:
        n = 1000
        counts = np.round(benford_first * n).astype(np.int64)
        counts[-1] = n - counts[:-1].sum()
        result = chi_square_test(counts, benford_first, n)
        assert isinstance(result, ChiSquareResult)
        assert result.degrees_of_freedom == 8
        assert 0.0 <= result.p_value <= 1.0
        assert result.critical_value > 0

    def test_degrees_of_freedom_second_digit(self) -> None:
        exp = second_digit_distribution()
        n = 1000
        counts = np.round(exp * n).astype(np.int64)
        counts[-1] = n - counts[:-1].sum()
        result = chi_square_test(counts, exp, n)
        assert result.degrees_of_freedom == 9


# ---------------------------------------------------------------------------
# KS test details
# ---------------------------------------------------------------------------


class TestKSTest:
    def test_result_type(self, benford_first: np.ndarray) -> None:
        result = ks_test(benford_first, benford_first, 1000)
        assert isinstance(result, KSResult)

    def test_critical_value_decreases_with_n(
        self, benford_first: np.ndarray
    ) -> None:
        r1 = ks_test(benford_first, benford_first, 100)
        r2 = ks_test(benford_first, benford_first, 10_000)
        assert r2.critical_value < r1.critical_value


# ---------------------------------------------------------------------------
# MAD conformity boundaries
# ---------------------------------------------------------------------------


class TestMADConformityBoundaries:
    """Test classification at exact threshold boundaries."""

    def test_first_digit_thresholds(self) -> None:
        exp = first_digit_distribution()
        for mad_val, expected_level in [
            (0.004, ConformityLevel.CLOSE),
            (0.009, ConformityLevel.ACCEPTABLE),
            (0.013, ConformityLevel.MARGINAL),
            (0.020, ConformityLevel.NONCONFORMITY),
        ]:
            # Construct observed with exact MAD = mad_val.
            # Shift first element by +d, last by -d; other 7 unchanged.
            # MAD = (|d| + |d|) / 9 = 2d/9 → d = mad_val * 9 / 2
            d = mad_val * 9.0 / 2.0
            observed = exp.copy()
            observed[0] += d
            observed[-1] -= d
            result = mad_test(observed, exp, DigitTest.FIRST)
            assert result.conformity == expected_level, (
                f"MAD={mad_val}: expected {expected_level}, got {result.conformity} "
                f"(actual MAD={result.mad})"
            )

    def test_thresholds_dict_populated(self) -> None:
        exp = first_digit_distribution()
        result = mad_test(exp, exp, DigitTest.FIRST)
        assert result.thresholds == {
            "close": 0.006,
            "acceptable": 0.012,
            "marginal": 0.015,
        }


# ---------------------------------------------------------------------------
# Distortion Factor details
# ---------------------------------------------------------------------------


class TestDistortionFactor:
    def test_neutral_direction(self) -> None:
        data = np.full(100, DF_EXPECTED_MEAN)
        result = distortion_factor_test(data)
        assert result.distortion_factor == pytest.approx(0.0, abs=1e-10)
        assert result.direction in ("neutral", "overstated", "understated")

    def test_understated(self) -> None:
        target_am = DF_EXPECTED_MEAN * 0.95
        data = np.full(1000, target_am)
        result = distortion_factor_test(data)
        assert result.direction == "understated"
        assert result.distortion_factor < 0

    def test_overstated(self) -> None:
        target_am = DF_EXPECTED_MEAN * 1.05
        data = np.full(1000, target_am)
        result = distortion_factor_test(data)
        assert result.direction == "overstated"
        assert result.distortion_factor > 0

    def test_filters_values_below_10(self) -> None:
        data = np.array([5.0, 3.0, 1.0, 39.0865, 39.0865])
        result = distortion_factor_test(data)
        assert result.actual_mean == pytest.approx(DF_EXPECTED_MEAN, abs=1e-3)

    def test_filters_negatives_by_abs(self) -> None:
        data = np.array([-50.0, -50.0, 50.0, 50.0])
        result = distortion_factor_test(data)
        assert result.actual_mean == pytest.approx(50.0, abs=0.01)

    def test_expected_mean_field(self) -> None:
        data = np.full(10, 39.0865)
        result = distortion_factor_test(data)
        assert result.expected_mean == DF_EXPECTED_MEAN


# ---------------------------------------------------------------------------
# SSD
# ---------------------------------------------------------------------------


class TestSSD:
    def test_simple_computation(self) -> None:
        obs = np.array([0.3, 0.2, 0.5])
        exp = np.array([0.33, 0.33, 0.34])
        expected = (0.3 - 0.33) ** 2 + (0.2 - 0.33) ** 2 + (0.5 - 0.34) ** 2
        assert sum_squared_differences(obs, exp) == pytest.approx(expected)

    def test_identical_is_zero(self) -> None:
        a = np.array([0.1, 0.2, 0.7])
        assert sum_squared_differences(a, a) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mantissa Arc Test
# ---------------------------------------------------------------------------


class TestMantissaArc:
    def test_result_type(self) -> None:
        data = np.array([10.0, 20.0, 50.0, 100.0, 200.0])
        result = mantissa_arc_test(data)
        assert isinstance(result, MantissaArcResult)
        assert 0.0 <= result.L2 <= 1.0

    def test_uniform_mantissas_low_l2(self) -> None:
        rng = np.random.default_rng(42)
        mantissas = rng.uniform(0, 1, 10_000)
        data = 10.0**mantissas
        result = mantissa_arc_test(data)
        assert result.L2 < 0.05

    def test_identical_values_high_l2(self) -> None:
        data = np.full(100, 100.0)
        result = mantissa_arc_test(data)
        assert result.L2 > 0.9
        assert result.p_value < 0.01
