"""Tests for pybenford.distributions module."""

from __future__ import annotations

import numpy as np
import pytest

from pybenford.constants import (
    FIRST_DIGIT_PROBS,
    FOURTH_DIGIT_PROBS,
    SECOND_DIGIT_PROBS,
    THIRD_DIGIT_PROBS,
)
from pybenford.distributions import (
    benford_distribution,
    digit_range,
    first_digit_distribution,
    second_digit_distribution,
    third_digit_distribution,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. digit_range
# ═══════════════════════════════════════════════════════════════════════════


class TestDigitRange:
    def test_k1(self) -> None:
        r = digit_range(1)
        assert r[0] == 1
        assert r[-1] == 9
        assert len(r) == 9
        np.testing.assert_array_equal(r, np.arange(1, 10))

    def test_k2(self) -> None:
        r = digit_range(2)
        assert r[0] == 10
        assert r[-1] == 99
        assert len(r) == 90
        np.testing.assert_array_equal(r, np.arange(10, 100))

    def test_k3(self) -> None:
        r = digit_range(3)
        assert r[0] == 100
        assert r[-1] == 999
        assert len(r) == 900
        np.testing.assert_array_equal(r, np.arange(100, 1000))

    def test_k0_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            digit_range(0)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            digit_range(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. benford_distribution
# ═══════════════════════════════════════════════════════════════════════════


class TestBenfordDistribution:
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_sums_to_one(self, k: int) -> None:
        assert benford_distribution(k).sum() == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Nigrini reference tables vs first-principles computations
# ═══════════════════════════════════════════════════════════════════════════


class TestReferenceTables:
    def test_first_digit_table(self) -> None:
        assert first_digit_distribution() == pytest.approx(FIRST_DIGIT_PROBS, abs=5e-6)

    def test_second_digit_table(self) -> None:
        assert second_digit_distribution() == pytest.approx(SECOND_DIGIT_PROBS, abs=5e-6)

    def test_third_digit_table(self) -> None:
        assert third_digit_distribution() == pytest.approx(THIRD_DIGIT_PROBS, abs=5e-6)

    def test_fourth_digit_table(self) -> None:
        fourth = benford_distribution(4).reshape(9, 10, 10, 10).sum(axis=(0, 1, 2))
        assert fourth == pytest.approx(FOURTH_DIGIT_PROBS, abs=5e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Marginal consistency of the dedicated wrappers
# ═══════════════════════════════════════════════════════════════════════════


class TestMarginalConsistency:
    def test_second_digit_is_marginal_of_joint(self) -> None:
        marginal = benford_distribution(2).reshape(9, 10).sum(axis=0)
        np.testing.assert_allclose(second_digit_distribution(), marginal, atol=1e-12)

    def test_third_digit_is_marginal_of_joint(self) -> None:
        marginal = benford_distribution(3).reshape(9, 10, 10).sum(axis=(0, 1))
        np.testing.assert_allclose(third_digit_distribution(), marginal, atol=1e-12)
