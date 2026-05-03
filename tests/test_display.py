"""Tests for __str__ methods on all result dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from pybenford.core import SummationResult, TestResult
from pybenford.statistics import DistortionResult, MantissaArcResult
from pybenford.utils import DuplicationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_result(
    *,
    test_name: str = "first_digit",
    n_digits: int = 9,
    flagged: list[int] | None = None,
    chi_significant: bool = False,
    ks_significant: bool = False,
    mad_conformity: str = "close_conformity",
) -> TestResult:
    digits = np.arange(1, 1 + n_digits, dtype=np.int64)
    counts = np.full(n_digits, 100, dtype=np.int64)
    obs = np.full(n_digits, 1.0 / n_digits, dtype=np.float64)
    exp = np.full(n_digits, 1.0 / n_digits, dtype=np.float64)
    z = np.full(n_digits, 0.5, dtype=np.float64)
    sig = np.zeros(n_digits, dtype=np.bool_)
    if flagged:
        for idx in flagged:
            sig[idx] = True
    return TestResult(
        test_name=test_name,
        digits=digits,
        counts=counts,
        observed=obs,
        expected=exp,
        z_scores=z,
        significant_flags=sig,
        chi_square=5.5,
        chi_square_critical=15.5,
        chi_square_significant=chi_significant,
        ks_statistic=0.01,
        ks_critical=0.02,
        ks_significant=ks_significant,
        mad=0.004,
        mad_conformity=mad_conformity,
        n=n_digits * 100,
        alpha=0.05,
    )


# ---------------------------------------------------------------------------
# TestResult.__str__
# ---------------------------------------------------------------------------


class TestTestResultStr:
    def test_full_table_header(self):
        r = _make_test_result()
        s = str(r)
        assert "First Digit Test" in s
        assert "n=900" in s
        assert "alpha=0.05" in s

    def test_full_table_columns(self):
        r = _make_test_result()
        s = str(r)
        assert "Digit" in s
        assert "Count" in s
        assert "Observed" in s
        assert "Expected" in s
        assert "Z-Score" in s

    def test_full_table_mad_conformity(self):
        r = _make_test_result(mad_conformity="close_conformity")
        assert "Close Conformity" in str(r)

        r2 = _make_test_result(mad_conformity="nonconformity")
        assert "Nonconformity" in str(r2)

        r3 = _make_test_result(mad_conformity="not_applicable")
        assert "N/A (no threshold defined)" in str(r3)

    def test_full_table_chi_pass(self):
        r = _make_test_result(chi_significant=False)
        s = str(r)
        assert "Chi-Square:" in s
        assert "Pass" in s

    def test_full_table_chi_fail(self):
        r = _make_test_result(chi_significant=True)
        assert "FAIL" in str(r)

    def test_full_table_ks_pass(self):
        r = _make_test_result(ks_significant=False)
        lines = str(r).split("\n")
        ks_line = [l for l in lines if "KS:" in l][0]
        assert "Pass" in ks_line

    def test_full_table_ks_fail(self):
        r = _make_test_result(ks_significant=True)
        lines = str(r).split("\n")
        ks_line = [l for l in lines if "KS:" in l][0]
        assert "FAIL" in ks_line

    def test_flagged_digit_star(self):
        r = _make_test_result(flagged=[0, 2])
        s = str(r)
        lines = s.split("\n")
        digit_lines = [l for l in lines if l.strip().startswith("1") or l.strip().startswith("3")]
        starred = [l for l in lines if "*" in l]
        assert len(starred) == 2

    def test_many_digits_flagged_only(self):
        r = _make_test_result(test_name="first_two_digits", n_digits=90, flagged=[0, 5, 10])
        s = str(r)
        assert "Flagged Digits (3 of 90):" in s
        assert "First Two Digits Test" in s

    def test_many_digits_no_flags(self):
        r = _make_test_result(test_name="first_two_digits", n_digits=90)
        s = str(r)
        assert "No individual digits flagged at alpha=0.05." in s

    def test_border_width(self):
        r = _make_test_result()
        s = str(r)
        lines = s.split("\n")
        assert lines[0] == "=" * 55
        assert lines[-1] == "=" * 55


# ---------------------------------------------------------------------------
# SummationResult.__str__
# ---------------------------------------------------------------------------


class TestSummationResultStr:
    def _make(self, *, flagged: list[int] | None = None) -> SummationResult:
        n_digits = 90
        digits = np.arange(10, 100, dtype=np.int64)
        sums = np.full(n_digits, 1000.0, dtype=np.float64)
        obs = np.full(n_digits, 1.0 / n_digits, dtype=np.float64)
        exp = np.full(n_digits, 1.0 / n_digits, dtype=np.float64)
        z = np.full(n_digits, 0.5, dtype=np.float64)
        sig = np.zeros(n_digits, dtype=np.bool_)
        if flagged:
            for idx in flagged:
                sig[idx] = True
        return SummationResult(
            test_name="summation",
            digits=digits,
            sums=sums,
            observed=obs,
            expected=exp,
            z_scores=z,
            significant_flags=sig,
            chi_square=200.0,
            chi_square_critical=112.0,
            chi_square_significant=True,
            grand_sum=90000.0,
            n=3000,
            alpha=0.05,
        )

    def test_header(self):
        r = self._make(flagged=[0])
        s = str(r)
        assert "Summation Test" in s
        assert "n=3,000" in s

    def test_grand_sum(self):
        r = self._make(flagged=[0])
        assert "Grand Sum: 90,000" in str(r)

    def test_flagged_digits(self):
        r = self._make(flagged=[0, 5, 10])
        s = str(r)
        assert "Flagged Digits (3 of 90):" in s
        assert "*" in s

    def test_no_flagged(self):
        r = self._make()
        assert "No individual digits flagged" in str(r)

    def test_chi_fail(self):
        r = self._make(flagged=[0])
        assert "FAIL" in str(r)

    def test_no_ks_or_mad(self):
        r = self._make(flagged=[0])
        s = str(r)
        assert "KS:" not in s
        assert "MAD:" not in s


# ---------------------------------------------------------------------------
# DistortionResult.__str__
# ---------------------------------------------------------------------------


class TestDistortionResultStr:
    def test_understated(self):
        r = DistortionResult(
            distortion_factor=-0.002,
            actual_mean=39.0,
            expected_mean=39.1,
            z_statistic=-0.5,
            p_value=0.6,
            significant=False,
            direction="understated",
            percentage=-0.2,
        )
        s = str(r)
        assert "Distortion Factor Test" in s
        assert "(Understated)" in s
        assert "Significant:         No" in s
        assert "-0.0020" in s

    def test_overstated(self):
        r = DistortionResult(
            distortion_factor=0.01,
            actual_mean=40.0,
            expected_mean=39.0,
            z_statistic=2.5,
            p_value=0.01,
            significant=True,
            direction="overstated",
            percentage=1.0,
        )
        s = str(r)
        assert "(Overstated)" in s
        assert "Significant:         Yes" in s

    def test_no_direction(self):
        r = DistortionResult(
            distortion_factor=0.0,
            actual_mean=39.0,
            expected_mean=39.0,
            z_statistic=0.0,
            p_value=1.0,
            significant=False,
            direction="none",
            percentage=0.0,
        )
        s = str(r)
        assert "(Understated)" not in s
        assert "(Overstated)" not in s


# ---------------------------------------------------------------------------
# MantissaArcResult.__str__
# ---------------------------------------------------------------------------


class TestMantissaArcResultStr:
    def test_output_fields(self):
        import math

        r = MantissaArcResult(
            mean_angle=math.pi,
            mean_x=-0.01,
            mean_y=0.02,
            L2=0.025,
            p_value=0.15,
        )
        s = str(r)
        assert "Mantissa Arc Test" in s
        assert "Gravity Center:" in s
        assert "-0.0100" in s
        assert "0.0200" in s
        assert "L2 Statistic:" in s
        assert "0.0250" in s
        assert "rad" in s
        assert "deg" in s
        assert "P-Value:" in s
        assert "0.1500" in s

    def test_angle_conversion(self):
        import math

        r = MantissaArcResult(
            mean_angle=math.pi / 2,
            mean_x=0.0,
            mean_y=0.0,
            L2=0.0,
            p_value=0.5,
        )
        s = str(r)
        assert "90.00 deg" in s


# ---------------------------------------------------------------------------
# DuplicationResult.__str__
# ---------------------------------------------------------------------------


class TestDuplicationResultStr:
    def test_with_duplicates(self):
        r = DuplicationResult(
            values=np.array([1234.0, 567.0], dtype=np.float64),
            counts=np.array([5, 3], dtype=np.int64),
            first_two_digits=np.array([12.0, 56.0], dtype=np.float64),
            total_unique=950,
            total_records=1000,
        )
        s = str(r)
        assert "Number Duplication Test" in s
        assert "Total Records: 1,000" in s
        assert "Unique Values: 950" in s
        assert "Top Duplicated Values:" in s
        assert "1,234" in s
        assert "12" in s

    def test_no_duplicates(self):
        r = DuplicationResult(
            values=np.array([100.0, 200.0], dtype=np.float64),
            counts=np.array([1, 1], dtype=np.int64),
            first_two_digits=np.array([10.0, 20.0], dtype=np.float64),
            total_unique=100,
            total_records=100,
        )
        s = str(r)
        assert "No duplicated values found." in s

    def test_border_width(self):
        r = DuplicationResult(
            values=np.array([100.0], dtype=np.float64),
            counts=np.array([2], dtype=np.int64),
            first_two_digits=np.array([10.0], dtype=np.float64),
            total_unique=99,
            total_records=100,
        )
        lines = str(r).split("\n")
        assert lines[0] == "=" * 55
        assert lines[-1] == "=" * 55
