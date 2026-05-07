"""Tests for pybenford.visualization module."""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pybenford.core import BenfordAnalysis
from pybenford.visualization import (
    plot_digit_test,
    plot_distortion_factor,
    plot_mantissa_arc,
    plot_ordered_mantissas,
    plot_summation,
    plot_z_scores,
)

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _benford_data(n: int = 5000, seed: int = 42) -> np.ndarray:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    mantissas = rng.uniform(0, 1, n)
    return 10.0 ** (mantissas + rng.integers(1, 5, n))


@pytest.fixture()
def ba() -> BenfordAnalysis:
    return BenfordAnalysis(_benford_data())


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    yield  # type: ignore[misc]
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# plot_digit_test
# ═══════════════════════════════════════════════════════════════════════════


class TestPlotDigitTest:
    def test_returns_fig_ax(self, ba: BenfordAnalysis) -> None:
        fig, ax = plot_digit_test(ba.first_digit())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_first_two_digits(self, ba: BenfordAnalysis) -> None:
        fig, _ax = plot_digit_test(ba.first_two_digits())
        assert isinstance(fig, Figure)

    def test_custom_title(self, ba: BenfordAnalysis) -> None:
        _fig, ax = plot_digit_test(ba.first_digit(), title="Custom Title")
        assert ax.get_title() == "Custom Title"

    def test_existing_ax(self, ba: BenfordAnalysis) -> None:
        _, host_ax = plt.subplots()
        _fig, ax = plot_digit_test(ba.first_digit(), ax=host_ax)
        assert ax is host_ax

    def test_no_confidence(self, ba: BenfordAnalysis) -> None:
        fig, _ax = plot_digit_test(ba.first_digit(), show_confidence=False)
        assert isinstance(fig, Figure)

    def test_save_to_file(self, ba: BenfordAnalysis, tmp_path: Path) -> None:
        fig, _ = plot_digit_test(ba.first_digit())
        path = tmp_path / "digit.png"
        fig.savefig(path)
        assert path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# plot_summation
# ═══════════════════════════════════════════════════════════════════════════


class TestPlotSummation:
    def test_returns_fig_ax(self, ba: BenfordAnalysis) -> None:
        fig, ax = plot_summation(ba.summation())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_title(self, ba: BenfordAnalysis) -> None:
        _fig, ax = plot_summation(ba.summation(), title="My Summation")
        assert ax.get_title() == "My Summation"

    def test_existing_ax(self, ba: BenfordAnalysis) -> None:
        _, host_ax = plt.subplots()
        _fig, ax = plot_summation(ba.summation(), ax=host_ax)
        assert ax is host_ax

    def test_save_to_file(self, ba: BenfordAnalysis, tmp_path: Path) -> None:
        fig, _ = plot_summation(ba.summation())
        path = tmp_path / "summation.png"
        fig.savefig(path)
        assert path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# plot_mantissa_arc
# ═══════════════════════════════════════════════════════════════════════════


class TestPlotMantissaArc:
    def test_returns_fig_ax(self, ba: BenfordAnalysis) -> None:
        fig, ax = plot_mantissa_arc(ba.mantissa_arc())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_title(self, ba: BenfordAnalysis) -> None:
        _fig, ax = plot_mantissa_arc(ba.mantissa_arc(), title="Arc")
        assert ax.get_title() == "Arc"

    def test_existing_ax(self, ba: BenfordAnalysis) -> None:
        _, host_ax = plt.subplots()
        _fig, ax = plot_mantissa_arc(ba.mantissa_arc(), ax=host_ax)
        assert ax is host_ax

    def test_save_to_file(self, ba: BenfordAnalysis, tmp_path: Path) -> None:
        fig, _ = plot_mantissa_arc(ba.mantissa_arc())
        path = tmp_path / "arc.png"
        fig.savefig(path)
        assert path.stat().st_size > 0

    def test_deprecated_data_arg(self, ba: BenfordAnalysis) -> None:
        with pytest.warns(DeprecationWarning, match="Passing 'data'"):
            fig, ax = plot_mantissa_arc(ba.mantissa_arc(), ba.clean_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


# ═══════════════════════════════════════════════════════════════════════════
# plot_ordered_mantissas
# ═══════════════════════════════════════════════════════════════════════════


class TestPlotOrderedMantissas:
    def test_returns_fig_ax(self, ba: BenfordAnalysis) -> None:
        fig, ax = plot_ordered_mantissas(ba.mantissa_arc())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_title(self, ba: BenfordAnalysis) -> None:
        _fig, ax = plot_ordered_mantissas(ba.mantissa_arc(), title="Mantissas")
        assert ax.get_title() == "Mantissas"

    def test_existing_ax(self, ba: BenfordAnalysis) -> None:
        _, host_ax = plt.subplots()
        _fig, ax = plot_ordered_mantissas(ba.mantissa_arc(), ax=host_ax)
        assert ax is host_ax

    def test_save_to_file(self, ba: BenfordAnalysis, tmp_path: Path) -> None:
        fig, _ = plot_ordered_mantissas(ba.mantissa_arc())
        path = tmp_path / "mantissas.png"
        fig.savefig(path)
        assert path.stat().st_size > 0

    def test_deprecated_raw_array(self, ba: BenfordAnalysis) -> None:
        with pytest.warns(DeprecationWarning, match="raw array"):
            fig, ax = plot_ordered_mantissas(ba.clean_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


# ═══════════════════════════════════════════════════════════════════════════
# plot_z_scores
# ═══════════════════════════════════════════════════════════════════════════


class TestPlotZScores:
    def test_returns_fig_ax(self, ba: BenfordAnalysis) -> None:
        fig, ax = plot_z_scores(ba.first_digit())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_title(self, ba: BenfordAnalysis) -> None:
        _fig, ax = plot_z_scores(ba.first_digit(), title="Z")
        assert ax.get_title() == "Z"

    def test_existing_ax(self, ba: BenfordAnalysis) -> None:
        _, host_ax = plt.subplots()
        _fig, ax = plot_z_scores(ba.first_digit(), ax=host_ax)
        assert ax is host_ax

    def test_save_to_file(self, ba: BenfordAnalysis, tmp_path: Path) -> None:
        fig, _ = plot_z_scores(ba.first_digit())
        path = tmp_path / "zscores.png"
        fig.savefig(path)
        assert path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# plot_distortion_factor
# ═══════════════════════════════════════════════════════════════════════════


class TestPlotDistortionFactor:
    def test_returns_fig_ax(self, ba: BenfordAnalysis) -> None:
        fig, ax = plot_distortion_factor(ba.distortion_factor())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_title(self, ba: BenfordAnalysis) -> None:
        _fig, ax = plot_distortion_factor(ba.distortion_factor(), title="DF")
        assert ax.get_title() == "DF"

    def test_existing_ax(self, ba: BenfordAnalysis) -> None:
        _, host_ax = plt.subplots()
        _fig, ax = plot_distortion_factor(ba.distortion_factor(), ax=host_ax)
        assert ax is host_ax

    def test_save_to_file(self, ba: BenfordAnalysis, tmp_path: Path) -> None:
        fig, _ = plot_distortion_factor(ba.distortion_factor())
        path = tmp_path / "df.png"
        fig.savefig(path)
        assert path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# No plt.show() is called
# ═══════════════════════════════════════════════════════════════════════════


class TestNoShow:
    def test_no_show_called(self, ba: BenfordAnalysis) -> None:
        with patch.object(plt, "show") as mock_show:
            plot_digit_test(ba.first_digit())
            plot_summation(ba.summation())
            plot_mantissa_arc(ba.mantissa_arc())
            plot_ordered_mantissas(ba.mantissa_arc())
            plot_z_scores(ba.first_digit())
            plot_distortion_factor(ba.distortion_factor())
            mock_show.assert_not_called()
