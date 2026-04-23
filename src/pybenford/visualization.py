"""Publication-quality plots for Benford's Law analysis results."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pybenford.core import SummationResult, TestResult
from pybenford.statistics import DistortionResult, MantissaArcResult

__all__ = [
    "plot_digit_test",
    "plot_distortion_factor",
    "plot_mantissa_arc",
    "plot_ordered_mantissas",
    "plot_summation",
    "plot_z_scores",
]

COLORS = {
    "observed": "#4C72B0",
    "expected": "#DD8452",
    "significant": "#C44E52",
    "confidence_fill": "#C44E52",
    "summation": "#55A868",
    "mantissa_point": "#4C72B0",
    "gravity_center": "#C44E52",
    "expected_line": "#999999",
    "grid": "#CCCCCC",
}


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _setup_axes(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    """Return (Figure, Axes), creating them when *ax* is ``None``."""
    if ax is not None:
        parent = ax.get_figure()
        if parent is None:  # pragma: no cover - defensive
            import matplotlib.pyplot as _plt

            parent = _plt.gcf()
        fig: Figure = parent  # type: ignore[assignment]
        return fig, ax

    import matplotlib.pyplot as _plt

    fig, new_ax = _plt.subplots(figsize=figsize)
    return fig, new_ax


def _clean_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def _auto_title(test_name: str, suffix: str = "Test") -> str:
    return test_name.replace("_", " ").title() + " " + suffix


# ---------------------------------------------------------------------------
# 1. plot_digit_test
# ---------------------------------------------------------------------------


def plot_digit_test(
    result: TestResult,
    *,
    show_confidence: bool = True,
    confidence_z: float = 1.96,
    highlight_significant: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Bar chart of observed vs. expected digit proportions.

    Parameters
    ----------
    result
        A :class:`~pybenford.core.TestResult` from any digit test.
    show_confidence
        Draw upper/lower Nigrini confidence bounds.
    confidence_z
        Z-value for the confidence band (default 1.96 = 95 %).
    highlight_significant
        Color bars red where ``result.significant_flags`` is True.
    title
        Plot title.  Auto-generated from *result.test_name* when None.
    figsize
        Figure dimensions in inches.
    ax
        Existing Axes to draw on (enables subplot composition).

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax_ = _setup_axes(ax, figsize)
    _clean_spines(ax_)

    digits = result.digits
    obs_pct = result.observed * 100.0
    exp_pct = result.expected * 100.0
    n = result.n

    many_bins = len(digits) >= 20
    bar_width = 0.4 if many_bins else 0.7
    x = np.arange(len(digits))

    bar_colors: NDArray[np.object_] = np.full(len(digits), COLORS["observed"], dtype=object)
    if highlight_significant:
        bar_colors[result.significant_flags] = COLORS["significant"]

    ax_.bar(x, obs_pct, width=bar_width, color=bar_colors, label="Observed", zorder=2)
    ax_.plot(x, exp_pct, color=COLORS["expected"], marker="o", markersize=3, linewidth=1.5,
             label="Expected", zorder=3)

    if show_confidence:
        ep = result.expected
        se = np.sqrt(ep * (1.0 - ep) / n)
        correction = 1.0 / (2.0 * n)
        upper_pct = (ep + confidence_z * se + correction) * 100.0
        lower_pct = np.maximum(ep - confidence_z * se - correction, 0.0) * 100.0
        ax_.fill_between(x, lower_pct, upper_pct, color=COLORS["confidence_fill"],
                         alpha=0.15, label="95 % CI", zorder=1)
        ax_.plot(x, upper_pct, color=COLORS["confidence_fill"], alpha=0.6,
                 linewidth=0.7, zorder=1)
        ax_.plot(x, lower_pct, color=COLORS["confidence_fill"], alpha=0.6,
                 linewidth=0.7, zorder=1)

    ax_.set_xticks(x)
    ax_.set_xticklabels(digits, rotation=90 if many_bins else 0)
    ax_.set_ylabel("Proportion (%)")
    ax_.set_title(title or _auto_title(result.test_name))
    ax_.legend(loc="upper right", fontsize=8, framealpha=0.8)

    info = (
        f"MAD={result.mad:.6f} ({result.mad_conformity})\n"
        f"χ²={result.chi_square:.2f} (sig={result.chi_square_significant})\n"
        f"n={result.n:,}"
    )
    ax_.text(0.98, 0.65, info, transform=ax_.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5})

    fig.tight_layout()
    return fig, ax_


# ---------------------------------------------------------------------------
# 2. plot_summation
# ---------------------------------------------------------------------------


def plot_summation(
    result: SummationResult,
    *,
    highlight_significant: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Bar chart of summation proportions vs. uniform expectation.

    Parameters
    ----------
    result
        A :class:`~pybenford.core.SummationResult`.
    highlight_significant
        Color bars red where ``result.significant_flags`` is True.
    title
        Plot title.
    figsize
        Figure dimensions.
    ax
        Existing Axes.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax_ = _setup_axes(ax, figsize)
    _clean_spines(ax_)

    digits = result.digits
    obs_pct = result.observed * 100.0
    exp_pct = float(result.expected[0]) * 100.0
    x = np.arange(len(digits))

    bar_colors: NDArray[np.object_] = np.full(len(digits), COLORS["summation"], dtype=object)
    if highlight_significant:
        bar_colors[result.significant_flags] = COLORS["significant"]

    ax_.bar(x, obs_pct, width=0.4, color=bar_colors, label="Observed", zorder=2)
    ax_.axhline(exp_pct, color=COLORS["expected"], linestyle="--", linewidth=1.2,
                label=f"Expected ({exp_pct:.2f} %)", zorder=3)

    ax_.set_xticks(x)
    ax_.set_xticklabels(digits, rotation=90, fontsize=6)
    ax_.set_ylabel("Sum Proportion (%)")
    ax_.set_title(title or "Summation Test")
    ax_.legend(loc="upper right", fontsize=8, framealpha=0.8)

    info = (
        f"Grand sum={result.grand_sum:,.2f}\n"
        f"χ²={result.chi_square:.2f}\n"
        f"n={result.n:,}"
    )
    ax_.text(0.98, 0.95, info, transform=ax_.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5})

    fig.tight_layout()
    return fig, ax_


# ---------------------------------------------------------------------------
# 3. plot_mantissa_arc
# ---------------------------------------------------------------------------


def plot_mantissa_arc(
    result: MantissaArcResult,
    data: NDArray[np.float64],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Scatter plot of mantissa points on the unit circle.

    Parameters
    ----------
    result
        A :class:`~pybenford.statistics.MantissaArcResult`.
    data
        The cleaned numeric array (``ba.clean_data``).
    title
        Plot title.
    figsize
        Figure dimensions.
    ax
        Existing Axes.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax_ = _setup_axes(ax, figsize)

    abs_data = np.abs(data.astype(np.float64))
    valid = np.isfinite(abs_data) & (abs_data > 0.0)
    log_v = np.log10(abs_data[valid])
    mantissas = log_v - np.floor(log_v)
    angles = 2.0 * np.pi * mantissas
    px = np.cos(angles)
    py = np.sin(angles)

    theta = np.linspace(0, 2 * np.pi, 300)
    ax_.plot(np.cos(theta), np.sin(theta), color=COLORS["grid"], linewidth=0.8)

    ax_.axhline(0, color=COLORS["grid"], linewidth=0.5)
    ax_.axvline(0, color=COLORS["grid"], linewidth=0.5)

    ax_.scatter(px, py, s=4, color=COLORS["mantissa_point"], alpha=0.4, zorder=2)
    ax_.plot(result.mean_x, result.mean_y, "D", color=COLORS["gravity_center"],
             markersize=10, zorder=3, label="Center of gravity")
    ax_.annotate(
        f"({result.mean_x:.4f}, {result.mean_y:.4f})",
        xy=(result.mean_x, result.mean_y),
        xytext=(10, 10), textcoords="offset points", fontsize=8,
        arrowprops={"arrowstyle": "->", "color": COLORS["gravity_center"]},
    )

    ax_.set_xlim(-1.3, 1.3)
    ax_.set_ylim(-1.3, 1.3)
    ax_.set_aspect("equal")
    ax_.set_title(title or "Mantissa Arc Test")
    ax_.legend(loc="upper left", fontsize=8)

    info = f"L²={result.L2:.6f}\np-value={result.p_value:.4f}"
    ax_.text(0.98, 0.05, info, transform=ax_.transAxes, fontsize=9,
             verticalalignment="bottom", horizontalalignment="right",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5})

    fig.tight_layout()
    return fig, ax_


# ---------------------------------------------------------------------------
# 4. plot_ordered_mantissas
# ---------------------------------------------------------------------------


def plot_ordered_mantissas(
    data: NDArray[np.float64],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Ordered-mantissa plot (Nigrini Fig 7.6).

    Parameters
    ----------
    data
        The cleaned numeric array (``ba.clean_data``).
    title
        Plot title.
    figsize
        Figure dimensions.
    ax
        Existing Axes.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax_ = _setup_axes(ax, figsize)
    _clean_spines(ax_)

    abs_data = np.abs(data.astype(np.float64))
    valid = np.isfinite(abs_data) & (abs_data > 0.0)
    log_v = np.log10(abs_data[valid])
    mantissas = np.sort(log_v - np.floor(log_v))
    n = len(mantissas)
    ranks = np.arange(1, n + 1)

    ax_.scatter(ranks, mantissas, s=1, color=COLORS["mantissa_point"], alpha=0.5, zorder=2)
    ax_.plot([1, n], [0.0, 1.0], color=COLORS["expected"], linestyle="--",
             linewidth=1.5, label="Expected (uniform)", zorder=3)

    ax_.set_xlabel("Rank")
    ax_.set_ylabel("Mantissa")
    ax_.set_title(title or "Ordered Mantissas")
    ax_.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig, ax_


# ---------------------------------------------------------------------------
# 5. plot_z_scores
# ---------------------------------------------------------------------------


def plot_z_scores(
    result: TestResult,
    *,
    critical_value: float = 1.96,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Bar chart of per-digit Z-scores with critical-value lines.

    Parameters
    ----------
    result
        A :class:`~pybenford.core.TestResult`.
    critical_value
        Horizontal threshold lines are drawn at +/- this value.
    title
        Plot title.
    figsize
        Figure dimensions.
    ax
        Existing Axes.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax_ = _setup_axes(ax, figsize)
    _clean_spines(ax_)

    digits = result.digits
    z = result.z_scores
    x = np.arange(len(digits))
    many_bins = len(digits) >= 20

    bar_colors = np.where(
        np.abs(z) > critical_value, COLORS["significant"], COLORS["observed"]
    )

    ax_.bar(x, z, width=0.5 if many_bins else 0.7, color=bar_colors, zorder=2)
    ax_.axhline(critical_value, color=COLORS["significant"], linestyle="--",
                linewidth=1, alpha=0.7, label=f"+/- {critical_value}")
    ax_.axhline(-critical_value, color=COLORS["significant"], linestyle="--",
                linewidth=1, alpha=0.7)

    ax_.set_xticks(x)
    ax_.set_xticklabels(digits, rotation=90 if many_bins else 0)
    ax_.set_ylabel("Z-statistic")
    ax_.set_title(title or _auto_title(result.test_name, "Z-Scores"))
    ax_.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig, ax_


# ---------------------------------------------------------------------------
# 6. plot_distortion_factor
# ---------------------------------------------------------------------------


def plot_distortion_factor(
    result: DistortionResult,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Gauge-style comparison of actual vs. expected collapsed mean.

    Parameters
    ----------
    result
        A :class:`~pybenford.statistics.DistortionResult`.
    title
        Plot title.
    figsize
        Figure dimensions.
    ax
        Existing Axes.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax_ = _setup_axes(ax, figsize)
    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)
    ax_.spines["left"].set_visible(False)
    ax_.set_yticks([])

    em = result.expected_mean
    am = result.actual_mean
    deviation = am - em
    margin = max(abs(deviation) * 2.0, 1.0)

    color = COLORS["significant"] if result.significant else COLORS["summation"]
    ax_.barh(0, deviation, left=em, height=0.4, color=color, alpha=0.7, zorder=2)
    ax_.axvline(em, color="black", linestyle="--", linewidth=1.2, label=f"Expected ({em:.2f})")
    ax_.plot(am, 0, "D", color=color, markersize=12, zorder=3,
             label=f"Actual ({am:.2f})")

    ax_.set_xlim(em - margin, em + margin)
    ax_.set_xlabel("Collapsed Mean")
    ax_.set_title(title or "Distortion Factor Model")
    ax_.legend(loc="upper left", fontsize=8)

    info = (
        f"DF={result.distortion_factor:.6f} ({result.direction})\n"
        f"Z={result.z_statistic:.4f}, p={result.p_value:.4f}\n"
        f"Significant={result.significant}"
    )
    ax_.text(0.98, 0.95, info, transform=ax_.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5})

    fig.tight_layout()
    return fig, ax_
