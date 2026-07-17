"""pybenford - Professional-grade Benford's Law analysis toolkit."""

from typing import TYPE_CHECKING

from pybenford.core import (
    BenfordAnalysis,
    SmallSampleWarning,
    SummationResult,
    TestResult,
)
from pybenford.utils import (
    CleaningReport,
    DataProfile,
    DigitFrequencies,
    DuplicationResult,
    Stratum,
    SummationFrequencies,
    clean_numeric_array,
    data_profile,
    digit_counts,
    number_duplication,
    second_order_differences,
    summation_by_digits,
    to_numeric_array,
)

if TYPE_CHECKING:
    # Type checkers see the real signatures; runtime stays lazy.
    from pybenford.visualization import (
        plot_digit_test,
        plot_distortion_factor,
        plot_mantissa_arc,
        plot_ordered_mantissas,
        plot_summation,
        plot_z_scores,
    )

_PLOT_EXPORTS = frozenset(
    {
        "plot_digit_test",
        "plot_distortion_factor",
        "plot_mantissa_arc",
        "plot_ordered_mantissas",
        "plot_summation",
        "plot_z_scores",
    }
)


def __getattr__(name: str) -> object:
    if name in _PLOT_EXPORTS:
        try:
            from pybenford import visualization
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib" or (
                exc.name is not None and exc.name.startswith("matplotlib.")
            ):
                raise ImportError(
                    f"{name} requires matplotlib. "
                    'Install the plot extra: pip install "pybenford[plot]"'
                ) from exc
            raise  # unrelated failure — never relabel as missing matplotlib
        attr = getattr(visualization, name)
        globals()[name] = attr  # cache: subsequent access skips __getattr__
        return attr
    raise AttributeError(f"module 'pybenford' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PLOT_EXPORTS)


__all__ = [
    "BenfordAnalysis",
    "CleaningReport",
    "DataProfile",
    "DigitFrequencies",
    "DuplicationResult",
    "SmallSampleWarning",
    "Stratum",
    "SummationFrequencies",
    "SummationResult",
    "TestResult",
    "clean_numeric_array",
    "data_profile",
    "digit_counts",
    "number_duplication",
    "plot_digit_test",
    "plot_distortion_factor",
    "plot_mantissa_arc",
    "plot_ordered_mantissas",
    "plot_summation",
    "plot_z_scores",
    "second_order_differences",
    "summation_by_digits",
    "to_numeric_array",
]
