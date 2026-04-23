"""pybenford - Professional-grade Benford's Law analysis toolkit."""

from pybenford.core import (
    BenfordAnalysis,
    SummationResult,
    TestResult,
)
from pybenford.visualization import (
    plot_digit_test,
    plot_distortion_factor,
    plot_mantissa_arc,
    plot_ordered_mantissas,
    plot_summation,
    plot_z_scores,
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

__all__ = [
    "BenfordAnalysis",
    "SummationResult",
    "TestResult",
    "plot_digit_test",
    "plot_distortion_factor",
    "plot_mantissa_arc",
    "plot_ordered_mantissas",
    "plot_summation",
    "plot_z_scores",
    "CleaningReport",
    "DataProfile",
    "DigitFrequencies",
    "DuplicationResult",
    "Stratum",
    "SummationFrequencies",
    "clean_numeric_array",
    "data_profile",
    "digit_counts",
    "number_duplication",
    "second_order_differences",
    "summation_by_digits",
    "to_numeric_array",
]
