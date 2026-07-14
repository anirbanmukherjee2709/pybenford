"""Census regression suite: locked outputs on real county-population data.

Expectations were computed at commit ``111b927`` and lock the end-to-end
behavior of ``BenfordAnalysis`` on the bundled Census county estimates
(``examples/data/co-est2025-alldata.csv``). Never adjust an expectation
to make a test pass — a mismatch is a behavior change to investigate.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pytest

from pybenford import BenfordAnalysis, SmallSampleWarning

CSV_PATH = Path(__file__).parent.parent / "examples" / "data" / "co-est2025-alldata.csv"

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="census example data not present (e.g. sdist without examples)",
)


def _load_county_populations() -> list[float]:
    """County-level (SUMLEV 050) 2025 population estimates.

    The file also contains 51 state aggregates, which the SUMLEV filter
    excludes.
    """
    with open(CSV_PATH, encoding="latin-1", newline="") as f:
        return [
            float(row["POPESTIMATE2025"])
            for row in csv.DictReader(f)
            if row["SUMLEV"].strip() == "050"
        ]


class TestCensusRegression:
    def test_record_count_after_filter(self) -> None:
        assert len(_load_county_populations()) == 3144

    def test_locked_analysis_outputs_warning_free(self) -> None:
        data = _load_county_populations()
        # n=3144 >= 1000: no SmallSampleWarning may fire on any call
        with warnings.catch_warnings():
            warnings.simplefilter("error", SmallSampleWarning)
            ba = BenfordAnalysis(data)
            fd = ba.first_digit()
            ft = ba.first_two_digits()
            so = ba.second_order()
            df = ba.distortion_factor()

        assert ba.n == 3144

        assert fd.mad == pytest.approx(0.003586242795, rel=1e-9)
        assert fd.mad_conformity == "close_conformity"
        assert fd.chi_square == pytest.approx(5.6231120182, rel=1e-9)

        assert ft.mad == pytest.approx(0.001613208349, rel=1e-9)
        assert ft.mad_conformity == "acceptable_conformity"

        assert so.mad == pytest.approx(0.005399086450, rel=1e-9)
        assert so.n == 3091

        assert df.distortion_factor == pytest.approx(-0.001881979793, rel=1e-9)
        assert df.z_statistic == pytest.approx(-0.1653344519, rel=1e-9)
