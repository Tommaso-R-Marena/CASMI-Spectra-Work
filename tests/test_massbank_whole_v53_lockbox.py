from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_massbank_whole_v53_lockbox import _parse_massbank_spectrum, _top1_vector

import numpy as np


def test_massbank_spectrum_parser_freezes_absolute_intensity_column(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text(
        """ACCESSION: X
MS$FOCUSED_ION: PRECURSOR_M/Z 123.456
PK$NUM_PEAK: 3
PK$PEAK: m/z int. rel.int.
  50.0000 22 3
  75.1000 145 15
  100.2000 9999 999
//
""",
        encoding="utf-8",
    )
    peaks, precursor = _parse_massbank_spectrum(p)
    assert precursor == 123.456
    assert peaks == [(50.0, 22.0), (75.1, 145.0), (100.2, 9999.0)]


def test_top1_credit_vector_is_tie_aware() -> None:
    got = _top1_vector(np.array([4.0, 4.0, 3.0, 1.0]))
    assert np.array_equal(got, np.array([0.5, 0.5, 0.0, 0.0]))
