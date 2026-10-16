import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"scripts"/"benchmark_massbank_external_energy_geometry.py"
spec=importlib.util.spec_from_file_location("benchmark_massbank_external_energy_geometry",PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_massbank_peak_parser_uses_raw_intensity_second_column():
    lines=["ACCESSION: TEST","PK$NUM_PEAK: 5","PK$PEAK: m/z int. rel.int.",
           "100.0000 10 100","101.0000 20 200","102.0000 30 300",
           "103.0000 40 400","104.0000 50 500","//"]
    obj=m.parse_peak_spectrum("\n".join(lines),record_path="x.txt")
    assert obj.mzs.tolist()==[100.,101.,102.,103.,104.]
    assert obj.intensities.tolist()==[10.,20.,30.,40.,50.]


def test_massbank_peak_parser_rejects_too_few_valid_peaks():
    with pytest.raises(ValueError):
        m.parse_peak_spectrum("PK$PEAK: m/z int.\n100 10\n//",record_path="bad")


def test_non_orbitrap_classification():
    assert m.non_orbitrap({"instrument":"Bruker maXis Impact","instrument_type":"LC-ESI-QTOF"})
    assert not m.non_orbitrap({"instrument":"Q Exactive","instrument_type":"LC-ESI-FT"})


def test_scoring_refuses_failed_external_gate_before_archive_access(tmp_path,monkeypatch):
    gate=tmp_path/"gate"
    gate.mkdir()
    (gate/"feasibility.json").write_text(json.dumps({
        "status":"EXTERNAL_MASSBANK_2026_03_FEASIBILITY_ONLY",
        "gate_pass":False,
    }))
    monkeypatch.setattr("sys.argv",[
        "script","/does/not/exist.zip",
        "--feasibility-dir",str(gate),
        "--out",str(tmp_path/"out"),
    ])
    with pytest.raises(RuntimeError,match="FROZEN_EXTERNAL_GATE_FAILED"):
        m.main()
    assert not (tmp_path/"out").exists()
