import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None or importlib.util.find_spec("pyarrow") is None,
    reason="optional parquet deps unavailable",
)


def test_library_pilot_cli_end_to_end(tmp_path):
    import polars as pl

    rows = []
    for key, dynamic in [("A", True), ("B", False)]:
        for idx, e in enumerate([10.0, 20.0, 30.0, 40.0]):
            if dynamic:
                ints = [[1, .5, .25], [.8, .7, .25], [.5, 1, .25], [.3, 1, .25]][idx]
            else:
                ints = [1, .5, .25]
            rows.append({
                "inchikey14": key,
                "molecular_formula": "C6H6",
                "spectrum_id": f"{key}{idx}",
                "adduct": "[M+H]+",
                "collision_energy_ev": [e],
                "ms2_mzs": [50.0, 100.0, 150.0],
                "ms2_normalized_intensities": ints,
                "base_peak_intensity": 10000.0,
                "precursor_mz": 200.0,
                "ingest_lib": "enveda-180",
                "instrument_type": "timsTOF",
                "ionization_mode": "positive",
            })
    path = tmp_path / "train.parquet"
    pl.DataFrame(rows).write_parquet(path)
    out = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_library_pilot.py"
    subprocess.run([
        sys.executable, str(script), str(path), "--out-dir", str(out),
        "--max-structures", "10", "--bootstrap", "50", "--max-energy-gap-ev", "10",
    ], check=True, env=env, capture_output=True, text=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_queries"] == 2
    assert summary["cohort"]["selected_structures"] == 2
    assert (out / "manifest.parquet").exists()
    assert (out / "per_query.csv").exists()
    for mode in ("static_best", "joint_energy", "two_energy_qfd"):
        assert f"mrr25_{mode}" in summary
