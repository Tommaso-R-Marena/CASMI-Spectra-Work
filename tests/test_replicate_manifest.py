import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None or importlib.util.find_spec("pyarrow") is None,
    reason="optional parquet deps unavailable",
)


def test_replicate_manifest_requires_two_replicates_per_energy_and_formula_competitors(tmp_path):
    import polars as pl
    from qfd.pilot_manifest import build_replicate_pilot_manifest

    rows = []
    for key in ["A", "B"]:
        for e in [20.0, 40.0]:
            for rep in [0, 1]:
                rows.append({
                    "inchikey14": key,
                    "molecular_formula": "C6H6",
                    "ingest_lib": "enveda-180",
                    "adduct": "[M+H]+",
                    "ionization_mode": "positive",
                    "collision_energy_ev": [e],
                    "spectrum_id": f"{key}{e}{rep}",
                })
    rows.append({
        "inchikey14": "C", "molecular_formula": "C7H8", "ingest_lib": "enveda-180",
        "adduct": "[M+H]+", "ionization_mode": "positive", "collision_energy_ev": [20.0],
        "spectrum_id": "c",
    })
    p = tmp_path / "x.parquet"
    pl.DataFrame(rows).write_parquet(p)
    got = build_replicate_pilot_manifest(p, min_energies=2, min_formula_structures=2)
    assert got["inchikey14"].to_list() == ["A", "B"]
    assert got["n_replicated_energies"].to_list() == [2, 2]
