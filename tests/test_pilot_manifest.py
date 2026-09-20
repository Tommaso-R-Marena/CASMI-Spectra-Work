import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None or importlib.util.find_spec("pyarrow") is None,
    reason="optional parquet deps unavailable",
)


def test_build_pilot_manifest_requires_formula_competitors(tmp_path):
    import polars as pl
    from qfd.pilot_manifest import build_pilot_manifest

    path = tmp_path / "tiny.parquet"
    pl.DataFrame({
        "inchikey14": ["A","A","A","D","D","D","B","B"],
        "molecular_formula": ["C2H4"]*6 + ["C3H6"]*2,
        "ingest_lib": ["enveda-180"]*8,
        "instrument_type": ["timsTOF"]*8,
        "adduct": ["[M+H]+"]*8,
        "ionization_mode": ["positive"]*8,
        "collision_energy_ev": [[20.0],[40.0],[60.0],[20.0],[40.0],[60.0],[20.0],[40.0]],
        "spectrum_id": ["a1","a2","a3","d1","d2","d3","b1","b2"],
    }).write_parquet(path)

    got = build_pilot_manifest(path, min_spectra=3, min_energies=2, max_structures=10, min_formula_structures=2)
    assert got["inchikey14"].to_list() == ["A", "D"]
    assert got["formula_structure_count"].to_list() == [2, 2]
