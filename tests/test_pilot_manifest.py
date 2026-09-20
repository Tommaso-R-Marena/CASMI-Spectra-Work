import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None or importlib.util.find_spec("pyarrow") is None,
    reason="optional parquet deps unavailable",
)


def test_build_pilot_manifest_selects_multi_energy_groups(tmp_path):
    import polars as pl
    from qfd.pilot_manifest import build_pilot_manifest

    path = tmp_path / "tiny.parquet"
    pl.DataFrame({
        "inchikey14": ["A", "A", "A", "B", "B", "C"],
        "molecular_formula": ["C2H4"] * 3 + ["C3H6"] * 2 + ["C4H8"],
        "ingest_lib": ["enveda-180"] * 6,
        "instrument_type": ["timsTOF"] * 6,
        "adduct": ["[M+H]+"] * 6,
        "ionization_mode": ["positive"] * 6,
        "collision_energy_ev": [[20.0], [40.0], [40.0], [20.0], [20.0], [20.0, 40.0]],
        "spectrum_id": ["a1", "a2", "a3", "b1", "b2", "c1"],
    }).write_parquet(path)

    got = build_pilot_manifest(path, min_spectra=3, min_energies=2, max_structures=10)
    assert got["inchikey14"].to_list() == ["A"]
    assert got["n_spectra"].to_list() == [3]
    assert got["n_energies"].to_list() == [2]
