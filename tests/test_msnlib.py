import json

from qfd.msnlib import iter_msnlib_jsonl, record_from_json


def example(energy=20.0):
    return {
        "inchikey": "ABCDEFGHIJKLMN-UHFFFAOYSA-N",
        "formula": "C10H12O2",
        "compound_name": "example",
        "adduct": "[M+H]+",
        "collision_energy": energy,
        "merge_type": "SINGLE_BEST_SCAN",
        "precursor_mz": 165.091,
        "usi": ["mzspec:test:scan:1"],
        "peaks": [[50.0, 100.0], [100.0, 50.0]],
    }


def test_record_adapter_handles_scalar_and_merged_energy():
    a = record_from_json(example(20.0))
    b = record_from_json(example([20.0, 40.0, 60.0]))
    assert a.inchikey14 == "ABCDEFGHIJKLMN"
    assert a.is_single_energy and not a.is_merged_energy
    assert b.is_merged_energy
    assert b.collision_energies == (20.0, 40.0, 60.0)
    assert a.view.base_peak_intensity == 100.0


def test_jsonl_streaming(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps(example(20.0)) + "\n" + json.dumps(example(40.0)) + "\n")
    got = list(iter_msnlib_jsonl(p))
    assert len(got) == 2
    assert [r.collision_energies for r in got] == [(20.0,), (40.0,)]
