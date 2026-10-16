"""Synthetic proof that posthoc alternate replicate choice uses metadata, not scores."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
PATH=ROOT/"scripts"/"audit_multims2_last_replicate_sensitivity.py"
spec=importlib.util.spec_from_file_location("audit_multims2_last_replicate_sensitivity",PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def row(index,energy):
    return {
       "record_index":index,"inchikey":"UHOVQNZJYSORNB-UHFFFAOYSA-N",
       "formula":"C8H10N4O2","adduct":"[M+H]+",
       "ion_mode":"POSITIVE","fragmentation":"CID",
       "source_dataset_id":"MSV000099369","energy_eV":energy,
    }

def original():
    r=row(0,40)
    for x in ("record_index","energy_eV"):
        r.pop(x)
    r.update({
        "low_index":1,"query_index":2,"high_index":3,
        "low_energy_eV":20,"query_energy_eV":40,"high_energy_eV":60,"t":.5,
    })
    return r

def test_alternate_last_uses_only_file_positions():
    candidates=[row(1,20),row(2,40),row(3,60),
                row(4,20),row(8,40),row(9,60)]
    chosen,info=m.last_replicate_manifest(candidates,[original()])
    assert len(chosen)==1
    assert (chosen[0]["low_index"],chosen[0]["query_index"],chosen[0]["high_index"])==(4,8,9)
    assert info["groups_with_any_alternate_scan"]==1
    assert info["low"]==1 and info["query"]==1 and info["high"]==1

def test_scanner_rejects_mismatch_to_original_first():
    candidates=[row(1,20),row(5,40),row(3,60)]
    with pytest.raises(RuntimeError,match="FIRST"):
        m.last_replicate_manifest(candidates,[original()])
    with pytest.raises(RuntimeError,match="lacks 20/40/60"):
        m.last_replicate_manifest([row(1,20),row(2,40)],[original()])

def test_model_methods_retained_unchanged():
    import benchmark_multims2_third_source_geometry as original_module
    assert "quarter" in original_module.METHODS
    assert "hellinger" in original_module.METHODS
    assert "adaptive_endpoint" in original_module.METHODS
