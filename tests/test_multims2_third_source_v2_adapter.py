"""Score adapter must not change alpha, formulas or benchmark metric code."""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
PATH=ROOT/"scripts"/"benchmark_multims2_third_source_geometry_v2.py"

def test_v2_scoring_only_rebinds_frozen_header_decoder():
    spec=importlib.util.spec_from_file_location("benchmark_multims2_third_source_geometry_v2",PATH)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from audit_multims2_third_source_metadata import eligible
    h={
      "INCHIAUX":"UHOVQNZJYSORNB-UHFFFAOYSA-N",
      "FORMULA":"C8H10N4O2","ADDUCT":"[M+H]+",
      "IONMODE":"positive","FRAGMENTATION_METHOD":"CID",
      "COLLISION_ENERGY":"[40.0]","NUM_PEAKS":"5","MSLEVEL":"2",
      "DATASET_ID":"MSV000099369",
    }
    r,reason=eligible(h,14)
    assert reason is None and r["energy_eV"]==40
    import benchmark_multims2_third_source_geometry as frozen
    assert frozen.main is module.main
    assert frozen.METHODS==("quarter","hellinger","adaptive_endpoint","support_only","linear","hellinger_oracle")
