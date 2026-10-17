"""EAD lockbox parser, missing-identity, and target-blind model tests."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import audit_multims2_ead_metadata as ead
from benchmark_multims2_ead_lockbox import score_one

KEY="UHOVQNZJYSORNB-UHFFFAOYSA-N"

def hdr(e,mode="POSITIVE",frag="EAD"):
    return {"INCHIAUX":KEY,"FORMULA":"C8H10N4O2",
            "IONMODE":mode,"ADDUCT":"[M+H]+",
            "FRAGMENTATION_METHOD":frag,"MSLEVEL":"2",
            "COLLISION_ENERGY":f"[{e}.0]","NUM_PEAKS":"5",
            "DATASET_ID":"test_original_source"}

def test_energy_and_method_fail_closed():
    for i,e in enumerate((12,16,24)):
        row,reason=ead.eligible(hdr(e),i)
        assert reason is None and row["energy_eV"]==e
    for text in ("[16.5]","16","[16.0 eV]","[40.0]","16 eV"):
        bad=hdr(16);bad["COLLISION_ENERGY"]=text
        assert ead.eligible(bad,1)[0] is None
    assert ead.eligible(hdr(16,frag="CID"),1)[0] is None
    assert ead.eligible(hdr(16,mode="UNKNOWN"),1)[0] is None
    incomplete=hdr(16);incomplete.pop("INCHIAUX")
    assert ead.eligible(incomplete,1)[0] is None

def test_header_only_manifest_deterministic_exclusion():
    r=[ead.eligible(hdr(e),idx)[0] for idx,e in enumerate((12,16,24))]
    newer=ead.eligible(hdr(16),99)[0]
    m,diag=ead.build_manifest([newer,*reversed(r)],blocked={})
    assert len(m)==1
    assert m[0]["low_index"]==0
    assert m[0]["query_index"]==1
    assert m[0]["high_index"]==2
    assert m[0]["t"]==1/3
    assert diag["duplicate_slots_discarded_first_scan_only"]==1
    assert not ead.build_manifest(r,{"prior":{KEY}})[0]
    assert not ead.build_manifest(r[:2],{})[0]

def test_selected_scoring_uses_endpoint_only_for_prediction():
    records=[ead.eligible(hdr(e),i)[0] for i,e in enumerate((12,16,24))]
    manifest,_=ead.build_manifest(records,{})
    a=SimpleNamespace(mzs=np.array([100.,110.,120.,130.,140.]),intensities=np.array([4.,2.,1.,1.,2.]))
    q=SimpleNamespace(mzs=np.array([100.,110.,120.,130.,180.]),intensities=np.array([2.,3.,1.,1.,3.]))
    b=SimpleNamespace(mzs=np.array([100.,110.,150.,160.,170.]),intensities=np.array([1.,3.,2.,2.,2.]))
    values=score_one(manifest[0],{0:(records[0],a),1:(records[1],q),2:(records[2],b)})
    for method in ("hybrid","quarter","support","hellinger","linear","oracle"):
        assert 0<=values[method]<=1
    assert values["unmatched_query_mass"]>0
    assert values["oracle"]+1e-8>=values["hellinger"]

def test_mismatched_chemical_identity_is_hard_failure():
    r=[ead.eligible(hdr(e),i)[0] for i,e in enumerate((12,16,24))]
    manifest,_=ead.build_manifest(r,{})
    s=SimpleNamespace(mzs=np.array([1.,2.,3.,4.,5.]),intensities=np.ones(5))
    r[1]["inchikey"]="AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    with pytest.raises(RuntimeError,match="identity"):
        score_one(manifest[0],{0:(r[0],s),1:(r[1],s),2:(r[2],s)})
