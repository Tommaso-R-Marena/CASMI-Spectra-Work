"""Frozen MultiMS2 v2 CE repair is exclusively a three-value bracket decoder."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
FILE=ROOT/"scripts"/"audit_multims2_third_source_metadata_v2.py"
spec=importlib.util.spec_from_file_location("audit_multims2_third_source_metadata_v2",FILE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def raw_header(ce,frag="CID"):
    return {"INCHIAUX":"UHOVQNZJYSORNB-UHFFFAOYSA-N",
            "FORMULA":"C8H10N4O2","ADDUCT":"[M+H]+",
            "IONMODE":"positive","FRAGMENTATION_METHOD":frag,
            "MSLEVEL":"2","COLLISION_ENERGY":ce,
            "NUM_PEAKS":"5","DATASET_ID":"MSV000099369"}

def test_only_originally_observed_bracketed_cid_values(monkeypatch):
    monkeypatch.setattr(m.original,"ENERGY",m.NEW_EXACT_CID_ENERGY)
    for i,(ce,expected) in enumerate((("[20.0]",20),("[40.0]",40),("[60.0]",60))):
        record,reason=m.original.eligible(raw_header(ce),i)
        assert reason is None and record["energy_eV"]==expected
    for ce in ("20.0","40 eV","[20.1]","[30.0]","[20,40]","[20%]","20-60","[20.0 eV]"):
        assert m.original.eligible(raw_header(ce),1)[0] is None
    assert m.original.eligible(raw_header("[20.0]",frag="EAD"),1)[0] is None

def test_original_scientific_gates_and_source_identity_are_inherited():
    assert m.original.THRESHOLDS=={"triplets":100,"structures":80,"formula_blocks":60,
                                    "positive_triplets":20,"negative_triplets":20}
    assert m.OBSERVED_CID=={"[20.0]":9538,"[40.0]":13089,"[60.0]":13046}
    assert m.SOURCE_SHA256=="4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"
