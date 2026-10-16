"""Adversarial metadata-only tests for third independent MultiMS2 source."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/"scripts"/"audit_multims2_third_source_metadata.py"
spec=importlib.util.spec_from_file_location("audit_multims2_third_source_metadata",PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def header(e,mode="Positive",key="UHOVQNZJYSORNB-UHFFFAOYSA-N"):
    return {
       "INCHIAUX":key,"FORMULA":"C8H10N4O2","ADDUCT":"[M+H]+",
       "IONMODE":mode,"FRAGMENTATION_METHOD":"CID",
       "COLLISION_ENERGY":str(e),"NUM_PEAKS":"10","MSLEVEL":"2",
       "DATASET_ID":"NEXUS",
    }

def test_source_metadata_parsing_rejects_wrong_cid_or_energy_units():
    rec,err=m.eligible(header("20.0"),7)
    assert err is None and rec["energy_eV"]==20 and rec["record_index"]==7
    for energy in ("10","30","50","20-40","[20,40]","20%","40 NCE","20.5","-20"):
        assert m.eligible(header(energy),7)[0] is None
    h=header("40")
    h["FRAGMENTATION_METHOD"]="EAD"
    assert m.eligible(h,0)[1]=="not_cid"
    h=header("60");h["NUM_PEAKS"]="4"
    assert m.eligible(h,0)[1]=="insufficient_declared_peaks"

def test_full_inchikey_and_collision_energy_alias():
    h=header("60 eV")
    h["INCHIAUX"]="InChIKey=UHOVQNZJYSORNB-UHFFFAOYSA-N"
    assert m.eligible(h,1)[0]["energy_eV"]==60
    h["INCHIAUX"]="UHOVQNZJYSORNB"
    assert m.eligible(h,1)[1]=="bad_full_inchikey"

def test_duplicate_energy_uses_first_file_order_and_rejects_any_prior_source():
    records=[]
    for i,e in enumerate([60,20,40,40]):
        r,_=m.eligible(header(e),i)
        records.append(r)
    manifest,meta=m.select_triplets(records,blocked={})
    assert len(manifest)==1
    assert manifest[0]["query_index"]==2
    assert meta["duplicate_energy_slots_discarded_first_in_file_wins"]==1
    key=records[0]["inchikey"]
    selected,why=m.select_triplets(records,{"prior":{"UHOVQNZJYSORNB-UHFFFAOYSA-N"}})
    assert selected==[] and why["blocked_records_by_source"]["prior"]==4

def test_header_stream_skips_peak_lines_including_untrusted_values(tmp_path):
    source=tmp_path/"fixture.mgf"
    lines=["BEGIN IONS","INCHIAUX=UHOVQNZJYSORNB-UHFFFAOYSA-N",
        "FORMULA=C8H10N4O2","ADDUCT=[M+H]+","IONMODE=Positive",
        "FRAGMENTATION_METHOD=CID","COLLISION_ENERGY=40.0","NUM_PEAKS=5",
        "120 50000000","130 0.1234","END IONS"]
    source.write_text("\n".join(lines)+"\n")
    found=list(m.stream_headers(source))
    assert len(found)==1 and found[0][1]["NUM_PEAKS"]=="5"
    assert "120" not in found[0][1]
    assert m.eligible(found[0][1],0)[0] is not None
    source.write_text("BEGIN IONS\nINCHIAUX=A\nEND IONS\nEND IONS\n")
    with pytest.raises(ValueError,match="end without beginning"):
        list(m.stream_headers(source))
