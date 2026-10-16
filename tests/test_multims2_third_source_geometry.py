"""Fail-closed synthetic tests for independently frozen third-source geometry."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
SCRIPT=ROOT/"scripts"/"benchmark_multims2_third_source_geometry.py"
spec=importlib.util.spec_from_file_location("benchmark_multims2_third_source_geometry",SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def record(i,e,mode="POSITIVE"):
    return {
        "record_index":i,"inchikey":"UHOVQNZJYSORNB-UHFFFAOYSA-N",
        "formula":"C8H10N4O2","adduct":"[M+H]+","ion_mode":mode,
        "fragmentation":"CID","energy_eV":e,"source_dataset_id":"NEXUS",
    }

def row():
    return {"inchikey":"UHOVQNZJYSORNB-UHFFFAOYSA-N","formula":"C8H10N4O2",
      "adduct":"[M+H]+","ion_mode":"POSITIVE","fragmentation":"CID",
      "source_dataset_id":"NEXUS","low_index":0,"query_index":1,"high_index":2,
      "low_energy_eV":20,"query_energy_eV":40,"high_energy_eV":60,"t":0.5,
    }

def spectrum(intensity,base=100):
    from types import SimpleNamespace
    return SimpleNamespace(mzs=np.array([base,base+10,base+20,base+30,base+40],float),
                           intensities=np.array(intensity,float))

def toy():
    return {
        0:(record(0,20),spectrum([80,20,15,4,5])),
        1:(record(1,40),spectrum([55,52,27,4,5])),
        2:(record(2,60),spectrum([30,75,40,4,5])),
    }

def test_all_predictors_are_bounded_and_alpha_never_uses_target():
    records=toy()
    one=m.score_one(row(),records)
    assert .25<=one["endpoint_adaptive_alpha"]<=.5
    for method in m.METHODS:
        assert 0<=one[method]<=1
    records[1]=(record(1,40),spectrum([20,40,50,10,5]))
    two=m.score_one(row(),records)
    assert one["endpoint_adaptive_alpha"]==two["endpoint_adaptive_alpha"]
    assert one["endpoint_bhattacharyya"]==two["endpoint_bhattacharyya"]

def test_strict_source_identity_and_20_40_60_energy_guard():
    records=toy()
    with pytest.raises(RuntimeError,match="identity differs"):
        m.score_one({**row(),"inchikey":"AAAAAAAAAAAAAA-UHFFFAOYSA-N"},records)
    with pytest.raises(RuntimeError,match="collision energy"):
        m.score_one({**row(),"query_energy_eV":42},records)
    with pytest.raises(RuntimeError,match="0.5"):
        m.score_one({**row(),"t":0.49},records)

def test_candidate_filename_reader_never_infers_metadata_from_target(tmp_path):
    content=[]
    for i,e in enumerate((20,40,60)):
        content += [
            "BEGIN IONS",
            "INCHIAUX=UHOVQNZJYSORNB-UHFFFAOYSA-N",
            "FORMULA=C8H10N4O2","ADDUCT=[M+H]+",
            "IONMODE=Positive","FRAGMENTATION_METHOD=CID",
            f"COLLISION_ENERGY={e}.0","NUM_PEAKS=5","MSLEVEL=2",
            "DATASET_ID=NEXUS",
            "100 100","110 30","120 15","130 10","140 5",
            "END IONS",
        ]
    p=tmp_path/"source.mgf";p.write_text("\n".join(content)+"\n")
    records=m.load_selected_source_spectra(p,{0,1,2})
    assert len(records)==3
    assert m.score_one(row(),records)["quarter"]==pytest.approx(1.0,abs=1e-12)
    with pytest.raises(RuntimeError,match="inventory mismatch"):
        m.load_selected_source_spectra(p,{5})

def test_metadata_gate_prevents_untouched_source_score_opening(tmp_path):
    root=tmp_path/"gate";root.mkdir()
    (root/"feasibility.json").write_text('{"status":"MULTIMS2_CID_20_40_60EV_METADATA_ONLY_V1","gate_pass":false}')
    (root/"frozen_triplets.jsonl").write_text("")
    with pytest.raises(RuntimeError,match="GATE CLOSED"):
        m.score_source(tmp_path/"source.mgf",root,tmp_path/"out")
