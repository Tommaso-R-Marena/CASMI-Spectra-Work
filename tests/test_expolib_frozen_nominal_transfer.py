"""Fail-closed synthetic tests for frozen independent ExpoLib nominal CE scorer."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"scripts"/"benchmark_expolib_frozen_nominal_transfer.py"
spec=importlib.util.spec_from_file_location("benchmark_expolib_frozen_nominal_transfer",PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def fixture_record(energy,peaks="100 10\n120 20\n130 30\n140 40\n150 50"):
    return "\n".join([
        "Name: Test example","Precursor_type: [M+H]+","Formula: C8H10N4O2",
        "InChIKey: UHOVQNZJYSORNB-UHFFFAOYSA-N",
        "Spectrum_type: 2","Method: HCD",f"Collision_energy: [{energy}]",
        "Ion_mode: POSITIVE","Num Peaks: 5",peaks,"",
    ])

def test_selective_parser_does_not_change_nonselected_records(tmp_path):
    path=tmp_path/"selected.msp"
    path.write_text(fixture_record(10)+"\n"+fixture_record(20)+"\n"+fixture_record(40)+"\n")
    parsed=m.load_selected_spectra(path,{0,2})
    assert set(parsed)=={0,2}
    assert parsed[0]["energy_nominal"]==10.0
    assert parsed[2]["spectrum"].intensities.tolist()==[10,20,30,40,50]
    with pytest.raises(ValueError,match="incomplete"):
        m.load_selected_spectra(path,{9})

def test_identity_mismatch_aborts_before_reporting_scores(tmp_path):
    path=tmp_path/"source.msp"
    path.write_text(fixture_record(10)+"\n"+fixture_record(20)+"\n"+fixture_record(40)+"\n")
    selected=m.load_selected_spectra(path,{0,1,2})
    row={
        "source":"ESI_POS","inchikey":"UHOVQNZJYSORNB-UHFFFAOYSA-N",
        "formula":"C8H10N4O2","adduct":"[M+H]+","mode":"POSITIVE",
        "fragmentation":"HCD","low_index":0,"query_index":1,"high_index":2,
        "low_energy_nominal":10.0,"query_energy_nominal":20.0,
        "high_energy_nominal":40.0,"t":1/3,
    }
    x=m.score_triplet(row,{"ESI_POS":selected})
    assert abs(x["quarter"]-1)<1e-10
    assert abs(x["hellinger"]-1)<1e-10
    assert abs(x["unmatched_target_fraction"])<1e-10
    with pytest.raises(RuntimeError,match="identity mismatch"):
        m.score_triplet({**row,"inchikey":"AAAAAAAAAAAAAA-UHFFFAOYSA-N"},{"ESI_POS":selected})

def test_query_off_grid_mass_counts_against_normalization(tmp_path):
    path=tmp_path/"source.msp"
    path.write_text(fixture_record(10)+"\n"+fixture_record(20,
        peaks="100 10\n120 20\n130 30\n140 40\n777 50")+"\n"+fixture_record(40)+"\n")
    selected=m.load_selected_spectra(path,{0,1,2})
    from qfd.target_blind_geometry import reference_only_grid,query_mass_on_reference_grid
    g=reference_only_grid(selected[0]["spectrum"],selected[2]["spectrum"],tolerance_da=.01)
    q,unmatched=query_mass_on_reference_grid(selected[1]["spectrum"],g,tolerance_da=.01)
    assert abs(unmatched-1/3)<1e-10
    assert abs(q.sum()+unmatched-1)<1e-10

def test_frozen_gate_enforces_complete_polarity_counts():
    d={
      "status":"EXPOLIB_V1_POSTDISCOVERY_LABEL_FREE_FEASIBILITY",
      "metadata_feasibility_pass":True,
      "physical_eV_unit_validation_pass":False,
      "scores_opened":False,
      "coverage":{"triplets":1125,"structures":122,"formula_blocks":114,"offcenter_triplets":103},
    }
    with pytest.raises(RuntimeError,match="frozen ExpoLib ion-polarity cohort"):
        m.validate_manifest(d,[])
    d["physical_eV_unit_validation_pass"]=True
    with pytest.raises(RuntimeError,match="wrong source gate"):
        m.validate_manifest(d,[])
