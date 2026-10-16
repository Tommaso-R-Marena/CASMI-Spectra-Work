"""ExpoLib label-free feasibility tests; no real spectral outcome is opened."""
import importlib.util
from pathlib import Path
import pytest

FILE=Path(__file__).resolve().parents[1]/"scripts"/"audit_expolib_nominal_energy_feasibility.py"
spec=importlib.util.spec_from_file_location("audit_expolib_nominal_energy_feasibility",FILE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def header(energy="[20.0]",key="UHOVQNZJYSORNB-UHFFFAOYSA-N",n="8"):
    return {
      "InChIKey":key,"Formula":"C8H10N4O2","Precursor_type":"[M+H]+",
      "Spectrum_type":"2","Method":"HCD","Collision_energy":energy,
      "Ion_mode":"POSITIVE","Num Peaks":n,
    }

def test_only_unique_nominal_one_value_ce_and_identity_eligible():
    assert m.accept(header(),"ESI_POS",2)["energy_nominal"]==20.
    for energy in ("20","[20, 30]","[20-40]","20 eV","[20%]","[20 V]","[0]"):
        assert m.accept(header(energy=energy),"ESI_POS",2) is None
    assert m.accept(header(key="bad"),"ESI_POS",2) is None
    assert m.accept(header(n="4"),"ESI_POS",2) is None
    assert m.accept(header(),"ESI_NEG",2) is None

def test_duplicate_nominal_slots_use_first_file_position_not_peak_scores():
    def row(energy,index):
        r=m.accept(header(energy=f"[{energy}]"),"ESI_POS",index)
        return r
    data=[row(10,1),row(20,2),row(20,3),row(40,4)]
    data[2]["n_peaks_declared"]=100
    trips,info=m.select(data,discovery=set())
    assert len(trips)==1
    t=trips[0]
    assert (t["low_index"],t["query_index"],t["high_index"])==(1,2,4)
    assert abs(t["t"]-1/3)<1e-12
    assert info["duplicate_nominal_energy_records_discarded_earliest_wins"]==1

def test_discovery_full_inchikey_excluded_all_settings():
    records=[m.accept(header(energy=f"[{e}]"),"ESI_POS",i) for i,e in enumerate((10,20,40))]
    trips,info=m.select(records,{records[0]["inchikey"]})
    assert trips==[] and info["excluded_discovery_records"]==3

def test_nominal_msp_parser_skips_peaks_and_treats_duplicate_headers_as_invalid(tmp_path):
    def make(name,eng,duplicate=False):
        lines=[f"Name: {name}","InChIKey: UHOVQNZJYSORNB-UHFFFAOYSA-N",
               "Formula: C8H10N4O2","Precursor_type: [M+H]+",
               "Spectrum_type: 2","Method: HCD",f"Collision_energy: [{eng}]",
               "Ion_mode: POSITIVE","Num Peaks: 8"]
        if duplicate:
            lines.append("Collision_energy: [35.0]")
        lines += ["100.0 999","101.0 1","",""]
        return "\n".join(lines)
    path=tmp_path/"source.msp"
    path.write_text(make("first",10)+"\n"+make("second",20)+"\n"+make("invalid",30,True))
    rec,n=m.parse_headers(path,"ESI_POS")
    assert n==3
    assert len(rec)==2
    assert [r["energy_nominal"] for r in rec]==[10.,20.]
