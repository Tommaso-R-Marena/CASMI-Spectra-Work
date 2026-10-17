"""Scientific guardrail regressions for a fourth-source CID transfer lockbox."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_phenolicsdb_independent_gate import header_only, build
from benchmark_phenolicsdb_independent_cid import (
    summary, load_selected, cmp
)
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity
)


def spec(mz, intensity):
    return SimpleNamespace(mzs=np.asarray(mz, float),
                           intensities=np.asarray(intensity, float))


def make_four_source(key="PFTAWBLQPZVEMU-UKRRQHHQSA-N"):
    return [
        {"record_index":i,"inchikey":key,"polarity":"Positive","energy_ev":ce,
         "precursor_mz":200.,"source_name":f"molecule {ce}eV"}
        for i,ce in enumerate((20,40,60,80))
    ]


def test_metadata_gate_ignores_target_peaks_and_uses_exact_keys(tmp_path):
    text=[]
    for i,e in enumerate((20,40,60,80)):
        text.extend([
            "BEGIN IONS",
            "NAME=test compound "+str(e)+"eV",
            "INCHI=PFTAWBLQPZVEMU-UKRRQHHQSA-N",
            "IONMODE=Positive","MSLEVEL=2","PEPMASS=200.0",
            # Intentionally nonnumeric: metadata-only gate MUST ignore this.
            "THIS HELDOUT PEAK INTENSITY IS UNTRUSTED",
            "END IONS",
        ])
    path=tmp_path/"fake.mgf";path.write_text("\n".join(text)+"\n")
    meta=header_only(path,"Positive")
    result,coverage=build(meta,[])
    assert len(result)==1
    assert coverage["target_queries"]==2
    assert result[0]["source_indices"]=={"20":0,"40":1,"60":2,"80":3}


def test_duplicate_energy_fails_closed():
    x=make_four_source()
    with pytest.raises(RuntimeError):
        build(x+[x[0]],[])


def test_inconsistent_precursor_fails_closed():
    x=make_four_source()
    x[2]["precursor_mz"]=201
    with pytest.raises(RuntimeError):
        build(x,[])


def test_metadata_no_four_energies_no_scored_query():
    x=make_four_source()
    result,coverage=build(x[:3],[])
    assert not result
    assert coverage["target_queries"]==0


def test_selected_source_parser_does_not_rescore_unselected_scans(tmp_path):
    text=[]
    for i,e in enumerate((20,40,60,80)):
        text.extend(["BEGIN IONS",f"NAME=test {e}eV",
                     "INCHI=PFTAWBLQPZVEMU-UKRRQHHQSA-N",
                     "IONMODE=Positive","MSLEVEL=2","PEPMASS=200.0",
                     "100 8","200 2" if i!=1 else "200 4",
                     "END IONS"])
    p=tmp_path/"test.mgf";p.write_text("\n".join(text)+"\n")
    scans=load_selected(p,"Positive",{0,2,3})
    assert set(scans)=={0,2,3}
    assert len(scans[2][1].mzs)==2
    assert scans[3][0]["ce"]==80


def test_target_does_not_change_reference_predictions():
    low=spec([100., 120.], [8,2])
    high=spec([100.001,130.], [2,8])
    grid=reference_only_grid(low,high,tolerance_da=.01)
    h=power_mean_distribution(grid.low,grid.high,1/3,.5)
    q=power_mean_distribution(grid.low,grid.high,1/3,.25)
    assert not np.allclose(h,q)
    s1=spec([100.001,120.0],[8,2])
    s2=spec([100.003,130.0],[8,2])
    a,_=query_mass_on_reference_grid(s1,grid)
    b,_=query_mass_on_reference_grid(s2,grid)
    assert affinity(a,h)>=0
    assert affinity(b,h)>=0
    assert np.allclose(h,power_mean_distribution(grid.low,grid.high,1/3,.5))


def test_structure_block_keeps_two_polarities_and_target_energies_together():
    rows=[]
    for p in ("Positive","Negative"):
        for e in (40,60):
            rows.append({"inchikey":"MOLECULE-ONE","formula":"not used",
                         "polarity":p,"quarter":.6,"hellinger":.5,
                         "support_only":.55})
    result=summary(rows,False)
    assert result["n_unique_structure_blocks"]==1
    assert result["n_mode_structure_groups"]==2
    assert result["n_queries"]==4
    assert result["quarter_minus_hellinger_descriptive"]==pytest.approx(.1)
