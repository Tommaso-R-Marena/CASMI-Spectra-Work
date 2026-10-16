"""Strict synthetic tests for target-blind MassBank identification ranking."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import reference_only_grid,query_mass_on_reference_grid,affinity,support_only_suppression

path=ROOT/"scripts"/"benchmark_massbank_isomer_identification.py"
spec=importlib.util.spec_from_file_location("benchmark_massbank_isomer_identification",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from types import SimpleNamespace as Spectrum

def spec_peaks(masses,weights):
    return Spectrum(mzs=np.array(masses,float),intensities=np.array(weights,float))

def example(i,key,formula="C3H6O"):
    return {
        "inchikey":key,"formula":formula,"ion_mode":"POSITIVE",
        "adduct":"[M+H]+","contributor":"Fixture",
        "instrument":"Instrument X","instrument_type":"QTOF",
        "fragmentation":"CID","low_eV":10.0,"query_eV":20.0,"high_eV":30.0,"t":.5,
        "low_path":f"{i}_low","query_path":f"{i}_mid","high_path":f"{i}_high",
    }

def test_label_free_pool_selection_rejects_stereochemical_ambiguity():
    one=example(1,"AAAAAAAAAAAAAA-UHFFFAOYSA-N")
    two=example(2,"BBBBBBBBBBBBBB-UHFFFAOYSA-N")
    three=example(3,"AAAAAAAAAAAAAA-ABCXYZABCD-Y")
    pools,bad=m.select_pools([one,two])
    assert len(pools)==1 and bad==0
    pools,bad=m.select_pools([one,two,three])
    assert len(pools)==0 and bad==1

def test_reference_prediction_rank_is_target_blind_and_self_consistent():
    a=example(1,"AAAAAAAAAAAAAA-UHFFFAOYSA-N")
    b=example(2,"BBBBBBBBBBBBBB-UHFFFAOYSA-N")
    specs={
      "1_low":spec_peaks([100,125,150,200,250],[80,20,35,10,5]),
      "1_high":spec_peaks([100,125,150,200,250],[30,90,65,5,5]),
      "1_mid":spec_peaks([100,125,150,200,250],[50,55,50,10,5]),
      "2_low":spec_peaks([110,135,160,210,260],[70,20,40,5,5]),
      "2_high":spec_peaks([110,135,160,210,260],[20,90,60,5,5]),
      "2_mid":spec_peaks([110,135,160,210,260],[50,50,50,10,5]),
    }
    originals={}
    for r in (a,b):
        grid=reference_only_grid(specs[r["low_path"]],specs[r["high_path"]],tolerance_da=.01)
        q,_=query_mass_on_reference_grid(specs[r["query_path"]],grid,tolerance_da=.01)
        original={
            "alpha_quarter":affinity(q,power_mean_distribution(grid.low,grid.high,.5,.25)),
            "hellinger_fixed":affinity(q,power_mean_distribution(grid.low,grid.high,.5,.5)),
            "support_only":affinity(q,support_only_suppression(grid.low,grid.high,.5,power=4)),
        }
        originals[r["query_path"]]=original
    results=m.rank_pool([a,b],specs.__getitem__,originals)
    assert len(results)==2
    for r in results:
        for method in m.METHODS:
            assert r[method+"_rank"]==1 and r[method+"_rr"]==1.
            assert r[method+"_margin"]>0
    # A single changed archived diagonal score must abort.
    tampered={k:dict(v) for k,v in originals.items()}
    tampered["1_mid"]["alpha_quarter"]=.1
    with pytest.raises(ValueError,match="original"):
        m.rank_pool([a,b],specs.__getitem__,tampered)

def test_challenge_expected_formula_block_counts_fail_closed():
    p=m.select_pools([example(1,"AAAAAAAAAAAAAA-UHFFFAOYSA-N"),
                      example(2,"BBBBBBBBBBBBBB-UHFFFAOYSA-N")])[0]
    with pytest.raises(ValueError,match="frozen number"):
        m.check_pools(p,set())

def test_correct_key_tie_break_does_not_use_answer():
    assert sorted([("BBBBBBBBBBBBBB",.5),("AAAAAAAAAAAAAA",.5)],
                  key=lambda pair:(-pair[1],pair[0]))[0][0]=="AAAAAAAAAAAAAA"
