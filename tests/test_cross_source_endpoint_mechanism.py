"""Third-source cross-source mechanism diagnostics must not inspect query spectra."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
PATH=ROOT/"scripts"/"audit_cross_source_endpoint_mechanism.py"
spec=importlib.util.spec_from_file_location("audit_cross_source_endpoint_mechanism",PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def spectrum(mzs,weights):
    return SimpleNamespace(mzs=np.array(mzs,dtype=float),intensities=np.array(weights,dtype=float))

def test_exact_disjoint_support_has_zero_endpoint_overlap():
    a=spectrum([100,120,140],[1,2,3])
    b=spectrum([101,121,141],[1,2,3])
    f=m.features(a,b)
    assert f["endpoint_bhattacharyya"]==0
    assert f["endpoint_shared_bins"]==0
    assert f["endpoint_unique_bins"]==6
    assert f["endpoint_exclusive_a_mass"]==pytest.approx(1.)
    assert f["endpoint_exclusive_b_mass"]==pytest.approx(1.)

def test_identical_endpoints_have_unit_overlap():
    a=spectrum([100,120,140],[1,2,3])
    f=m.features(a,a)
    assert f["endpoint_bhattacharyya"]==pytest.approx(1.)
    assert f["endpoint_exclusive_a_mass"]==0
    assert f["endpoint_exclusive_b_mass"]==0
    assert f["endpoint_min_overlap_mass"]==pytest.approx(1.)

def test_binning_has_fixed_intervals_including_endpoints():
    assert m.bin_label(0)=="[0.00,0.01)"
    assert m.bin_label(0.11)=="[0.10,0.20)"
    assert m.bin_label(1)=="[0.70,1.00)"

def test_feature_statistics_do_not_hide_negative_delta():
    rows=[
      {"inchikey":"A","endpoint_bhattacharyya":0.1,
       "endpoint_exclusive_a_mass":0.5,"endpoint_exclusive_b_mass":0.4,
       "delta_quarter_hellinger":-0.02,"delta_quarter_support":-0.01},
      {"inchikey":"B","endpoint_bhattacharyya":0.5,
       "endpoint_exclusive_a_mass":0.1,"endpoint_exclusive_b_mass":0.2,
       "delta_quarter_hellinger":0.01,"delta_quarter_support":0.00},
    ]
    s=m.source_stats(rows)
    assert s["n_queries"]==2
    assert s["mean_quarter_minus_hellinger"]==pytest.approx(-.005)
    assert len(s["endpoint_BC_bins_posthoc"])==2
