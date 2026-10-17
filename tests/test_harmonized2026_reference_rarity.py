"""Reference-only spectral-rarity invariants and formula-cluster statistics."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_harmonized2026_reference_rarity import (
    molecular_documents, ref_idf, weighted_hellinger, grouped_comparison)


def spectrum(masses,intensities):
    return np.asarray(masses,float),np.asarray(intensities,float)


def test_common_fragment_downweighted_vs_rare_fragment():
    docs=[np.array([100.,150.]),np.array([100.,200.]),np.array([100.,250.])]
    w=ref_idf([100.,150.,900.],docs)
    assert w[0]==pytest.approx(1.)
    assert w[1]==pytest.approx(1+math.log(4/2))
    assert w[2]==pytest.approx(1+math.log(4))
    assert w[0]<w[1]<w[2]


def test_replication_cannot_inflate_formula_molecular_doc_count():
    refs={1:{"source":"GNPS","inchikey":"A"},
          2:{"source":"GNPS","inchikey":"A"},
          3:{"source":"GNPS","inchikey":"B"}}
    scans={1:spectrum([100.,150.],[1.,1.]),
           2:spectrum([100.,150.],[2.,1.]),
           3:spectrum([100.,200.],[1.,1.])}
    docs=molecular_documents([1,2,3],refs,scans)
    assert len(docs)==2
    assert ref_idf([150.],docs)[0]==pytest.approx(1+math.log(3/2))


def test_identical_spectrum_has_unit_weighted_affinity():
    docs=[np.array([100.,150.]),np.array([100.,200.])]
    s=spectrum([100.,150.],[1.,3.])
    assert weighted_hellinger(s,s,docs)==pytest.approx(1.)


def test_unmatched_query_peaks_still_penalized():
    docs=[np.array([100.,150.]),np.array([100.,200.])]
    a=spectrum([100.,150.],[1.,1.])
    b=spectrum([100.,200.],[1.,1.])
    assert 0<weighted_hellinger(a,b,docs)<1
    assert weighted_hellinger(spectrum([700.],[1.]),b,docs)==0.


def test_query_peak_intensities_never_appear_in_reference_document_idf():
    docs=[np.array([100.,150.]),np.array([100.,200.])]
    w=ref_idf([100.,150.],docs)
    for scale in (0.001,1.,100000.):
        assert np.array_equal(w,ref_idf([100.,150.],docs))
        assert 0<=weighted_hellinger(
            spectrum([100.,150.],[scale,1.]),
            spectrum([100.,200.],[1.,1.]),docs)<=1


def test_fixed_mass_tolerance():
    docs=[np.array([100.,140.]),np.array([120.])]
    assert ref_idf([100.009],docs)[0]==pytest.approx(1+math.log(3/2))
    assert ref_idf([100.011],docs)[0]==pytest.approx(1+math.log(3))
    assert weighted_hellinger(spectrum([100.011],[1.]),
                              spectrum([100.],[1.]),docs)==0


def test_block_inference_clusters_on_formula_not_query_or_structure():
    def row(formula,d):
        return {"formula":formula,"metrics":{
            "rarity_blend":{"mrr":.5+d},"hellinger":{"mrr":.5}}}
    items=[row("X",.2),row("X",.2),row("Y",-.1),row("Z",.1)]
    analysis=grouped_comparison(items,"rarity_blend",seed=1)
    assert analysis["formula_bootstrap_95"]["n_blocks"]==3
    assert analysis["formula_signflip"]["n_blocks"]==3
    assert analysis["delta_mrr"]==pytest.approx(.1)
