"""Direct fixed-score test of cross-provider actual same-formula ID endpoint."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_harmonized2026_crossprovider import (
    scores_ion_spectra,tie_aware_rank)


def spec(m,z):
    a=np.asarray(m,float);b=np.asarray(z,float)
    k=np.argsort(a)
    return (a[k],b[k])


def test_identity_and_entire_unmatched_mass_penalty():
    a=spec([100.,120.],[1,1])
    b=spec([100.,140.],[1,1])
    scores=scores_ion_spectra(a,b)
    assert scores["cosine"]==pytest.approx(.5)
    assert scores["hellinger"]==pytest.approx(.5)
    assert scores["fragment_overlap"]==pytest.approx(.5)
    assert scores["support_composite"]==pytest.approx(.5)
    assert all(v==pytest.approx(1) for v in scores_ion_spectra(a,a).values())


def test_no_overlap_is_zero_and_scores_remain_normalized():
    a=spec([100.,120.],[3,7])
    b=spec([140.,160.],[3,7])
    assert all(v==0 for v in scores_ion_spectra(a,b).values())


def test_ties_correctly_reduce_top1_and_reciprocal_rank():
    vals=tie_aware_rank({"A":.75,"B":.75,"C":.3},"A")
    assert vals["top1_fractional"]==.5
    assert vals["mrr"]==pytest.approx(2/3)
    with pytest.raises(ValueError):
        tie_aware_rank({"A":.5},"A")
