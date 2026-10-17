"""Unweighted Jensen-Shannon spectral entropy identity baseline, independently tested."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_harmonized2026_entropy_control import unweighted_entropy


def spec(m,z):
    return np.asarray(m,float),np.asarray(z,float)


def direct_js_affinity(p,q):
    p=np.asarray(p,float)
    q=np.asarray(q,float)
    p=p/p.sum()
    q=q/q.sum()
    m=(p+q)/2
    def kl(a):
        keep=a>0
        return float(np.sum(a[keep]*np.log(a[keep]/m[keep])))
    return 1-(kl(p)+kl(q))/(2*math.log(2))


def test_identical_spectra_are_one():
    s=spec([100.,150.,200.],[1.,3.,2.])
    assert unweighted_entropy(s,s)==pytest.approx(1.)


def test_disjoint_spectra_are_zero():
    assert unweighted_entropy(
        spec([100.,120.],[1.,2.]),spec([140.,160.],[2.,1.]))==0


def test_matched_fragment_and_unmatched_mass_exact_js_formula():
    a=spec([100.,120.],[2.,3.])
    b=spec([100.,140.],[1.,4.])
    expected=direct_js_affinity([2.,3.,0.],[1.,0.,4.])
    assert unweighted_entropy(a,b)==pytest.approx(expected,abs=1e-12)


def test_symmetric_when_one_to_one_matching_unique():
    a=spec([100.,150.,200.],[1.,3.,2.])
    b=spec([100.003,160.,200.004],[2.,5.,4.])
    assert unweighted_entropy(a,b)==pytest.approx(unweighted_entropy(b,a))


def test_norm_includes_unmatched_peaks():
    a=spec([100.,120.],[1.,9.])
    b=spec([100.,140.],[1.,9.])
    v=unweighted_entropy(a,b)
    assert 0<v<.2


def test_entropy_validates_spectrum_intensity():
    with pytest.raises(ValueError):
        unweighted_entropy(spec([100.],[0.]),spec([100.],[1.]))


def test_perfect_discrimination_mass_tolerance_boundaries():
    s=spec([100.],[1.])
    assert unweighted_entropy(s,spec([100.009],[1.]))==pytest.approx(1.)
    assert unweighted_entropy(s,spec([100.011],[1.]))==0
