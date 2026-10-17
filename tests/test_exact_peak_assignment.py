"""Exact matching theorem checked by exhaustive reference enumeration.

This is algorithmic correctness of the weighted bipartite matching objective,
NOT physical kinetics or external laboratory rank superiority.
"""
from __future__ import annotations
from itertools import permutations
import math
import numpy as np
import pytest
from qfd.exact_peak_assignment import (
    METHODS,optimal_ion_spectra,admissible_edges)
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_harmonized2026_crossprovider import scores_ion_spectra


def mk(mz,intensity):
    mz=np.asarray(mz,float);intensity=np.asarray(intensity,float)
    order=np.argsort(mz,kind="mergesort")
    return mz[order],intensity[order]


def weights(q,r):
    qm,qi=q;rm,ri=r
    ql=qi/qi.sum();rl=ri/ri.sum()
    q2=qi/np.linalg.norm(qi);r2=ri/np.linalg.norm(ri)
    result={}
    for i,j in admissible_edges(qm,rm,.01):
        a=q2[i]*r2[j];b=math.sqrt(ql[i]*rl[j]);c=min(ql[i],rl[j])
        result[i,j]={"cosine":a,"hellinger":b,"fragment_overlap":c,
                     "support_composite":(a+c)/2}
    return result


def brute_force(q,r):
    """Exhaustively enumerate all injections allowing unmatched query peaks."""
    n=len(q[0]);m=len(r[0]);allowed=weights(q,r)
    ans={name:0. for name in METHODS}
    def rec(i,used,total):
        if i==n:
            for name in METHODS:ans[name]=max(ans[name],total[name])
            return
        rec(i+1,used,total)
        for j in range(m):
            if (i,j) in allowed and j not in used:
                w=allowed[i,j]
                rec(i+1,used|{j},{k:total[k]+w[k] for k in METHODS})
    rec(0,set(),{k:0. for k in METHODS})
    return ans


def test_conflicting_peaks_greedy_nearest_mass_is_suboptimal():
    q=mk([100,100.007],[1,100])
    r=mk([100.001],[100])
    greedy=scores_ion_spectra(q,r)
    best,info=optimal_ion_spectra(q,r)
    assert info["n_ambiguous_components"]==1
    assert best["hellinger"]>greedy["hellinger"]+.7
    assert best["cosine"]>greedy["cosine"]+.9
    assert best["fragment_overlap"]>greedy["fragment_overlap"]+.9


def test_exhaustive_all_injections_for_300_randomized_small_spectra():
    rng=np.random.default_rng(20261017)
    for _ in range(300):
        n=int(rng.integers(1,5));m=int(rng.integers(1,5))
        qm=np.sort(100+rng.uniform(0,.04,size=n))
        rm=np.sort(100+rng.uniform(0,.04,size=m))
        q=mk(qm,rng.lognormal(size=n))
        r=mk(rm,rng.lognormal(size=m))
        truth=brute_force(q,r)
        actual,_=optimal_ion_spectra(q,r)
        for k in METHODS:
            assert actual[k]==pytest.approx(truth[k],abs=1e-12)


def test_monotonicity_against_published_greedy_for_mixed_support():
    rng=np.random.default_rng(123)
    for _ in range(400):
        q=mk(100+rng.uniform(0,.08,10),rng.uniform(.01,10,10))
        r=mk(100+rng.uniform(0,.08,10),rng.uniform(.01,10,10))
        fixed=scores_ion_spectra(q,r)
        exact,_=optimal_ion_spectra(q,r)
        for k in METHODS:
            assert exact[k]+1e-10>=fixed[k]


def test_alignment_may_be_identity_ambiguous_but_all_unmatched_denominators_preserved():
    q=mk([100,200,300],[1,1,8])
    r=mk([100,400],[1,9])
    actual,_=optimal_ion_spectra(q,r)
    assert actual["fragment_overlap"]==pytest.approx(.1)
    assert actual["hellinger"]==pytest.approx(.1)
    assert actual["cosine"]==pytest.approx(.1/math.sqrt(.66))


@pytest.mark.parametrize("q,r",[
    (([100],[0]),([100],[1])),
    (([100],[-1]),([100],[1])),
    (([100,float("nan")],[1,1]),([100],[1])),
    (([200,100],[1,1]),([100],[1])),
    (([100],[1]),([0],[1])),
])
def test_invalid_peak_data_fail_closed(q,r):
    with pytest.raises(ValueError):
        optimal_ion_spectra(mk(q[0],q[1]) if False else
                            (np.asarray(q[0]),np.asarray(q[1])),
                            (np.asarray(r[0]),np.asarray(r[1])))


def test_no_match_stays_zero_without_renormalizing():
    q=mk([100,120],[1,1])
    r=mk([101,121],[1,1])
    actual,info=optimal_ion_spectra(q,r)
    assert info["n_admissible_edges"]==0
    assert all(v==0 for v in actual.values())


def test_assignments_repeat_deterministically():
    q=mk([100,100.003,100.006],[10,20,30])
    r=mk([100.001,100.004,100.007],[8,20,40])
    a,_=optimal_ion_spectra(q,r)
    b,_=optimal_ion_spectra(q,r)
    assert a==b
