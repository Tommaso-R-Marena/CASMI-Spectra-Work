"""Stress an independent, analytic oracle against numerical/dense optimization."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from qfd.hellinger_analytic_oracle import analytic_hellinger_chord_oracle


def chord_affinity(q, p0, p1, w):
    a, b = np.sqrt(p0/p0.sum()), np.sqrt(p1/p1.sum())
    v = (1-w)*a+w*b
    return float(np.dot(np.sqrt(q), v)/np.linalg.norm(v))


def test_exact_on_hellinger_chord_and_at_endpoints():
    p0=np.array([.5,.3,.2,0])
    p1=np.array([.1,.5,.1,.3])
    a, b = np.sqrt(p0), np.sqrt(p1)
    for w in (0., .03, .28, .5, .91, 1.):
        v=(1-w)*a+w*b
        q=v*v/float(np.dot(v,v))
        result=analytic_hellinger_chord_oracle(q,p0,p1)
        assert result["affinity"] == pytest.approx(1.0,abs=1e-12)
        if 0<w<1:
            assert result["oracle_weight"] == pytest.approx(w,abs=1e-10)


def test_global_oracle_dominates_numerical_and_dense_search():
    rng=np.random.default_rng(20261016)
    for iteration in range(500):
        n=int(rng.integers(2,45))
        p0, p1, q = rng.random((3,n))
        p0[rng.random(n)<.25]=0
        p1[rng.random(n)<.25]=0
        q[rng.random(n)<.3]=0
        if not p0.any() or not p1.any():
            continue
        # Keep unmatched mass outside the reference bins in the denominator.
        q=q/(q.sum() or 1)*float(rng.uniform(.02,1))
        if iteration%10==0:
            p1=p0.copy()
        oracle=analytic_hellinger_chord_oracle(q,p0,p1)
        dense=max(chord_affinity(q,p0,p1,float(w))
                  for w in np.linspace(0,1,201))
        fit=minimize_scalar(lambda w:-chord_affinity(q,p0,p1,w),
                            bounds=(0,1),method="bounded")
        numeric=max(chord_affinity(q,p0,p1,0),
                    chord_affinity(q,p0,p1,1),-float(fit.fun))
        assert oracle["affinity"] >= dense-2e-12
        assert oracle["affinity"] >= numeric-2e-12
        assert 0<=oracle["oracle_weight"]<=1


@pytest.mark.parametrize("q,p0,p1", [
    ([1.,0.],[0.,0.],[0.,1.]),
    ([1.,0.],[0.,1.],[float("nan"),1.]),
    ([-1.,0.],[0.,1.],[1.,0.]),
    ([1.,0.],[0.,1.,1.],[1.,0.]),
])
def test_fail_closed_on_invalid_input(q,p0,p1):
    with pytest.raises(ValueError):
        analytic_hellinger_chord_oracle(q,p0,p1)
