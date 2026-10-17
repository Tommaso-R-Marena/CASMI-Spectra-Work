"""Adversarial and randomized tests of exact support-factorization predictions."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from qfd.power_geometry import power_mean_distribution
from qfd.persistent_survival import persistent_hellinger_survival
from qfd.support_factorization import (
    unnormalized_components,factorized_distribution,
    fixed_mechanism_predictions,reference_support_diagnostics)
from qfd.target_blind_geometry import support_only_suppression
from audit_phenolicsdb_support_mechanism import exact_block_signflip


def test_1000_randomized_exact_factorization_identities():
    rng=np.random.default_rng(20261017)
    for _ in range(1000):
        x=rng.dirichlet(np.ones(10))
        y=rng.dirichlet(np.ones(10))
        x[rng.random(10)<.35]=0
        y[rng.random(10)<.35]=0
        if x.sum()==0 or y.sum()==0:
            continue
        t=float(rng.uniform(0,1))
        for alpha in (.125,.25,.5,1.):
            actual=factorized_distribution(
                x,y,t,shared_alpha=alpha,exclusive_power=1/alpha)
            expected=power_mean_distribution(x,y,t,alpha)
            assert np.allclose(actual,expected,rtol=1e-11,atol=1e-11)
        preds=fixed_mechanism_predictions(x,y,t)
        assert np.allclose(preds["quarter_factored"],
                           power_mean_distribution(x,y,t,.25),atol=1e-11)
        assert np.allclose(preds["shared_linear_exclusive_fourth"],
                           support_only_suppression(x,y,t,power=4),atol=1e-11)
        assert np.allclose(preds["shared_hellinger_exclusive_fourth"],
                           persistent_hellinger_survival(x,y,t),atol=1e-11)


@pytest.mark.parametrize("t",[0,1,.1,.5,.9])
def test_disjoint_support_all_shared_variants_identical(t):
    a=np.array([1,3,0,0,0],float)
    b=np.array([0,0,2,3,1],float)
    p=fixed_mechanism_predictions(a,b,t)
    assert np.allclose(p["quarter_factored"],p["shared_linear_exclusive_fourth"])
    assert np.allclose(p["quarter_factored"],p["shared_hellinger_exclusive_fourth"])
    diag=reference_support_diagnostics(a,b)
    assert diag["shared_bin_count"]==0


def test_exclusive_coefficients_exact_before_normalization():
    a=np.array([.3,.7,0.,0.])
    b=np.array([.2,0.,.8,0.])
    t=.3
    s,l,r=unnormalized_components(a,b,t,shared_alpha=.25,exclusive_power=4)
    assert np.isclose(s[0],((1-t)*a[0]**.25+t*b[0]**.25)**4)
    assert np.isclose(l[1],(1-t)**4*a[1])
    assert np.isclose(r[2],t**4*b[2])
    assert np.all(s[[1,2,3]]==0)
    assert np.all(l[[0,2,3]]==0)
    assert np.all(r[[0,1,3]]==0)


def test_symmetry_and_interpolation_endpoints():
    a=np.array([.5,.3,.2,0])
    b=np.array([.2,.2,0,.6])
    for t in (0,.2,.51,1):
        for alpha in (.25,.5,1.):
            p=factorized_distribution(a,b,t,shared_alpha=alpha,exclusive_power=4.)
            reverse=factorized_distribution(b,a,1-t,shared_alpha=alpha,exclusive_power=4.)
            assert np.allclose(p,reverse,atol=1e-12)
            assert np.isclose(p.sum(),1.)
    assert np.allclose(factorized_distribution(a,b,0,shared_alpha=.25,exclusive_power=4.),a)
    assert np.allclose(factorized_distribution(a,b,1,shared_alpha=.25,exclusive_power=4.),b)


@pytest.mark.parametrize("a,b,t,alpha,power",[
    ([0,0],[1,0],.3,.5,4),
    ([1,float("nan")],[0,1],.5,.5,4),
    ([-.1,1.1],[1,0],.5,.5,4),
    ([1,0],[0,1],float("nan"),.5,4),
    ([1,0],[0,1],.5,0,4),
    ([1,0],[0,1],.5,.5,0),
    ([1,0],[0,1],1.1,.5,4),
])
def test_invalid_mass_and_parameter_fail_closed(a,b,t,alpha,power):
    with pytest.raises(ValueError):
        factorized_distribution(a,b,t,shared_alpha=alpha,exclusive_power=power)


def test_exact_randomization_all_sign_assignments_and_clustering():
    x=[.12,.12,-.01,-.01]
    groups=["A","A","B","B"]
    stat=exact_block_signflip(x,groups)
    assert stat["exact_assignments"]==4
    assert stat["distinct_structure_blocks"]==2
    assert stat["extreme_assignments"]==2
    assert stat["p_two_sided"]==.5


def test_exact_signflip_all_zero_gives_p_one():
    stat=exact_block_signflip([0,0,0],["A","A","B"])
    assert stat["exact_assignments"]==4
    assert stat["p_two_sided"]==1.0


def test_invalid_exact_signflip_fails_closed():
    with pytest.raises(ValueError):
        exact_block_signflip([float("nan")],["A"])
    with pytest.raises(ValueError):
        exact_block_signflip([.2],["A","B"])
