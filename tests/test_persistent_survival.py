import numpy as np
import pytest
from qfd.persistent_survival import persistent_hellinger_survival as hybrid
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import support_only_suppression

def test_exact_endpoints():
    a=np.array([.4,.6,0.]); b=np.array([.2,.3,.5])
    np.testing.assert_allclose(hybrid(a,b,0),a,atol=1e-15)
    np.testing.assert_allclose(hybrid(a,b,1),b,atol=1e-15)

def test_energy_reversal_and_normalization():
    rng=np.random.default_rng(1453)
    for _ in range(100):
        a=rng.exponential(size=30); b=rng.exponential(size=30)
        a[rng.random(30)<.5]=0; b[rng.random(30)<.5]=0
        if not a.any() or not b.any(): continue
        t=float(rng.uniform(0,1))
        p=hybrid(a,b,t)
        np.testing.assert_allclose(p,hybrid(b,a,1-t),atol=3e-15)
        assert (p>=0).all() and abs(p.sum()-1)<1e-12

def test_disjoint_reduces_exactly_to_quarter_power():
    a=np.array([.3,.7,0.,0.]); b=np.array([0.,0.,.1,.9])
    for t in (0,.15,1/3,.5,.95,1):
        np.testing.assert_allclose(hybrid(a,b,t),power_mean_distribution(a,b,t,.25),atol=1e-14)

def test_shared_equal_endpoints_stay_fixed():
    p=np.array([.1,.2,.7])
    for t in (0,.1,1/3,.5,.9,1):
        np.testing.assert_allclose(hybrid(p,p,t),p,atol=2e-15)

def test_is_not_relabelled_baseline():
    a=np.array([.4,.2,.4,0.]); b=np.array([.1,.4,0.,.5])
    h=hybrid(a,b,1/3)
    assert not np.allclose(h,power_mean_distribution(a,b,1/3,.25))
    assert not np.allclose(h,support_only_suppression(a,b,1/3,power=4))

@pytest.mark.parametrize("a,b,t",[
    ([0,0],[1,0],.5),([1,0],[0,0],.5),
    ([-1,2],[1,0],.5),([1,0],[0,1],-0.1),
    ([1,0],[0,1],1.1),([1,0],[0,1],float("nan")),
    ([1,0],[0,1,0],.5),
])
def test_fail_closed(a,b,t):
    with pytest.raises(ValueError):
        hybrid(a,b,t)
