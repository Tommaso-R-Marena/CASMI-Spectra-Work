"""Falsification-first reference-only spectral bound and ranking guarantees."""
import numpy as np
import pytest

from qfd.power_geometry import power_mean_distribution
from qfd.support_certificates import (
    disjoint_support_energy_reparameterization,
    score_perturbation_certificate,certify_top1_unchanged)
from qfd.support_factorization import factorized_distribution
from qfd.target_blind_geometry import affinity


def test_disjoint_power_geometries_exactly_nonidentifiable_under_energy_warp():
    rng=np.random.default_rng(20261017)
    for _ in range(600):
        a=np.zeros(12);b=np.zeros(12)
        a[:6]=rng.dirichlet(np.ones(6))
        b[6:]=rng.dirichlet(np.ones(6))
        t=float(rng.uniform(.005,.995))
        for alpha in (.125,.25,.5,1.):
            for beta in (.125,.25,.5,1.):
                s=disjoint_support_energy_reparameterization(t,alpha,beta)
                assert 0<=s<=1
                assert np.allclose(
                    power_mean_distribution(a,b,t,alpha),
                    power_mean_distribution(a,b,s,beta),
                    rtol=1e-11,atol=1e-11)


def test_energy_endpoints_and_reverse_involution():
    for alpha,beta in ((.25,.5),(.5,1.),(.125,.3)):
        assert disjoint_support_energy_reparameterization(0,alpha,beta)==0
        assert disjoint_support_energy_reparameterization(1,alpha,beta)==1
        for t in (.01,.2,.5,.8,.99):
            s=disjoint_support_energy_reparameterization(t,alpha,beta)
            inv=disjoint_support_energy_reparameterization(s,beta,alpha)
            assert inv==pytest.approx(t,abs=1e-12)


def test_random_query_affinity_differences_respect_proved_bound():
    rng=np.random.default_rng(9)
    for _ in range(1000):
        a=rng.random(16);b=rng.random(16)
        a[rng.random(16)<.4]=0
        b[rng.random(16)<.4]=0
        if a.sum()<=0 or b.sum()<=0:continue
        t=float(rng.uniform(.02,.98))
        result=score_perturbation_certificate(
            a,b,t,from_shared_alpha=.5,to_shared_alpha=.25)
        p=factorized_distribution(a,b,t,shared_alpha=.5,exclusive_power=4)
        s=factorized_distribution(a,b,t,shared_alpha=.25,exclusive_power=4)
        obs=rng.random(16)
        # Query may have unmatched peak mass and sums <=1.
        obs=obs/obs.sum()*float(rng.uniform(.05,1.))
        actual=abs(affinity(obs,p)-affinity(obs,s))
        assert actual<=result["affinity_uniform_upper"]+1e-10
        assert result["tv_actual"]<=result["tv_proved_upper"]+1e-10


def test_zero_shared_support_has_zero_score_change_certificate():
    x=np.array([.8,.2,0,0])
    y=np.array([0,0,.25,.75])
    cert=score_perturbation_certificate(
        x,y,.33,from_shared_alpha=.5,to_shared_alpha=.25)
    assert cert["shared_reference_bin_count"]==0
    assert cert["shared_unnormalized_l1_change"]==0
    assert cert["affinity_uniform_upper"]==0


def test_safe_top1_guarantee_cannot_be_wrong_under_bounded_perturbations():
    scores={"a":.9,"b":.5,"c":.3}
    eps={"a":.05,"b":.05,"c":.15}
    cert=certify_top1_unchanged(scores,eps)
    assert cert["certified_unchanged"]
    rng=np.random.default_rng(8)
    for _ in range(1000):
        perturbed={k:min(1.,max(0.,scores[k]+rng.uniform(-eps[k],eps[k])))
                   for k in scores}
        assert max(perturbed,key=perturbed.get)=="a"


def test_close_or_tied_candidates_annotated_unknown_not_false_claim():
    res=certify_top1_unchanged({"a":.5,"b":.49},{"a":.03,"b":.03})
    assert res["status"]=="UNKNOWN_NOT_PROVED"
    assert not res["certified_unchanged"]
    tied=certify_top1_unchanged({"a":.5,"b":.5},{"a":0.,"b":0.})
    assert not tied["certified_unchanged"]


@pytest.mark.parametrize("t,a,b",[(float("nan"),.25,.5),(.5,0.,.5),
                                  (.5,.25,float("nan")),(1.01,.25,.5)])
def test_invalid_energy_exponent_refused(t,a,b):
    with pytest.raises(ValueError):
        disjoint_support_energy_reparameterization(t,a,b)


def test_invalid_candidate_bound_inventory_fails_closed():
    with pytest.raises(ValueError):
        certify_top1_unchanged({"a":.5,"b":.4},{"a":0.01})
    with pytest.raises(ValueError):
        certify_top1_unchanged({"a":.5,"b":.4},{"a":float("nan"),"b":.02})
