import numpy as np

from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    best_hellinger_chord_oracle, best_convex_oracle, affinity,
)


def test_hellinger_oracle_dominates_any_scalar_energy_reparameterization():
    p0=np.array([.68,.24,.08,0.])
    p1=np.array([.12,.27,.36,.25])
    q=np.array([.22,.30,.41,.07])
    best=best_hellinger_chord_oracle(q,p0,p1)
    values=[
        affinity(q,power_mean_distribution(p0,p1,float(t),.5))
        for t in np.linspace(0.,1.,201)
    ]
    assert best["affinity"] >= max(values) - 1e-8
    assert 0 <= best["oracle_weight"] <= 1


def test_hellinger_oracle_exact_for_target_on_hellinger_path():
    a=np.array([.5,.3,.2])
    b=np.array([.1,.5,.4])
    for t in (.12,.45,.85):
        q=power_mean_distribution(a,b,t,.5)
        x=best_hellinger_chord_oracle(q,a,b)
        assert np.isclose(x["affinity"],1.0,atol=1e-9)
        assert np.isclose(x["oracle_weight"],t,atol=1e-5)


def test_hellinger_oracle_and_convex_oracle_are_distinct_families():
    a=np.array([.72,.18,.10])
    b=np.array([.1,.35,.55])
    q=power_mean_distribution(a,b,.35,.5)
    h=best_hellinger_chord_oracle(q,a,b)
    p=best_convex_oracle(q,a,b)
    assert np.isclose(h["affinity"],1.0,atol=1e-9)
    assert h["affinity"]>=p
