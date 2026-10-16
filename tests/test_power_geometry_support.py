"""Algebraic falsification controls for claims about power-geometry mechanisms."""
import numpy as np
from qfd.power_geometry import power_mean_distribution
from qfd.energy_geometry_controls import geometry_control_scores


def test_disjoint_support_power_path_reduces_exactly_to_warped_mixture():
    p0 = np.array([0.8, 0.2, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.25, 0.75])
    for r in (0.125, 0.25, 0.5, 1.0):
        for t in (0.1, 0.35, 0.65, 0.9):
            aa, bb = (1-t)**(1/r), t**(1/r)
            w = bb/(aa+bb)
            expected = (1-w)*p0+w*p1
            assert np.allclose(power_mean_distribution(p0,p1,t,r),expected,rtol=1e-12,atol=1e-12)


def test_endpoint_exclusive_fragments_have_power_law_unnormalized_weight():
    p0 = np.array([0.75, 0.25, 0.0])
    p1 = np.array([0.1, 0.4, 0.5])
    t = 0.3
    for r in (0.25,0.5,1.0):
        val = ((1-t)*p0[2]**r+t*p1[2]**r)**(1/r)
        assert np.isclose(val, t**(1/r)*p1[2])


def test_shared_support_can_escape_convex_probability_mixture_oracle():
    p0 = np.array([0.7, 0.2, 0.1])
    p1 = np.array([0.15, 0.3, 0.55])
    t = 0.35
    q = power_mean_distribution(p0, p1, t, 0.25)
    # The answer-aware convex-mixing oracle cannot attain this non-convex path point.
    control = geometry_control_scores(q,p0,p1,t)
    assert control.oracle_probability_mixture < 0.9999
    assert np.isclose(np.sum(np.sqrt(q*q)),1.0)
