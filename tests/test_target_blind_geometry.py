from types import SimpleNamespace
import numpy as np
import pytest

from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, support_only_suppression,
    target_blind_scores, affinity, best_convex_oracle,
)
from qfd.power_geometry import power_mean_distribution


def spec(mzs, intensities):
    return SimpleNamespace(mzs=np.asarray(mzs, dtype=float),
                           intensities=np.asarray(intensities, dtype=float))


def test_query_cannot_change_reference_bins_or_predictions():
    lo=spec([100.000, 130.000], [80,20])
    hi=spec([100.020, 150.000], [60,40])
    before=reference_only_grid(lo, hi)
    # A query peak at 100.009 previously affected a three-way strict grid.
    qa=spec([100.009, 160.000],[100,10])
    qb=spec([100.019, 170.000],[100,10])
    query_mass_on_reference_grid(qa,before)
    query_mass_on_reference_grid(qb,before)
    after=reference_only_grid(lo,hi)
    assert np.array_equal(before.mz,after.mz)
    assert np.array_equal(before.low,after.low)
    assert np.array_equal(before.high,after.high)
    assert len(before.mz)==4


def test_peaks_are_conserved_even_when_reference_masses_cluster():
    lo=spec([100.000,100.006,140.0],[3,7,10])
    hi=spec([100.004,150.0],[6,4])
    grid=reference_only_grid(lo,hi)
    assert np.isclose(grid.low.sum(),1)
    assert np.isclose(grid.high.sum(),1)
    assert np.isclose(grid.low[0],0.5)
    assert np.isclose(grid.high[0],0.6)
    assert grid.mz[0]>=100.0 and grid.mz[0]<=100.006


def test_query_is_normalized_with_unmatched_mass():
    grid=reference_only_grid(spec([100],[1]),spec([100],[1]))
    q,unmatched=query_mass_on_reference_grid(spec([100.005,200],[1,3]),grid)
    assert np.isclose(q[0],.25)
    assert np.isclose(unmatched,.75)
    assert np.isclose(affinity(q,grid.low),.5)


def test_one_to_one_matching_prevents_duplicating_prediction_mass():
    grid=reference_only_grid(spec([100],[1]),spec([100],[1]))
    q,unmatched=query_mass_on_reference_grid(spec([99.998,100.002],[3,1]),grid)
    assert np.isclose(q.sum(),.75)
    assert np.isclose(unmatched,.25)


def test_support_only_distinguishes_power_curvature_with_shared_peaks():
    a=np.array([.7,.2,.1,0.])
    b=np.array([.2,.3,0.,.5])
    q=support_only_suppression(a,b,.35,power=4)
    target=power_mean_distribution(a,b,.35,.25)
    assert np.isclose(q.sum(),1)
    assert np.any(abs(q-target)>1e-5)


def test_disjoint_support_exactly_equals_warped_probability_mixture():
    a=np.array([.7,.3,0.,0.])
    b=np.array([0.,0.,.4,.6])
    t=.3
    power=4
    q=support_only_suppression(a,b,t,power=power)
    w=t**power/(t**power+(1-t)**power)
    assert np.allclose(q,(1-w)*a+w*b)
    assert np.allclose(q,power_mean_distribution(a,b,t,.25))


def test_oracle_is_at_least_every_convex_weight():
    q=np.array([.2,.5,.3])
    a=np.array([.7,.2,.1])
    b=np.array([.1,.3,.6])
    z=best_convex_oracle(q,a,b)
    assert z >= max(affinity(q,(1-w)*a+w*b) for w in np.linspace(0,1,31))-1e-8


def test_end_to_end_all_methods_in_bounds():
    q=spec([100,120,150],[4,3,5])
    lo=spec([100,125,140],[3,7,2])
    hi=spec([100,120,150],[5,1,6])
    scores=target_blind_scores(q,lo,hi,.35)
    for key,val in scores.items():
        if key in {"reference_bins","reference_shared_bin_count"}:
            continue
        assert np.isfinite(val)
        assert -1e-10<=val<=1+1e-10, (key,val)
    assert scores["reference_bins"]>=3


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        reference_only_grid(spec([100],[1]),spec([100],[1]),tolerance_da=0)
    with pytest.raises(ValueError):
        target_blind_scores(spec([100],[1]),spec([100],[1]),spec([100],[1]),0)
