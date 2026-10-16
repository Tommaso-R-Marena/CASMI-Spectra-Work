"""Exhaustive small-case proof checks for sharp source-overlap bounds."""
import importlib.util
from itertools import combinations
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/"scripts"/"bound_massbank_overlap_effect.py"
spec=importlib.util.spec_from_file_location("bound_massbank_overlap_effect",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_extremal_means_match_exhaustive_all_overlap_allocations():
    values=[-3.,-1.,0.,.3,4.,8.]
    for removed in range(len(values)):
        result=m.exact_bounds(values,removed)
        kept_means=[
            sum(values[i] for i in range(len(values)) if i not in removed_set)/(len(values)-removed)
            for removed_set in map(set,combinations(range(len(values)),removed))
        ]
        assert abs(result["lower_possible_mean"]-min(kept_means))<1e-12
        assert abs(result["upper_possible_mean"]-max(kept_means))<1e-12

def test_guaranteed_wins_survive_arbitrary_exclusion():
    values=[.1,.2,.3,.4,-.2,-.1]
    b=m.exact_bounds(values,2)
    assert b["minimum_remaining_wins"]==2
    assert b["minimum_remaining_win_fraction"]==.5

def test_invalid_exclusion_or_nan_fails_closed():
    for n in (-1,3,1.5):
        with pytest.raises(ValueError,match="excluded"):
            m.exact_bounds([1,2,3],n)
    with pytest.raises(ValueError,match="invalid affinity"):
        m.exact_bounds([float("nan"),2,3],1)
