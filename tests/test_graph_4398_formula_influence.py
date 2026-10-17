"""Exact formula-group influence invariants for retrospective rank interpretation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_graph_4398_formula_influence import influence


def row(formula,id,d,t=0):
    return {
        "formula":formula,"identity":id,"query_index":id,
        "hellinger_mrr":.5,"graph_mrr":.5+d,
        "hellinger_top1":.5,"graph_top1":.5+t,
    }


def test_correct_exact_worst_formula_deletion():
    rows=[
        row("A",1,.1,.5),
        row("A",2,.2),
        row("B",3,-.1,-.5),
        row("C",4,.1),
    ]
    a=influence(rows)
    assert a["n_formula_clusters"]==3
    assert a["mean_mrr_delta"]==pytest.approx(.075)
    assert a["rank_wins"]==3
    assert a["rank_losses"]==1
    assert a["fractional_top1_wins"]==1
    assert a["fractional_top1_losses"]==1
    # Delete positive-heavy A: retained B+C sum=0, n=2.
    assert a["exact_leave_one_formula_out"]["min_mrr_delta"]==pytest.approx(0)
    assert not a["exact_leave_one_formula_out"]["all_leave_one_out_deltas_positive"]


def test_posthoc_deletions_cannot_renormalize_removed_queries_as_wins():
    x=influence([row("A",1,.4),row("A",2,.3),
                 row("B",3,.05),row("C",4,-.1)])
    d=x["posthoc_largest_gain_group_deletion"]["1"]
    assert d["n_queries_removed"]==2
    assert d["remaining_mrr_delta"]==pytest.approx(-.025)


def test_invariants_reject_missing_and_nonfinite():
    with pytest.raises(ValueError):
        influence([])
    with pytest.raises(ValueError):
        influence([{"formula":"A"}])
    with pytest.raises(ValueError):
        influence([row("A",1,float("nan"))])
