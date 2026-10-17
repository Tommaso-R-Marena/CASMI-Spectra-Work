"""Clustering of MassBank graph-only gain uses native formulas rather than replicates."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_massbank_graph_stratified import grouped_stats,influence


def row(i,formula,delta,mode="POSITIVE",contributor="X"):
    return {"query_path":str(i),"formula":formula,
        "graph_mrr":.5+delta,"exact_random_label_null_mrr":.5,
        "graph_top1_fractional":.5,"uniform_label_top1_fractional":.5,
        "ion_mode":mode,"contributor":contributor}


def test_one_formula_is_one_resampling_block():
    rows=[row(1,"A",.2),row(2,"A",.2),row(3,"B",-.1),
          row(4,"C",.1)]
    result=grouped_stats(rows)
    assert result["n_queries"]==4
    assert result["n_unique_formula_blocks"]==3
    assert result["formula_cluster_bootstrap_95"]["n_blocks"]==3
    assert result["mean_delta"]==pytest.approx(.1)


def test_worst_formula_removal_and_posthoc_gains():
    rows=[row(1,"A",.2),row(2,"A",.2),row(3,"B",-.1),
          row(4,"C",.1)]
    s=influence(rows)
    assert s["all_single_formula_deletions_remain_positive"] is False
    assert s["smallest_leave_one_formula_out_delta"]==pytest.approx(0)
    deletion=s["top_favorable_formula_group_deletions_posthoc"]["1"]
    assert deletion["removed_query_records"]==2
    assert deletion["remaining_mean_delta"]==pytest.approx(0)


def test_missing_formula_or_nonpositive_independence_rejected():
    with pytest.raises(ValueError):
        influence([])
    with pytest.raises(ValueError):
        influence([row(1,"A",.1)])
