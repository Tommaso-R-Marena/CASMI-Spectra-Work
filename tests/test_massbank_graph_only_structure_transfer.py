"""MassBank graph-only isomer ranking exact null and no-spectrum invariants."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_massbank_graph_only_structure_transfer import (
    exact_label_null,tieaware,summarize)


def test_exact_random_label_expected_mrr_and_ties():
    assert exact_label_null({"A":1.,"B":0.})==pytest.approx(.75)
    assert exact_label_null({"A":0.,"B":0.})==pytest.approx(2/3)
    assert exact_label_null({"A":.8,"B":.8,"C":.1})==pytest.approx(
        ((2/3)+(2/3)+(1/3))/3)


def test_scored_molecule_competitor_tie_policy_same_as_original():
    scores={"A":.8,"B":.8,"C":.1}
    rr,top1=tieaware(scores,"A")
    assert rr==pytest.approx(2/3)
    assert top1==pytest.approx(.5)
    rr,top1=tieaware(scores,"C")
    assert rr==pytest.approx(1/3)
    assert top1==0.


def test_averaging_resamples_native_formula_not_each_acquisition():
    def row(i,formula,graph,null):
        return {"inchikey":str(i),"formula":formula,
            "graph_mrr":graph,"exact_random_label_null_mrr":null,
            "graph_top1_fractional":.5,
            "uniform_label_top1_fractional":.5}
    stat=summarize([
        row(1,"C2H6",.8,.7),row(2,"C2H6",.8,.7),
        row(3,"C3H8",.5,.6),row(4,"C4H10",.6,.6),
    ])
    assert stat["n_queries"]==4
    assert stat["n_formulas"]==3
    assert stat["formula_block_bootstrap_95"]["n_blocks"]==3
    assert stat["formula_block_bootstrap_95"]["delta"]==pytest.approx(.025)


def test_malformed_scores_reject_finite_null():
    with pytest.raises(ValueError):
        exact_label_null({"A":1.})
    with pytest.raises(ValueError):
        exact_label_null({"A":float("nan"),"B":.2})
    with pytest.raises(ValueError):
        tieaware({"A":.1,"B":.2},"C")
