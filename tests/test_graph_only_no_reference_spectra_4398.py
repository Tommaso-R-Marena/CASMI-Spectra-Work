"""Reference-spectrum-free exact molecule-label null regression checks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_graph_only_no_reference_spectra_4398 import (
    exact_random_label_expected_rank, formula_cluster_delta)


def test_two_candidates_exact_uniform_label_expectation():
    assert exact_random_label_expected_rank({"A":1.,"B":0.})==pytest.approx(.75)
    assert exact_random_label_expected_rank({"B":0.,"A":1.})==pytest.approx(.75)


def test_tie_aware_uniform_answer_is_not_arbitrary_winner():
    # Midrank=1.5, reciprocal=2/3, under the original metric policy.
    assert exact_random_label_expected_rank({"A":0.,"B":0.})==pytest.approx(2/3)
    assert exact_random_label_expected_rank({"A":1.,"B":1.,"C":1.})==pytest.approx(.5)


def test_nontrivial_score_ties_have_exact_expected_reciprocal_rank():
    # rank(A)=rank(B)=1.5, rank(C)=3
    v=exact_random_label_expected_rank({"A":.8,"B":.8,"C":.1})
    assert v==pytest.approx(((2/3)+(2/3)+(1/3))/3)


def test_invariant_score_scale_and_label_permutation():
    x={"A":.1,"B":.3,"C":.9,"D":.3}
    old=exact_random_label_expected_rank(x)
    assert exact_random_label_expected_rank({k:3*v+2 for k,v in x.items()})==pytest.approx(old)
    assert exact_random_label_expected_rank(dict(reversed(list(x.items()))))==pytest.approx(old)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        exact_random_label_expected_rank({"A":1.})
    with pytest.raises(ValueError):
        exact_random_label_expected_rank({"A":float("nan"),"B":1.})


def test_formula_not_query_is_bernoulli_resampling_block():
    def row(formula,a,b):
        return {"formula":formula,"graph_only_mrr":a,
                "uniform_label_null_mrr":b}
    test=formula_cluster_delta([
        row("A",.8,.7),row("A",.8,.7),
        row("B",.5,.6),row("C",.6,.6),
    ])
    assert test["n_blocks"]==3
    assert test["delta"]==pytest.approx(.025)
