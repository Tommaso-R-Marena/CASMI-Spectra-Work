import pytest

from qfd.metrics import rank_by_score, tie_aware_truth_metrics


def test_rank_by_score_preserves_input_order_for_descending_ties():
    ids = ["first", "second", "third"]
    assert rank_by_score(ids, [1.0, 1.0, 0.0]) == ["first", "second", "third"]


def test_rank_by_score_preserves_input_order_for_ascending_ties():
    ids = ["first", "second", "third"]
    assert rank_by_score(ids, [1.0, 1.0, 0.0], higher_is_better=False) == ["third", "first", "second"]


def test_tie_aware_truth_metrics_are_order_invariant():
    a = tie_aware_truth_metrics(["truth", "peer", "low"], [1.0, 1.0, 0.0], "truth")
    b = tie_aware_truth_metrics(["peer", "low", "truth"], [1.0, 0.0, 1.0], "truth")
    assert a == b
    assert a["rank_min"] == 1
    assert a["rank_max"] == 2
    assert a["tie_size"] == 2
    assert a["expected_rank"] == 1.5
    assert a["rr_at_k"] == pytest.approx(0.75)
    assert a["top1_credit"] == pytest.approx(0.5)


def test_tie_aware_truth_metrics_handle_cutoff_inside_tie():
    stats = tie_aware_truth_metrics(
        ["better", "truth", "peer1", "peer2"],
        [2.0, 1.0, 1.0, 1.0],
        "truth",
        k=2,
    )
    assert stats["rank_min"] == 2
    assert stats["rank_max"] == 4
    assert stats["rr_at_k"] == pytest.approx((0.5 + 0.0 + 0.0) / 3.0)
    assert stats["top1_credit"] == 0.0
