from qfd.metrics import rank_by_score


def test_rank_by_score_preserves_input_order_for_descending_ties():
    ids = ["first", "second", "third"]
    assert rank_by_score(ids, [1.0, 1.0, 0.0]) == ["first", "second", "third"]


def test_rank_by_score_preserves_input_order_for_ascending_ties():
    ids = ["first", "second", "third"]
    assert rank_by_score(ids, [1.0, 1.0, 0.0], higher_is_better=False) == ["third", "first", "second"]
