import numpy as np

from qfd.protected_tail_v53 import (
    expected_descending_ranks,
    protected_tail_v53_scores,
)


def _tie_topk_credit(scores: np.ndarray, truth_index: int, k: int) -> float:
    truth = float(scores[truth_index])
    better = int(np.sum(scores > truth))
    tied = int(np.sum(scores == truth))
    ranks = np.arange(better + 1, better + tied + 1)
    return float(np.mean(ranks <= k))


def test_expected_descending_ranks_are_tie_aware_midranks():
    scores = np.array([9.0, 8.0, 8.0, 5.0])
    assert np.array_equal(
        expected_descending_ranks(scores),
        np.array([1.0, 2.5, 2.5, 4.0]),
    )


def test_v53_falls_back_exactly_to_static_when_head_guard_fails():
    static = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    expert = np.array([1.0, 9.0, 8.0, 7.0, 6.0])
    chosen, _, used, protected, survives, max_rank = protected_tail_v53_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=3,
        head_survival_slack=1,
    )
    assert not survives
    assert max_rank == 5
    assert not used
    assert np.array_equal(chosen, static)
    assert np.array_equal(protected, np.array([True, False, False, False, False]))


def test_v53_rejects_unilateral_expert_tail_promotion():
    static = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
    # Candidate 4 is the expert's favorite tail candidate but static ranks it
    # last in the tail. Minimax consensus therefore cannot promote it to the
    # front of the tail on expert enthusiasm alone.
    expert = np.array([10.0, 9.0, 8.0, 7.0, 100.0])
    chosen, _, used, protected, survives, _ = protected_tail_v53_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=3,
        head_survival_slack=10,
    )
    assert survives and used
    assert np.array_equal(protected, np.array([True, False, False, False, False]))
    assert chosen[1] > chosen[4]
    assert chosen[2] > chosen[4]


def test_v53_can_promote_when_both_rankings_avoid_strong_objection():
    static = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
    expert = np.array([10.0, 7.0, 9.0, 8.0, 6.0])
    chosen, _, used, _, survives, _ = protected_tail_v53_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=3,
        head_survival_slack=1,
    )
    assert survives and used
    # Candidate 2 is static-tail rank 2 and expert-tail rank 1; candidate 1 is
    # static-tail rank 1 but expert-tail rank 3. Worst-rank consensus prefers 2.
    assert chosen[2] > chosen[1]


def test_v53_preserves_tie_aware_top1_for_every_possible_truth():
    static = np.array([9.0, 9.0, 8.0, 8.0, 7.0, 6.0, 5.0])
    expert = np.array([9.0, 8.5, 9.5, 8.0, 7.0, 6.0, 5.0])
    chosen, _, used, _, survives, _ = protected_tail_v53_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=5,
        head_survival_slack=10,
    )
    assert survives and used
    for truth_index in range(static.size):
        assert _tie_topk_credit(static, truth_index, 1) == _tie_topk_credit(
            chosen, truth_index, 1
        )


def test_v53_is_candidate_permutation_equivariant_with_ties():
    static = np.array([10.0, 10.0, 9.0, 8.0, 8.0, 7.0, 6.0, 6.0])
    expert = np.array([10.0, 10.0, 8.0, 9.0, 8.0, 6.0, 7.0, 6.0])
    baseline = protected_tail_v53_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=5,
        head_survival_slack=1,
    )
    base_scores, base_overlap, base_used, base_protected, base_survives, base_max_rank = baseline

    rng = np.random.default_rng(20260921)
    for _ in range(200):
        perm = rng.permutation(static.size)
        inv = np.argsort(perm)
        result = protected_tail_v53_scores(
            static[perm],
            expert[perm],
            overlap_threshold=0.0,
            protected_k=1,
            overlap_k=5,
            head_survival_slack=1,
        )
        scores, overlap, used, protected, survives, max_rank = result
        assert overlap == base_overlap
        assert used == base_used
        assert survives == base_survives
        assert max_rank == base_max_rank
        assert np.array_equal(protected[inv], base_protected)
        assert np.array_equal(scores[inv], base_scores)
