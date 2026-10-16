import numpy as np

from qfd.consensus_ranker import (
    protected_tail_consensus_scores,
    tie_aware_topk_overlap_fraction,
    topk_overlap_fraction,
)


def test_tie_aware_overlap_matches_ordinary_overlap_without_cutoff_ties():
    a = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    b = np.array([8.0, 6.0, 7.0, 4.0, 5.0, 3.0])
    for k in (1, 2, 3, 4, 5, 6):
        assert tie_aware_topk_overlap_fraction(a, b, k=k) == topk_overlap_fraction(a, b, k=k)


def test_tie_aware_overlap_is_candidate_order_invariant_at_cutoff_ties():
    static = np.array([10.0, 9.0, 8.0, 8.0, 8.0, 7.0, 6.0])
    expert = np.array([10.0, 1.0, 8.0, 8.0, 0.0, 9.0, 7.0])
    baseline = tie_aware_topk_overlap_fraction(static, expert, k=3)
    rng = np.random.default_rng(20260921)
    for _ in range(200):
        perm = rng.permutation(static.size)
        observed = tie_aware_topk_overlap_fraction(static[perm], expert[perm], k=3)
        assert observed == baseline


def test_protected_tail_gate_and_scores_are_permutation_equivariant_with_ties():
    static = np.array([10.0, 9.0, 8.0, 8.0, 8.0, 7.0, 6.0, 5.0])
    expert = np.array([0.0, 9.0, 8.0, 8.0, 1.0, 10.0, 7.0, 6.0])
    chosen, overlap, used, protected = protected_tail_consensus_scores(
        static,
        expert,
        overlap_threshold=0.5,
        protected_k=1,
        overlap_k=3,
    )
    rng = np.random.default_rng(17)
    for _ in range(200):
        perm = rng.permutation(static.size)
        c2, o2, u2, p2 = protected_tail_consensus_scores(
            static[perm],
            expert[perm],
            overlap_threshold=0.5,
            protected_k=1,
            overlap_k=3,
        )
        assert o2 == overlap
        assert u2 == used
        restored_scores = np.empty_like(c2)
        restored_scores[perm] = c2
        restored_protected = np.empty_like(p2)
        restored_protected[perm] = p2
        assert np.array_equal(restored_scores, chosen)
        assert np.array_equal(restored_protected, protected)
