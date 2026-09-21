import numpy as np

from qfd.consensus_ranker import (
    CONSENSUS_FEATURE_NAMES,
    conditional_softmax_fit,
    consensus_feature_matrix,
    deterministic_fold,
    protected_tail_consensus_scores,
    selective_consensus_scores,
    topk_overlap_fraction,
)


def test_consensus_features_reward_cross_energy_consistency():
    X = consensus_feature_matrix(
        [0.9, 0.7, 0.4],
        [0.9, 0.7, 0.4],
        [0.9, 0.7, 0.4],
        [-0.1, -0.2, -0.3],
    )
    assert X.shape == (3, len(CONSENSUS_FEATURE_NAMES))
    assert np.all(np.isfinite(X))
    # Identical energy scores have zero spread, hence a constant third feature.
    assert np.allclose(X[:, 2], 0.0)


def test_topk_overlap_and_selective_fallback():
    static = np.array([4.0, 3.0, 2.0, 1.0])
    expert = np.array([4.0, 2.0, 3.0, 1.0])
    assert topk_overlap_fraction(static, expert, k=2) == 0.5
    chosen, overlap, used = selective_consensus_scores(
        static, expert, overlap_threshold=0.6, k=2
    )
    assert overlap == 0.5
    assert not used
    assert np.array_equal(chosen, static)
    chosen, overlap, used = selective_consensus_scores(
        static, expert, overlap_threshold=0.5, k=2
    )
    assert used
    assert np.array_equal(chosen, expert)


def test_protected_tail_freezes_static_max_tie_block_and_reranks_tail():
    static = np.array([5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    # The expert strongly dislikes the first static winner and strongly favors
    # lower static candidates. The protected maximum tie block must still stay
    # tied at the top while the tail follows expert order.
    expert = np.array([-10.0, -20.0, 0.0, 3.0, 2.0, 1.0])
    chosen, overlap, used, protected = protected_tail_consensus_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=3,
    )
    assert used
    assert 0.0 <= overlap <= 1.0
    assert np.array_equal(protected, np.array([True, True, False, False, False, False]))
    assert chosen[0] == chosen[1]
    assert chosen[0] > np.max(chosen[~protected])
    # Tail order is candidate 3, 4, 5, 2 under the expert.
    assert list(np.argsort(-chosen, kind="stable")) == [0, 1, 3, 4, 5, 2]


def test_protected_tail_includes_all_ties_at_cutoff():
    static = np.array([9.0, 8.0, 8.0, 8.0, 7.0, 6.0])
    expert = np.array([0.0, -1.0, -2.0, -3.0, 10.0, 9.0])
    chosen, _, used, protected = protected_tail_consensus_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=2,
        overlap_k=3,
    )
    assert used
    assert np.array_equal(protected, np.array([True, True, True, True, False, False]))
    assert chosen[1] == chosen[2] == chosen[3]
    assert np.min(chosen[protected]) > np.max(chosen[~protected])


def test_protected_tail_falls_back_to_static_when_gate_fails():
    static = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    expert = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    chosen, overlap, used, protected = protected_tail_consensus_scores(
        static,
        expert,
        overlap_threshold=1.0,
        protected_k=1,
        overlap_k=2,
    )
    assert overlap == 0.0
    assert not used
    assert np.array_equal(chosen, static)
    assert np.array_equal(protected, np.array([True, False, False, False, False, False]))


def _tie_aware_topk_credit(scores: np.ndarray, truth_index: int, k: int) -> float:
    truth = float(scores[truth_index])
    better = int(np.sum(scores > truth))
    tied = int(np.sum(scores == truth))
    ranks = np.arange(better + 1, better + tied + 1)
    return float(np.mean(ranks <= k))


def test_protected_k_preserves_tie_aware_topk_credit_for_every_truth():
    static = np.array([9.0, 8.0, 8.0, 8.0, 7.0, 6.0, 5.0])
    expert = np.array([-2.0, -3.0, -4.0, -5.0, 100.0, 90.0, 80.0])
    for k in (1, 2, 3, 4, 5):
        chosen, _, used, _ = protected_tail_consensus_scores(
            static,
            expert,
            overlap_threshold=0.0,
            protected_k=k,
            overlap_k=3,
        )
        assert used
        for truth_index in range(static.size):
            assert _tie_aware_topk_credit(static, truth_index, k) == _tie_aware_topk_credit(
                chosen, truth_index, k
            )


def test_conditional_softmax_learns_truth_direction():
    # Feature 0 is consistently larger on the true row in both queries.
    groups = [
        (np.array([[2.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]), 0),
        (np.array([[1.5, 0.0], [0.2, 0.0], [-0.5, 0.0]]), 0),
    ]
    fit = conditional_softmax_fit(groups, l2=0.1)
    assert fit.success
    assert fit.weights.shape == (2,)
    assert fit.weights[0] > 0
    assert abs(fit.weights[1]) < 1e-8


def test_deterministic_fold_is_reproducible():
    a = deterministic_fold(12345, salt="x", n_folds=5)
    b = deterministic_fold(12345, salt="x", n_folds=5)
    assert a == b
    assert 0 <= a < 5
