import numpy as np

from qfd.consensus_ranker import (
    CONSENSUS_FEATURE_NAMES,
    conditional_softmax_fit,
    consensus_feature_matrix,
    deterministic_fold,
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
