import numpy as np

from qfd.protected_tail_v52 import (
    protected_tail_v52_scores,
    static_protected_head_survives_expert,
)


def _tie_topk_credit(scores: np.ndarray, truth_index: int, k: int) -> float:
    truth = float(scores[truth_index])
    better = int(np.sum(scores > truth))
    tied = int(np.sum(scores == truth))
    ranks = np.arange(better + 1, better + tied + 1)
    return float(np.mean(ranks <= k))


def test_head_survival_accepts_full_tie_closed_head_with_slack_one():
    static = np.array([9.0, 9.0, 7.0, 6.0, 5.0])
    expert = np.array([8.0, 7.0, 9.0, 6.0, 5.0])
    survives, protected, head_size, max_rank = static_protected_head_survives_expert(
        static, expert, protected_k=1, slack=1
    )
    assert survives
    assert head_size == 2
    assert max_rank == 3
    assert np.array_equal(protected, np.array([True, True, False, False, False]))


def test_head_survival_rejects_when_any_protected_tie_block_falls_too_low():
    static = np.array([9.0, 9.0, 7.0, 6.0, 5.0, 4.0])
    # Candidate 1 is in the static max tie block but its expert score occupies
    # rank 5, below head_size + slack = 3.
    expert = np.array([8.0, 6.0, 10.0, 9.0, 7.0, 5.0])
    survives, protected, head_size, max_rank = static_protected_head_survives_expert(
        static, expert, protected_k=1, slack=1
    )
    assert not survives
    assert head_size == 2
    assert max_rank == 5
    assert np.array_equal(protected, np.array([True, True, False, False, False, False]))


def test_head_survival_uses_expert_tie_block_max_rank_conservatively():
    static = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    # The protected candidate is tied with two decoys for expert ranks 1..3.
    # Stable row order must not be allowed to make that look like expert rank 1.
    expert = np.array([10.0, 10.0, 10.0, 4.0, 3.0])
    survives, _, head_size, max_rank = static_protected_head_survives_expert(
        static, expert, protected_k=1, slack=1
    )
    assert head_size == 1
    assert max_rank == 3
    assert not survives


def test_v52_falls_back_exactly_to_static_when_head_guard_fails():
    static = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    expert = np.array([1.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    chosen, _, used, protected, survives, max_rank = protected_tail_v52_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=3,
        head_survival_slack=1,
    )
    assert not survives
    assert max_rank == 6
    assert not used
    assert np.array_equal(chosen, static)
    assert np.array_equal(protected, np.array([True, False, False, False, False, False]))


def test_v52_reranks_tail_when_both_label_free_guards_pass():
    static = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    expert = np.array([8.0, 9.0, 4.0, 7.0, 6.0, 5.0])
    chosen, overlap, used, protected, survives, max_rank = protected_tail_v52_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=1,
        overlap_k=3,
        head_survival_slack=1,
    )
    assert survives
    assert max_rank == 2
    assert used
    assert 0.0 <= overlap <= 1.0
    assert np.array_equal(protected, np.array([True, False, False, False, False, False]))
    assert int(np.argmax(chosen)) == 0
    # Candidate 1 leads the tail under the expert, then 3,4,5,2.
    assert list(np.argsort(-chosen, kind="stable")) == [0, 1, 3, 4, 5, 2]


def test_v52_preserves_declared_tie_aware_topk_for_every_possible_truth():
    static = np.array([9.0, 8.0, 8.0, 8.0, 7.0, 6.0, 5.0])
    expert = np.array([9.0, 8.0, 7.0, 6.0, 100.0, 90.0, 80.0])
    for k in (1, 2, 3, 4, 5):
        chosen, _, _, _, _, _ = protected_tail_v52_scores(
            static,
            expert,
            overlap_threshold=0.0,
            protected_k=k,
            overlap_k=3,
            head_survival_slack=100,
        )
        for truth_index in range(static.size):
            assert _tie_topk_credit(static, truth_index, k) == _tie_topk_credit(
                chosen, truth_index, k
            )


def test_v52_decision_and_scores_are_candidate_permutation_equivariant():
    static = np.array([9.0, 9.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    expert = np.array([8.0, 8.0, 9.0, 7.0, 7.0, 6.0, 5.0, 4.0])
    baseline = protected_tail_v52_scores(
        static,
        expert,
        overlap_threshold=0.55,
        protected_k=1,
        overlap_k=5,
        head_survival_slack=1,
    )
    base_scores, base_overlap, base_used, base_protected, base_survives, base_max_rank = baseline

    rng = np.random.default_rng(20260921)
    for _ in range(200):
        perm = rng.permutation(static.size)
        inv = np.argsort(perm)
        result = protected_tail_v52_scores(
            static[perm],
            expert[perm],
            overlap_threshold=0.55,
            protected_k=1,
            overlap_k=5,
            head_survival_slack=1,
        )
        scores, overlap, used, protected, survives, max_rank = result
        assert np.isclose(overlap, base_overlap, rtol=0.0, atol=1e-15)
        assert used == base_used
        assert survives == base_survives
        assert max_rank == base_max_rank
        assert np.array_equal(protected[inv], base_protected)
        assert np.array_equal(scores[inv], base_scores)


def test_v52_overlap_gate_is_permutation_equivariant_at_cutoff_ties():
    # Both score vectors have cutoff ties crossing top-3. The gate must be
    # independent of arbitrary candidate row order. The overlap itself is a
    # floating expectation and may differ at the final machine bit because
    # permutation changes summation order, so compare it at machine precision.
    static = np.array([10.0, 9.0, 8.0, 8.0, 8.0, 1.0])
    expert = np.array([10.0, 8.0, 9.0, 8.0, 8.0, 1.0])
    baseline = protected_tail_v52_scores(
        static,
        expert,
        overlap_threshold=0.7,
        protected_k=1,
        overlap_k=3,
        head_survival_slack=10,
    )
    rng = np.random.default_rng(53)
    for _ in range(100):
        perm = rng.permutation(static.size)
        inv = np.argsort(perm)
        result = protected_tail_v52_scores(
            static[perm],
            expert[perm],
            overlap_threshold=0.7,
            protected_k=1,
            overlap_k=3,
            head_survival_slack=10,
        )
        assert np.isclose(result[1], baseline[1], rtol=0.0, atol=1e-15)
        assert result[2] == baseline[2]
        assert np.array_equal(result[3][inv], baseline[3])
        assert np.array_equal(result[0][inv], baseline[0])
