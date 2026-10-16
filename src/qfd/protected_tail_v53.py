from __future__ import annotations

from typing import Iterable

import numpy as np

from qfd.protected_tail_v52 import _protected_tail_guard


def expected_descending_ranks(scores: Iterable[float]) -> np.ndarray:
    """Return tie-aware expected descending ranks, invariant to row ordering.

    For a score occupying a tie block from ranks ``rank_min`` through
    ``rank_max``, the returned rank is the midpoint of that block. This is the
    expected rank under uniform random tie breaking and depends only on score
    values, never candidate row order.
    """
    s = np.asarray(list(scores), dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("finite non-empty score vector required")

    # np.unique is ascending. For each unique score value, the number of
    # candidates with a strictly greater score is n - cumulative_count(value).
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    greater = int(s.size) - cumulative
    rank_min = greater + 1
    rank_max = greater + counts
    expected = 0.5 * (rank_min + rank_max)
    return expected[inverse].astype(float, copy=False)


def protected_tail_v53_scores(
    static_scores: Iterable[float],
    expert_scores: Iterable[float],
    *,
    overlap_threshold: float = 0.8,
    protected_k: int = 1,
    overlap_k: int = 10,
    head_survival_slack: int = 1,
) -> tuple[np.ndarray, float, bool, np.ndarray, bool, int]:
    """Frozen v5.3 candidate: guarded minimax consensus on the unprotected tail.

    v5.3 keeps the two label-free v5.2 query guards:

    1. tie-aware static/expert top-``overlap_k`` overlap must reach
       ``overlap_threshold``;
    2. the full tie-closed static protected head must survive within
       ``head_size + head_survival_slack`` under the expert.

    If either guard fails, or if the protected head covers the whole pool,
    static scores are returned unchanged. Otherwise the tie-closed static
    top-``protected_k`` prefix remains strictly above the tail. Tail candidates
    are ranked by the *worse* of their tie-aware expected static rank and
    expert rank:

        consensus_cost_i = max(rank_static_i, rank_expert_i).

    Lower cost is better. Thus the trajectory expert cannot promote a tail
    candidate merely because it alone ranks that candidate highly; both views
    must avoid strongly objecting. The construction is score-scale invariant,
    candidate-permutation equivariant, and retains exact ties in consensus
    cost. With ``protected_k=1`` it inherits exact tie-aware Top-1 invariance
    for every possible truth label.

    Returns ``(scores, overlap, used_consensus_tail, protected_mask,
    head_survives, max_protected_expert_rank)``.
    """
    static, expert, overlap, use_tail, protected, head_survives, max_rank = _protected_tail_guard(
        static_scores,
        expert_scores,
        overlap_threshold=overlap_threshold,
        protected_k=protected_k,
        overlap_k=overlap_k,
        head_survival_slack=head_survival_slack,
    )
    if not use_tail:
        return static.copy(), overlap, False, protected, head_survives, max_rank

    tail = ~protected
    static_rank = expected_descending_ranks(static[tail])
    expert_rank = expected_descending_ranks(expert[tail])
    consensus_cost = np.maximum(static_rank, expert_rank)

    # Rank-surrogate output. Protected static score groups occupy strictly
    # positive levels. The best tail consensus-cost group is zero and lower
    # groups are negative. This preserves all exact consensus ties and avoids
    # any dependence on the numerical scales of the two input scores.
    out = np.empty_like(static)
    protected_values = np.unique(static[protected])[::-1]
    for group_rank, value in enumerate(protected_values):
        out[protected & (static == value)] = float(len(protected_values) - group_rank + 1)

    tail_indices = np.flatnonzero(tail)
    cost_values = np.unique(consensus_cost)
    for group_rank, value in enumerate(cost_values):
        members = tail_indices[consensus_cost == value]
        out[members] = -float(group_rank)

    return out, overlap, True, protected, head_survives, max_rank
