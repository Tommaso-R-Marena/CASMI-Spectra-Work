from __future__ import annotations

from typing import Iterable

import numpy as np

from qfd.consensus_ranker import protected_tail_consensus_scores, topk_overlap_fraction


def _aligned_finite_scores(
    static_scores: Iterable[float], expert_scores: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    static = np.asarray(list(static_scores), dtype=float)
    expert = np.asarray(list(expert_scores), dtype=float)
    if static.shape != expert.shape or static.ndim != 1 or static.size == 0:
        raise ValueError("aligned non-empty score vectors required")
    if not np.all(np.isfinite(static)) or not np.all(np.isfinite(expert)):
        raise ValueError("finite score vectors required")
    return static, expert


def static_protected_head_survives_expert(
    static_scores: Iterable[float],
    expert_scores: Iterable[float],
    *,
    protected_k: int = 1,
    slack: int = 1,
) -> tuple[bool, np.ndarray, int, int]:
    """Check whether the full tie-closed static head survives under the expert.

    The static top-``protected_k`` prefix is closed under ties exactly as in the
    v5 protected-tail rule.  For each protected candidate, its expert
    ``rank_max`` is the last rank occupied by that candidate's complete expert
    tie block, equivalently ``count(expert >= expert[i])``.  The head survives
    only when every protected candidate has ``rank_max <= head_size + slack``.

    This guard is label-free and invariant to candidate row permutation.  Using
    tie-block maximum rank rather than an arbitrary stable-sort position makes
    the decision conservative in the presence of exact expert ties.

    Returns ``(survives, protected_mask, head_size, max_protected_expert_rank)``.
    """
    static, expert = _aligned_finite_scores(static_scores, expert_scores)
    pk = int(protected_k)
    sl = int(slack)
    if pk < 1:
        raise ValueError("protected_k must be positive")
    if sl < 0:
        raise ValueError("slack must be non-negative")

    static_order = np.argsort(-static, kind="stable")
    cutoff = static[static_order[min(pk, int(static.size)) - 1]]
    protected = static >= cutoff
    head_size = int(np.sum(protected))

    protected_indices = np.flatnonzero(protected)
    rank_max = np.asarray(
        [int(np.sum(expert >= expert[i])) for i in protected_indices], dtype=int
    )
    max_rank = int(np.max(rank_max))
    survives = bool(max_rank <= head_size + sl)
    return survives, protected, head_size, max_rank


def protected_tail_v52_scores(
    static_scores: Iterable[float],
    expert_scores: Iterable[float],
    *,
    overlap_threshold: float = 0.8,
    protected_k: int = 1,
    overlap_k: int = 10,
    head_survival_slack: int = 1,
) -> tuple[np.ndarray, float, bool, np.ndarray, bool, int]:
    """Frozen v5.2: overlap gate + protected-head survival + expert-only tail.

    The expert may rerank only when both label-free conditions hold:

    1. tie-aware static/expert top-``overlap_k`` overlap is at least
       ``overlap_threshold``;
    2. the complete tie-closed static protected head survives within
       ``head_size + head_survival_slack`` under the expert, measured with
       expert tie-block maximum ranks.

    If either guard fails, the static scores are returned unchanged.  If both
    pass, the v5 protected-tail construction is used, so the protected static
    prefix remains exactly above the expert-reranked tail.  Consequently
    ``protected_k=1`` retains the v5 theorem of exact tie-aware Top-1
    invariance for every possible truth label.

    Returns ``(scores, overlap, used_expert_tail, protected_mask,
    head_survives, max_protected_expert_rank)``.
    """
    static, expert = _aligned_finite_scores(static_scores, expert_scores)
    overlap = topk_overlap_fraction(static, expert, k=overlap_k)
    head_survives, protected, _, max_rank = static_protected_head_survives_expert(
        static,
        expert,
        protected_k=protected_k,
        slack=head_survival_slack,
    )
    use_expert = bool(overlap >= float(overlap_threshold) and head_survives)
    if not use_expert:
        return static.copy(), float(overlap), False, protected, head_survives, max_rank

    chosen, overlap_check, used, protected_check = protected_tail_consensus_scores(
        static,
        expert,
        overlap_threshold=0.0,
        protected_k=protected_k,
        overlap_k=overlap_k,
    )
    if not used or not np.array_equal(protected, protected_check):
        raise RuntimeError("v5.2/v5 protected-head construction mismatch")
    if not np.isclose(overlap, overlap_check, rtol=0.0, atol=1e-15):
        raise RuntimeError("v5.2/v5 overlap calculation mismatch")
    return chosen, float(overlap), True, protected, True, max_rank
