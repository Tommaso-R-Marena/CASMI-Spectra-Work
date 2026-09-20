from __future__ import annotations

from collections.abc import Iterable, Sequence
import numpy as np


def reciprocal_rank_at_k(ranked: Sequence[str], truth: str, k: int = 25) -> float:
    for i, candidate in enumerate(ranked[:k], start=1):
        if candidate == truth:
            return 1.0 / i
    return 0.0


def mrr_at_k(rankings: Iterable[Sequence[str]], truths: Iterable[str], k: int = 25) -> float:
    scores = [reciprocal_rank_at_k(r, t, k=k) for r, t in zip(rankings, truths)]
    if not scores:
        raise ValueError("at least one query is required")
    return float(np.mean(scores))


def rank_by_score(candidate_ids: Sequence[str], scores: Sequence[float], *, higher_is_better: bool = True):
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores must have equal length")
    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")
    if higher_is_better:
        order = order[::-1]
    return [candidate_ids[int(i)] for i in order]
