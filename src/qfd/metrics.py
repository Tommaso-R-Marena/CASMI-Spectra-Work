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
    values = np.asarray(scores, dtype=float)
    # Sort in the requested direction directly rather than sorting ascending and
    # reversing the indices. Reversing a stable ascending sort also reverses
    # exact ties, which makes tie order depend on an implementation accident.
    order = np.argsort(-values if higher_is_better else values, kind="stable")
    return [candidate_ids[int(i)] for i in order]


def tie_aware_truth_metrics(
    candidate_ids: Sequence[str],
    scores: Sequence[float],
    truth: str,
    *,
    k: int = 25,
    higher_is_better: bool = True,
):
    """Order-invariant ranking credit when the truth is tied with other candidates.

    Exact score ties occupy a rank interval. This reports the interval and the
    expectation obtained by uniformly randomizing only within that tied block.
    It therefore cannot change merely because candidate IDs or input order change.
    """
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores must have equal length")
    if truth not in candidate_ids:
        raise ValueError("truth must be present in candidate_ids")
    if k < 1:
        raise ValueError("k must be positive")

    values = np.asarray(scores, dtype=float)
    truth_index = candidate_ids.index(truth)
    truth_score = values[truth_index]
    if np.isnan(truth_score):
        raise ValueError("truth score must not be NaN")

    if higher_is_better:
        better = values > truth_score
    else:
        better = values < truth_score
    tied = values == truth_score
    tie_size = int(np.sum(tied))
    rank_min = int(np.sum(better)) + 1
    rank_max = rank_min + tie_size - 1

    rr_values = [1.0 / r if r <= k else 0.0 for r in range(rank_min, rank_max + 1)]
    expected_rr = float(np.mean(rr_values))
    top1_credit = (1.0 / tie_size) if rank_min == 1 else 0.0
    top5_credit = float(np.mean([r <= 5 for r in range(rank_min, rank_max + 1)]))

    return {
        "rank_min": rank_min,
        "rank_max": rank_max,
        "expected_rank": 0.5 * (rank_min + rank_max),
        "tie_size": tie_size,
        "rr_at_k": expected_rr,
        "top1_credit": float(top1_credit),
        "top5_credit": top5_credit,
    }
