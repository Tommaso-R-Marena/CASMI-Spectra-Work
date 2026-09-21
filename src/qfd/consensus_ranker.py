from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


CONSENSUS_FEATURE_NAMES = (
    "z_static_best",
    "z_ordered_path",
    "z_neg_energy_spread",
)


@dataclass(frozen=True)
class ConditionalRankerFit:
    weights: np.ndarray
    l2: float
    success: bool
    objective: float


def conditional_softmax_fit(
    groups: Sequence[tuple[np.ndarray, int]],
    *,
    l2: float = 1.0,
    initial: np.ndarray | None = None,
    maxiter: int = 500,
) -> ConditionalRankerFit:
    """Fit one shared linear score using a query-conditional softmax loss.

    Each group contains a candidate feature matrix X_q and the row index of the
    true candidate. The loss is

        sum_q [logsumexp(X_q w) - x_truth^T w] + (l2/2)||w||^2.

    Treating each query as one likelihood term prevents large candidate pools
    from receiving disproportionate weight merely because they contain more
    decoys.
    """
    if not groups:
        raise ValueError("at least one query group is required")
    p = int(np.asarray(groups[0][0]).shape[1])
    w0 = np.zeros(p, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    if w0.shape != (p,):
        raise ValueError("initial weight shape mismatch")
    checked: list[tuple[np.ndarray, int]] = []
    for X, y in groups:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != p or X.shape[0] < 1:
            raise ValueError("invalid feature matrix")
        if not np.all(np.isfinite(X)):
            raise ValueError("features must be finite")
        y = int(y)
        if not 0 <= y < X.shape[0]:
            raise ValueError("truth index out of range")
        checked.append((X, y))

    def objective_gradient(w: np.ndarray):
        loss = 0.0
        grad = np.zeros_like(w)
        for X, y in checked:
            s = X @ w
            z = logsumexp(s)
            loss += float(z - s[y])
            prob = np.exp(s - z)
            grad += prob @ X - X[y]
        loss += 0.5 * float(l2) * float(w @ w)
        grad += float(l2) * w
        return loss, grad

    result = minimize(
        lambda w: objective_gradient(w),
        w0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-11},
    )
    return ConditionalRankerFit(
        weights=np.asarray(result.x, dtype=float),
        l2=float(l2),
        success=bool(result.success),
        objective=float(result.fun),
    )


def standardize_within_query(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("finite one-dimensional values required")
    sd = float(np.std(x))
    if sd <= 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


def consensus_feature_matrix(
    cos10: Iterable[float],
    cos20: Iterable[float],
    cos40: Iterable[float],
    ordered_path_score: Iterable[float],
) -> np.ndarray:
    """Build the frozen three-feature CASMI consensus representation.

    Features are standardized within each query/candidate pool:
      1. best individual-energy cosine,
      2. ordered 10->20->40 path score,
      3. negative across-energy cosine spread.

    The third term rewards candidates whose predicted fragmentation agrees
    coherently across energies instead of obtaining one isolated high match.
    """
    c10 = np.asarray(list(cos10), dtype=float)
    c20 = np.asarray(list(cos20), dtype=float)
    c40 = np.asarray(list(cos40), dtype=float)
    path = np.asarray(list(ordered_path_score), dtype=float)
    if not (c10.shape == c20.shape == c40.shape == path.shape) or c10.ndim != 1:
        raise ValueError("aligned one-dimensional candidate vectors required")
    if c10.size == 0 or not all(np.all(np.isfinite(x)) for x in (c10, c20, c40, path)):
        raise ValueError("finite non-empty candidate vectors required")
    C = np.column_stack([c10, c20, c40])
    static = np.max(C, axis=1)
    neg_spread = -np.std(C, axis=1)
    return np.column_stack([
        standardize_within_query(static),
        standardize_within_query(path),
        standardize_within_query(neg_spread),
    ])


def topk_overlap_fraction(a: Iterable[float], b: Iterable[float], *, k: int = 10) -> float:
    """Fractional overlap between two descending top-k candidate sets."""
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.shape != bv.shape or av.ndim != 1 or av.size == 0:
        raise ValueError("aligned non-empty score vectors required")
    kk = min(int(k), int(av.size))
    if kk < 1:
        raise ValueError("k must be positive")
    ia = set(np.argsort(-av, kind="stable")[:kk].tolist())
    ib = set(np.argsort(-bv, kind="stable")[:kk].tolist())
    return float(len(ia & ib) / kk)


def selective_consensus_scores(
    static_scores: Iterable[float],
    expert_scores: Iterable[float],
    *,
    overlap_threshold: float,
    k: int = 10,
) -> tuple[np.ndarray, float, bool]:
    """Use expert scores only when expert/static top-k sets sufficiently agree."""
    static = np.asarray(list(static_scores), dtype=float)
    expert = np.asarray(list(expert_scores), dtype=float)
    if static.shape != expert.shape or static.ndim != 1 or static.size == 0:
        raise ValueError("aligned non-empty score vectors required")
    if not np.all(np.isfinite(static)) or not np.all(np.isfinite(expert)):
        raise ValueError("finite score vectors required")
    overlap = topk_overlap_fraction(static, expert, k=k)
    use_expert = bool(overlap >= float(overlap_threshold))
    return (expert.copy() if use_expert else static.copy()), float(overlap), use_expert


def deterministic_fold(key: int, *, salt: str, n_folds: int) -> int:
    import hashlib

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    digest = hashlib.sha256(f"{salt}:{int(key)}".encode("utf-8")).digest()
    return int(int.from_bytes(digest[:8], "big") % int(n_folds))
