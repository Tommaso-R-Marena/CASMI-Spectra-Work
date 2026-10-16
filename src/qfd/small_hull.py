from __future__ import annotations

from itertools import combinations
import numpy as np


def convex_hull_distance_active_set(y, columns, *, feasibility_tol: float = 1e-10):
    """Deterministic Euclidean projection onto a small convex hull.

    Enumerates every non-empty face of the simplex. This is intended for the
    three-energy CFM transfer setting, where it avoids optimizer tolerance and
    initialization effects from a generic nonlinear solver while solving the
    same convex problem: min ||y - Cw||_2 subject to w >= 0 and sum(w)=1.
    """
    y = np.asarray(y, dtype=float)
    C = np.asarray(columns, dtype=float)
    if y.ndim != 1 or C.ndim != 2 or C.shape[0] != y.size:
        raise ValueError("columns must be a 2D matrix aligned with y")
    m = C.shape[1]
    if m < 1:
        raise ValueError("at least one convex-hull column is required")
    if m > 8:
        raise ValueError("active-set enumeration is intended for at most 8 columns")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(C)):
        raise ValueError("finite inputs required")

    best_dist = np.inf
    best_w = None
    all_idx = range(m)

    for size in range(1, m + 1):
        for subset in combinations(all_idx, size):
            idx = np.asarray(subset, dtype=int)
            A = C[:, idx]
            gram = A.T @ A
            rhs0 = A.T @ y
            K = np.empty((size + 1, size + 1), dtype=float)
            K[:size, :size] = gram
            K[:size, size] = 1.0
            K[size, :size] = 1.0
            K[size, size] = 0.0
            rhs = np.concatenate([rhs0, [1.0]])
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            w_face = sol[:size]
            if np.min(w_face) < -feasibility_tol:
                continue
            w_face = np.maximum(w_face, 0.0)
            total = float(w_face.sum())
            if total <= 0:
                continue
            w_face = w_face / total
            residual = y - A @ w_face
            dist = float(np.linalg.norm(residual))
            if dist < best_dist:
                w = np.zeros(m, dtype=float)
                w[idx] = w_face
                best_dist = dist
                best_w = w

    if best_w is None:
        raise RuntimeError("failed to find a feasible convex-hull projection")
    return best_dist, best_w
