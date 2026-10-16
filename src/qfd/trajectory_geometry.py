from __future__ import annotations

import numpy as np


def segment_projection(y, a, b):
    """Project y onto the closed line segment [a,b].

    Returns (distance, t), where projection=(1-t)*a+t*b and t is clipped to
    [0,1]. Degenerate segments are handled deterministically with t=0.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if y.ndim != 1 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("y, a, and b must be aligned 1D arrays")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("finite inputs required")
    v = b - a
    denom = float(v @ v)
    if denom <= np.finfo(float).eps:
        t = 0.0
        p = a
    else:
        t = float(np.clip(((y - a) @ v) / denom, 0.0, 1.0))
        p = a + t * v
    return float(np.linalg.norm(y - p)), t


def ordered_path_projection(y, columns):
    """Project y onto an ordered piecewise-linear energy trajectory.

    ``columns`` is shape (features, energies), ordered by increasing collision
    energy. Only adjacent energy states are connected. This intentionally
    differs from projection onto the full convex hull: non-adjacent/interior
    mixtures are not treated as physically reachable single-energy states.

    Returns (distance, segment_index, t). The segment index is the left endpoint
    index, and t is the interpolation coordinate on that adjacent segment.
    """
    y = np.asarray(y, dtype=float)
    C = np.asarray(columns, dtype=float)
    if y.ndim != 1 or C.ndim != 2 or C.shape[0] != y.size:
        raise ValueError("columns must be a 2D matrix aligned with y")
    if C.shape[1] < 2:
        raise ValueError("ordered path requires at least two energy states")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(C)):
        raise ValueError("finite inputs required")

    best = None
    for j in range(C.shape[1] - 1):
        d, t = segment_projection(y, C[:, j], C[:, j + 1])
        candidate = (d, j, t)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best
