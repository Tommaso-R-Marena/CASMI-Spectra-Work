from __future__ import annotations

import numpy as np


def paired_bootstrap_delta(a, b, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap CI for mean(a-b)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("a and b must be nonempty one-dimensional arrays of equal length")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    n = len(x)
    deltas = np.empty(n_boot, dtype=float)
    d = x - y
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = d[idx].mean()
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "seed": int(seed),
        "n_boot": int(n_boot),
    }


def paired_win_loss_tie(a, b, *, atol: float = 1e-12):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("a and b must have equal shape")
    d = x - y
    return {
        "wins": int(np.sum(d > atol)),
        "losses": int(np.sum(d < -atol)),
        "ties": int(np.sum(np.abs(d) <= atol)),
    }
