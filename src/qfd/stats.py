from __future__ import annotations

from collections import defaultdict
import hashlib
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


def paired_block_bootstrap_delta(
    a,
    b,
    blocks,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
):
    """Cluster/block bootstrap CI for the query-weighted mean paired delta.

    ``blocks`` identifies the independent sampling unit for each paired observation.
    A bootstrap replicate samples blocks with replacement and includes every query in
    each sampled block. This preserves within-block dependence while retaining the
    query-weighted estimand used by the benchmark summaries.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    block_labels = list(blocks)
    if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("a and b must be nonempty one-dimensional arrays of equal length")
    if len(block_labels) != len(x):
        raise ValueError("blocks must have the same length as a and b")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    by_block = defaultdict(list)
    for i, label in enumerate(block_labels):
        by_block[label].append(i)
    unique_blocks = list(by_block)
    if not unique_blocks:
        raise ValueError("at least one block is required")

    d = x - y
    block_deltas = [d[np.asarray(by_block[label], dtype=int)] for label in unique_blocks]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    n_blocks = len(unique_blocks)
    for i in range(n_boot):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        values = np.concatenate([block_deltas[int(j)] for j in sampled])
        deltas[i] = values.mean()

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(len(d)),
        "n_blocks": int(n_blocks),
        "seed": int(seed),
        "n_boot": int(n_boot),
        "resampling_unit": "block",
    }


def stable_group_is_validation(
    group_label: str,
    *,
    validation_fraction: float = 0.5,
    salt: str = "msnlib-group-validation-v1",
) -> bool:
    """Deterministically assign a whole group to discovery or validation.

    SHA-256 makes the split independent of input order, Python hash randomization,
    and candidate identifiers outside the supplied group label. The salt is part of
    the protocol and should be frozen before looking at validation outcomes.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    payload = f"{salt}\x1f{group_label}".encode("utf-8")
    u = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)
    return u < validation_fraction


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
