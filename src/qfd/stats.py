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


def _validated_paired_blocks(a, b, blocks):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    block_labels = list(blocks)
    if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("a and b must be nonempty one-dimensional arrays of equal length")
    if len(block_labels) != len(x):
        raise ValueError("blocks must have the same length as a and b")
    by_block = defaultdict(list)
    for i, label in enumerate(block_labels):
        by_block[label].append(i)
    unique_blocks = list(by_block)
    if not unique_blocks:
        raise ValueError("at least one block is required")
    return x, y, by_block, unique_blocks


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
    x, y, by_block, unique_blocks = _validated_paired_blocks(a, b, blocks)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

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


def paired_block_signflip_test(
    a,
    b,
    blocks,
    *,
    seed: int = 0,
    n_permutations: int = 200_000,
    exact_max_blocks: int = 20,
    atol: float = 1e-15,
):
    """Two-sided paired randomization test that flips whole blocks together.

    Under the sharp paired null, exchanging method labels within an independent block
    negates every query-level difference in that block.  The test statistic remains
    the benchmark's query-weighted mean difference, so blocks with more queries retain
    their contribution to the reported estimand.  For at most ``exact_max_blocks``
    blocks every sign assignment is enumerated; otherwise a seeded Monte Carlo test is
    used.  This is a robustness test complementary to the block bootstrap CI.
    """
    x, y, by_block, unique_blocks = _validated_paired_blocks(a, b, blocks)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if exact_max_blocks < 1:
        raise ValueError("exact_max_blocks must be positive")

    d = x - y
    block_sums = np.asarray(
        [d[np.asarray(by_block[label], dtype=int)].sum() for label in unique_blocks],
        dtype=float,
    )
    n = len(d)
    n_blocks = len(block_sums)
    observed = float(d.mean())
    threshold = abs(observed) - atol

    if n_blocks <= exact_max_blocks:
        total = 1 << n_blocks
        extreme = 0
        # Chunk enumeration keeps memory bounded while remaining exact.
        chunk = 65_536
        bit_positions = np.arange(n_blocks, dtype=np.uint64)
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            ids = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = ((ids >> bit_positions) & 1).astype(np.float64)
            signs = bits * 2.0 - 1.0
            stats = (signs @ block_sums) / float(n)
            extreme += int(np.sum(np.abs(stats) >= threshold))
        p_value = extreme / float(total)
        return {
            "delta": observed,
            "p_value_two_sided": float(p_value),
            "n": int(n),
            "n_blocks": int(n_blocks),
            "mode": "exact",
            "n_sign_assignments": int(total),
            "extreme": int(extreme),
        }

    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    batch = 10_000
    while done < n_permutations:
        m = min(batch, n_permutations - done)
        signs = rng.integers(0, 2, size=(m, n_blocks), dtype=np.int8).astype(np.float64)
        signs = signs * 2.0 - 1.0
        stats = (signs @ block_sums) / float(n)
        extreme += int(np.sum(np.abs(stats) >= threshold))
        done += m
    # Add-one correction gives a valid finite Monte Carlo randomization p-value.
    p_value = (extreme + 1.0) / (n_permutations + 1.0)
    return {
        "delta": observed,
        "p_value_two_sided": float(p_value),
        "n": int(n),
        "n_blocks": int(n_blocks),
        "mode": "monte_carlo",
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "extreme": int(extreme),
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
