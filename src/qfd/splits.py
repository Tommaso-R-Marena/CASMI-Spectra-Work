from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _bucket(key: str, seed: int) -> float:
    payload = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def structure_holdout(structures: Iterable[str], *, val_fraction: float = 0.2, seed: int = 0):
    """Deterministically split unique 2D structure IDs into train/validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    unique = sorted(set(str(x) for x in structures))
    val = {x for x in unique if _bucket(x, seed) < val_fraction}
    train = set(unique) - val
    return train, val


def assert_no_structure_leakage(train_structures, val_structures) -> None:
    overlap = set(train_structures) & set(val_structures)
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(f"structure leakage detected ({len(overlap)} IDs), e.g. {sample}")
