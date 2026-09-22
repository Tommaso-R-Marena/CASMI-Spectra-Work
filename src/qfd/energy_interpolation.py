from __future__ import annotations

import math
import numpy as np

from .trajectory import SpectrumView


def _common_grid(spectra: list[SpectrumView], tolerance_da: float) -> np.ndarray:
    """Return nonnegative intensity columns on a shared tolerance-clustered m/z grid.

    Historical heldout30 implementation. Kept bit-for-bit in semantics for frozen
    replay; use ``_common_grid_strict`` for the post-hoc anti-chaining robustness audit.
    """
    vals: list[tuple[float, int, int]] = []
    for sidx, spec in enumerate(spectra):
        for pidx, mz in enumerate(np.asarray(spec.mzs, dtype=float)):
            if np.isfinite(mz):
                vals.append((float(mz), sidx, pidx))
    vals.sort()
    clusters: list[dict[str, object]] = []
    for mz, sidx, pidx in vals:
        if not clusters or mz - float(clusters[-1]["max_mz"]) > float(tolerance_da):
            clusters.append({"mzs": [mz], "max_mz": mz, "members": [(sidx, pidx, mz)]})
        else:
            c = clusters[-1]
            c["mzs"].append(mz)  # type: ignore[index]
            c["max_mz"] = mz
            c["members"].append((sidx, pidx, mz))  # type: ignore[index]

    if not clusters:
        return np.zeros((0, len(spectra)), dtype=float)

    grid = np.asarray([np.mean(c["mzs"]) for c in clusters], dtype=float)
    X = np.zeros((len(grid), len(spectra)), dtype=float)
    for i, c in enumerate(clusters):
        by_spec: dict[int, tuple[float, int]] = {}
        for sidx, pidx, mz in c["members"]:  # type: ignore[assignment]
            dist = abs(float(mz) - float(grid[i]))
            prev = by_spec.get(int(sidx))
            if prev is None or dist < prev[0]:
                by_spec[int(sidx)] = (dist, int(pidx))
        for sidx, (_, pidx) in by_spec.items():
            val = float(spectra[sidx].intensities[pidx])
            if np.isfinite(val) and val > 0:
                X[i, sidx] = val
    return X


def _common_grid_strict(spectra: list[SpectrumView], tolerance_da: float) -> np.ndarray:
    """Shared grid whose every cluster has diameter <= ``tolerance_da``.

    Unlike the historical sequential-linkage grid, this cannot join A and C merely
    because A~B and B~C when |A-C| exceeds tolerance.  This function is intentionally
    separate so the original prospective heldout30 result remains exactly replayable.
    """
    vals: list[tuple[float, int, int]] = []
    for sidx, spec in enumerate(spectra):
        for pidx, mz in enumerate(np.asarray(spec.mzs, dtype=float)):
            if np.isfinite(mz):
                vals.append((float(mz), sidx, pidx))
    vals.sort()
    clusters: list[dict[str, object]] = []
    for mz, sidx, pidx in vals:
        if not clusters or mz - float(clusters[-1]["min_mz"]) > float(tolerance_da):
            clusters.append({"mzs": [mz], "min_mz": mz, "members": [(sidx, pidx, mz)]})
        else:
            c = clusters[-1]
            c["mzs"].append(mz)  # type: ignore[index]
            c["members"].append((sidx, pidx, mz))  # type: ignore[index]

    if not clusters:
        return np.zeros((0, len(spectra)), dtype=float)

    grid = np.asarray([np.mean(c["mzs"]) for c in clusters], dtype=float)
    X = np.zeros((len(grid), len(spectra)), dtype=float)
    for i, c in enumerate(clusters):
        by_spec: dict[int, tuple[float, int]] = {}
        for sidx, pidx, mz in c["members"]:  # type: ignore[assignment]
            dist = abs(float(mz) - float(grid[i]))
            prev = by_spec.get(int(sidx))
            if prev is None or dist < prev[0]:
                by_spec[int(sidx)] = (dist, int(pidx))
        for sidx, (_, pidx) in by_spec.items():
            val = float(spectra[sidx].intensities[pidx])
            if np.isfinite(val) and val > 0:
                X[i, sidx] = val
    return X


def _l1_columns(X: np.ndarray) -> np.ndarray | None:
    sums = np.asarray(X, dtype=float).sum(axis=0)
    if X.ndim != 2 or X.shape[1] == 0 or np.any(~np.isfinite(sums)) or np.any(sums <= 0):
        return None
    return X / sums[None, :]


def _single_energy(spec: SpectrumView) -> float:
    if len(spec.collision_energy_ev) != 1:
        raise ValueError("single-energy spectrum required")
    return float(spec.collision_energy_ev[0])


def _check_triplet(query: SpectrumView, low: SpectrumView, high: SpectrumView) -> tuple[float, float, float]:
    eq = _single_energy(query)
    elo = _single_energy(low)
    ehi = _single_energy(high)
    if not elo < eq < ehi:
        raise ValueError(f"heldout query energy must be bracketed: low={elo}, query={eq}, high={ehi}")
    if query.adduct is not None and (query.adduct != low.adduct or query.adduct != high.adduct):
        raise ValueError("query/reference adduct mismatch")
    return eq, elo, ehi


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("aligned one-dimensional distributions required")
    return float(np.sum(np.sqrt(np.clip(p, 0.0, None) * np.clip(q, 0.0, None))))


def _pair_affinity_strict(a: SpectrumView, b: SpectrumView, tolerance_da: float) -> float:
    X = _l1_columns(_common_grid_strict([a, b], tolerance_da))
    if X is None:
        return float("nan")
    return _bhattacharyya(X[:, 0], X[:, 1])


def endpoint_static_score(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> float:
    """Best endpoint Hellinger/Bhattacharyya affinity using only bracketing references."""
    _check_triplet(query, low, high)
    X = _l1_columns(_common_grid([query, low, high], tolerance_da))
    if X is None:
        return float("nan")
    q, lo, hi = X[:, 0], X[:, 1], X[:, 2]
    return max(_bhattacharyya(q, lo), _bhattacharyya(q, hi))


def endpoint_static_score_strict(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> float:
    """Anti-chaining baseline: score each endpoint independently on a strict grid."""
    _check_triplet(query, low, high)
    return max(
        _pair_affinity_strict(query, low, tolerance_da),
        _pair_affinity_strict(query, high, tolerance_da),
    )


def energy_interpolation_score(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> float:
    """Score a held-out energy against the linearly energy-constrained spectral path.

    Each spectrum is placed on one shared m/z grid and L1-normalized. If the held-out
    query energy ``e`` lies between bracketing reference energies ``e0 < e < e1``, the
    predicted fragment distribution is

        p_hat(e) = (1-t) p(e0) + t p(e1),  t=(e-e0)/(e1-e0).

    The returned score is the Bhattacharyya affinity between the observed query and
    ``p_hat``. It lies in [0, 1] up to floating-point roundoff. No truth label or
    candidate rank enters the construction.
    """
    eq, elo, ehi = _check_triplet(query, low, high)
    X = _l1_columns(_common_grid([query, low, high], tolerance_da))
    if X is None:
        return float("nan")
    q, lo, hi = X[:, 0], X[:, 1], X[:, 2]
    t = (eq - elo) / (ehi - elo)
    pred = (1.0 - t) * lo + t * hi
    score = _bhattacharyya(q, pred)
    if not math.isfinite(score):
        return float("nan")
    return float(min(1.0, max(0.0, score)))


def energy_interpolation_score_strict(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> float:
    """Post-hoc robustness version using only diameter-bounded m/z clusters."""
    eq, elo, ehi = _check_triplet(query, low, high)
    X = _l1_columns(_common_grid_strict([query, low, high], tolerance_da))
    if X is None:
        return float("nan")
    q, lo, hi = X[:, 0], X[:, 1], X[:, 2]
    t = (eq - elo) / (ehi - elo)
    pred = (1.0 - t) * lo + t * hi
    score = _bhattacharyya(q, pred)
    if not math.isfinite(score):
        return float("nan")
    return float(min(1.0, max(0.0, score)))


def equal_mix_control_score(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> float:
    """Energy-blind 50/50 mixture control on the same endpoint spectra."""
    _check_triplet(query, low, high)
    X = _l1_columns(_common_grid([query, low, high], tolerance_da))
    if X is None:
        return float("nan")
    q, lo, hi = X[:, 0], X[:, 1], X[:, 2]
    return _bhattacharyya(q, 0.5 * lo + 0.5 * hi)


def equal_mix_control_score_strict(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> float:
    """Energy-blind control under the same anti-chaining grid as strict interpolation."""
    _check_triplet(query, low, high)
    X = _l1_columns(_common_grid_strict([query, low, high], tolerance_da))
    if X is None:
        return float("nan")
    q, lo, hi = X[:, 0], X[:, 1], X[:, 2]
    return _bhattacharyya(q, 0.5 * lo + 0.5 * hi)
