from __future__ import annotations

import math
import numpy as np

from .merged import _cluster_grid
from .spectra import matched_cosine
from .trajectory import SpectrumView


def _energy_map(spectra: list[SpectrumView]):
    out = {}
    for s in spectra:
        if len(s.collision_energy_ev) == 1:
            out[float(s.collision_energy_ev[0])] = s
    return out


def exact_energy_cosines(
    query: list[SpectrumView],
    reference: list[SpectrumView],
    *,
    tolerance_da: float = 0.01,
):
    """Cosines at exactly shared collision energies, in increasing energy order."""
    qmap = _energy_map(query)
    rmap = _energy_map(reference)
    energies = sorted(set(qmap) & set(rmap))
    vals = []
    for e in energies:
        q, r = qmap[e], rmap[e]
        if q.adduct is not None and r.adduct is not None and q.adduct != r.adduct:
            continue
        vals.append(
            matched_cosine(
                q.mzs, q.intensities, r.mzs, r.intensities,
                tolerance_da=tolerance_da,
            )
        )
    return vals


def geometric_joint_score(query, reference, *, tolerance_da: float = 0.01, eps: float = 1e-12):
    vals = exact_energy_cosines(query, reference, tolerance_da=tolerance_da)
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.maximum(vals, eps)))))


def minimum_joint_score(query, reference, *, tolerance_da: float = 0.01):
    vals = exact_energy_cosines(query, reference, tolerance_da=tolerance_da)
    if not vals:
        return float("nan")
    return float(np.min(vals))


def multi_energy_delta_score(
    query: list[SpectrumView],
    reference: list[SpectrumView],
    *,
    mode: str = "hellinger",
    tolerance_da: float = 0.01,
    eps: float = 1e-4,
) -> float:
    """Compare energy derivatives on a common fragment grid.

    Unlike strict log-fold QFD, this keeps appearing/disappearing peaks as zeros.
    Each spectrum is first L1-normalized. ``hellinger`` uses sqrt-probability
    coordinates; ``clr`` uses a pseudocount-stabilized centered log-ratio map.
    Consecutive differences are divided by their collision-energy gaps and then
    compared by cosine similarity.
    """
    qmap = _energy_map(query)
    rmap = _energy_map(reference)
    energies = sorted(set(qmap) & set(rmap))
    if len(energies) < 2:
        return float("nan")
    if any(qmap[e].adduct != rmap[e].adduct for e in energies):
        return float("nan")

    spectra = [qmap[e] for e in energies] + [rmap[e] for e in energies]
    _, M = _cluster_grid(spectra, tolerance_da)
    M = np.maximum(np.asarray(M, dtype=float), 0.0)
    sums = M.sum(axis=0)
    if np.any(sums <= 0):
        return float("nan")
    P = M / sums[None, :]
    k = len(energies)

    if mode == "hellinger":
        T = np.sqrt(P)
    elif mode == "clr":
        T = np.log(P + float(eps))
        T = T - T.mean(axis=0, keepdims=True)
    else:
        raise ValueError("mode must be 'hellinger' or 'clr'")

    gaps = np.diff(np.asarray(energies, dtype=float))
    if np.any(gaps <= 0):
        return float("nan")
    dq = np.diff(T[:, :k], axis=1) / gaps[None, :]
    dr = np.diff(T[:, k:], axis=1) / gaps[None, :]
    a = dq.ravel()
    b = dr.ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def candidate_zscores(score_map: dict[str, float]):
    """Within-query z-scores; nonfinite candidates receive zero dynamic evidence."""
    finite = np.array([v for v in score_map.values() if math.isfinite(v)], dtype=float)
    if len(finite) < 2:
        return {k: 0.0 for k in score_map}
    mu = float(finite.mean())
    sd = float(finite.std())
    if sd <= 1e-12:
        return {k: 0.0 for k in score_map}
    return {k: ((v - mu) / sd if math.isfinite(v) else 0.0) for k, v in score_map.items()}
