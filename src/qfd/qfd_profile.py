from __future__ import annotations

import math
import numpy as np

from .fragment_matrix import align_fragment_matrix, centered_log_fold_change
from .spectra import match_peaks
from .trajectory import SpectrumView


def _single_energy_same_adduct(spectra: list[SpectrumView]):
    groups: dict[str | None, list[SpectrumView]] = {}
    for s in spectra:
        if len(s.collision_energy_ev) == 1:
            groups.setdefault(s.adduct, []).append(s)
    return groups


def widest_energy_pair(spectra: list[SpectrumView]):
    """Choose the same-adduct single-energy pair with the widest eV separation."""
    best = None
    best_gap = -math.inf
    for group in _single_energy_same_adduct(spectra).values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda s: s.collision_energy_ev[0])
        pair = (ordered[0], ordered[-1])
        gap = pair[1].collision_energy_ev[0] - pair[0].collision_energy_ev[0]
        if gap > best_gap:
            best, best_gap = pair, gap
    return best


def matching_energy_pair(reference: list[SpectrumView], target_pair, *, max_gap_ev: float = 5.0):
    """Find same-adduct reference spectra nearest the two target energies."""
    if target_pair is None:
        return None
    q0, q1 = target_pair
    refs = [r for r in reference if len(r.collision_energy_ev) == 1 and r.adduct == q0.adduct]
    if len(refs) < 2:
        return None
    chosen = []
    used = set()
    for q in (q0, q1):
        ranked = sorted(
            enumerate(refs),
            key=lambda x: abs(x[1].collision_energy_ev[0] - q.collision_energy_ev[0]),
        )
        pick = next(((idx, r) for idx, r in ranked if idx not in used), None)
        if pick is None:
            return None
        idx, r = pick
        if abs(r.collision_energy_ev[0] - q.collision_energy_ev[0]) > max_gap_ev:
            return None
        used.add(idx)
        chosen.append(r)
    return tuple(chosen)


def two_energy_qfd_score(
    query: list[SpectrumView],
    reference: list[SpectrumView],
    *,
    tolerance_da: float = 0.01,
    max_energy_gap_ev: float = 5.0,
    min_shared_fragments: int = 2,
) -> float:
    """Negative RMSE between nuisance-invariant two-energy log-fold profiles."""
    qpair = widest_energy_pair(query)
    rpair = matching_energy_pair(reference, qpair, max_gap_ev=max_energy_gap_ev)
    if qpair is None or rpair is None:
        return float("nan")

    Q = align_fragment_matrix(list(qpair), tolerance_da=tolerance_da, min_presence=2)
    R = align_fragment_matrix(list(rpair), tolerance_da=tolerance_da, min_presence=2)
    qmz, qv = centered_log_fold_change(Q)
    rmz, rv = centered_log_fold_change(R)
    pairs = match_peaks(qmz, rmz, tolerance_da=tolerance_da)
    if len(pairs) < min_shared_fragments:
        return float("nan")
    d = np.array([qv[i] - rv[j] for i, j in pairs], dtype=float)
    return -float(np.sqrt(np.mean(d * d)))
