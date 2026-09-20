from __future__ import annotations

import numpy as np

from .cone import convex_hull_distance
from .trajectory import SpectrumView


def _choose_energy_columns(reference: list[SpectrumView], target_energies, *, adduct=None, max_gap_ev=5.0):
    refs = [r for r in reference if len(r.collision_energy_ev) == 1 and (adduct is None or r.adduct == adduct)]
    chosen = []
    used = set()
    for e in target_energies:
        ranked = sorted(enumerate(refs), key=lambda x: abs(x[1].collision_energy_ev[0] - e))
        pick = next(((idx, r) for idx, r in ranked if idx not in used), None)
        if pick is None:
            return None
        idx, r = pick
        if abs(r.collision_energy_ev[0] - e) > max_gap_ev:
            return None
        used.add(idx)
        chosen.append(r)
    return chosen


def _cluster_grid(spectra: list[SpectrumView], tolerance_da: float):
    vals = []
    for sidx, s in enumerate(spectra):
        for pidx, mz in enumerate(np.asarray(s.mzs, float)):
            if np.isfinite(mz):
                vals.append((float(mz), sidx, pidx))
    vals.sort()
    clusters = []
    for mz, sidx, pidx in vals:
        if not clusters or mz - clusters[-1]["max_mz"] > tolerance_da:
            clusters.append({"mzs": [mz], "max_mz": mz, "members": [(sidx, pidx, mz)]})
        else:
            c = clusters[-1]
            c["mzs"].append(mz)
            c["max_mz"] = mz
            c["members"].append((sidx, pidx, mz))
    grid = np.array([np.mean(c["mzs"]) for c in clusters], float)
    X = np.zeros((len(grid), len(spectra)), float)
    for i, c in enumerate(clusters):
        by_spec = {}
        for sidx, pidx, mz in c["members"]:
            prev = by_spec.get(sidx)
            dist = abs(mz - grid[i])
            if prev is None or dist < prev[0]:
                by_spec[sidx] = (dist, pidx)
        for sidx, (_, pidx) in by_spec.items():
            val = float(spectra[sidx].intensities[pidx])
            if np.isfinite(val) and val > 0:
                X[i, sidx] = val
    return grid, X


def merged_convex_score(
    query: SpectrumView,
    reference: list[SpectrumView],
    *,
    tolerance_da: float = 0.01,
    max_energy_gap_ev: float = 5.0,
):
    """Score a merged query against the convex hull of candidate energy spectra.

    Intensities must be on a linear nonnegative scale (raw counts or ordinary
    relative intensities), not sqrt/log transformed. Each spectrum is L1-normalized
    internally, so this implements the normalized convex-hull formulation.
    Returns negative residual norm; higher is better.
    """
    if len(query.collision_energy_ev) < 2:
        return float("nan")
    chosen = _choose_energy_columns(
        reference, query.collision_energy_ev, adduct=query.adduct, max_gap_ev=max_energy_gap_ev,
    )
    if chosen is None:
        return float("nan")
    _, M = _cluster_grid([query] + chosen, tolerance_da)
    q = M[:, 0]
    C = M[:, 1:]
    qsum = q.sum()
    csums = C.sum(axis=0)
    if qsum <= 0 or np.any(csums <= 0):
        return float("nan")
    q = q / qsum
    C = C / csums[None, :]
    dist, w = convex_hull_distance(q, C)
    if not np.all(np.isfinite(w)):
        return float("nan")
    return -dist
