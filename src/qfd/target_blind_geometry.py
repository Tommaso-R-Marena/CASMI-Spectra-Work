"""Target-blind reference geometry and m/z matching for falsification of MSnLib LOEO.

Every prediction grid is constructed from endpoint references only; the withheld
spectrum may influence the *evaluation score* but not binning or prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize_scalar

from .power_geometry import power_mean_distribution


@dataclass(frozen=True)
class ReferenceGrid:
    mz: np.ndarray
    low: np.ndarray
    high: np.ndarray


def _positive_peaks(spec):
    mz = np.asarray(spec.mzs, dtype=float)
    intensity = np.asarray(spec.intensities, dtype=float)
    if mz.ndim != 1 or intensity.ndim != 1 or mz.shape != intensity.shape:
        raise ValueError("m/z and intensity vectors must match")
    mask = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    return mz[mask], intensity[mask]


def reference_only_grid(low, high, *, tolerance_da: float = 0.01) -> ReferenceGrid:
    """Deterministic complete-link reference bins, never reading the held-out query.

    All positive reference peak intensities are retained. Within a reference bin
    the intensities of same-spectrum peaks are summed rather than discarded.
    """
    if not math.isfinite(tolerance_da) or tolerance_da <= 0:
        raise ValueError("tolerance must be positive and finite")
    mz0, y0 = _positive_peaks(low)
    mz1, y1 = _positive_peaks(high)
    if not len(y0) or not len(y1):
        raise ValueError("both reference spectra must contain positive peaks")
    values = [(float(m), 0, float(v)) for m, v in zip(mz0, y0)]
    values += [(float(m), 1, float(v)) for m, v in zip(mz1, y1)]
    values.sort(key=lambda x: (x[0], x[1], x[2]))
    centers, row0, row1 = [], [], []
    cluster = []
    anchor = None

    def flush():
        if not cluster:
            return
        centers.append(float(np.mean([z[0] for z in cluster])))
        row0.append(sum(z[2] for z in cluster if z[1] == 0))
        row1.append(sum(z[2] for z in cluster if z[1] == 1))

    for point in values:
        if anchor is None or point[0] - anchor > tolerance_da:
            flush()
            cluster = []
            anchor = point[0]
        cluster.append(point)
    flush()
    a, b = np.asarray(row0, float), np.asarray(row1, float)
    return ReferenceGrid(np.asarray(centers, float), a / a.sum(), b / b.sum())


def query_mass_on_reference_grid(query, grid: ReferenceGrid, *, tolerance_da: float = 0.01):
    """Deterministic one-to-one smallest-error peak matching.

    Query intensities are normalized by *all* query peaks, including unmatched
    peaks. Matching is only used to compute affinity and cannot change grid.low,
    grid.high, or grid.mz. Unmatched query mass earns zero prediction affinity.
    """
    mz, intensity = _positive_peaks(query)
    if not len(intensity):
        raise ValueError("held-out query requires nonzero positive mass")
    q = intensity / intensity.sum()
    edges = []
    for i, mass in enumerate(mz):
        left = int(np.searchsorted(grid.mz, mass - tolerance_da, side="left"))
        right = int(np.searchsorted(grid.mz, mass + tolerance_da, side="right"))
        for j in range(left, right):
            err = abs(float(grid.mz[j] - mass))
            if err <= tolerance_da:
                edges.append((err, j, i))
    edges.sort()
    used_ref, used_query = set(), set()
    aligned = np.zeros(len(grid.mz), dtype=float)
    for _, j, i in edges:
        if j not in used_ref and i not in used_query:
            aligned[j] = float(q[i])
            used_ref.add(j)
            used_query.add(i)
    return aligned, float(sum(q[i] for i in range(len(q)) if i not in used_query))


def affinity(query_aligned, prediction) -> float:
    a = np.asarray(query_aligned, dtype=float)
    b = np.asarray(prediction, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or np.any(a < 0) or np.any(b < 0):
        raise ValueError("invalid affinity vectors")
    return float(np.sum(np.sqrt(a * b)))


def support_only_suppression(low, high, t: float, *, power: float = 4.0):
    """Alternative to nonlinear power curvature.

    Keep ordinary linear mixing on bins present at BOTH energies; suppress
    endpoint-exclusive bins with (1-t)^power and t^power respectively.
    This tests if the empirical gain is just exclusion of transient peaks.
    """
    a, b = np.asarray(low, float), np.asarray(high, float)
    if a.shape != b.shape or a.ndim != 1 or not 0 <= t <= 1 or power <= 0:
        raise ValueError("invalid support-only path")
    shared = (a > 0) & (b > 0)
    z = np.where(shared, (1-t)*a+t*b, (1-t)**power*a+t**power*b)
    total = float(z.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("degenerate path")
    return z / total


def best_convex_oracle(q, p0, p1):
    """Answer-aware nondeployable upper bound over all convex endpoint mixes."""
    objective = lambda w: affinity(q, (1-w)*p0+w*p1)
    fit = minimize_scalar(lambda w: -objective(float(w)), bounds=(0, 1),
                          method="bounded", options={"xatol": 1e-11})
    candidates = [objective(0.0), objective(1.0)]
    if fit.success:
        candidates.append(objective(float(fit.x)))
    return float(max(candidates))


def target_blind_scores(query, low, high, t: float, *, tolerance_da=0.01):
    if not (0 < t < 1):
        raise ValueError("strict interior collision-energy query required")
    grid = reference_only_grid(low, high, tolerance_da=tolerance_da)
    q, unmatched = query_mass_on_reference_grid(query, grid, tolerance_da=tolerance_da)
    p0, p1 = grid.low, grid.high
    score = lambda pred: affinity(q, pred)
    return {
        "linear": score(power_mean_distribution(p0,p1,t,1.0)),
        "hellinger_chord": score(power_mean_distribution(p0,p1,t,0.5)),
        "alpha_quarter": score(power_mean_distribution(p0,p1,t,0.25)),
        "alpha_eighth_exploratory": score(power_mean_distribution(p0,p1,t,0.125)),
        "support_only": score(support_only_suppression(p0,p1,t,power=4)),
        "oracle_probability_mixture": best_convex_oracle(q,p0,p1),
        "best_endpoint": max(score(p0),score(p1)),
        "query_unmatched_mass": unmatched,
        "reference_shared_min_overlap": float(np.minimum(p0,p1).sum()),
        "reference_shared_bin_count": int(np.sum((p0>0)&(p1>0))),
        "reference_bins": len(p0),
    }
