from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .energy_interpolation import _common_grid_strict, _l1_columns, _check_triplet, _bhattacharyya
from .trajectory import SpectrumView


@dataclass(frozen=True)
class EnergyPredictionScores:
    query_energy_ev: float
    low_energy_ev: float
    high_energy_ev: float
    t: float
    best_endpoint: float
    low_endpoint: float
    high_endpoint: float
    linear: float
    equal_mix: float
    reversed_energy: float
    hellinger_geodesic: float


def _normalized_nonnegative(x: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total = float(y.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("positive finite mass required")
    return y / total


def hellinger_geodesic_distribution(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Fisher-Rao/Hellinger great-circle interpolation on the probability simplex.

    The square-root embedding maps normalized discrete distributions to the positive
    orthant of the unit sphere.  This returns the squared great-circle interpolation
    at fraction ``t``.  Endpoint values are exact up to floating-point roundoff.
    """
    a = _normalized_nonnegative(p0)
    b = _normalized_nonnegative(p1)
    tau = float(t)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    x0 = np.sqrt(a)
    x1 = np.sqrt(b)
    dot = float(np.clip(np.dot(x0, x1), -1.0, 1.0))
    theta = math.acos(dot)
    if theta <= 1e-12:
        x = (1.0 - tau) * x0 + tau * x1
    else:
        denom = math.sin(theta)
        x = (
            math.sin((1.0 - tau) * theta) / denom * x0
            + math.sin(tau * theta) / denom * x1
        )
    norm = float(np.linalg.norm(x))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("degenerate geodesic state")
    x = x / norm
    return _normalized_nonnegative(x * x)


def energy_prediction_scores_strict(
    query: SpectrumView,
    low: SpectrumView,
    high: SpectrumView,
    *,
    tolerance_da: float = 0.01,
) -> EnergyPredictionScores:
    """Compute all frozen generalized held-out-energy predictors on one strict grid."""
    eq, elo, ehi = _check_triplet(query, low, high)
    X = _l1_columns(_common_grid_strict([query, low, high], tolerance_da))
    if X is None:
        raise ValueError("could not construct positive normalized triplet")
    q, p0, p1 = X[:, 0], X[:, 1], X[:, 2]
    t = float((eq - elo) / (ehi - elo))

    lo_score = _bhattacharyya(q, p0)
    hi_score = _bhattacharyya(q, p1)
    linear = _bhattacharyya(q, (1.0 - t) * p0 + t * p1)
    equal = _bhattacharyya(q, 0.5 * p0 + 0.5 * p1)
    reversed_energy = _bhattacharyya(q, t * p0 + (1.0 - t) * p1)
    geo = _bhattacharyya(q, hellinger_geodesic_distribution(p0, p1, t))

    vals = [lo_score, hi_score, linear, equal, reversed_energy, geo]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("non-finite prediction score")

    return EnergyPredictionScores(
        query_energy_ev=float(eq),
        low_energy_ev=float(elo),
        high_energy_ev=float(ehi),
        t=t,
        best_endpoint=float(max(lo_score, hi_score)),
        low_endpoint=float(lo_score),
        high_endpoint=float(hi_score),
        linear=float(linear),
        equal_mix=float(equal),
        reversed_energy=float(reversed_energy),
        hellinger_geodesic=float(geo),
    )
