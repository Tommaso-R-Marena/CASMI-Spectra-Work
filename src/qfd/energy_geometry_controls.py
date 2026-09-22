from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .energy_geometry import hellinger_geodesic_distribution
from .energy_interpolation import _bhattacharyya


@dataclass(frozen=True)
class GeometryControlScores:
    hellinger_chord: float
    matched_coeff_mixture: float
    orthogonal_warp_mixture: float
    oracle_probability_mixture: float
    oracle_probability_weight: float
    oracle_geodesic: float
    oracle_geodesic_coordinate: float


def _normalize_prob(x: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total = float(y.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("positive finite probability mass required")
    return y / total


def hellinger_chord_distribution(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    a = _normalize_prob(p0)
    b = _normalize_prob(p1)
    tau = float(t)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("t must lie in [0,1]")
    x = (1.0 - tau) * np.sqrt(a) + tau * np.sqrt(b)
    norm = float(np.linalg.norm(x))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("degenerate Hellinger chord")
    x = x / norm
    return _normalize_prob(x * x)


def _slerp_coefficients(p0: np.ndarray, p1: np.ndarray, t: float) -> tuple[float, float, float]:
    a = _normalize_prob(p0)
    b = _normalize_prob(p1)
    tau = float(t)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("t must lie in [0,1]")
    x0 = np.sqrt(a)
    x1 = np.sqrt(b)
    theta = math.acos(float(np.clip(np.dot(x0, x1), -1.0, 1.0)))
    if theta <= 1e-12:
        return 1.0 - tau, tau, theta
    den = math.sin(theta)
    return math.sin((1.0 - tau) * theta) / den, math.sin(tau * theta) / den, theta


def matched_coefficient_probability_mixture(p0: np.ndarray, p1: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    a = _normalize_prob(p0)
    b = _normalize_prob(p1)
    c0, c1, _ = _slerp_coefficients(a, b, t)
    den = c0 * c0 + c1 * c1
    if den <= 0 or not math.isfinite(den):
        raise ValueError("degenerate slerp coefficients")
    w = float((c1 * c1) / den)
    return _normalize_prob((1.0 - w) * a + w * b), w


def orthogonal_warp_probability_mixture(p0: np.ndarray, p1: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    a = _normalize_prob(p0)
    b = _normalize_prob(p1)
    tau = float(t)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("t must lie in [0,1]")
    w = float(math.sin(0.5 * math.pi * tau) ** 2)
    return _normalize_prob((1.0 - w) * a + w * b), w


def _bounded_oracle(objective, *, xatol: float = 1e-10) -> tuple[float, float]:
    res = minimize_scalar(lambda z: -float(objective(float(z))), bounds=(0.0, 1.0), method="bounded", options={"xatol": xatol})
    candidates = [(0.0, float(objective(0.0))), (1.0, float(objective(1.0)))]
    if res.success and math.isfinite(float(res.fun)):
        z = float(np.clip(res.x, 0.0, 1.0))
        candidates.append((z, float(objective(z))))
    z, val = max(candidates, key=lambda item: item[1])
    return float(val), float(z)


def geometry_control_scores(q: np.ndarray, p0: np.ndarray, p1: np.ndarray, t: float) -> GeometryControlScores:
    qn = _normalize_prob(q)
    a = _normalize_prob(p0)
    b = _normalize_prob(p1)
    tau = float(t)

    chord = hellinger_chord_distribution(a, b, tau)
    coeff, _ = matched_coefficient_probability_mixture(a, b, tau)
    orth, _ = orthogonal_warp_probability_mixture(a, b, tau)

    oracle_prob, w_star = _bounded_oracle(
        lambda w: _bhattacharyya(qn, (1.0 - w) * a + w * b)
    )
    oracle_geo, s_star = _bounded_oracle(
        lambda s: _bhattacharyya(qn, hellinger_geodesic_distribution(a, b, s))
    )

    return GeometryControlScores(
        hellinger_chord=float(_bhattacharyya(qn, chord)),
        matched_coeff_mixture=float(_bhattacharyya(qn, coeff)),
        orthogonal_warp_mixture=float(_bhattacharyya(qn, orth)),
        oracle_probability_mixture=float(oracle_prob),
        oracle_probability_weight=float(w_star),
        oracle_geodesic=float(oracle_geo),
        oracle_geodesic_coordinate=float(s_star),
    )
