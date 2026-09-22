from __future__ import annotations

import math

import numpy as np

from .energy_interpolation import _bhattacharyya


def _normalize_prob(x: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total = float(y.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("positive finite probability mass required")
    return y / total


def power_mean_distribution(
    p0: np.ndarray,
    p1: np.ndarray,
    t: float,
    alpha: float,
) -> np.ndarray:
    """Normalized power-mean interpolation between endpoint probability spectra.

    P_alpha(t) ∝ ((1-t) p0^alpha + t p1^alpha)^(1/alpha).

    alpha=1 is ordinary probability interpolation. alpha=1/2 is exactly the
    normalized Hellinger/square-root chord.
    """
    a = _normalize_prob(p0)
    b = _normalize_prob(p1)
    tau = float(t)
    al = float(alpha)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("t must lie in [0,1]")
    if not math.isfinite(al) or al <= 0:
        raise ValueError("alpha must be positive and finite")
    z = (1.0 - tau) * np.power(a, al) + tau * np.power(b, al)
    out = np.power(np.clip(z, 0.0, None), 1.0 / al)
    return _normalize_prob(out)


def power_mean_score(
    q: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    t: float,
    alpha: float,
) -> float:
    qn = _normalize_prob(q)
    pred = power_mean_distribution(p0, p1, t, alpha)
    return float(_bhattacharyya(qn, pred))


def logit_energy_warp(t: float, a: float, b: float, *, eps: float = 1e-8) -> float:
    """Monotone logistic recalibration of the collision-energy coordinate."""
    tau = float(t)
    aa = float(a)
    bb = float(b)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("t must lie in [0,1]")
    if not math.isfinite(aa) or aa <= 0 or not math.isfinite(bb):
        raise ValueError("warp parameters must be finite with a>0")
    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0,0.5)")
    tc = float(np.clip(tau, eps, 1.0 - eps))
    logit = math.log(tc / (1.0 - tc))
    z = aa * logit + bb
    if z >= 0:
        e = math.exp(-z)
        return float(1.0 / (1.0 + e))
    e = math.exp(z)
    return float(e / (1.0 + e))


def warped_power_mean_score(
    q: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    t: float,
    alpha: float,
    warp_a: float,
    warp_b: float,
) -> float:
    return power_mean_score(q, p0, p1, logit_energy_warp(t, warp_a, warp_b), alpha)
