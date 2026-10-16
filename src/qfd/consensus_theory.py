from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def subgaussian_consensus_pair_failure_bound(
    mean_margin: Sequence[float],
    covariance_proxy: Sequence[Sequence[float]],
) -> float:
    """Chernoff upper bound for one decoy beating the truth by consensus sum.

    Let X_e = s_e(truth) - s_e(decoy). Assume the centered margin vector is
    jointly sub-Gaussian with positive-semidefinite covariance proxy Sigma:

        E exp(u^T (X-mu)) <= exp(0.5 * u^T Sigma u)

    for all u. If 1^T mu > 0, then

        P(1^T X <= 0)
          <= exp(-(1^T mu)^2 / (2 * 1^T Sigma 1)).

    A nonpositive aggregate mean gives only the vacuous bound 1. If the
    aggregate variance proxy is exactly zero while the aggregate mean is
    positive, the sum is deterministic under the stated sub-Gaussian proxy and
    the failure probability is zero.
    """
    mu = np.asarray(mean_margin, dtype=float)
    sigma = np.asarray(covariance_proxy, dtype=float)
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("mean_margin must be a non-empty vector")
    if sigma.shape != (mu.size, mu.size):
        raise ValueError("covariance_proxy shape mismatch")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("finite inputs required")
    if not np.allclose(sigma, sigma.T, atol=1e-12, rtol=0.0):
        raise ValueError("covariance_proxy must be symmetric")
    if float(np.min(np.linalg.eigvalsh(sigma))) < -1e-10:
        raise ValueError("covariance_proxy must be positive semidefinite")

    ones = np.ones(mu.size, dtype=float)
    mean_sum = float(ones @ mu)
    var_proxy = float(ones @ sigma @ ones)
    if mean_sum <= 0.0:
        return 1.0
    if var_proxy < -1e-10:
        raise ValueError("covariance proxy gives a negative aggregate variance")
    if var_proxy <= 1e-15:
        return 0.0
    exponent = -(mean_sum * mean_sum) / (2.0 * var_proxy)
    return float(min(1.0, math.exp(exponent)))


def subgaussian_consensus_top1_failure_bound(
    mean_margins: Sequence[Sequence[float]],
    covariance_proxies: Sequence[Sequence[Sequence[float]]],
) -> float:
    """Union-bound probability that any supplied decoy beats the truth.

    Each entry describes a truth-minus-decoy margin vector and its joint
    sub-Gaussian covariance proxy. No independence between different decoys is
    required for the union bound.
    """
    if len(mean_margins) != len(covariance_proxies):
        raise ValueError("one covariance proxy is required per decoy")
    total = 0.0
    for mu, sigma in zip(mean_margins, covariance_proxies):
        total += subgaussian_consensus_pair_failure_bound(mu, sigma)
    return float(min(1.0, total))


def iid_consensus_top1_failure_bound(
    *,
    n_candidates: int,
    n_conditions: int,
    mean_margin: float,
    sigma: float,
) -> float:
    """Simplified union bound for iid sub-Gaussian per-condition margins.

    With N total candidates, m conditions, positive truth-minus-decoy mean
    margin mu, and per-condition sub-Gaussian scale sigma,

        P(truth is not Top-1) <= (N-1) exp(-m mu^2/(2 sigma^2)).

    This is a conditional statistical theorem, not an unconditional guarantee
    about molecular identification. Its usefulness depends on positive margins
    and on the stated sub-Gaussian model being appropriate to the deployment
    distribution.
    """
    n_candidates = int(n_candidates)
    n_conditions = int(n_conditions)
    mean_margin = float(mean_margin)
    sigma = float(sigma)
    if n_candidates < 1 or n_conditions < 1:
        raise ValueError("candidate and condition counts must be positive")
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    if not math.isfinite(mean_margin):
        raise ValueError("mean_margin must be finite")
    if n_candidates == 1:
        return 0.0
    if mean_margin <= 0.0:
        return 1.0
    bound = (n_candidates - 1) * math.exp(
        -n_conditions * mean_margin * mean_margin / (2.0 * sigma * sigma)
    )
    return float(min(1.0, bound))


def bounded_correction_order_is_protected(
    baseline_gap: float,
    correction_amplitude: float,
) -> bool:
    """Whether a pairwise baseline order is guaranteed under bounded fusion.

    For scores B_i + alpha h_i with h_i in [-1,1], candidate i cannot be
    reordered below j whenever B_i-B_j > 2*alpha. Equality is intentionally
    not declared protected because the worst-case correction can create a tie.
    """
    gap = float(baseline_gap)
    alpha = float(correction_amplitude)
    if not math.isfinite(gap) or not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("finite gap and nonnegative correction amplitude required")
    return bool(gap > 2.0 * alpha)


def selective_utility_gain(
    gate_probability: float,
    conditional_expert_minus_baseline_utility: float,
) -> float:
    """Exact expected-gain identity for a binary selective expert gate.

    If G is the event that the expert is used and the baseline is used on G^c,

        E[U_selective-U_baseline]
          = P(G) E[U_expert-U_baseline | G].

    The identity by itself does not imply a positive gain: the conditional
    expert advantage on the gate-selected region must itself be positive.
    """
    p = float(gate_probability)
    delta = float(conditional_expert_minus_baseline_utility)
    if not 0.0 <= p <= 1.0 or not math.isfinite(delta):
        raise ValueError("gate probability must lie in [0,1] and utility must be finite")
    return float(p * delta)
