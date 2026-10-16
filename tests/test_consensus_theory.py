import math

import numpy as np
import pytest

from qfd.consensus_theory import (
    bounded_correction_order_is_protected,
    iid_consensus_top1_failure_bound,
    selective_utility_gain,
    subgaussian_consensus_pair_failure_bound,
    subgaussian_consensus_top1_failure_bound,
)


def test_pair_bound_matches_iid_single_decoy_formula():
    m = 4
    mu = 0.6
    sigma = 1.3
    pair = subgaussian_consensus_pair_failure_bound(
        [mu] * m,
        np.eye(m) * sigma**2,
    )
    expected = math.exp(-m * mu**2 / (2.0 * sigma**2))
    assert pair == pytest.approx(expected, rel=1e-12)


def test_iid_bound_improves_with_more_independent_conditions():
    b1 = iid_consensus_top1_failure_bound(
        n_candidates=20, n_conditions=1, mean_margin=1.5, sigma=1.0
    )
    b3 = iid_consensus_top1_failure_bound(
        n_candidates=20, n_conditions=3, mean_margin=1.5, sigma=1.0
    )
    b8 = iid_consensus_top1_failure_bound(
        n_candidates=20, n_conditions=8, mean_margin=1.5, sigma=1.0
    )
    assert b8 < b3 < b1


def test_positive_cross_energy_correlation_weakens_consensus_bound():
    mu = [0.8, 0.8, 0.8]
    independent = np.eye(3)
    correlated = np.full((3, 3), 0.7)
    np.fill_diagonal(correlated, 1.0)
    b_ind = subgaussian_consensus_pair_failure_bound(mu, independent)
    b_corr = subgaussian_consensus_pair_failure_bound(mu, correlated)
    assert b_corr > b_ind


def test_nonpositive_margin_is_explicitly_vacuous():
    assert subgaussian_consensus_pair_failure_bound(
        [-0.1, 0.0, 0.05], np.eye(3)
    ) == 1.0
    assert iid_consensus_top1_failure_bound(
        n_candidates=100, n_conditions=3, mean_margin=0.0, sigma=1.0
    ) == 1.0


def test_zero_variance_positive_margin_is_deterministically_safe_under_assumption():
    assert subgaussian_consensus_pair_failure_bound(
        [0.2, 0.3], np.zeros((2, 2))
    ) == 0.0


def test_top1_union_bound_sums_and_clips_without_decoy_independence():
    means = [[1.0, 1.0], [0.8, 0.8]]
    covs = [np.eye(2), np.eye(2)]
    pair_sum = sum(
        subgaussian_consensus_pair_failure_bound(m, c)
        for m, c in zip(means, covs)
    )
    top1 = subgaussian_consensus_top1_failure_bound(means, covs)
    assert top1 == pytest.approx(min(1.0, pair_sum))


def test_invalid_covariance_proxy_is_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        subgaussian_consensus_pair_failure_bound(
            [1.0, 1.0], [[1.0, 0.5], [0.0, 1.0]]
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        subgaussian_consensus_pair_failure_bound(
            [1.0, 1.0], [[1.0, 2.0], [2.0, 1.0]]
        )


def test_bounded_correction_protection_is_strict_at_two_alpha():
    assert bounded_correction_order_is_protected(1.000001, 0.5)
    assert not bounded_correction_order_is_protected(1.0, 0.5)
    assert not bounded_correction_order_is_protected(0.999999, 0.5)


def test_selective_utility_identity_preserves_sign():
    assert selective_utility_gain(0.4, 0.25) == pytest.approx(0.1)
    assert selective_utility_gain(0.4, -0.25) == pytest.approx(-0.1)
    assert selective_utility_gain(0.0, 100.0) == 0.0
