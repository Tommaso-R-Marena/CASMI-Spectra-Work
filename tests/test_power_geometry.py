import numpy as np

from qfd.energy_geometry_controls import hellinger_chord_distribution
from qfd.power_geometry import (
    logit_energy_warp,
    power_mean_distribution,
    power_mean_score,
)


def test_power_family_normalizes_and_recovers_endpoints():
    p0 = np.array([0.8, 0.2, 0.0])
    p1 = np.array([0.1, 0.4, 0.5])
    for alpha in [0.25, 0.5, 1.0, 1.25]:
        mid = power_mean_distribution(p0, p1, 0.3, alpha)
        assert np.all(mid >= 0)
        assert np.isclose(mid.sum(), 1.0)
        assert np.allclose(power_mean_distribution(p0, p1, 0.0, alpha), p0)
        assert np.allclose(power_mean_distribution(p0, p1, 1.0, alpha), p1)


def test_alpha_one_is_probability_linear():
    p0 = np.array([0.75, 0.25, 0.0])
    p1 = np.array([0.15, 0.35, 0.5])
    t = 0.37
    expected = (1.0 - t) * p0 + t * p1
    assert np.allclose(power_mean_distribution(p0, p1, t, 1.0), expected)


def test_alpha_half_is_hellinger_chord():
    p0 = np.array([0.75, 0.25, 0.0])
    p1 = np.array([0.15, 0.35, 0.5])
    for t in [0.1, 0.37, 0.5, 0.9]:
        got = power_mean_distribution(p0, p1, t, 0.5)
        expected = hellinger_chord_distribution(p0, p1, t)
        assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_identity_logit_warp_and_monotonicity():
    ts = np.linspace(0.01, 0.99, 99)
    identity = np.asarray([logit_energy_warp(float(t), 1.0, 0.0) for t in ts])
    assert np.allclose(identity, ts)
    warped = np.asarray([logit_energy_warp(float(t), 1.7, -0.2) for t in ts])
    assert np.all(np.diff(warped) > 0)


def test_score_is_bounded():
    q = np.array([0.5, 0.3, 0.2])
    p0 = np.array([0.8, 0.2, 0.0])
    p1 = np.array([0.1, 0.4, 0.5])
    s = power_mean_score(q, p0, p1, 0.3, 0.5)
    assert 0.0 <= s <= 1.0 + 1e-12
