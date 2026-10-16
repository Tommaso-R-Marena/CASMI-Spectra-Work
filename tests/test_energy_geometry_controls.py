import numpy as np

from qfd.energy_geometry import hellinger_geodesic_distribution
from qfd.energy_geometry_controls import (
    geometry_control_scores,
    hellinger_chord_distribution,
    matched_coefficient_probability_mixture,
    orthogonal_warp_probability_mixture,
)
from qfd.energy_interpolation import _bhattacharyya


def test_control_distributions_normalize_and_keep_nonnegative():
    p0 = np.array([0.8, 0.2, 0.0])
    p1 = np.array([0.1, 0.4, 0.5])
    for fn in [hellinger_chord_distribution]:
        p = fn(p0, p1, 0.3)
        assert np.all(p >= 0)
        assert np.isclose(p.sum(), 1.0)
    for fn in [matched_coefficient_probability_mixture, orthogonal_warp_probability_mixture]:
        p, w = fn(p0, p1, 0.3)
        assert np.all(p >= 0)
        assert np.isclose(p.sum(), 1.0)
        assert 0.0 <= w <= 1.0


def test_oracles_dominate_corresponding_fixed_points():
    q = np.array([0.7, 0.2, 0.1])
    p0 = np.array([0.9, 0.1, 0.0])
    p1 = np.array([0.1, 0.3, 0.6])
    t = 0.25
    s = geometry_control_scores(q, p0, p1, t)
    fixed_linear = _bhattacharyya(q, (1.0 - t) * p0 + t * p1)
    fixed_geo = _bhattacharyya(q, hellinger_geodesic_distribution(p0, p1, t))
    assert s.oracle_probability_mixture + 1e-10 >= fixed_linear
    assert s.oracle_geodesic + 1e-10 >= fixed_geo


def test_endpoints_are_recovered():
    p0 = np.array([0.75, 0.25])
    p1 = np.array([0.2, 0.8])
    for t, expected in [(0.0, p0), (1.0, p1)]:
        assert np.allclose(hellinger_chord_distribution(p0, p1, t), expected)
        pc, _ = matched_coefficient_probability_mixture(p0, p1, t)
        po, _ = orthogonal_warp_probability_mixture(p0, p1, t)
        assert np.allclose(pc, expected)
        assert np.allclose(po, expected)
