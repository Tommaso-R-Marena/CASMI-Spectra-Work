import numpy as np

from qfd.energy_geometry import hellinger_geodesic_distribution, energy_prediction_scores_strict
from qfd.energy_interpolation import _common_grid, _common_grid_strict
from qfd.trajectory import SpectrumView


def _v(mz, it, e):
    return SpectrumView.make(mz, it, [e], adduct='[M+H]+')


def test_hellinger_geodesic_endpoints_and_normalization():
    p0 = np.array([0.8, 0.2, 0.0])
    p1 = np.array([0.1, 0.3, 0.6])
    a = hellinger_geodesic_distribution(p0, p1, 0.0)
    b = hellinger_geodesic_distribution(p0, p1, 1.0)
    mid = hellinger_geodesic_distribution(p0, p1, 0.5)
    assert np.allclose(a, p0)
    assert np.allclose(b, p1)
    assert np.isclose(mid.sum(), 1.0)
    assert np.all(mid >= 0)


def test_strict_grid_blocks_transitive_chaining():
    q = _v([100.000], [1.0], 30)
    lo = _v([100.009], [1.0], 20)
    hi = _v([100.018], [1.0], 60)
    historical = _common_grid([q, lo, hi], 0.01)
    strict = _common_grid_strict([q, lo, hi], 0.01)
    assert historical.shape[0] == 1
    assert strict.shape[0] == 2


def test_energy_prediction_scores_use_energy_fraction_and_are_finite():
    q = _v([50.0, 75.0], [0.75, 0.25], 30)
    lo = _v([50.0, 75.0], [1.0, 0.0], 20)
    hi = _v([50.0, 75.0], [0.0, 1.0], 60)
    s = energy_prediction_scores_strict(q, lo, hi)
    assert np.isclose(s.t, 0.25)
    assert 0 <= s.best_endpoint <= 1
    assert 0 <= s.linear <= 1
    assert 0 <= s.equal_mix <= 1
    assert 0 <= s.reversed_energy <= 1
    assert 0 <= s.hellinger_geodesic <= 1
    assert s.linear > s.reversed_energy


def test_geodesic_is_symmetric_under_endpoint_swap_with_t_swap():
    q = _v([10.0, 20.0, 30.0], [0.2, 0.5, 0.3], 30)
    lo = _v([10.0, 20.0, 30.0], [0.8, 0.1, 0.1], 20)
    hi = _v([10.0, 20.0, 30.0], [0.1, 0.2, 0.7], 60)
    a = energy_prediction_scores_strict(q, lo, hi).hellinger_geodesic

    # Direct distribution-level endpoint-swap symmetry.
    p0 = np.array([0.8, 0.1, 0.1])
    p1 = np.array([0.1, 0.2, 0.7])
    x = hellinger_geodesic_distribution(p0, p1, 0.25)
    y = hellinger_geodesic_distribution(p1, p0, 0.75)
    assert np.allclose(x, y)
    assert np.isfinite(a)
