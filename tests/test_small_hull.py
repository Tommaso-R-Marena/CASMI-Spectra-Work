import numpy as np

from qfd.cone import convex_hull_distance
from qfd.small_hull import convex_hull_distance_active_set


def test_small_hull_projects_inside_triangle_exactly():
    C = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0.25, 0.25])
    d, w = convex_hull_distance_active_set(y, C)
    assert d < 1e-12
    assert np.all(w >= -1e-12)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.allclose(C @ w, y, atol=1e-12)


def test_small_hull_projects_to_edge():
    C = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0.7, -0.2])
    d, w = convex_hull_distance_active_set(y, C)
    assert np.all(w >= -1e-12)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.allclose(C @ w, [0.7, 0.0], atol=1e-10)
    assert abs(d - 0.2) < 1e-10


def test_small_hull_matches_slsqp_on_random_three_vertex_cases():
    rng = np.random.default_rng(20260920)
    for _ in range(25):
        C = rng.normal(size=(20, 3))
        y = rng.normal(size=20)
        d_active, w = convex_hull_distance_active_set(y, C)
        d_slsqp, _ = convex_hull_distance(y, C)
        assert np.all(w >= -1e-10)
        assert abs(w.sum() - 1.0) < 1e-10
        assert abs(d_active - d_slsqp) < 2e-6
