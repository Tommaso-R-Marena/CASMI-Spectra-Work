import numpy as np

from qfd.small_hull import convex_hull_distance_active_set
from qfd.trajectory_geometry import ordered_path_projection, segment_projection


def test_segment_projection_interior_and_endpoint():
    y = np.array([0.25, 1.0])
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    d, t = segment_projection(y, a, b)
    assert abs(t - 0.25) < 1e-12
    assert abs(d - 1.0) < 1e-12

    d2, t2 = segment_projection(np.array([-2.0, 0.0]), a, b)
    assert t2 == 0.0
    assert abs(d2 - 2.0) < 1e-12


def test_segment_projection_degenerate_is_deterministic():
    a = np.array([1.0, 2.0])
    d, t = segment_projection(np.array([4.0, 6.0]), a, a)
    assert t == 0.0
    assert abs(d - 5.0) < 1e-12


def test_ordered_path_uses_only_adjacent_segments():
    # Triangle vertices. The point is inside the full triangle but away from
    # the ordered path a->b->c, so full-hull distance is zero while path
    # distance is strictly positive.
    C = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0.25, 0.25])
    hull_d, _ = convex_hull_distance_active_set(y, C)
    path_d, seg, t = ordered_path_projection(y, C)
    assert hull_d < 1e-12
    assert path_d > 0.0
    assert seg in (0, 1)
    assert 0.0 <= t <= 1.0


def test_ordered_path_matches_best_adjacent_two_point_hull():
    rng = np.random.default_rng(20260920)
    for _ in range(40):
        C = rng.normal(size=(25, 3))
        y = rng.normal(size=25)
        d_path, seg, t = ordered_path_projection(y, C)
        d01, _ = convex_hull_distance_active_set(y, C[:, :2])
        d12, _ = convex_hull_distance_active_set(y, C[:, 1:])
        assert abs(d_path - min(d01, d12)) < 1e-10
        assert seg == (0 if d01 <= d12 else 1)
        assert 0.0 <= t <= 1.0
