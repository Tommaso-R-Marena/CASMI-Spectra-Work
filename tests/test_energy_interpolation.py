import numpy as np

from qfd.energy_interpolation import (
    endpoint_static_score,
    energy_interpolation_score,
    equal_mix_control_score,
)
from qfd.trajectory import SpectrumView


def _spec(energy, intensities):
    return SpectrumView.make(
        [100.0, 150.0],
        intensities,
        [float(energy)],
        adduct="[M+H]+",
    )


def test_exact_linear_path_scores_one():
    low = _spec(20, [3.0, 1.0])
    high = _spec(60, [1.0, 3.0])
    # L1 endpoints are (0.75,0.25) and (0.25,0.75); at 30 eV t=0.25,
    # so the linear path is (0.625,0.375).
    query = _spec(30, [5.0, 3.0])
    score = energy_interpolation_score(query, low, high)
    assert np.isclose(score, 1.0, atol=1e-12)
    assert score > endpoint_static_score(query, low, high)


def test_equal_mix_differs_from_energy_constrained_mix():
    low = _spec(20, [9.0, 1.0])
    high = _spec(60, [1.0, 9.0])
    query = _spec(30, [7.0, 3.0])
    constrained = energy_interpolation_score(query, low, high)
    control = equal_mix_control_score(query, low, high)
    assert constrained > control


def test_scores_are_bounded():
    low = _spec(20, [10.0, 0.0])
    high = _spec(60, [0.0, 10.0])
    query = _spec(30, [8.0, 2.0])
    for score in [
        endpoint_static_score(query, low, high),
        energy_interpolation_score(query, low, high),
        equal_mix_control_score(query, low, high),
    ]:
        assert 0.0 <= score <= 1.0
