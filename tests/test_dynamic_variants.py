import numpy as np

from qfd.dynamic_variants import geometric_joint_score, minimum_joint_score, multi_energy_delta_score
from qfd.trajectory import SpectrumView


def view(e, vals):
    return SpectrumView.make([50.0, 100.0, 150.0], vals, [float(e)], adduct="[M+H]+")


def test_hellinger_delta_prefers_matching_energy_response():
    query = [
        view(10, [0.8, 0.15, 0.05]),
        view(20, [0.45, 0.45, 0.10]),
        view(30, [0.15, 0.65, 0.20]),
    ]
    true = [
        view(10, [0.78, 0.17, 0.05]),
        view(20, [0.46, 0.43, 0.11]),
        view(30, [0.16, 0.64, 0.20]),
    ]
    decoy = [
        view(10, [0.80, 0.15, 0.05]),
        view(20, [0.80, 0.15, 0.05]),
        view(30, [0.80, 0.15, 0.05]),
    ]
    for mode, kwargs in [("hellinger", {}), ("clr", {"eps": 1e-4})]:
        s_true = multi_energy_delta_score(query, true, mode=mode, **kwargs)
        s_decoy = multi_energy_delta_score(query, decoy, mode=mode, **kwargs)
        assert np.isfinite(s_true)
        assert (not np.isfinite(s_decoy)) or s_true > s_decoy


def test_joint_aggregators_are_bounded():
    query = [view(10, [1, 0, 0]), view(20, [0, 1, 0])]
    ref = [view(10, [1, 0, 0]), view(20, [0, 0.8, 0.2])]
    geom = geometric_joint_score(query, ref)
    minimum = minimum_joint_score(query, ref)
    assert 0 <= geom <= 1
    assert 0 <= minimum <= 1
    assert minimum <= geom
