import numpy as np

from qfd.fragment_matrix import align_fragment_matrix, centered_log_fold_change
from qfd.quotient import cycle_rank
from qfd.trajectory import SpectrumView


def S(mz, inten, e):
    return SpectrumView.make(mz, inten, [e], adduct="[M+H]+")


def test_alignment_preserves_missingness_and_cycle_rank():
    specs = [
        S([50.000, 100.000], [1.0, 0.5], 20),
        S([50.004, 100.006, 150.0], [0.5, 1.0, 0.2], 40),
        S([50.002, 150.005], [0.25, 1.0], 60),
    ]
    M = align_fragment_matrix(specs, tolerance_da=0.01, min_presence=2)
    assert M.intensities.shape == (3, 3)
    assert np.isnan(M.intensities[1, 2])
    assert np.isnan(M.intensities[2, 0])
    assert cycle_rank(M.mask) == 2


def test_two_energy_centered_log_fold_change_cancels_global_gain():
    a = align_fragment_matrix([
        S([50, 100], [1.0, 2.0], 20),
        S([50, 100], [2.0, 2.0], 40),
    ], min_presence=2)
    b = align_fragment_matrix([
        S([50, 100], [7.0, 14.0], 20),
        S([50, 100], [22.0, 22.0], 40),
    ], min_presence=2)
    _, ra = centered_log_fold_change(a)
    _, rb = centered_log_fold_change(b)
    np.testing.assert_allclose(ra, rb)
