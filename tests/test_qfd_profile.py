from qfd.qfd_profile import two_energy_qfd_score
from qfd.trajectory import SpectrumView


def S(inten, e):
    return SpectrumView.make([50, 100, 150], inten, [e], adduct="[M+H]+")


def test_two_energy_qfd_prefers_matching_relative_dynamics():
    query = [S([1.0, 0.5, 0.25], 20), S([0.5, 1.0, 0.25], 40)]
    correct = [S([7.0, 3.5, 1.75], 20), S([11.0, 22.0, 5.5], 40)]
    decoy = [S([1.0, 0.5, 0.25], 20), S([1.0, 0.5, 0.25], 40)]
    assert two_energy_qfd_score(query, correct, max_energy_gap_ev=0) > two_energy_qfd_score(query, decoy, max_energy_gap_ev=0)
