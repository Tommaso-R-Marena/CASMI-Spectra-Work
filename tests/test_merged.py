from qfd.merged import merged_convex_score
from qfd.trajectory import SpectrumView


def S(ints, energy):
    es = energy if isinstance(energy, list) else [energy]
    return SpectrumView.make([50, 100, 150], ints, es, adduct="[M+H]+")


def test_exact_normalized_mixture_has_near_zero_residual():
    r1 = S([1, 0, 0], 20)
    r2 = S([0, 1, 0], 40)
    q = S([1/3, 1, 0], [20, 40])
    score = merged_convex_score(q, [r1, r2], max_energy_gap_ev=0)
    assert score > -1e-8


def test_wrong_candidate_is_farther_from_convex_hull():
    q = S([.5, 1, 0], [20, 40])
    good = [S([1, 0, 0], 20), S([0, 1, 0], 40)]
    bad = [S([0, 0, 1], 20), S([0, 0, 1], 40)]
    assert merged_convex_score(q, good, max_energy_gap_ev=0) > merged_convex_score(q, bad, max_energy_gap_ev=0)
