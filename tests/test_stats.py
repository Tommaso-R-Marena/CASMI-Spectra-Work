from qfd.stats import paired_bootstrap_delta, paired_win_loss_tie


def test_paired_bootstrap_delta_direction_and_reproducibility():
    a = [1.0, 0.5, 1.0, 0.25]
    b = [0.5, 0.5, 0.5, 0.25]
    x = paired_bootstrap_delta(a, b, n_boot=200, seed=7)
    y = paired_bootstrap_delta(a, b, n_boot=200, seed=7)
    assert x == y
    assert x["delta"] > 0


def test_win_loss_tie():
    got = paired_win_loss_tie([1, .5, .2], [.5, .5, .3])
    assert got == {"wins": 1, "losses": 1, "ties": 1}
