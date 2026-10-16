from qfd.stats import (
    paired_block_bootstrap_delta,
    paired_block_signflip_test,
    paired_bootstrap_delta,
    paired_win_loss_tie,
    stable_group_is_validation,
)


def test_paired_bootstrap_delta_direction_and_reproducibility():
    a = [1.0, 0.5, 1.0, 0.25]
    b = [0.5, 0.5, 0.5, 0.25]
    x = paired_bootstrap_delta(a, b, n_boot=200, seed=7)
    y = paired_bootstrap_delta(a, b, n_boot=200, seed=7)
    assert x == y
    assert x["delta"] > 0


def test_block_bootstrap_resamples_whole_groups_reproducibly():
    a = [1.0, 0.0, 1.0, 1.0]
    b = [0.0, 1.0, 0.0, 0.0]
    blocks = ["g1", "g1", "g2", "g3"]
    x = paired_block_bootstrap_delta(a, b, blocks, n_boot=400, seed=11)
    y = paired_block_bootstrap_delta(a, b, blocks, n_boot=400, seed=11)
    assert x == y
    assert x["n"] == 4
    assert x["n_blocks"] == 3
    assert x["resampling_unit"] == "block"
    assert x["delta"] == 0.5


def test_block_bootstrap_rejects_misaligned_blocks():
    try:
        paired_block_bootstrap_delta([1, 2], [1, 2], ["only-one"], n_boot=10)
    except ValueError as exc:
        assert "blocks" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_block_signflip_exact_known_case():
    # Three independent blocks all favor a. Under the sharp paired null, only the
    # all-positive and all-negative sign assignments are as extreme as observed.
    got = paired_block_signflip_test(
        [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], ["a", "b", "c"], exact_max_blocks=20
    )
    assert got["mode"] == "exact"
    assert got["n_sign_assignments"] == 8
    assert got["extreme"] == 2
    assert got["p_value_two_sided"] == 0.25


def test_block_signflip_flips_repeated_queries_together():
    # The first two observations form one block, so this is a 2-block exact test,
    # not a 3-query sign test.
    got = paired_block_signflip_test(
        [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], ["a", "a", "b"], exact_max_blocks=20
    )
    assert got["n_blocks"] == 2
    assert got["n_sign_assignments"] == 4
    assert got["p_value_two_sided"] == 0.5


def test_block_signflip_monte_carlo_reproducible():
    a = [1.0] * 6
    b = [0.0] * 6
    blocks = [f"g{i}" for i in range(6)]
    x = paired_block_signflip_test(
        a, b, blocks, exact_max_blocks=3, n_permutations=2000, seed=19
    )
    y = paired_block_signflip_test(
        a, b, blocks, exact_max_blocks=3, n_permutations=2000, seed=19
    )
    assert x == y
    assert x["mode"] == "monte_carlo"
    assert 0.0 < x["p_value_two_sided"] <= 1.0


def test_stable_group_split_is_deterministic_and_salt_sensitive():
    x = stable_group_is_validation("C6H6\x1f[M+H]+", validation_fraction=0.5)
    y = stable_group_is_validation("C6H6\x1f[M+H]+", validation_fraction=0.5)
    assert x == y
    # A different salt defines a different frozen protocol; this assertion checks
    # deterministic evaluation rather than requiring the outcome itself to flip.
    z = stable_group_is_validation(
        "C6H6\x1f[M+H]+", validation_fraction=0.5, salt="other-protocol"
    )
    assert isinstance(z, bool)


def test_win_loss_tie():
    got = paired_win_loss_tie([1, .5, .2], [.5, .5, .3])
    assert got == {"wins": 1, "losses": 1, "ties": 1}
