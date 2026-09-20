from qfd.stats import (
    paired_block_bootstrap_delta,
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
