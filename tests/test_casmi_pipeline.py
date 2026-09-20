import importlib.util
import math
import numpy as np
import pytest

from qfd.benchmark import rank_library
from qfd.casmi_schema import is_merged_energy, normalize_energy_list, reconstruct_raw_intensities, validate_peak_arrays
from qfd.metrics import reciprocal_rank_at_k, mrr_at_k, rank_by_score
from qfd.splits import assert_no_structure_leakage, structure_holdout
from qfd.spectra import match_peaks, matched_cosine, preprocess_spectrum
from qfd.trajectory import SpectrumView, joint_energy_score


def test_energy_list_and_merge_detection():
    assert normalize_energy_list([20, 40]) == (20.0, 40.0)
    assert normalize_energy_list(30) == (30.0,)
    assert normalize_energy_list(None) == ()
    assert is_merged_energy([20, 40])
    assert not is_merged_energy([20])


def test_reconstruct_raw_intensities():
    got = reconstruct_raw_intensities([1.0, 0.25, 0.01], 4000)
    np.testing.assert_allclose(got, [4000, 1000, 40])
    assert reconstruct_raw_intensities([1.0], None) is None


def test_validate_peak_arrays_rejects_mismatch():
    with pytest.raises(ValueError):
        validate_peak_arrays([1, 2], [1])


def test_rr_and_mrr25_hand_verified():
    assert reciprocal_rank_at_k(["b", "a", "c"], "a", 25) == 0.5
    assert reciprocal_rank_at_k([str(i) for i in range(30)], "26", 25) == 0.0
    assert mrr_at_k([["a"], ["x", "b"], ["x"]], ["a", "b", "c"], 25) == 0.5
    assert rank_by_score(["a", "b", "c"], [0.1, 0.9, 0.2]) == ["b", "c", "a"]


def test_structure_holdout_is_disjoint_and_deterministic():
    ids = [f"KEY{i:03d}" for i in range(100)]
    a, b = structure_holdout(ids, val_fraction=0.2, seed=7)
    a2, b2 = structure_holdout(ids, val_fraction=0.2, seed=7)
    assert (a, b) == (a2, b2)
    assert a | b == set(ids)
    assert_no_structure_leakage(a, b)


def test_peak_matching_and_cosine():
    a = np.array([50.0, 100.0, 150.0])
    b = np.array([50.005, 100.008, 180.0])
    assert match_peaks(a, b, tolerance_da=0.01) == [(0, 0), (1, 1)]
    score = matched_cosine(a, [1, 1, 0], b, [1, 1, 0], tolerance_da=0.01)
    assert abs(score - 1.0) < 1e-12


def test_preprocess_filters_precursor_and_caps():
    mz, it = preprocess_spectrum([10, 20, 30, 120], [0.01, 1, 0.5, 1], precursor_mz=100, intensity_floor=0.02, max_peaks=2, intensity_transform="none")
    np.testing.assert_allclose(mz, [20, 30])
    assert abs(np.linalg.norm(it) - 1.0) < 1e-12


def _s(mz, inten, e):
    return SpectrumView.make(mz, inten, [e], adduct="[M+H]+", base_peak_intensity=10000)


def test_joint_energy_score_prefers_condition_consistent_candidate():
    q = [_s([50, 100], [1, 0], 20), _s([50, 100], [0, 1], 40)]
    correct = [_s([50, 100], [1, 0], 20), _s([50, 100], [0, 1], 40)]
    swapped = [_s([50, 100], [0, 1], 20), _s([50, 100], [1, 0], 40)]
    a = joint_energy_score(q, correct, max_energy_gap_ev=0)
    b = joint_energy_score(q, swapped, max_energy_gap_ev=0)
    assert a > b
    assert math.isclose(a, 1.0)


def test_library_ranking_exposes_dynamic_advantage():
    q = [_s([50, 100], [1, 0], 20), _s([50, 100], [0, 1], 40)]
    library = {
        "correct": [_s([50, 100], [1, 0], 20), _s([50, 100], [0, 1], 40)],
        "static-decoy": [_s([50, 100], [1, 0], 20), _s([50, 100], [1, 0], 40)],
    }
    ranked, scores = rank_library(q, library, mode="joint_energy", max_energy_gap_ev=0)
    assert ranked[0] == "correct"
    assert scores["correct"] > scores["static-decoy"]


@pytest.mark.skipif(importlib.util.find_spec("polars") is None or importlib.util.find_spec("pyarrow") is None, reason="optional parquet deps unavailable")
def test_scan_train_filters_ingest_lib(tmp_path):
    import polars as pl
    from qfd.data import scan_train

    path = tmp_path / "tiny.parquet"
    pl.DataFrame({"spectrum_id": ["s1", "s2"], "ingest_lib": ["enveda-180", "other"]}).write_parquet(path)
    got = scan_train(path, columns=["spectrum_id", "ingest_lib"], ingest_lib="enveda-180").collect()
    assert got["spectrum_id"].to_list() == ["s1"]
