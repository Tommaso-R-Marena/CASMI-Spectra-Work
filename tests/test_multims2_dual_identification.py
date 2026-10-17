"""Guardrails for the exploratory CID/EAD isomer identification experiment."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_multims2_dual_identification_gate import build, unique
from benchmark_multims2_dual_identification import (
    rank_scores, predictor, candidate_affinity
)


def item(key, formula="C2H4O2", kind="CID", start=0, adduct="[M+H]+"):
    low, middle, high = (20, 40, 60) if kind == "CID" else (12, 16, 24)
    return {
        "inchikey": key, "formula": formula, "adduct": adduct,
        "ion_mode": "POSITIVE", "source_dataset_id": "L1",
        "fragmentation": kind, "low_index": start,
        "query_index": start+1, "high_index": start+2,
        "low_energy_eV": low, "query_energy_eV": middle, "high_energy_eV": high,
        "t": (middle-low)/(high-low),
    }


def spec(masses, weights):
    return SimpleNamespace(
        mzs=np.asarray(masses, dtype=float),
        intensities=np.asarray(weights, dtype=float),
    )


def test_gate_requires_actual_distinct_isomers_not_singletons():
    cid = [item("FULLKEY-A", start=0), item("FULLKEY-B", start=3),
           item("SINGLE-C", formula="C3H6O2", start=6)]
    ead = [item("FULLKEY-A", kind="EAD", start=10),
           item("FULLKEY-B", kind="EAD", start=13),
           item("SINGLE-C", formula="C3H6O2", kind="EAD", start=16)]
    rows, cov = build(cid, ead)
    assert len(rows) == 2
    assert cov["same_formula_isomer_groups"] == 1
    assert cov["eligible_isomer_queries"] == 2
    assert cov["all_matched_singletons_excluded_by_label_free_rule"] == 1
    assert all(row["candidate_count"] == 2 for row in rows)
    assert {r["inchikey"] for r in rows} == {"FULLKEY-A", "FULLKEY-B"}


def test_cannot_match_on_truncated_inchikey_or_wrong_adduct():
    cid = [item("AAAAAAAAAAAAAA-AAAAAAA-1", start=0),
           item("AAAAAAAAAAAAAA-BBBBBBB-2", start=3)]
    ead = [item("AAAAAAAAAAAAAA-AAAAAAA-1", kind="EAD", start=10),
           item("AAAAAAAAAAAAAA-BBBBBBB-2", kind="EAD", start=13, adduct="[M+Na]+")]
    rows, cov = build(cid, ead)
    assert rows == []
    assert cov["exact_identity_matched"] == 1
    assert cov["eligible_isomer_queries"] == 0


def test_duplicate_full_structure_or_reused_scans_abort():
    x = item("FULLKEY-A", start=0)
    with pytest.raises(RuntimeError):
        unique([x, x], "CID")
    with pytest.raises(RuntimeError):
        unique([x, item("FULLKEY-B", start=1)], "CID")


def test_no_duplicate_or_wrong_energy_accepted():
    a = item("FULLKEY-A", start=0)
    b = item("FULLKEY-B", start=3)
    a_e = item("FULLKEY-A", kind="EAD", start=10)
    b_e = item("FULLKEY-B", kind="EAD", start=13)
    b_e["t"] = .5
    with pytest.raises(RuntimeError):
        build([a, b], [a_e, b_e])


def test_tie_aware_rank_and_fractional_top1():
    top = rank_scores({"a": .7, "b": .7, "c": .2}, "a")
    assert top["top1_fractional"] == .5
    assert top["reciprocal_rank"] == pytest.approx(2/3)
    second = rank_scores({"a": .8, "b": .6}, "b")
    assert second["top1_fractional"] == 0.0
    assert second["reciprocal_rank"] == .5


def test_bad_scores_and_missing_competitor_fail_closed():
    with pytest.raises(ValueError):
        rank_scores({"a": 1.0}, "a")
    with pytest.raises(ValueError):
        rank_scores({"a": .5, "b": float("nan")}, "a")
    with pytest.raises(ValueError):
        rank_scores({"a": .5, "b": .2}, "none")


def test_prediction_grid_never_depends_on_heldout_query():
    lo = spec([100.0, 120.0], [8, 2])
    hi = spec([100.002, 130.0], [2, 8])
    pr = predictor(lo, hi, .5, .5)
    initial_centers = pr[0].mz.copy()
    initial_probability = pr[1].copy()
    score_one = candidate_affinity(spec([100.0, 200.0], [8, 2]), pr)
    score_two = candidate_affinity(spec([100.0, 210.0], [8, 2]), pr)
    assert score_one == pytest.approx(score_two)
    assert np.array_equal(initial_centers, pr[0].mz)
    assert np.array_equal(initial_probability, pr[1])


def test_second_modality_can_resolve_cid_tie_without_fitting():
    cid_same = spec([100.0, 200.0], [5, 5])
    ead_a = spec([110.0, 210.0], [9, 1])
    ead_b = spec([110.0, 210.0], [1, 9])
    cid_a = predictor(cid_same, cid_same, .5, .5)
    cid_b = predictor(cid_same, cid_same, .5, .5)
    ea = predictor(ead_a, ead_a, 1/3, .5)
    eb = predictor(ead_b, ead_b, 1/3, .5)
    target_cid = cid_same
    target_ead = ead_a
    cid = {"A": candidate_affinity(target_cid, cid_a),
           "B": candidate_affinity(target_cid, cid_b)}
    fused = {
        "A": .5*(cid["A"]+candidate_affinity(target_ead, ea)),
        "B": .5*(cid["B"]+candidate_affinity(target_ead, eb))
    }
    assert rank_scores(cid, "A")["top1_fractional"] == .5
    assert rank_scores(fused, "A")["top1_fractional"] == 1.0
