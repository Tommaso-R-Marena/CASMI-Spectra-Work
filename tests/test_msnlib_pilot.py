import pytest

from qfd.msnlib import record_from_json
from qfd.msnlib_pilot import evaluate_merged_convex, evaluate_same_energy_trajectory, index_records


def rec(key, formula, energy, intensities, merge_type, name="x"):
    obj = {
        "inchikey": f"{key}-UHFFFAOYSA-N",
        "formula": formula,
        "compound_name": name,
        "adduct": "[M+H]+",
        "collision_energy": energy,
        "merge_type": merge_type,
        "precursor_mz": 200.0,
        "usi": [f"mzspec:{key}:{merge_type}:{energy}"],
        "peaks": [[50.0, intensities[0]], [100.0, intensities[1]], [150.0, intensities[2]]],
    }
    return record_from_json(obj)


def test_index_records_rejects_duplicate_single_energy_slots():
    A = "AAAAAAAAAAAAAA"
    a = rec(A, "C6H6", 20.0, [100, 50, 25], "SINGLE_BEST_SCAN")
    b = rec(A, "C6H6", 20.0, [90, 60, 25], "SINGLE_BEST_SCAN")
    with pytest.raises(ValueError, match="duplicate SINGLE_BEST_SCAN"):
        index_records([a, b])


def test_index_records_rejects_formula_inconsistency():
    A = "AAAAAAAAAAAAAA"
    a = rec(A, "C6H6", 20.0, [100, 50, 25], "SINGLE_BEST_SCAN")
    b = rec(A, "C7H8", 40.0, [90, 60, 25], "SINGLE_BEST_SCAN")
    with pytest.raises(ValueError, match="inconsistent formulas"):
        index_records([a, b])


def test_same_energy_trajectory_qfd_uses_dynamics():
    A = "AAAAAAAAAAAAAA"
    B = "BBBBBBBBBBBBBB"
    formula = "C6H6"
    records = [
        rec(A, formula, 20.0, [100, 50, 25], "SINGLE_BEST_SCAN"),
        rec(A, formula, 40.0, [50, 100, 25], "SINGLE_BEST_SCAN"),
        rec(A, formula, 20.0, [200, 100, 50], "SAME_ENERGY"),
        rec(A, formula, 40.0, [75, 150, 37.5], "SAME_ENERGY"),
        rec(B, formula, 20.0, [100, 50, 25], "SINGLE_BEST_SCAN"),
        rec(B, formula, 40.0, [100, 50, 25], "SINGLE_BEST_SCAN"),
        rec(B, formula, 20.0, [200, 100, 50], "SAME_ENERGY"),
        rec(B, formula, 40.0, [200, 100, 50], "SAME_ENERGY"),
    ]
    out = evaluate_same_energy_trajectory(records, hard_static_threshold=0.8)
    rows = {r["inchikey14"]: r for r in out["rows"]}
    assert rows[A]["rank_two_energy_qfd"] == 1
    assert out["all"]["n_queries"] == 2


def test_merged_convex_prefers_true_constituent_family():
    A = "AAAAAAAAAAAAAA"
    B = "BBBBBBBBBBBBBB"
    formula = "C6H6"
    records = [
        rec(A, formula, 20.0, [100, 1, 1], "SINGLE_BEST_SCAN"),
        rec(A, formula, 40.0, [1, 100, 1], "SINGLE_BEST_SCAN"),
        rec(A, formula, [20.0, 40.0], [50, 50, 1], "ALL_ENERGIES"),
        rec(B, formula, 20.0, [1, 1, 100], "SINGLE_BEST_SCAN"),
        rec(B, formula, 40.0, [1, 1, 100], "SINGLE_BEST_SCAN"),
        rec(B, formula, [20.0, 40.0], [1, 1, 100], "ALL_ENERGIES"),
    ]
    out = evaluate_merged_convex(records)
    rows = {r["inchikey14"]: r for r in out["rows"]}
    assert rows[A]["rank_merged_convex"] == 1
    assert out["all"]["n_queries"] == 2
