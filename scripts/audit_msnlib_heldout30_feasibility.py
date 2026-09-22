from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _unambiguous_groups, index_records
from qfd.stats import stable_group_is_validation


TARGET_EV = 30.0
LOW_EV = 20.0
HIGH_EV = 60.0
SPLIT_SALT = "msnlib-heldout30-formula-v1"
VALIDATION_FRACTION = 0.5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "queries": 0,
            "formula_blocks": 0,
            "queries_pool_ge_5": 0,
            "queries_pool_ge_10": 0,
            "pool_size_min": 0,
            "pool_size_median": 0.0,
            "pool_size_mean": 0.0,
            "pool_size_max": 0,
        }
    pools = np.asarray([int(r["candidate_count"]) for r in rows], dtype=int)
    return {
        "queries": int(len(rows)),
        "formula_blocks": int(len({str(r["formula"]) for r in rows})),
        "queries_pool_ge_5": int(np.sum(pools >= 5)),
        "queries_pool_ge_10": int(np.sum(pools >= 10)),
        "pool_size_min": int(np.min(pools)),
        "pool_size_median": float(np.median(pools)),
        "pool_size_mean": float(np.mean(pools)),
        "pool_size_max": int(np.max(pools)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Label-free feasibility audit for the predeclared 30 eV heldout-energy task.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    ref_eligible = {
        skey: g
        for skey, g in groups.items()
        if LOW_EV in g["same_energy"] and HIGH_EV in g["same_energy"]
    }
    by_formula_adduct: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for skey, g in ref_eligible.items():
        by_formula_adduct[(str(g["formula"]), str(skey[1]))].append(skey)

    rows: list[dict[str, object]] = []
    for truth, g in sorted(ref_eligible.items()):
        if TARGET_EV not in g["single_best"]:
            continue
        formula = str(g["formula"])
        candidates = by_formula_adduct[(formula, str(truth[1]))]
        if len(candidates) < 2:
            continue
        split = (
            "validation"
            if stable_group_is_validation(
                formula,
                validation_fraction=VALIDATION_FRACTION,
                salt=SPLIT_SALT,
            )
            else "discovery"
        )
        rows.append({
            "formula": formula,
            "split": split,
            "candidate_count": int(len(candidates)),
        })

    discovery = [r for r in rows if r["split"] == "discovery"]
    validation = [r for r in rows if r["split"] == "validation"]
    if {str(r["formula"]) for r in discovery} & {str(r["formula"]) for r in validation}:
        raise RuntimeError("formula leakage across heldout-energy split")

    summary = {
        "status": "MSNLIB_HELDOUT30_LABEL_FREE_FEASIBILITY",
        "dataset": str(args.dataset),
        "input": {"path": str(args.jsonl), "sha256": _sha256(args.jsonl), "records": int(len(records))},
        "protocol": {
            "query": "SINGLE_BEST_SCAN at exactly 30 eV",
            "candidate_references": "SAME_ENERGY at exactly 20 and 60 eV only; 30 eV is withheld from every candidate reference",
            "candidate_pool": "same molecular formula and same adduct",
            "identity": "full InChIKey + adduct",
            "split_unit": "molecular formula",
            "split_salt": SPLIT_SALT,
            "validation_fraction": VALIDATION_FRACTION,
            "ranking_scores_computed": False,
            "truth_ranks_computed": False,
        },
        "coverage": {
            "indexed_full_identity_groups": int(len(indexed)),
            "unambiguous_full_identity_groups": int(len(groups)),
            "endpoint_reference_eligible_groups": int(len(ref_eligible)),
            "all": _summarize(rows),
            "discovery": _summarize(discovery),
            "validation": _summarize(validation),
        },
        "predeclared_gate": {
            "validation_queries_min": 300,
            "validation_formula_blocks_min": 100,
            "validation_queries_pool_ge_5_min": 100,
            "rationale": (
                "Primary heldout-energy inference requires a large formula-disjoint validation cohort and enough "
                "candidate pools with at least five structures to avoid another saturated two-candidate test."
            ),
        },
        "guardrail": (
            "No spectrum similarity, interpolation score, truth rank, selector output, or efficacy metric is computed "
            "in this pass. The 30 eV rankings may be opened only if every validation gate is satisfied."
        ),
    }
    v = summary["coverage"]["validation"]
    g = summary["predeclared_gate"]
    summary["gate_pass"] = bool(
        v["queries"] >= g["validation_queries_min"]
        and v["formula_blocks"] >= g["validation_formula_blocks_min"]
        and v["queries_pool_ge_5"] >= g["validation_queries_pool_ge_5_min"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
