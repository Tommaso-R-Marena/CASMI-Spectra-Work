from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _unambiguous_groups, index_records


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Label-free feasibility audit for an unseen MSnLib compound collection.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--exclude-jsonl", type=Path, required=True,
                   help="Previously used collection; all overlapping full InChIKeys are removed before feasibility is counted.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude-list-out", type=Path, required=True)
    p.add_argument("--collection", required=True)
    args = p.parse_args()

    target_records = load_msnlib_jsonl(args.jsonl)
    prior_records = load_msnlib_jsonl(args.exclude_jsonl)
    prior_structures = {r.full_inchikey for r in prior_records if r.full_inchikey}
    args.exclude_list_out.parent.mkdir(parents=True, exist_ok=True)
    args.exclude_list_out.write_text("\n".join(sorted(prior_structures)) + "\n")

    indexed = index_records(target_records)
    groups = _unambiguous_groups(indexed)
    target_structures = {key[0] for key in groups}
    overlap_structures = target_structures & prior_structures

    eligible = {}
    for skey, g in groups.items():
        if skey[0] in prior_structures:
            continue
        common = sorted(set(g["single_best"]) & set(g["same_energy"]))
        if len(common) < 2:
            continue
        eligible[skey] = {
            "formula": g["formula"],
            "energies": common,
        }

    by_formula_adduct = defaultdict(list)
    for skey, g in eligible.items():
        by_formula_adduct[(g["formula"], skey[1])].append(skey)

    query_rows = []
    for truth, g in sorted(eligible.items()):
        required = set(g["energies"])
        candidate_keys = [
            cid for cid in by_formula_adduct[(g["formula"], truth[1])]
            if required.issubset(set(eligible[cid]["energies"]))
        ]
        if len(candidate_keys) < 2:
            continue
        query_rows.append({
            "truth_id": truth[0],
            "formula": str(g["formula"]),
            "adduct": str(truth[1]),
            "candidate_count": len(candidate_keys),
            "n_energies": len(required),
        })

    counts = np.asarray([r["candidate_count"] for r in query_rows], dtype=int)
    formulas = {r["formula"] for r in query_rows}
    summary = {
        "status": "MSNLIB_UNSEEN_COLLECTION_LABEL_FREE_FEASIBILITY",
        "collection": args.collection,
        "guardrails": {
            "ranking_scores_computed": False,
            "truth_ranks_computed": False,
            "full_inchikey_overlap_with_prior_collection_excluded": True,
        },
        "input": {
            "target_path": str(args.jsonl),
            "target_sha256": _sha256(args.jsonl),
            "target_records": len(target_records),
            "prior_path": str(args.exclude_jsonl),
            "prior_sha256": _sha256(args.exclude_jsonl),
            "prior_records": len(prior_records),
        },
        "coverage": {
            "indexed_full_identity_groups": len(indexed),
            "unambiguous_full_identity_groups": len(groups),
            "target_unique_full_inchikeys": len(target_structures),
            "overlap_full_inchikeys_removed": len(overlap_structures),
            "eligible_structure_adduct_groups_after_overlap_removal": len(eligible),
            "eligible_ambiguous_queries": len(query_rows),
            "eligible_formula_blocks": len(formulas),
            "queries_pool_ge_5": int(np.sum(counts >= 5)) if len(counts) else 0,
            "queries_pool_ge_10": int(np.sum(counts >= 10)) if len(counts) else 0,
            "queries_pool_ge_20": int(np.sum(counts >= 20)) if len(counts) else 0,
            "pool_size": ({
                "min": int(counts.min()),
                "median": float(np.median(counts)),
                "mean": float(np.mean(counts)),
                "q75": float(np.quantile(counts, 0.75)),
                "max": int(counts.max()),
            } if len(counts) else None),
        },
        "predeclared_gate": {
            "eligible_ambiguous_queries_ge": 150,
            "eligible_formula_blocks_ge": 50,
            "queries_pool_ge_5_ge": 25,
        },
    }
    c = summary["coverage"]
    g = summary["predeclared_gate"]
    summary["gate_passed"] = bool(
        c["eligible_ambiguous_queries"] >= g["eligible_ambiguous_queries_ge"]
        and c["eligible_formula_blocks"] >= g["eligible_formula_blocks_ge"]
        and c["queries_pool_ge_5"] >= g["queries_pool_ge_5_ge"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
