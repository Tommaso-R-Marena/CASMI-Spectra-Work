from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _unambiguous_groups, index_records


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Label-free feasibility audit for a prospective MSnLib external chemical-library lockbox. "
            "This script deliberately computes no ranking score and no truth rank."
        )
    )
    p.add_argument("jsonl", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    eligible = {}
    for skey, g in groups.items():
        common = tuple(sorted(set(g["single_best"]) & set(g["same_energy"])))
        if len(common) < 2:
            continue
        eligible[skey] = {
            "formula": str(g["formula"]),
            "energies": common,
        }

    by_formula_adduct: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for skey, g in eligible.items():
        by_formula_adduct[(g["formula"], str(skey[1]))].append(skey)

    query_pool_sizes: list[int] = []
    query_energy_counts: list[int] = []
    formula_blocks: set[str] = set()
    pool_hist: Counter[int] = Counter()

    for truth, g in sorted(eligible.items()):
        required = set(g["energies"])
        candidates = [
            cid
            for cid in by_formula_adduct[(g["formula"], str(truth[1]))]
            if required.issubset(set(eligible[cid]["energies"]))
        ]
        if len(candidates) < 2:
            continue
        n = len(candidates)
        query_pool_sizes.append(n)
        query_energy_counts.append(len(required))
        formula_blocks.add(g["formula"])
        pool_hist[n] += 1

    def nge(k: int) -> int:
        return int(sum(n >= k for n in query_pool_sizes))

    summary = {
        "status": "MSNLIB_EXTERNAL_LOCKBOX_LABEL_FREE_FEASIBILITY",
        "dataset": str(args.dataset),
        "input": {
            "path": str(args.jsonl),
            "sha256": _sha256(args.jsonl),
            "records": int(len(records)),
        },
        "identity_definition": "full InChIKey + adduct",
        "protocol": {
            "query": "SINGLE_BEST_SCAN exact-condition spectra",
            "reference": "SAME_ENERGY exact-condition spectra",
            "candidate_pool": (
                "same molecular formula and same adduct; every candidate must contain every exact collision "
                "energy required by the query"
            ),
            "minimum_exact_energies": 2,
            "ranking_scores_computed": False,
            "truth_ranks_computed": False,
        },
        "coverage": {
            "indexed_full_identity_groups": int(len(indexed)),
            "unambiguous_full_identity_groups": int(len(groups)),
            "eligible_structure_adduct_groups": int(len(eligible)),
            "eligible_queries": int(len(query_pool_sizes)),
            "eligible_formula_blocks": int(len(formula_blocks)),
            "queries_pool_ge_2": nge(2),
            "queries_pool_ge_5": nge(5),
            "queries_pool_ge_10": nge(10),
            "queries_pool_ge_20": nge(20),
            "pool_size_min": int(min(query_pool_sizes)) if query_pool_sizes else 0,
            "pool_size_median": float(__import__("numpy").median(query_pool_sizes)) if query_pool_sizes else 0.0,
            "pool_size_mean": float(__import__("numpy").mean(query_pool_sizes)) if query_pool_sizes else 0.0,
            "pool_size_max": int(max(query_pool_sizes)) if query_pool_sizes else 0,
            "pool_size_histogram": {str(k): int(v) for k, v in sorted(pool_hist.items())},
            "energy_count_min": int(min(query_energy_counts)) if query_energy_counts else 0,
            "energy_count_max": int(max(query_energy_counts)) if query_energy_counts else 0,
        },
        "predeclared_gate": {
            "eligible_queries_min": 100,
            "eligible_formula_blocks_min": 30,
            "queries_pool_ge_5_min": 30,
            "rationale": (
                "Require a nontrivial ambiguous cohort, multiple independent formula blocks, and enough pools "
                "with at least five candidates for a Top-1-preserving tail rule to have meaningful room to act."
            ),
        },
        "guardrail": (
            "This feasibility pass is label-free with respect to ranking outcomes. No static, expert, v5.2, or "
            "v5.3 score is computed here. Ranking may be opened only if every predeclared gate is satisfied."
        ),
    }
    gate = summary["predeclared_gate"]
    cov = summary["coverage"]
    summary["gate_pass"] = bool(
        cov["eligible_queries"] >= gate["eligible_queries_min"]
        and cov["eligible_formula_blocks"] >= gate["eligible_formula_blocks_min"]
        and cov["queries_pool_ge_5"] >= gate["queries_pool_ge_5_min"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
