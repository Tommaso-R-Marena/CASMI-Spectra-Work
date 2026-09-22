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


def _coverage(groups, low: float, target: float, high: float) -> dict[str, object]:
    ref_eligible = {
        skey: g
        for skey, g in groups.items()
        if low in g["same_energy"] and high in g["same_energy"]
    }
    by_formula_adduct = defaultdict(list)
    for skey, g in ref_eligible.items():
        by_formula_adduct[(str(g["formula"]), str(skey[1]))].append(skey)

    pools: list[int] = []
    formulas: set[str] = set()
    for truth, g in ref_eligible.items():
        if target not in g["single_best"]:
            continue
        formula = str(g["formula"])
        candidates = by_formula_adduct[(formula, str(truth[1]))]
        if len(candidates) < 2:
            continue
        pools.append(len(candidates))
        formulas.add(formula)

    a = np.asarray(pools, dtype=int)
    t = (target - low) / (high - low)
    return {
        "low_ev": float(low),
        "target_ev": float(target),
        "high_ev": float(high),
        "interpolation_t": float(t),
        "low_weight": float(1.0 - t),
        "high_weight": float(t),
        "endpoint_reference_eligible_groups": int(len(ref_eligible)),
        "eligible_queries": int(len(pools)),
        "eligible_formula_blocks": int(len(formulas)),
        "queries_pool_ge_5": int(np.sum(a >= 5)) if len(a) else 0,
        "queries_pool_ge_10": int(np.sum(a >= 10)) if len(a) else 0,
        "pool_size_min": int(a.min()) if len(a) else 0,
        "pool_size_median": float(np.median(a)) if len(a) else 0.0,
        "pool_size_mean": float(a.mean()) if len(a) else 0.0,
        "pool_size_max": int(a.max()) if len(a) else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Label-free audit of all bracketing collision-energy triplets in an MSnLib library. "
            "No spectral similarity, truth rank, or outcome metric is computed."
        )
    )
    p.add_argument("jsonl", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    single_energies = sorted({float(e) for g in groups.values() for e in g["single_best"]})
    same_energies = sorted({float(e) for g in groups.values() for e in g["same_energy"]})

    candidates: list[dict[str, object]] = []
    for target in single_energies:
        for low in same_energies:
            if not low < target:
                continue
            for high in same_energies:
                if not target < high:
                    continue
                t = (target - low) / (high - low)
                # Keep interpolation meaningfully bracketed rather than an almost-endpoint extrapolation surrogate.
                if t < 0.20 - 1e-12 or t > 0.80 + 1e-12:
                    continue
                candidates.append(_coverage(groups, low, target, high))

    if candidates:
        # Frozen label-free selection rule: maximize room for a tail-ranking test first,
        # then total ambiguous queries, then independent formula blocks. Exact ties are
        # broken deterministically by the lexicographically smallest energy triplet.
        candidates.sort(
            key=lambda r: (
                -int(r["queries_pool_ge_5"]),
                -int(r["eligible_queries"]),
                -int(r["eligible_formula_blocks"]),
                float(r["low_ev"]),
                float(r["target_ev"]),
                float(r["high_ev"]),
            )
        )
        selected = candidates[0]
    else:
        selected = None

    result = {
        "status": "MSNLIB_LABEL_FREE_ENERGY_TRIPLET_AUDIT",
        "dataset": str(args.dataset),
        "input": {"path": str(args.jsonl), "sha256": _sha256(args.jsonl), "records": int(len(records))},
        "indexed_full_identity_groups": int(len(indexed)),
        "unambiguous_full_identity_groups": int(len(groups)),
        "single_best_energies_ev": single_energies,
        "same_energy_energies_ev": same_energies,
        "selection_rule": {
            "bracketing_constraint": "low < target < high and interpolation t in [0.20, 0.80]",
            "objective_order": [
                "maximize queries_pool_ge_5",
                "maximize eligible_queries",
                "maximize eligible_formula_blocks",
                "lexicographically minimize (low, target, high) only for exact coverage ties",
            ],
            "uses_scores_or_truth_ranks": False,
        },
        "selected_triplet": selected,
        "all_triplets_ranked_by_label_free_coverage": candidates,
        "guardrail": (
            "Triplet selection uses acquisition metadata and candidate-pool cardinalities only. "
            "No spectrum similarity, interpolation score, truth identity rank, selector output, or efficacy outcome is computed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "dataset": result["dataset"],
        "single_best_energies_ev": single_energies,
        "same_energy_energies_ev": same_energies,
        "selected_triplet": selected,
        "top_10_triplets": candidates[:10],
        "guardrail": result["guardrail"],
    }, indent=2))


if __name__ == "__main__":
    main()
