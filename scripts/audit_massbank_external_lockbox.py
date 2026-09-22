from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_massbank_postrelease_lockbox import (
    QUERY_ENERGY,
    REFERENCE_ENERGIES,
    _in_scope,
    _is_energy,
    _quantiles,
    _tree_hash,
    parse_record,
)


def build_external_audit(root: Path, *, dataset_sha: str) -> tuple[dict, list[dict]]:
    paths = sorted(root.rglob("*.txt"))
    records = [r for p in paths if (r := parse_record(p, root)) is not None and _in_scope(r)]

    by_structure: dict[str, list] = defaultdict(list)
    for r in records:
        by_structure[r.connectivity_key].append(r)

    complete_refs: dict[str, dict[float, object]] = {}
    structure_formula: dict[str, str] = {}
    for key, rows in by_structure.items():
        formulas = sorted(set(r.formula for r in rows))
        if len(formulas) != 1:
            continue
        structure_formula[key] = formulas[0]
        refs: dict[float, object] = {}
        for energy in REFERENCE_ENERGIES:
            cands = sorted(
                (r for r in rows if _is_energy(r.collision_energy_ev, energy)),
                key=lambda r: (r.accession, r.path),
            )
            if cands:
                refs[energy] = cands[0]
        if len(refs) == len(REFERENCE_ENERGIES):
            complete_refs[key] = refs

    by_formula: dict[str, list[str]] = defaultdict(list)
    for key in complete_refs:
        by_formula[structure_formula[key]].append(key)
    for formula in by_formula:
        by_formula[formula] = sorted(set(by_formula[formula]))

    # Query selection is deterministic and label-free: one 30 eV record per
    # 2D structure, ordered by accession/path. The 30 eV query is never used as
    # a candidate reference, which remains fixed to 10/20/40 eV.
    query_by_structure = {}
    for key, rows in sorted(by_structure.items()):
        q = sorted(
            (r for r in rows if _is_energy(r.collision_energy_ev, QUERY_ENERGY)),
            key=lambda r: (r.accession, r.path),
        )
        if q:
            query_by_structure[key] = q[0]

    manifest: list[dict] = []
    for key, q in sorted(query_by_structure.items()):
        if key not in complete_refs or key not in structure_formula:
            continue
        formula = structure_formula[key]
        pool = by_formula[formula]
        if len(pool) < 2:
            continue
        refs = complete_refs[key]
        contributor = q.path.split("/", 1)[0]
        manifest.append(
            {
                "formula": formula,
                "truth_connectivity_key": key,
                "query_accession": q.accession,
                "query_path": q.path,
                "query_contributor": contributor,
                "query_instrument": q.instrument,
                "query_instrument_type": q.instrument_type,
                "query_energy_ev": QUERY_ENERGY,
                "candidate_count": len(pool),
                "candidate_connectivity_keys": ";".join(pool),
                "truth_ref_10_path": refs[10.0].path,
                "truth_ref_20_path": refs[20.0].path,
                "truth_ref_40_path": refs[40.0].path,
            }
        )

    pools = [int(r["candidate_count"]) for r in manifest]
    thresholds = (("ge_2", 2), ("ge_5", 5), ("ge_10", 10), ("ge_20", 20))
    contributor_counts = Counter(r["query_contributor"] for r in manifest)
    instrument_type_counts = Counter(r["query_instrument_type"] for r in manifest)
    formula_contributor = defaultdict(set)
    for r in manifest:
        formula_contributor[r["formula"]].add(r["query_contributor"])

    coverage = {
        "queries": len(manifest),
        "formula_blocks": len({r["formula"] for r in manifest}),
        "contributors": len(contributor_counts),
        "instrument_types": len(instrument_type_counts),
        "pool_size": _quantiles(pools),
        "queries_by_min_pool_size": {k: sum(n >= t for n in pools) for k, t in thresholds},
        "formula_blocks_by_min_pool_size": {
            k: len({r["formula"] for r in manifest if int(r["candidate_count"]) >= t})
            for k, t in thresholds
        },
        "formula_blocks_with_multiple_query_contributors": sum(len(v) >= 2 for v in formula_contributor.values()),
        "query_contributor_counts": dict(sorted(contributor_counts.items())),
        "query_instrument_type_counts": dict(sorted(instrument_type_counts.items())),
    }

    summary = {
        "status": "MASSBANK_WHOLE_LIBRARY_LABEL_FREE_EXTERNAL_FEASIBILITY_AUDIT",
        "dataset": {
            "sha": dataset_sha,
            "txt_files": len(paths),
            "tree_sha256": _tree_hash(paths, root),
            "in_scope_records": len(records),
            "complete_reference_structures": len(complete_refs),
            "structures_with_30ev_query": len(query_by_structure),
        },
        "design": {
            "performance_unopened": True,
            "query_rule": "one deterministic measured 30 eV query per 2D structure",
            "candidate_rule": "all 2D structures sharing exact molecular formula with complete measured 10/20/40 eV references",
            "scope": "MS2, POSITIVE, [M+H]+",
            "structure_identity": "first InChIKey block (2D connectivity)",
            "query_energy_ev": QUERY_ENERGY,
            "reference_energies_ev": list(REFERENCE_ENERGIES),
            "primary_inference_unit": "molecular-formula block; query-level metrics are descriptive and formula-clustered inference is required",
            "ranking_scores_computed": False,
        },
        "coverage": coverage,
        "guardrails": {
            "label_free": True,
            "massbank_performance_used_for_v53_selection": False,
            "selector_frozen": {
                "frozen_parent_sha": "0006c7f81c7dfac53da48690ca0fe627108b9767",
                "overlap_threshold": 0.8,
                "protected_k": 1,
                "overlap_k": 10,
                "head_survival_slack": 1,
                "tail_rule": "minimax rank aggregation",
            },
            "required_negative_controls_if_opened": [
                "within-formula expert permutation",
                "expert-rank inversion",
                "within-query expert-score shuffle",
            ],
        },
    }
    return summary, manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--dataset-sha", required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    args = p.parse_args()

    summary, manifest = build_external_audit(args.root, dataset_sha=args.dataset_sha)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    import csv
    fields = list(manifest[0]) if manifest else [
        "formula", "truth_connectivity_key", "query_accession", "query_path",
        "query_contributor", "query_instrument", "query_instrument_type",
        "query_energy_ev", "candidate_count", "candidate_connectivity_keys",
        "truth_ref_10_path", "truth_ref_20_path", "truth_ref_40_path",
    ]
    with args.out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
