from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from qfd.benchmark import static_best_score
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.stats import stable_group_is_validation
from qfd.trajectory import joint_energy_score


SPLIT_SALT = "msnlib-group-validation-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export the exact-condition MSnLib candidate pools used by the frozen "
            "static_best vs joint_mean trajectory comparison."
        )
    )
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=0.5)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    eligible = {}
    for skey, g in groups.items():
        common = sorted(set(g["single_best"]) & set(g["same_energy"]))
        if len(common) < 2:
            continue
        eligible[skey] = {
            "formula": g["formula"],
            "inchikey14": g["inchikey14"],
            "energies": common,
            "qsqrt": {e: _clean_view(g["single_best"][e], transform="sqrt") for e in common},
            "rsqrt": {e: _clean_view(g["same_energy"][e], transform="sqrt") for e in common},
        }

    by_formula_adduct = defaultdict(list)
    for skey, g in eligible.items():
        by_formula_adduct[(g["formula"], skey[1])].append(skey)

    rows: list[dict[str, object]] = []
    query_count = 0
    validation_queries = 0
    discovery_queries = 0
    formula_counts = {"validation": set(), "discovery": set()}

    for truth, g in sorted(eligible.items()):
        required = tuple(g["energies"])
        candidate_keys = sorted(
            cid
            for cid in by_formula_adduct[(g["formula"], truth[1])]
            if set(required).issubset(set(eligible[cid]["energies"]))
        )
        if len(candidate_keys) < 2:
            continue

        formula = str(g["formula"])
        split = (
            "validation"
            if stable_group_is_validation(
                formula,
                validation_fraction=args.validation_fraction,
                salt=SPLIT_SALT,
            )
            else "discovery"
        )
        formula_counts[split].add(formula)
        query_count += 1
        if split == "validation":
            validation_queries += 1
        else:
            discovery_queries += 1

        qsqrt = [g["qsqrt"][e] for e in required]
        query_id = f"{truth[0]}|{truth[1]}"
        for cid in candidate_keys:
            cg = eligible[cid]
            rsqrt = [cg["rsqrt"][e] for e in required]
            candidate_id = str(cid[0])
            rows.append(
                {
                    "spec_id": query_id,
                    "formula": formula,
                    "split": split,
                    "adduct": str(truth[1]),
                    "truth_id": str(truth[0]),
                    "candidate_id": candidate_id,
                    "is_truth": candidate_id == str(truth[0]),
                    "candidate_count": len(candidate_keys),
                    "n_energies": len(required),
                    "energies": ";".join(str(x) for x in required),
                    "static_best": static_best_score(qsqrt, rsqrt),
                    "expert_score": joint_energy_score(qsqrt, rsqrt, max_energy_gap_ev=0),
                }
            )

    if formula_counts["validation"] & formula_counts["discovery"]:
        raise RuntimeError("formula leakage across discovery/validation split")
    if not rows:
        raise RuntimeError("no eligible candidate rows produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with gzip.open(args.out, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "MSNLIB_EXACT_CONDITION_CANDIDATE_EXPORT",
        "input": {
            "path": str(args.jsonl),
            "sha256": _sha256(args.jsonl),
            "records": len(records),
        },
        "identity_definition": "full InChIKey + adduct",
        "candidate_protocol": (
            "Same-formula, same-adduct candidate pools; every candidate has the exact same collision-energy "
            "support required by the query. Query spectra use SINGLE_BEST_SCAN and candidate references use "
            "SAME_ENERGY, matching evaluate_exact_condition_variants."
        ),
        "scores": {
            "static_best": "best sqrt-cosine across exact matching energies",
            "expert_score": "frozen primary trajectory method joint_mean",
        },
        "split": {
            "unit": "molecular formula",
            "salt": SPLIT_SALT,
            "validation_fraction": args.validation_fraction,
            "validation_queries": validation_queries,
            "discovery_queries": discovery_queries,
            "validation_formula_blocks": len(formula_counts["validation"]),
            "discovery_formula_blocks": len(formula_counts["discovery"]),
        },
        "eligible_structure_adduct_groups": len(eligible),
        "queries": query_count,
        "candidate_rows": len(rows),
        "output": {
            "path": str(args.out),
            "sha256": _sha256(args.out),
        },
        "guardrail": (
            "This export does not use truth ranks to select or tune candidates, scores, selector thresholds, "
            "or validation membership. v5.3 parameters remain frozen from the NIST development analysis."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
