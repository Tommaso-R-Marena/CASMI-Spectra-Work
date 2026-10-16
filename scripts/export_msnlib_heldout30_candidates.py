from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from qfd.energy_interpolation import endpoint_static_score, energy_interpolation_score, equal_mix_control_score
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
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


def main() -> None:
    p = argparse.ArgumentParser(description="Export predeclared 30 eV heldout-energy candidate scores.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    eligible = {}
    for skey, g in groups.items():
        if LOW_EV not in g["same_energy"] or HIGH_EV not in g["same_energy"]:
            continue
        eligible[skey] = {
            "formula": str(g["formula"]),
            "query": _clean_view(g["single_best"][TARGET_EV], transform="none") if TARGET_EV in g["single_best"] else None,
            "low": _clean_view(g["same_energy"][LOW_EV], transform="none"),
            "high": _clean_view(g["same_energy"][HIGH_EV], transform="none"),
        }

    by_formula_adduct: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for skey, g in eligible.items():
        by_formula_adduct[(g["formula"], str(skey[1]))].append(skey)

    rows: list[dict[str, object]] = []
    split_queries = {"discovery": 0, "validation": 0}
    split_formulas = {"discovery": set(), "validation": set()}

    for truth, g in sorted(eligible.items()):
        query = g["query"]
        if query is None:
            continue
        formula = str(g["formula"])
        candidates = sorted(by_formula_adduct[(formula, str(truth[1]))])
        if len(candidates) < 2:
            continue
        split = (
            "validation"
            if stable_group_is_validation(formula, validation_fraction=VALIDATION_FRACTION, salt=SPLIT_SALT)
            else "discovery"
        )
        split_queries[split] += 1
        split_formulas[split].add(formula)
        spec_id = f"{truth[0]}|{truth[1]}|q{int(TARGET_EV)}"

        for cid in candidates:
            cg = eligible[cid]
            static = endpoint_static_score(query, cg["low"], cg["high"])
            interp = energy_interpolation_score(query, cg["low"], cg["high"])
            control = equal_mix_control_score(query, cg["low"], cg["high"])
            rows.append({
                "spec_id": spec_id,
                "formula": formula,
                "split": split,
                "adduct": str(truth[1]),
                "truth_id": str(truth[0]),
                "candidate_id": str(cid[0]),
                "is_truth": str(cid[0]) == str(truth[0]),
                "candidate_count": int(len(candidates)),
                "query_energy_ev": TARGET_EV,
                "reference_low_ev": LOW_EV,
                "reference_high_ev": HIGH_EV,
                "static_best": float(static),
                "expert_score": float(interp),
                "equal_mix_control": float(control),
            })

    if split_formulas["discovery"] & split_formulas["validation"]:
        raise RuntimeError("formula leakage across heldout-energy split")
    if not rows:
        raise RuntimeError("no heldout-energy candidate rows produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with gzip.open(args.out, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "MSNLIB_HELDOUT30_CANDIDATE_EXPORT",
        "input": {"path": str(args.jsonl), "sha256": _sha256(args.jsonl), "records": int(len(records))},
        "protocol": {
            "query": "SINGLE_BEST_SCAN at 30 eV",
            "candidate_references": "SAME_ENERGY at 20 and 60 eV only",
            "static_best": "maximum endpoint Bhattacharyya/Hellinger affinity",
            "expert_score": "30 eV energy-constrained linear spectral interpolation affinity (75% 20 eV + 25% 60 eV)",
            "equal_mix_control": "energy-blind 50/50 endpoint mixture affinity",
            "candidate_pool": "same formula and same adduct",
            "identity": "full InChIKey + adduct",
            "split_unit": "molecular formula",
            "split_salt": SPLIT_SALT,
            "validation_fraction": VALIDATION_FRACTION,
        },
        "queries": int(sum(split_queries.values())),
        "candidate_rows": int(len(rows)),
        "split": {
            "discovery_queries": int(split_queries["discovery"]),
            "validation_queries": int(split_queries["validation"]),
            "discovery_formula_blocks": int(len(split_formulas["discovery"])),
            "validation_formula_blocks": int(len(split_formulas["validation"])),
        },
        "output": {"path": str(args.out), "sha256": _sha256(args.out)},
        "guardrail": (
            "The interpolation formula, energies, split salt, validation fraction, and selector parameters were "
            "fixed before heldout-energy ranking outcomes were opened."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
