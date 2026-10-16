"""Independent audit of the frozen numerical Hellinger oracle on MSnLib.

This does not select/train an exponent or create a new scientific holdout.
It replays previously eligible cohorts with an exact closed-form oracle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid,
    best_hellinger_chord_oracle, affinity,
)
from qfd.hellinger_analytic_oracle import analytic_hellinger_chord_oracle


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()
    groups=_unambiguous_groups(index_records(load_msnlib_jsonl(args.jsonl)))
    rows=[]
    for skey, g in sorted(groups.items()):
        refs=sorted(float(e) for e in g["same_energy"])
        if len(refs)<2:
            continue
        for eq in sorted(float(e) for e in g["single_best"]):
            before=[e for e in refs if e<eq]
            after=[e for e in refs if e>eq]
            if not before or not after:
                continue
            e0,e1=max(before),min(after)
            t=(eq-e0)/(e1-e0)
            query=_clean_view(g["single_best"][eq],transform="none")
            low=_clean_view(g["same_energy"][e0],transform="none")
            high=_clean_view(g["same_energy"][e1],transform="none")
            grid=reference_only_grid(low,high,tolerance_da=.01)
            q,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=.01)
            exact=analytic_hellinger_chord_oracle(q,grid.low,grid.high)
            numerical=best_hellinger_chord_oracle(q,grid.low,grid.high)
            quarter=affinity(q,power_mean_distribution(grid.low,grid.high,t,.25))
            rows.append({
                "structure_adduct":f"{skey[0]}|{skey[1]}",
                "formula":str(g["formula"]),
                "eV":eq,"t":t,
                "exact_affinity":exact["affinity"],
                "numeric_affinity":numerical["affinity"],
                "exact_weight":exact["oracle_weight"],
                "numeric_weight":numerical["oracle_weight"],
                "quarter_minus_exact_oracle":quarter-exact["affinity"],
                "quarter_minus_numeric_oracle":quarter-numerical["affinity"],
                "query_unmatched_mass":unmatched,
            })
    if not rows:
        raise RuntimeError("no eligible query triplets")
    gaps=np.asarray([r["exact_affinity"]-r["numeric_affinity"] for r in rows])
    if not np.isfinite(gaps).all():
        raise ValueError("nonfinite oracle comparison")
    sha=hashlib.sha256(args.jsonl.read_bytes()).hexdigest()
    result={
        "status":"POSTHOC_NUMERICAL_ORACLE_INDEPENDENT_AUDIT",
        "dataset":args.dataset,"input_sha256":sha,
        "n_queries":len(rows),
        "n_formula_blocks":len(set(r["formula"] for r in rows)),
        "analytic_minus_numerical_mean":float(gaps.mean()),
        "analytic_minus_numerical_min":float(gaps.min()),
        "analytic_minus_numerical_max":float(gaps.max()),
        "max_absolute_affinity_discrepancy":float(np.max(np.abs(gaps))),
        "numerical_exceeds_true_global_max_by_over_1e-9":int(np.count_nonzero(gaps < -1e-9)),
        "quarter_minus_exact_global_oracle_mean":float(np.mean(
            [r["quarter_minus_exact_oracle"] for r in rows])),
        "quarter_minus_original_numeric_oracle_mean":float(np.mean(
            [r["quarter_minus_numeric_oracle"] for r in rows])),
        "pass_tolerance":1e-6,
        "audit_pass":bool(np.max(np.abs(gaps))<=1e-6),
        "interpretation":"Optimizer verification only; cohorts were previously used; no new efficacy confirmation.",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"analytic_oracle_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.out/"analytic_oracle_rows.jsonl").write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
    print(json.dumps(result,indent=2))
    if not result["audit_pass"]:
        raise SystemExit("Analytic-vs-numeric oracle comparison failed: inspect full per-query audit")


if __name__=="__main__":
    main()
