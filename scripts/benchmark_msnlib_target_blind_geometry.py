"""Audit frozen alpha-quarter collision-energy transfer without target-grid leakage.

Post-hoc robustness study. The target-blind alignment is NOT numerically the
same benchmark as the original shared query/reference alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qfd.energy_interpolation import _bhattacharyya, _common_grid_strict, _l1_columns
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.power_geometry import power_mean_distribution
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie
from qfd.target_blind_geometry import target_blind_scores


METHODS = (
    "linear", "hellinger_chord", "alpha_quarter", "support_only",
    "oracle_probability_mixture", "best_endpoint", "alpha_eighth_exploratory",
)
COMPARISONS = (
    ("alpha_quarter", "hellinger_chord"),
    ("alpha_quarter", "support_only"),
    ("alpha_quarter", "oracle_probability_mixture"),
    ("support_only", "hellinger_chord"),
    ("alpha_eighth_exploratory", "alpha_quarter"),
    ("original_alpha_quarter", "original_hellinger_chord"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def old_shared_grid_scores(query, low, high, t):
    x = _l1_columns(_common_grid_strict([query, low, high], 0.01))
    if x is None:
        raise ValueError("non-normalizable original grid")
    q, a, b = x[:, 0], x[:, 1], x[:, 2]
    return {
        "original_alpha_quarter": float(_bhattacharyya(q, power_mean_distribution(a,b,t,0.25))),
        "original_hellinger_chord": float(_bhattacharyya(q, power_mean_distribution(a,b,t,0.5))),
    }


def comparison(rows, left, right, seed):
    av = np.asarray([float(r[left]) for r in rows])
    bv = np.asarray([float(r[right]) for r in rows])
    blocks = [r["formula"] for r in rows]
    return {
        "a": left, "b": right, "mean_delta": float(np.mean(av-bv)),
        "n_queries": len(rows), "n_formula_blocks": len(set(blocks)),
        "win_loss_tie": paired_win_loss_tie(av,bv,atol=1e-15),
        "formula_block_bootstrap_95": paired_block_bootstrap_delta(
            av,bv,blocks,n_boot=3000,seed=seed
        ),
        "formula_block_signflip_two_sided": paired_block_signflip_test(
            av,bv,blocks,n_permutations=50_000,seed=seed
        ),
    }


def qsummary(rows, key):
    x=np.asarray([float(r[key]) for r in rows])
    return {"mean": float(x.mean()), "median":float(np.median(x)),
            "q05":float(np.quantile(x,.05)), "q95":float(np.quantile(x,.95))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path,required=True)
    p.add_argument("--seed", type=int,default=20261016)
    args=p.parse_args()

    records=load_msnlib_jsonl(args.jsonl)
    groups=_unambiguous_groups(index_records(records))
    rows=[]
    for skey,g in sorted(groups.items()):
        refs=sorted(float(e) for e in g["same_energy"])
        if len(refs)<2:
            continue
        for eq in sorted(float(e) for e in g["single_best"]):
            lower=[e for e in refs if e<eq]
            higher=[e for e in refs if e>eq]
            if not lower or not higher:
                continue
            e0,e1=max(lower),min(higher)
            t=float((eq-e0)/(e1-e0))
            query=_clean_view(g["single_best"][eq],transform="none")
            low=_clean_view(g["same_energy"][e0],transform="none")
            high=_clean_view(g["same_energy"][e1],transform="none")
            s=target_blind_scores(query,low,high,t,tolerance_da=0.01)
            s.update(old_shared_grid_scores(query,low,high,t))
            if any(not np.isfinite(float(v)) for v in s.values()):
                raise ValueError(f"nonfinite outcome: {skey} {eq}")
            rows.append({
                "structure_adduct":f"{skey[0]}|{skey[1]}",
                "formula":str(g["formula"]),
                "query_energy_ev":eq, "low_energy_ev":e0, "high_energy_ev":e1,
                "t":t, **s,
            })
    if not rows:
        raise RuntimeError("no eligible target-blind triplets")

    comparisons={}
    for i,(a,b) in enumerate(COMPARISONS):
        comparisons[f"{a}_vs_{b}"]=comparison(rows,a,b,args.seed+i)

    out={
        "status":"MSNLIB_TARGET_BLIND_POSTHOC_FALSIFICATION",
        "dataset":args.dataset,
        "input":{"sha256":sha256(args.jsonl),"records":len(records)},
        "protocol":"docs/TARGET_BLIND_ENERGY_GEOMETRY_PROTOCOL.md",
        "posthoc":True,
        "reference_grid":"low/high only, strictly diameter-bounded 0.01 Da",
        "query_alignment":"one-to-one nearest mass within 0.01 Da; unmatched mass retained in denominator",
        "coverage":{"query_triplets":len(rows),
                    "formula_blocks":len(set(r["formula"] for r in rows)),
                    "structure_adduct_groups":len(set(r["structure_adduct"] for r in rows))},
        "methods":{m:qsummary(rows,m) for m in METHODS},
        "target_blind_diagnostics":{
            "unmatched_query_fraction":qsummary(rows,"query_unmatched_mass"),
            "reference_shared_min_overlap":qsummary(rows,"reference_shared_min_overlap"),
            "reference_bins":qsummary(rows,"reference_bins"),
        },
        "comparisons":comparisons,
        "interpretation_guardrail":"post-hoc: no universal physical law, no external instrument, no exponent optimality",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(out,indent=2)+"\n")
    with (args.out/"rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
