"""Exact worst-case source-overlap bounds on frozen MassBank affinity deltas.

No overlap identity is assumed: for any unknown set of k excluded queries the
smallest remaining mean removes the k largest gains, and conversely for maximum.
Not an independent validation or a confidence interval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROW_SHA="2c973a7b83c1a48f95f923107ee72bb44a2a95f7d7da403f9d6f541d662efc8b"
OVERLAP_SHA="dd594c1b1a3acefd94ece3a23166bd98754b87dadcde1b1c5998d02296e42050"
CONTROLS=("hellinger_fixed","support_only","hellinger_oracle","convex_oracle")

def exact_bounds(differences,excluded):
    if not isinstance(excluded,int) or not 0<=excluded<len(differences):
        raise ValueError("invalid excluded query count")
    values=sorted(float(x) for x in differences)
    if any(x!=x or abs(x)==float("inf") for x in values):
        raise ValueError("invalid affinity delta")
    keep=len(values)-excluded
    wins=sum(x>1e-12 for x in values)
    return {
        "full_mean":sum(values)/len(values),
        "excluded_n":excluded,
        "remaining_n":keep,
        "lower_possible_mean":sum(values[:keep])/keep,
        "upper_possible_mean":sum(values[excluded:])/keep,
        "guaranteed_positive_mean":sum(values[:keep])>0,
        "original_wins":wins,
        "original_losses":sum(x < -1e-12 for x in values),
        "minimum_remaining_wins":max(wins-excluded,0),
        "minimum_remaining_win_fraction":max(wins-excluded,0)/keep,
        "proof":"Delete the k largest paired scores for minimum and k smallest for maximum; this contains every feasible identity allocation.",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--efficacy-dir",type=Path,required=True)
    p.add_argument("--overlap-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    source=(a.efficacy_dir/"rows.jsonl").read_bytes()
    overlap=(a.overlap_dir/"source_overlap.json").read_bytes()
    if hashlib.sha256(source).hexdigest()!=ROW_SHA:
        raise ValueError("frozen scored rows changed")
    if hashlib.sha256(overlap).hexdigest()!=OVERLAP_SHA:
        raise ValueError("independent overlap audit changed")
    rows=[json.loads(x) for x in source.splitlines() if x.strip()]
    audit=json.loads(overlap)
    original=json.loads((a.efficacy_dir/"result.json").read_text())
    if len(rows)!=2138 or original["all"]["coverage"]["queries"]!=len(rows):
        raise ValueError("original complete efficacy cohort changed")
    k=audit["n_overlapping_prior_msnlib_queries"]
    if k!=106 or audit["n_nonoverlapping_prior_msnlib_queries"]!=2032 or audit["n_query_triplets"]!=len(rows):
        raise ValueError("previous structure overlap count changed")
    result={
        "status":"POSTHOC_EXACT_WORST_CASE_SOURCE_OVERLAP_BOUNDS",
        "evaluation_class":"mathematical post-hoc sensitivity, NOT new independent efficacy",
        "original_efficacy_run":38918658343,
        "original_overlap_run":38920155068,
        "original_n":len(rows),"excluded_n":k,"remaining_n":len(rows)-k,
        "comparisons":{m:exact_bounds([r["alpha_quarter"]-r[m] for r in rows],k) for m in CONTROLS},
        "limitations":"A bound on mean efficacy, not a subset CI, contributor-level replication, independent-acquisition demonstration or prospective confirmation.",
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"bounds.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Exact worst-case prior-MSnLib-structure-overlap sensitivity","",
           "**POST-HOC: NOT A NEW HELD-OUT EXPERIMENT.**","",
           "| Comparator | Frozen full mean | Worst remaining mean | Best remaining mean | Guaranteed positive? |",
           "|---|---:|---:|---:|---|"]
    for m,b in result["comparisons"].items():
        lines.append(f"| {m} | {b['full_mean']:+.6f} | {b['lower_possible_mean']:+.6f} | {b['upper_possible_mean']:+.6f} | {b['guaranteed_positive_mean']} |")
    lines += ["",result["limitations"],""]
    (a.out/"bounds.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__=="__main__":
    main()
