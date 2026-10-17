"""Independent artifact-only influence audit of published graph-conditioned result.

No new spectra, chemical predictions, model tuning, or query filtering.
This is POST-HOC sensitivity to concentration in chemical formula clusters.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_RUN=39058512753
SOURCE_SHA="aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e"


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def influence(rows):
    if not rows:
        raise ValueError("nonempty genuine query-level graph result required")
    group=defaultdict(list)
    wins=losses=ties=0
    top1_win=top1_loss=top1_tie=0
    for row in rows:
        if not all(k in row for k in (
            "identity","formula","query_index",
            "graph_mrr","hellinger_mrr","graph_top1","hellinger_top1"
        )):
            raise ValueError("missing original frozen per-query result")
        d=float(row["graph_mrr"])-float(row["hellinger_mrr"])
        t=float(row["graph_top1"])-float(row["hellinger_top1"])
        if not np.isfinite(d) or not np.isfinite(t):
            raise ValueError("nonfinite result")
        group[row["formula"]].append(d)
        wins+=int(d>1e-12)
        losses+=int(d< -1e-12)
        ties+=int(abs(d)<=1e-12)
        top1_win+=int(t>1e-12)
        top1_loss+=int(t< -1e-12)
        top1_tie+=int(abs(t)<=1e-12)
    n=len(rows)
    total=sum(sum(v) for v in group.values())
    mean=total/n
    formula_sums={k:float(sum(v)) for k,v in group.items()}
    formula_sizes={k:len(v) for k,v in group.items()}
    if n<=max(formula_sizes.values()):
        raise ValueError("at least two formula clusters required")
    leave_one={
        k:(total-formula_sums[k])/(n-formula_sizes[k]) for k in group
    }
    worst_formula=min(leave_one,key=lambda k:leave_one[k])
    best_formula=max(leave_one,key=lambda k:leave_one[k])
    # Posthoc prespecified descriptive stress sets: remove the largest
    # positive raw formula contributions, not a fair new test/CI.
    rankings=sorted(formula_sums,key=lambda k:(-formula_sums[k],k))
    dropped={}
    for take in (1,3,5,10,20):
        keep_drop=rankings[:min(take,len(rankings)-1)]
        num=sum(formula_sums[k] for k in keep_drop)
        removed_n=sum(formula_sizes[k] for k in keep_drop)
        dropped[str(take)]={
            "n_formulas_removed":len(keep_drop),
            "n_queries_removed":removed_n,
            "remaining_mrr_delta":(total-num)/(n-removed_n),
            "removed_positive_contribution":num,
        }
    per_formula=[
        {"formula":k,"n_queries":formula_sizes[k],"sum_mrr_delta":formula_sums[k],
         "mean_mrr_delta":formula_sums[k]/formula_sizes[k]}
        for k in sorted(group)
    ]
    return {
        "n_queries":n,"n_formula_clusters":len(group),
        "mean_mrr_delta":mean,
        "rank_wins":wins,"rank_losses":losses,"rank_ties":ties,
        "fractional_top1_wins":top1_win,
        "fractional_top1_losses":top1_loss,
        "fractional_top1_ties":top1_tie,
        "n_formula_positive":sum(v>1e-12 for v in formula_sums.values()),
        "n_formula_negative":sum(v< -1e-12 for v in formula_sums.values()),
        "n_formula_zero":sum(abs(v)<=1e-12 for v in formula_sums.values()),
        "exact_leave_one_formula_out":{
            "min_mrr_delta":leave_one[worst_formula],
            "min_formula":worst_formula,
            "min_formula_n":formula_sizes[worst_formula],
            "max_mrr_delta":leave_one[best_formula],
            "max_formula":best_formula,
            "max_formula_n":formula_sizes[best_formula],
            "all_leave_one_out_deltas_positive":all(v>0 for v in leave_one.values()),
        },
        "posthoc_largest_gain_group_deletion":dropped,
        "formula_contribution_rows":per_formula,
    }


def evaluate(prior:Path,out:Path):
    result=json.loads((prior/"result.json").read_text())
    rows=read_lines(prior/"query_metrics.jsonl")
    if (result.get("source_sha256")!=SOURCE_SHA
        or result.get("n_original_queries")!=4398
        or result.get("n_native_formula_blocks")!=2254
        or result.get("retrospective_fixed_primary_passed") is not True
        or len(rows)!=4398
        or len({(r["query_index"],r["identity"]) for r in rows})!=4398):
        raise RuntimeError("original unchanged positive graph outcome missing or drifted")
    audit=influence(rows)
    if (audit["n_queries"]!=4398
        or audit["n_formula_clusters"]!=2254
        or abs(audit["mean_mrr_delta"]-result["retrospective_formula_block_inference"]["mean_mrr_delta"])>1e-12
        or abs(audit["mean_mrr_delta"]-(result["graph_conditioned_mrr"]-result["original_historical_hellinger_mrr"]))>1e-12):
        raise RuntimeError("original published graph MRR not independently reproduced")
    source_bytes=(prior/"result.json").read_bytes()
    query_bytes=(prior/"query_metrics.jsonl").read_bytes()
    compact={
        **{k:v for k,v in audit.items() if k!="formula_contribution_rows"},
        "status":"POSTHOC_GRAPH_4398_FORMULA_INFLUENCE_AUDIT_V1",
        "original_graph_workflow":SOURCE_RUN,
        "original_result_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "original_per_query_sha256":hashlib.sha256(query_bytes).hexdigest(),
        "original_formula_ci":result["retrospective_formula_block_inference"]["formula_block_bootstrap_95"],
        "original_two_sided_formula_p":result["retrospective_formula_block_inference"]["formula_block_signflip_2sided"]["p_value_two_sided"],
        "new_independent_validation":False,
        "note":"Retrospective artifact-only robustness, NOT independent structural prediction or causal chemistry mechanism; all original MRR outcomes preserved.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(compact,sort_keys=True,indent=2)+"\n")
    with (out/"per_formula_contributions.jsonl").open("w") as f:
        for row in audit["formula_contribution_rows"]:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    lines=[
        "# Graph-conditioned 4398-query formula-influence audit",
        "",
        "Posthoc review of the **same original scored per-query artifact**; no new source or fitting.",
        f"Unchanged ΔMRR +{audit['mean_mrr_delta']:.9f} over {audit['n_queries']} queries and {audit['n_formula_clusters']} formulas.",
        f"Rank wins/losses/ties: {audit['rank_wins']}/{audit['rank_losses']}/{audit['rank_ties']}.",
        f"Top1 wins/losses/ties: {audit['fractional_top1_wins']}/{audit['fractional_top1_losses']}/{audit['fractional_top1_ties']}.",
        f"Formula sums positive/negative/zero: {audit['n_formula_positive']}/{audit['n_formula_negative']}/{audit['n_formula_zero']}.",
        "",
        f"Exact worst leave-one-formula-out ΔMRR {audit['exact_leave_one_formula_out']['min_mrr_delta']:+.9f} "
        f"({audit['exact_leave_one_formula_out']['min_formula']}); "
        f"all LOFO positive: {audit['exact_leave_one_formula_out']['all_leave_one_out_deltas_positive']}.",
        "",
        "| Remove top positive-contribution formulas | Removed queries | Remaining ΔMRR |",
        "|---|---:|---:|",
    ]
    for k,row in audit["posthoc_largest_gain_group_deletion"].items():
        lines.append(f"| {k} | {row['n_queries_removed']} | {row['remaining_mrr_delta']:+.9f} |")
    lines+=["",compact["note"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return compact


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.prior,a.out)
