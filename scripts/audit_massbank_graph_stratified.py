"""Independent post-hoc robustness of graph-only MassBank second-source signal.

Artifact-only: use exactly the original completed 252 query-rank records.
No raw MassBank peaks or reference chemistry, no recomputed graph features,
no parameter refitting and no new eligibility selection. Every original
query is retained in the primary; subgroup analyses are DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from qfd.stats import paired_block_bootstrap_delta,paired_block_signflip_test

EXACT_SOURCE="12c6a922929f9bf2484c46586b6101512a415e44680558883b0630c6e47905fb"
ORIGINAL_RUN=39061071789
RESULT_STATUS="MASSBANK_GRAPH_ONLY_NATIVE_STRUCTURE_ID_SECOND_SOURCE_V1"


def read_jsonl(path:Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def grouped_stats(rows,seed=20261017):
    if not rows:return None
    a=np.array([float(x["graph_mrr"]) for x in rows],float)
    b=np.array([float(x["exact_random_label_null_mrr"]) for x in rows],float)
    formulas=[x["formula"] for x in rows]
    boot=paired_block_bootstrap_delta(
        a,b,formulas,n_boot=10000,seed=seed)
    dif=a-b
    return {
        "n_queries":len(rows),
        "n_unique_formula_blocks":len(set(formulas)),
        "n_contributors":len({x["contributor"] for x in rows}),
        "mean_graph_mrr":float(a.mean()),
        "mean_exact_null_mrr":float(b.mean()),
        "mean_delta":float(dif.mean()),
        "mean_graph_top1_fractional":float(np.mean([
            x["graph_top1_fractional"] for x in rows])),
        "mean_uniform_label_top1_fractional":float(np.mean([
            x["uniform_label_top1_fractional"] for x in rows])),
        "formula_cluster_bootstrap_95":boot,
        "delta_positive_and_95_ci_lower_positive":bool(
            dif.mean()>0 and boot["ci_low"]>0),
        "formula_summed_signal_positive":sum(
            sum(x["graph_mrr"]-x["exact_random_label_null_mrr"]
                for x in rows if x["formula"]==f)>1e-12 for f in set(formulas)),
        "formula_summed_signal_negative":sum(
            sum(x["graph_mrr"]-x["exact_random_label_null_mrr"]
                for x in rows if x["formula"]==f)<-1e-12 for f in set(formulas)),
        "formula_summed_signal_tie":sum(
            abs(sum(x["graph_mrr"]-x["exact_random_label_null_mrr"]
                for x in rows if x["formula"]==f))<=1e-12 for f in set(formulas)),
    }


def influence(rows):
    if not rows:raise ValueError("nonempty fixed query evidence expected")
    formula=defaultdict(list)
    for x in rows:
        formula[x["formula"]].append(x)
    if len(formula)<2:
        raise ValueError("at least two independent formula groups required")
    n=len(rows)
    byformula={f:sum(x["graph_mrr"]-x["exact_random_label_null_mrr"] for x in rs)
               for f,rs in formula.items()}
    total=sum(byformula.values())
    leaveout={f:(total-v)/(n-len(formula[f]))
              for f,v in byformula.items() if n>len(formula[f])}
    min_key=min(leaveout,key=leaveout.get)
    top=sorted(byformula,key=lambda f:(-byformula[f],f))
    deletion={}
    for count in (1,3,5,10):
        selected=top[:min(count,len(top)-1)]
        n_removed=sum(len(formula[f]) for f in selected)
        sum_removed=sum(byformula[f] for f in selected)
        deletion[str(count)]={
            "removed_formula_groups":len(selected),
            "removed_query_records":n_removed,
            "remaining_mean_delta":float((total-sum_removed)/(n-n_removed)),
        }
    return {
        "all_single_formula_deletions_remain_positive":all(x>0 for x in leaveout.values()),
        "smallest_leave_one_formula_out_delta":float(leaveout[min_key]),
        "smallest_leave_one_formula_out_formula":min_key,
        "top_favorable_formula_group_deletions_posthoc":deletion,
        "per_formula_contributions":[
            {"formula":f,"n_queries":len(formula[f]),
             "sum_delta":float(byformula[f]),
             "mean_delta":float(byformula[f]/len(formula[f]))}
            for f in sorted(formula)
        ],
    }


def evaluate(prior:Path,out:Path):
    old=json.loads((prior/"results.json").read_text())
    rows=read_jsonl(prior/"per_query_ranking.jsonl")
    if (old.get("status")!=RESULT_STATUS
        or old.get("source_sha256")!=EXACT_SOURCE
        or old.get("all_original_252",{}).get("n_queries")!=252
        or old.get("all_original_252",{}).get("n_formulas")!=44
        or old.get("all_original_252",{}).get("exploratory_positive_delta_and_ci") is not True
        or len(rows)!=252
        or len({r["query_path"] for r in rows})!=252
        or len({r["formula"] for r in rows})!=44):
        raise RuntimeError("original full MassBank graph transfer result changed or missing")
    for x in rows:
        for k in ("graph_mrr","exact_random_label_null_mrr",
                  "graph_top1_fractional","uniform_label_top1_fractional"):
            if not np.isfinite(x[k]) or not 0<=x[k]<=1:
                raise RuntimeError("nonfinite or out-of-range original result")
    full=grouped_stats(rows)
    if (abs(full["mean_delta"]-
            old["all_original_252"]["formula_block_bootstrap_95"]["delta"])>1e-12
        or abs(full["mean_graph_mrr"]-
               old["all_original_252"]["mean_graph_mrr"])>1e-12):
        raise RuntimeError("published source MRR not exactly reproduced")
    positive=[x for x in rows if x["ion_mode"]=="POSITIVE"]
    negative=[x for x in rows if x["ion_mode"]=="NEGATIVE"]
    if len(positive)!=160 or len(negative)!=92:
        raise RuntimeError("original positive/negative subgroup denominator drifted")
    no_prior=[x for x in rows if x["no_prior_msnlib_structures_in_pool"]]
    if len(no_prior)!=230 or len({x["formula"] for x in no_prior})!=40:
        raise RuntimeError("original no prior MSnLib subgroup changed")
    modes={
        "positive":grouped_stats(positive,20261017),
        "negative":grouped_stats(negative,20261018),
        "positive_and_no_prior_MSnLib_structure":grouped_stats(
            [x for x in no_prior if x["ion_mode"]=="POSITIVE"],20261019),
        "negative_and_no_prior_MSnLib_structure":grouped_stats(
            [x for x in no_prior if x["ion_mode"]=="NEGATIVE"],20261020),
    }
    contributors=Counter(r["contributor"] for r in rows)
    contributors_sorted=sorted(contributors)
    leave_contrib={}
    for c in contributors_sorted:
        remain=[x for x in rows if x["contributor"]!=c]
        leave_contrib[c]=grouped_stats(remain,20261021)
    bycontrib={}
    for c in contributors_sorted:
        bycontrib[c]=grouped_stats(
            [x for x in rows if x["contributor"]==c],20261022)
    influences=influence(rows)
    result={
        "status":"POSTHOC_MASSBANK_GRAPH_ONLY_STRATIFIED_ROBUSTNESS_20261017",
        "original_real_data_run":ORIGINAL_RUN,
        "original_result_sha256":hashlib.sha256(
            (prior/"results.json").read_bytes()).hexdigest(),
        "original_query_results_sha256":hashlib.sha256(
            (prior/"per_query_ranking.jsonl").read_bytes()).hexdigest(),
        "no_raw_spectra_or_graph_predictions_opened":True,
        "all_original_252":full,
        "by_native_ion_mode_and_nonoverlap":modes,
        "by_original_contributor":bycontrib,
        "leave_one_contributor_out":leave_contrib,
        "worst_case_formula_deletion":{
            k:v for k,v in influences.items() if k!="per_formula_contributions"},
        "not_new_independent_replication":True,
        "limitations":"All subgroup and deletion tests are retrospective after source "
            "graph score revealed. The original graph fragment ion shifts are "
            "from a positive-ion design and lack a validated negative-ion mechanism; "
            "conditional exact-label baseline is NOT an official chemical database "
            "candidate retrieval experiment.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (out/"per_formula_contributions.jsonl").open("w") as f:
        for r in influences["per_formula_contributions"]:
            f.write(json.dumps(r,sort_keys=True)+"\n")
    report=[
        "# MassBank graph-only second archive: independent artifact-only robustness",
        "",
        f"Exact original {len(rows)} query records, {full['n_unique_formula_blocks']} formula groups, original +{full['mean_delta']:.6f} MRR versus exact conditional random-label null.",
        "",
        "| Population | n | formulas | ΔMRR graph−null | 95% formula bootstrap |",
        "|---|---:|---:|---:|---|",
    ]
    for name,stats in [
        ("All unchanged",full),
        ("Positive ion only (mechanism-relevant)",modes["positive"]),
        ("Negative ion only (mechanism unvalidated)",modes["negative"]),
        ("Positive, no prior MSnLib structures",
         modes["positive_and_no_prior_MSnLib_structure"]),
        ("Negative, no prior MSnLib structures",
         modes["negative_and_no_prior_MSnLib_structure"]),
    ]:
        if stats is None:continue
        ci=stats["formula_cluster_bootstrap_95"]
        report.append(f"| {name} | {stats['n_queries']} | {stats['n_unique_formula_blocks']} | "
                      f"{stats['mean_delta']:+.6f} | [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}] |")
    lo=influences
    report+=["",f"All single-formula deletions retain positive ΔMRR: {lo['all_single_formula_deletions_remain_positive']}; "
             f"worst ΔMRR {lo['smallest_leave_one_formula_out_delta']:+.6f} after removing "
             f"{lo['smallest_leave_one_formula_out_formula']}.",
             "", "| Deliberately delete top positive formulas | Removed queries | Remaining ΔMRR |",
             "|---|---:|---:|"]
    for k,s in lo["top_favorable_formula_group_deletions_posthoc"].items():
        report.append(f"| {k} | {s['removed_query_records']} | {s['remaining_mean_delta']:+.6f} |")
    report+=["","Original contributors and formula-preserving leave-one-contributor-out:"]
    for c in contributors_sorted:
        b=bycontrib[c]; l=leave_contrib[c]
        report.append(f"- {c}: {b['n_queries']} original queries, own ΔMRR {b['mean_delta']:+.6f}; "
                      f"when removed remaining {l['n_queries']} queries ΔMRR {l['mean_delta']:+.6f}")
    report+=["",result["limitations"],""]
    (out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.prior,a.out)
