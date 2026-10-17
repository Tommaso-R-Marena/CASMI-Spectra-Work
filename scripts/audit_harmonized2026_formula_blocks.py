"""Original cohort-only audit: formula-block uncertainty and hard-pool strata.

POSTHOC following frozen real first-look run 39056094950. Never reopen the
443905-peak source; no new methods or tuning allowed. Null results retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from qfd.stats import (
    paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie)

ORIGINAL_RUN=39056094950
SHA256="aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e"
ORIGINAL_MRR={
    "cosine":.808541,"hellinger":.811962,
    "fragment_overlap":.806652,"support_composite":.807687,
}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def compare(rr,left,right,seed):
    a=np.asarray([r["metrics"][left]["mrr"] for r in rr],float)
    b=np.asarray([r["metrics"][right]["mrr"] for r in rr],float)
    return {
        "mean_delta":float(np.mean(a-b)),
        "wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
        "formula_cluster_bootstrap_95":paired_block_bootstrap_delta(
            a,b,[r["formula"] for r in rr],n_boot=10000,seed=seed),
        "formula_cluster_signflip":paired_block_signflip_test(
            a,b,[r["formula"] for r in rr],n_permutations=100000,seed=seed),
    }


def evaluate(prior:Path,out:Path):
    original=json.loads((prior/"results.json").read_text())
    data=rows(prior/"query_metrics.jsonl")
    if (original.get("status")!="FIRST_LOOK_HARMONIZED2026_GNPS_TO_MSG_SAME_FORMULA_IDENTIFICATION"
        or original.get("source_sha256")!=SHA256
        or original.get("original_two_polarity_gate_failed") is not True
        or original.get("new_positive_only_v2_prescore_gate_passed") is not True
        or original.get("predeclared_joint_mrr_test_passed") is not False
        or original.get("n_hard_queries")!=4398 or len(data)!=4398
        or original.get("n_distinct_formulas")!=2254
        or original.get("exact_identical_query_reference_spectra")!=0):
        raise RuntimeError("original negative firstlook artifact or entire 4398 cohort altered")
    for method,v in ORIGINAL_MRR.items():
        x=float(np.mean([r["metrics"][method]["mrr"] for r in data]))
        if abs(x-v)>1e-6 or abs(x-original["mean_mrr"][method])>1e-12:
            raise RuntimeError("cannot reproduce original frozen full-cohort MRR")
    if len({r["identity"] for r in data})!=4398:
        raise RuntimeError("original one-query-per-structure invariant altered")
    groupbins={
        "2_candidates":lambda n:n==2,
        "3_to_4_candidates":lambda n:3<=n<=4,
        "5_to_9_candidates":lambda n:5<=n<=9,
        "10_or_more_candidates":lambda n:n>=10,
    }
    bins={}
    for name,rule in groupbins.items():
        rr=[r for r in data if rule(r["candidate_count"])]
        bins[name]={
            "n":len(rr),
            "mean_cosine_mrr":float(np.mean([r["metrics"]["cosine"]["mrr"] for r in rr])) if rr else None,
            "mean_hellinger_mrr":float(np.mean([r["metrics"]["hellinger"]["mrr"] for r in rr])) if rr else None,
            "mean_support_mrr":float(np.mean([r["metrics"]["support_composite"]["mrr"] for r in rr])) if rr else None,
        }
    if sum(x["n"] for x in bins.values())!=len(data):
        raise RuntimeError("not all hard candidate groups represented")
    tests={
        "support_vs_cosine":compare(data,"support_composite","cosine",20261017),
        "support_vs_hellinger":compare(data,"support_composite","hellinger",20261018),
    }
    report={
        "status":"POSTHOC_HARMONIZED2026_FORMULA_CLUSTER_AUDIT",
        "source_original_real_scorer":ORIGINAL_RUN,
        "n_hard_queries":4398,
        "n_distinct_native_formulas":2254,
        "n_distinct_full_inchikeys":4398,
        "source_sha256":SHA256,
        "new_scoring_or_optimization_performed":False,
        "posthoc_not_preregistered":True,
        "all_original_negatives_retained":True,
        "formula_block_sensitivity":tests,
        "candidate_count_strata":bins,
        "limitations":"Conservative formula grouping shares correlated candidate pools. This is a retrospective secondary uncertainty audit, not first-look confirmation, and only positive-ion cross-archive retrieval.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    lines=["# Original 4398-query frozen GNPS/MassSpecGym negative firstlook — formula-cluster audit",
           "",
           "No spectral source data opened or rescored. All original chemical structures retained.",
           "",
           "| Contrast | Mean MRR difference | Formula-block bootstrap 95% CI | Formula-block signflip p |",
           "|---|---:|---|---:|"]
    for name,t in tests.items():
        v=t["formula_cluster_bootstrap_95"]
        lines.append(f"| {name} | {t['mean_delta']:+.6f} | "
                     f"[{v['ci_low']:+.6f},{v['ci_high']:+.6f}] | "
                     f"{t['formula_cluster_signflip']['p_value_two_sided']:.6f} |")
    lines+=["","| Hard candidate pool | n | MRR cosine | MRR Hellinger | MRR composite |",
            "|---|---:|---:|---:|---:|"]
    for name,v in bins.items():
        x=lambda key:"—" if v[key] is None else f"{v[key]:.6f}"
        lines.append(f"| {name} | {v['n']} | {x('mean_cosine_mrr')} | "
                     f"{x('mean_hellinger_mrr')} | {x('mean_support_mrr')} |")
    lines+=["","**Secondary retrospective audit; cannot rescue or overturn original failed primary.**",""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.prior_dir,a.out)

if __name__=="__main__":main()
