"""Replay actual 63-query CID isomer ranking and certify model-change stability.

Uses the ORIGINAL metadata-only CID+EAD exact-formula candidate inventory from
run 39047245864 and unchanged MultiMS2 raw source. The candidate reference
spectra only determine predictions and universal perturbation bounds; target
40-eV peaks are used only for retrospective scoring and rank comparisons.

This is a post-outcome mathematical mechanism audit, NOT a new CASMI test set.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np

import audit_multims2_third_source_metadata as source
from audit_multims2_dual_identification_gate import SOURCE_SHA
from benchmark_multims2_dual_identification import (
    validate_source, rank_scores, candidate_affinity, predictor)
from benchmark_multims2_third_source_geometry import load_selected_source_spectra
from qfd.power_geometry import power_mean_distribution
from qfd.support_factorization import fixed_mechanism_predictions
from qfd.support_certificates import (
    distribution_affinity_difference_upper,
    score_perturbation_certificate,certify_top1_unchanged)
from qfd.target_blind_geometry import reference_only_grid,query_mass_on_reference_grid,affinity

METHODS=("cid_h","cid_q","cid_hybrid","cid_support")
KEY=("group_id","inchikey")


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def replay_reference_predictions(low,high):
    grid=reference_only_grid(low,high,tolerance_da=.01)
    a,b=grid.low,grid.high
    factored=fixed_mechanism_predictions(a,b,.5)
    preds={
        "cid_h":power_mean_distribution(a,b,.5,.5),
        "cid_q":factored["quarter_factored"],
        "cid_hybrid":factored["shared_hellinger_exclusive_fourth"],
        "cid_support":factored["shared_linear_exclusive_fourth"],
    }
    bounds={
        "h_to_q":distribution_affinity_difference_upper(preds["cid_h"],preds["cid_q"]),
        "hybrid_to_q":score_perturbation_certificate(
            a,b,.5,from_shared_alpha=.5,to_shared_alpha=.25)["affinity_uniform_upper"],
        "hybrid_to_support":score_perturbation_certificate(
            a,b,.5,from_shared_alpha=.5,to_shared_alpha=1.)["affinity_uniform_upper"],
    }
    # Validate the specialized shared-only bound never exceeds the universal
    # direct Hellinger distance apart from numeric roundoff.
    assert abs(bounds["hybrid_to_q"]-
               distribution_affinity_difference_upper(
                   preds["cid_hybrid"],preds["cid_q"]))<1e-10
    return grid,preds,bounds


def evaluate(mgf:Path,gate_dir:Path,original_query_scores:Path,out:Path):
    gate=json.loads((gate_dir/"feasibility.json").read_text())
    rows=read(gate_dir/"eligible_isomer_queries.jsonl")
    baseline=read(original_query_scores)
    if (gate.get("status")!="MULTIMS2_DUAL_MODAL_ISOMER_IDENTIFICATION_METADATA_V1"
        or gate.get("gate_pass") is not True
        or gate.get("source_sha256")!=SOURCE_SHA
        or gate.get("target_peak_positions_or_intensities_opened") is not False
        or gate.get("candidate_selection_uses_outcome_scores") is not False
        or gate.get("coverage",{}).get("same_formula_isomer_groups")!=29
        or gate.get("coverage",{}).get("eligible_isomer_queries")!=63
        or len(rows)!=63 or len(baseline)!=63):
        raise RuntimeError("original scored 63-query chemical candidate cohort changed")
    if (mgf.stat().st_size!=source.UPSTREAM_BYTES
        or source.git_blob_sha1(mgf)!=source.UPSTREAM_BLOB_SHA1
        or source.digest(mgf)!=SOURCE_SHA):
        raise RuntimeError("original 77 MB MultiMS2 source mismatched")
    original={tuple(r[x] for x in KEY):r for r in baseline}
    if len(original)!=63:
        raise RuntimeError("repeated baseline true compounds")
    bygroup=defaultdict(list)
    for row in rows:
        bygroup[row["group_id"]].append(row)
    if len(bygroup)!=29:
        raise RuntimeError("exact-formula original group inventory altered")
    cid_ids={r["cid_indices"][role] for r in rows for role in ("low","query","high")}
    # Adapter is deliberately imported only INSIDE actual source-scoring
    # code to avoid mutating shared original parser when pytest imports.
    import benchmark_multims2_third_source_geometry_v2  # noqa: F401
    scans=load_selected_source_spectra(mgf,cid_ids)
    refs={}
    truth={}
    for group_id,members in bygroup.items():
        if len(members)<2 or len(members)!=members[0]["candidate_count"]:
            raise RuntimeError("real same-formula competition required")
        if len({x["inchikey"] for x in members})!=len(members):
            raise RuntimeError("duplicate full chemical identity")
        for row in members:
            low,query,high=validate_source(row,"CID",scans)
            refs[(group_id,row["inchikey"])]=replay_reference_predictions(low,high)
            truth[(group_id,row["inchikey"])]=query
    scored=[]
    for group_id,members in sorted(bygroup.items()):
        for row in members:
            truth_key=row["inchikey"]
            query=truth[(group_id,truth_key)]
            scores={m:{} for m in METHODS}
            eps={m:{} for m in ("h_to_q","hybrid_to_q","hybrid_to_support")}
            for candidate in members:
                candidate_id=candidate["inchikey"]
                grid,preds,bounds=refs[(group_id,candidate_id)]
                aligned,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=.01)
                if abs(float(aligned.sum())+unmatched-1)>1e-7:
                    raise RuntimeError("query is not correctly mass-conserved")
                for method in METHODS:
                    scores[method][candidate_id]=affinity(aligned,preds[method])
                for key in eps:
                    eps[key][candidate_id]=bounds[key]
            ranked={m:rank_scores(scores[m],truth_key) for m in METHODS}
            key=(group_id,truth_key)
            if key not in original:
                raise RuntimeError("candidate target absent from originally scored experiment")
            old=original[key]
            for method in ("cid_h","cid_q"):
                for stat in ("reciprocal_rank","top1_fractional"):
                    if abs(ranked[method][stat]-float(old[method][stat]))>1e-12:
                        raise RuntimeError(f"original identification rank replay FAILED: {method} {stat}")
            proofs={
                "h_to_q":certify_top1_unchanged(scores["cid_h"],eps["h_to_q"]),
                "hybrid_to_q":certify_top1_unchanged(scores["cid_hybrid"],eps["hybrid_to_q"]),
                "hybrid_to_support":certify_top1_unchanged(scores["cid_hybrid"],eps["hybrid_to_support"]),
            }
            for name,target in (("h_to_q","cid_q"),("hybrid_to_q","cid_q"),
                                ("hybrid_to_support","cid_support")):
                cert=proofs[name]
                if cert["certified_unchanged"]:
                    observed_winner=max(sorted(scores[target]),key=lambda k:(scores[target][k],k))
                    if observed_winner!=cert["baseline_winner"]:
                        raise RuntimeError("target-independent proved stability contradicted by real data")
            scored.append({
                "group_id":group_id,"inchikey":truth_key,"formula":row["formula"],
                "candidate_count":len(members),
                "scores_rank":ranked,
                "certification":proofs,
                "ranking_winner":{
                    m:max(sorted(scores[m]),key=lambda k:(scores[m][k],k))
                    for m in METHODS},
            })
    if len(scored)!=63:
        raise RuntimeError("incomplete fixed-score 63-query audit")
    result={
        "status":"POSTHOC_MULTIMS2_CID_ISOMER_RANK_STABILITY_CERTIFICATES",
        "inferential_status":"RETROSPECTIVE. Existing 63 structure rankings and same-lab target data already known. Bounds are mathematical, not new independent efficacy.",
        "replayed_prior_scoring_run":39047245864,
        "source_sha256":SOURCE_SHA,
        "n_heldout_queries":len(scored),
        "n_formula_groups":len(bygroup),
        "n_independent_formula_blocks":len({r["formula"] for r in scored}),
        "original_baseline_replay":"EXACT_CID_H_AND_Q_MRR_TOP1_PER_QUERY",
        "predictor_source":"candidate endpoint reference spectra only; score bounds never inspect target",
        "method_metrics":{m:{
            "mrr":float(np.mean([r["scores_rank"][m]["reciprocal_rank"] for r in scored])),
            "top1_fractional":float(np.mean([r["scores_rank"][m]["top1_fractional"] for r in scored])),
        } for m in METHODS},
        "comparisons":{
            name:{
                "n_certified_unchanged":sum(r["certification"][name]["certified_unchanged"] for r in scored),
                "n_observed_top1_changed":sum(
                    r["ranking_winner"][left]!=r["ranking_winner"][right] for r in scored),
                "n_queries":len(scored),
                "status":"All certifications sound; uncertified cases UNKNOWN, not predicted failures.",
            } for name,left,right in (
                ("h_to_q","cid_h","cid_q"),
                ("hybrid_to_q","cid_hybrid","cid_q"),
                ("hybrid_to_support","cid_hybrid","cid_support"))
        },
        "limitations":(
            "Only 63 queries/29 formula groups of previously studied source, not unknown-molecule "
            "test; these are sufficient reference-only score perturbation certificates, not "
            "guaranteed tight, not a causal model, not demonstrated CASMI generalization."
        ),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    with (out/"rank_certificates.jsonl").open("w") as f:
        for row in scored:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    lines=[
        "# MultiMS2 exact-formula candidate rank robustness (retrospective)",
        "",
        "Original 63 query ranks reproduced **exactly** for CID-Hellinger and CID-quarter.",
        "",
        "| Model | MRR | Fractional top-1 |",
        "|---|---:|---:|",
    ]
    for model,v in result["method_metrics"].items():
        lines.append(f"| {model} | {v['mrr']:.6f} | {v['top1_fractional']:.6f} |")
    lines += ["",
              "| Comparison | Certified invariant top1 | Actual top1 changed |",
              "|---|---:|---:|"]
    for name,r in result["comparisons"].items():
        lines.append(f"| {name} | {r['n_certified_unchanged']}/{r['n_queries']} | "
                     f"{r['n_observed_top1_changed']}/{r['n_queries']} |")
    lines += ["","Certificates need only endpoint reference distributions and baseline "
              "candidate affinity gaps. An uncertified change is UNKNOWN, not a false claim.",
              "",result["limitations"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("mgf","gate_dir","original_query_scores","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    evaluate(a.mgf,a.gate_dir,a.original_query_scores,a.out)


if __name__=="__main__":
    main()
