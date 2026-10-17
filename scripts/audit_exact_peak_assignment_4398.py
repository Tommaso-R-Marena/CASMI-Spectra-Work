"""Exact MS/MS peak matching vs historical greedy alignment on frozen 4398 queries.

After-outcome methodological analysis. Do not call this a pristine lockbox.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_harmonized2026_crossprovider_gate import (
    SHA256,MD5,sha256file,meta_only,freeze)
from benchmark_harmonized2026_crossprovider import (
    read_rows,extract_selected_peaks,scores_ion_spectra,tie_aware_rank)
from qfd.exact_peak_assignment import METHODS,optimal_ion_spectra
from qfd.stats import (
    paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie)

ORIGINAL_RUN=39056094950
BASELINE_NAMES=METHODS
PRIMARY=(
    ("exact_hellinger_vs_greedy_hellinger","exact_hellinger","greedy_hellinger"),
    ("exact_hellinger_vs_greedy_cosine","exact_hellinger","greedy_cosine"),
)


def digest(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(rows,left,right,seed):
    a=np.asarray([r["metrics"][left]["mrr"] for r in rows],float)
    b=np.asarray([r["metrics"][right]["mrr"] for r in rows],float)
    blocks=[r["formula"] for r in rows]
    return {
        "mean_mrr_delta":float(np.mean(a-b)),
        "wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,blocks,n_boot=10000,seed=seed),
        "formula_block_signflip":paired_block_signflip_test(
            a,b,blocks,n_permutations=50000,seed=seed)
    }


def evaluate(source:Path,gate_dir:Path,v2_dir:Path,original_dir:Path,out:Path):
    gate=json.loads((gate_dir/"feasibility.json").read_text())
    v2=json.loads((v2_dir/"feasibility.json").read_text())
    orig=json.loads((original_dir/"results.json").read_text())
    original_rows=read_rows(original_dir/"query_metrics.jsonl")
    references=read_rows(gate_dir/"reference_manifest.jsonl")
    queries=read_rows(gate_dir/"hard_queries.jsonl")
    if (
        gate.get("status")!="HARMONIZED2026_CROSS_PROVIDER_NATIVE_HEADER_GATE_V1"
        or gate.get("gate_pass") is not False
        or gate.get("coverage",{}).get("hard_positive")!=4398
        or gate.get("coverage",{}).get("hard_negative",0)!=0
        or gate.get("source_sha256")!=SHA256
        or gate.get("query_peak_locations_or_intensities_opened") is not False
        or v2.get("status")!="HARMONIZED2026_POSITIVE_ONLY_NATIVE_COHORT_V2"
        or v2.get("gate_pass") is not True
        or v2.get("original_v1_two_polarity_scientific_pass") is not False
        or v2.get("original_v1_feasibility_sha256")!=digest(gate_dir/"feasibility.json")
        or v2.get("original_v1_queries_sha256")!=digest(gate_dir/"hard_queries.jsonl")
        or v2.get("original_v1_reference_sha256")!=digest(gate_dir/"reference_manifest.jsonl")
        or v2.get("source_query_spectra_opened") is not False
        or orig.get("status")!="FIRST_LOOK_HARMONIZED2026_GNPS_TO_MSG_SAME_FORMULA_IDENTIFICATION"
        or orig.get("source_sha256")!=SHA256
        or orig.get("n_hard_queries")!=4398
        or orig.get("n_distinct_formulas")!=2254
        or orig.get("predeclared_joint_mrr_test_passed") is not False
        or len(original_rows)!=4398 or len(queries)!=4398
        or len(references)!=119863
        or any(q["mode"]!="positive" for q in queries)
    ):
        raise RuntimeError("original negative first-look sources/gates changed; no new peaks authorized")
    original={(r["identity"],r["query_index"]):r for r in original_rows}
    if len(original)!=4398:
        raise RuntimeError("duplicate original full-key query records")
    sha256file(source)
    metadata,raw=meta_only(source)
    ref_now,q_now,cov,all_polarity_pass=freeze(metadata)
    if (all_polarity_pass or raw!=gate["raw_header_audit"] or
        cov!=gate["coverage"] or ref_now!=references or q_now!=queries):
        raise RuntimeError("cross-provider exact native chemical cohort replay failed")
    needed={q["index"] for q in queries}
    needed.update(k for q in queries for k in q["selected_reference_indices"])
    spectra=extract_selected_peaks(source,needed)
    refs={r["index"]:r for r in references}
    rows=[]
    comparisons=Counter()
    ambiguous_query_count=0
    rank_switch_counts=Counter()
    for q in queries:
        obs=spectra[q["index"]]
        true=q["inchikey"]
        candidates={f"{tag}_{method}":{} for tag in ("greedy","exact") for method in METHODS}
        ambiguous_pairs=0
        identities=set()
        for scan in q["selected_reference_indices"]:
            meta=refs[scan]
            if (meta["source"]!="GNPS" or meta["mode"]!=q["mode"] or
                meta["formula"]!=q["formula"] or
                abs(meta["precursor_mz"]-q["precursor_mz"])>
                    max(.02,20e-6*q["precursor_mz"])+1e-12):
                raise RuntimeError("reference formula, precursor, or polarity mismatch")
            key=meta["inchikey"]
            identities.add(key)
            reference=spectra[scan]
            greedy=scores_ion_spectra(obs,reference)
            exact,cert=optimal_ion_spectra(obs,reference)
            comparisons["candidate_scan_pairs"]+=1
            if cert["n_ambiguous_components"]>0:
                ambiguous_pairs+=1
                comparisons["candidate_scan_pairs_with_ambiguous_edges"]+=1
            for name in METHODS:
                if exact[name]+1e-10<greedy[name]:
                    raise RuntimeError("global objective lower than feasible greedy matching")
                if exact[name]>greedy[name]+1e-10:
                    comparisons[f"{name}_pairs_strict_gain"]+=1
                for tag,values in (("greedy",greedy),("exact",exact)):
                    slot=candidates[f"{tag}_{name}"]
                    slot[key]=max(slot.get(key,-1.),values[name])
        if len(identities)!=q["candidate_structures"] or len(identities)<2 or true not in identities:
            raise RuntimeError("original exact native candidate inventory changed")
        ranks={method:tie_aware_rank(score,true) for method,score in candidates.items()}
        prior=original.get((true,q["index"]))
        if prior is None or prior["candidate_count"]!=len(identities):
            raise RuntimeError("original query identity/candidate inventory changed")
        for name in METHODS:
            for key in ("mrr","top1_fractional","midrank","candidate_count"):
                if abs(ranks[f"greedy_{name}"][key]-prior["metrics"][name][key])>1e-12:
                    raise RuntimeError(f"ORIGINAL 4398-QUERY RANK REPLAY FAILED for {name}, {key}")
        if ambiguous_pairs:ambiguous_query_count+=1
        for name in METHODS:
            if ranks[f"exact_{name}"]["midrank"]!=ranks[f"greedy_{name}"]["midrank"]:
                rank_switch_counts[name]+=1
        rows.append({
            "identity":true,
            "query_index":q["index"],
            "formula":q["formula"],
            "candidate_count":len(identities),
            "number_ambiguous_candidate_scan_pairs":ambiguous_pairs,
            "metrics":ranks,
        })
    if len(rows)!=4398 or len({r["formula"] for r in rows})!=2254:
        raise RuntimeError("original complete population changed")
    metrics={name:{
        "mrr":float(np.mean([r["metrics"][name]["mrr"] for r in rows])),
        "fractional_top1":float(np.mean([r["metrics"][name]["top1_fractional"] for r in rows])),
    } for name in candidates}
    for method in METHODS:
        if abs(metrics[f"greedy_{method}"]["mrr"]-orig["mean_mrr"][method])>1e-12:
            raise RuntimeError("original aggregate source result not exactly reproduced")
    tests={name:infer(rows,left,right,20261017+129*i)
           for i,(name,left,right) in enumerate(PRIMARY)}
    order=sorted((v["formula_block_signflip"]["p_value_two_sided"],name)
                 for name,v in tests.items())
    last=0.
    for i,(p,name) in enumerate(order):
        last=max(last,min(1.,(len(order)-i)*p))
        tests[name]["holm_p"]=last
    both=all(
        value["mean_mrr_delta"]>0 and
        value["formula_block_bootstrap_95"]["ci_low"]>0 and value["holm_p"]<.05
        for value in tests.values())
    result={
        "status":"RETROSPECTIVE_2026_EXACT_GLOBAL_PEAK_MATCHING_4398",
        "scientific_result_is_post_outcome_exploratory":True,
        "original_negative_run":ORIGINAL_RUN,
        "original_two_polarity_v1_failed":True,
        "positive_only_v2_prescore_passed":True,
        "source_sha256":SHA256,
        "n_queries":len(rows),"n_formula_blocks":2254,
        "n_native_full_inchikeys":4398,
        "identical_original_greedy_rank_replay":"ALL_4398_QUERY_TIMES_FOUR_METHODS",
        "no_fitted_parameters":True,
        "m_z_tolerance_da":.01,
        "peak_matching":"per-score exact maximum-weight one-to-one bipartite assignment",
        "n_queries_with_any_ambiguous_candidate_pair":ambiguous_query_count,
        "candidate_pair_diagnostics":dict(comparisons),
        "queries_with_changed_true_structure_midrank":dict(rank_switch_counts),
        "metrics":metrics,
        "formula_cluster_retrospective_primary_diagnostics":tests,
        "all_diagnostics_positive":both,
        "would_require_untouched_independent_external_acquisition_for_breakthrough":True,
        "limitations":"Same 2026 GNPS/MassSpecGym positive-ion benchmark already examined; computationally harmonized identity labels, overlapping original physical source possible, no negative-ion evidence, no energy-model kinetic law.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (out/"rank_rows.jsonl").open("w") as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
    lines=[
        "# Globally optimal peak assignment vs greedy peak-mass matching",
        "",
        "**RETROSPECTIVE after earlier 4,398-query outcome. NOT independent external confirmation.**",
        f"Original all-query MRR/top1 replay exact. {ambiguous_query_count} of 4,398 queries have an ambiguous matching component in at least one candidate scan pair.",
        "",
        "| Fixed similarity functional | Historical greedy MRR | Globally optimal MRR | Optimal Top-1 |",
        "|---|---:|---:|---:|",
    ]
    for name in METHODS:
        lines.append(f"| {name} | {metrics['greedy_'+name]['mrr']:.6f} | "
                     f"{metrics['exact_'+name]['mrr']:.6f} | "
                     f"{metrics['exact_'+name]['fractional_top1']:.6f} |")
    lines += ["","### Frozen exploratory diagnostics, formula blocks", ""]
    for name,v in tests.items():
        ci=v["formula_block_bootstrap_95"]
        lines.append(f"- {name}: MRR difference {v['mean_mrr_delta']:+.6f}, "
                     f"95% CI [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], "
                     f"Holm p={v['holm_p']:.6f}; {v['wins_losses_ties']}")
    lines += ["",f"Both tests pass: **{both}**.",
              "Optimal pair affinity is mathematically nondecreasing for each method. "
              "Molecular-rank effects are empirical and may regress.",
              "",result["limitations"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--positive-v2-dir",type=Path,required=True)
    p.add_argument("--original-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.source,a.gate_dir,a.positive_v2_dir,a.original_dir,a.out)

if __name__=="__main__":main()
