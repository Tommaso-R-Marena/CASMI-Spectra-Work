"""Spectral entropy comparator: test rarity against a stronger real-data baseline.

One original source, immutable targets/candidates, no method weights fitted.
Predeclared formula-block Holm-adjusted requirement that the rarity blend beat
BOTH frozen Hellinger and unweighted spectral entropy, not just Hellinger.
This remains retrospective and never an independent laboratory replication.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_harmonized2026_crossprovider_gate import SHA256
from benchmark_harmonized2026_crossprovider import (
    evaluate as replay_original, read_rows, extract_selected_peaks,
    scores_ion_spectra, tie_aware_rank,
)
from benchmark_harmonized2026_reference_rarity import (
    molecular_documents, weighted_hellinger, grouped_comparison)

LN2=math.log(2.0)
TOL=.01


def unweighted_entropy(query,reference,tol=TOL):
    """1-JensenShannon(p,q)/ln2 via positive contributions for matched ions.

    Full unmatched intensities are present in original L1 normalization and
    implicitly lower similarity. One-to-one greedy mass alignment is unchanged.
    This is unweighted entropy, not the published entropy intensity transform.
    """
    qm,qi=query
    rm,ri=reference
    if (not len(qm) or not len(rm)
        or np.any(qi<0) or np.any(ri<0)
        or not np.all(np.isfinite(qi)) or not np.all(np.isfinite(ri))
        or qi.sum()<=0 or ri.sum()<=0):
        raise ValueError("nonempty positive finite spectra required")
    qp=qi/qi.sum()
    rp=ri/ri.sum()
    edges=[]
    for i,mass in enumerate(qm):
        left=int(np.searchsorted(rm,mass-tol,side="left"))
        right=int(np.searchsorted(rm,mass+tol,side="right"))
        for j in range(left,right):
            error=abs(float(rm[j]-mass))
            if error<=tol:edges.append((error,j,i))
    edges.sort()
    taken_q,taken_r=set(),set()
    affinity=0.
    for _,j,i in edges:
        if i in taken_q or j in taken_r:continue
        taken_q.add(i);taken_r.add(j)
        a=float(qp[i]);b=float(rp[j])
        if a and b:
            # From 1 - normalized JS on the union peak grid; exact and
            # numerically stable for positive matched intensities.
            affinity+=((a+b)*math.log(a+b)-a*math.log(a)-b*math.log(b))/(2*LN2)
    if not -1e-9<=affinity<=1+1e-9:
        raise RuntimeError("JS-derived entropy similarity outside [0,1]")
    return max(0.,min(1.,affinity))


def evaluate(source:Path,gate:Path,v2:Path,out:Path):
    original=replay_original(source,gate,v2,out/"original_control")
    if (original["source_sha256"]!=SHA256
        or original["n_hard_queries"]!=4398
        or original["n_distinct_formulas"]!=2254
        or original["predeclared_joint_mrr_test_passed"] is not False
        or abs(original["mean_mrr"]["hellinger"]-.811962)>2e-6):
        raise RuntimeError("original frozen negative hypothesis not reproduced")
    qs=read_rows(gate/"hard_queries.jsonl")
    refs={r["index"]:r for r in read_rows(gate/"reference_manifest.jsonl")}
    old=read_rows(out/"original_control"/"query_metrics.jsonl")
    selected={q["index"] for q in qs}
    selected.update(idx for q in qs for idx in q["selected_reference_indices"])
    spectra=extract_selected_peaks(source,selected)
    records=[]
    for q,prior in zip(qs,old):
        if (q["index"]!=prior["query_index"] or
            q["inchikey"]!=prior["identity"] or q["formula"]!=prior["formula"]):
            raise RuntimeError("native query/candidate manifest changed")
        pool=q["selected_reference_indices"]
        docs=molecular_documents(pool,refs,spectra)
        if len(docs)!=q["candidate_structures"]:
            raise RuntimeError("distinct candidate structure count drift")
        escores,rarity_scores={},{}
        for idx in pool:
            label=refs[idx]["inchikey"]
            candidate=spectra[idx]
            observed=spectra[q["index"]]
            ent=unweighted_entropy(observed,candidate)
            rare=weighted_hellinger(observed,candidate,docs)
            hel=scores_ion_spectra(observed,candidate)["hellinger"]
            blend=.5*rare+.5*hel
            escores[label]=max(escores.get(label,-1.),ent)
            rarity_scores[label]=max(rarity_scores.get(label,-1.),blend)
        records.append({
            "identity":q["inchikey"],"formula":q["formula"],
            "query_index":q["index"],"candidate_count":len(docs),
            "metrics":{
                **prior["metrics"],
                "unweighted_entropy":tie_aware_rank(escores,q["inchikey"]),
                "rarity_blend":tie_aware_rank(rarity_scores,q["inchikey"]),
            },
        })
    if len(records)!=4398 or len({r["identity"] for r in records})!=4398:
        raise RuntimeError("not the complete frozen cohort")
    tests={
        "rarity_blend_vs_hellinger":grouped_comparison(
            records,"rarity_blend",seed=20261017),
        "rarity_blend_vs_entropy":grouped_comparison(
            records,"rarity_blend","unweighted_entropy",seed=20261020),
    }
    ordered=sorted((v["formula_signflip"]["p_value_two_sided"],k)
                   for k,v in tests.items())
    adjusted=0.
    for i,(p,name) in enumerate(ordered):
        adjusted=max(adjusted,min(1.,p*(len(ordered)-i)))
        tests[name]["holm_two_sided_p"]=adjusted
    passed=all(
        x["delta_mrr"]>0 and
        x["formula_bootstrap_95"]["ci_low"]>0 and
        x["holm_two_sided_p"]<.05 for x in tests.values())
    entropy_delta=grouped_comparison(
        records,"unweighted_entropy","hellinger",seed=20261021)
    methods=("cosine","hellinger","support_composite",
             "unweighted_entropy","rarity_blend")
    result={
        "status":"EXPLORATORY_RARITY_VS_JS_ENTROPY_STRONGER_CONTROL_20261017",
        "source_doi":"10.5281/zenodo.20042904",
        "source_sha256":SHA256,
        "n_queries":len(records),
        "n_formula_blocks":len({x["formula"] for x in records}),
        "original_harmonized_source_outcomes_previously_published":True,
        "historical_support_composite_hypothesis_failed":True,
        "entropy_type":"UNWEIGHTED JSD 1-JSD/ln2, not weighted spectral entropy",
        "entropy_peak_matching":"0.01Da one-to-one mass matching, full unmatched-intensity denominator",
        "reference_only_rarity_weight":"1+ln((N+1)/(df+1))",
        "fixed_rarity_blend":.5,
        "mean_mrr":{m:float(np.mean([r["metrics"][m]["mrr"] for r in records]))
                    for m in methods},
        "top1_fractional":{m:float(np.mean([r["metrics"][m]["top1_fractional"]
                                          for r in records])) for m in methods},
        "predeclared_two_baseline_joint_tests":tests,
        "joint_exploratory_success":passed,
        "secondary_entropy_vs_hellinger":entropy_delta,
        "novel_physical_law_or_independent_breakthrough_established":False,
        "limitations":"Retrospective previously scored cross-archive source; no independent "
                      "new physical acquisition proven, chemical annotations not curator-verified, "
                      "negative ion absent, weighted entropy and modern learned methods not tested.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    with (out/"complete_molecule_ranks.jsonl").open("w") as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+"\n")
    report=["# Frozen same-source comparison: reference rarity vs unweighted entropy",
            "",f"{len(records)} actual molecule queries; {result['n_formula_blocks']} native formulas.",
            "Retrospective: source outcomes have already been read and cannot confirm a breakthrough.",
            "", "| Method | MRR | Fractional top1 |",
            "|---|---:|---:|"]
    for m in methods:
        report.append(f"| {m} | {result['mean_mrr'][m]:.6f} | "
                      f"{result['top1_fractional'][m]:.6f} |")
    report += [""]
    for name,v in tests.items():
        ci=v["formula_bootstrap_95"]
        report += [f"- {name}: MRR difference {v['delta_mrr']:+.6f}, "
                   f"formula-block bootstrap [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], "
                   f"Holm p={v['holm_two_sided_p']:.6f}, wins/losses/ties {v['wins_losses_ties']}."]
    report += ["",f"**Joint two-baseline exploratory success: {passed}**.",
               "",result["limitations"],""]
    (out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zst",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--positive-v2-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.zst,a.gate_dir,a.positive_v2_dir,a.out)
