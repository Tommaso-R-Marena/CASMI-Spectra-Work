"""Retrospective independent-lab support-mechanism falsification and grid audit.

Compares the SAME original 50 complete PhenolicsDB CID query targets. Replays
SHA-fixed original 20/80 endpoint predictions, never picks a favorable subset,
and checks original published 0.01 Da per-query scores before new diagnostics.
This is AFTER the prior source outcomes; not independent prospective inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from audit_phenolicsdb_independent_gate import verify_source,CE_VALUES,SOURCE_BLOB_SHA1
from benchmark_phenolicsdb_independent_cid import load_selected
from qfd.power_geometry import power_mean_distribution
from qfd.persistent_survival import persistent_hellinger_survival
from qfd.support_factorization import (
    fixed_mechanism_predictions,reference_support_diagnostics)
from qfd.target_blind_geometry import (
    reference_only_grid,query_mass_on_reference_grid,affinity,
    support_only_suppression)
from qfd.stats import paired_block_bootstrap_delta,paired_win_loss_tie

TOLERANCES=(0.0025,0.005,0.01,0.02)
METHODS=("quarter","hellinger","support","hybrid")
COMPARISONS=(("quarter_vs_hellinger","quarter","hellinger"),
             ("quarter_vs_support","quarter","support"),
             ("quarter_vs_hybrid","quarter","hybrid"),
             ("hybrid_vs_support","hybrid","support"))
KEY=("inchikey","polarity","target_energy_ev")


def exact_block_signflip(delta,blocks):
    """Enumerate ALL 2^K chemical-identity sign flips, if K<=20.

    Each compound's two energies and both polarities share ONE sign.
    No Monte Carlo p-value floor or pseudoreplication of target scans.
    """
    xs=np.asarray(delta,dtype=float)
    if xs.ndim!=1 or len(xs)!=len(blocks) or not len(xs) or not np.isfinite(xs).all():
        raise ValueError("nonfinite or incomplete paired identity-block contrast")
    names=sorted(set(blocks))
    if len(names)>20:
        raise ValueError("exact enumeration reserved for at most 20 independent identity blocks")
    contributions=np.asarray([sum(float(d) for d,b in zip(xs,blocks) if b==name)
                              for name in names],float)
    observed=abs(float(contributions.sum()))
    numbers=np.arange(1<<len(names),dtype=np.uint32)
    totals=np.zeros(len(numbers),float)
    for j,v in enumerate(contributions):
        totals+=v*(1.-2.*((numbers>>j)&1))
    extreme=int(np.count_nonzero(np.abs(totals)>=observed-1e-12))
    return {"p_two_sided":float(extreme/len(numbers)),
            "exact_assignments":len(numbers),
            "extreme_assignments":extreme,
            "distinct_structure_blocks":len(names)}


def compare(rows,left,right,seed):
    a=np.asarray([r[left] for r in rows],float)
    b=np.asarray([r[right] for r in rows],float)
    blocks=[r["inchikey"] for r in rows]
    return {
        "mean_delta":float(np.mean(a-b)),
        "wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
        "identity_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,blocks,n_boot=5000,seed=seed),
        "exact_structure_signflip":exact_block_signflip(a-b,blocks),
    }


def report(rows):
    by_tolerance={}
    for j,tol in enumerate(TOLERANCES):
        subset=[r for r in rows if r["tolerance_da"]==tol]
        if len(subset)!=50:
            raise RuntimeError("missing a query in tolerance sensitivity, must retain all 50")
        comparisons={name:compare(subset,a,b,20261017+i+31*j)
                     for i,(name,a,b) in enumerate(COMPARISONS)}
        by_tolerance[str(tol)]={
            "n_queries":len(subset),
            "n_independent_structures":len(set(r["inchikey"] for r in subset)),
            "mean_scores":{m:float(np.mean([r[m] for r in subset])) for m in METHODS},
            "mean_shared_endpoint_bc":float(np.mean([r["reference_endpoint_bc"] for r in subset])),
            "n_zero_shared_reference_bins":sum(r["shared_bin_count"]==0 for r in subset),
            "mean_target_unmatched_mass":float(np.mean([r["unmatched_target_mass"] for r in subset])),
            "contrasts":comparisons,
            "by_polarity":{mode:{
                "n":len([r for r in subset if r["polarity"]==mode]),
                "quarter_minus_hellinger":float(np.mean([
                    r["quarter"]-r["hellinger"] for r in subset if r["polarity"]==mode])),
                "quarter_minus_support":float(np.mean([
                    r["quarter"]-r["support"] for r in subset if r["polarity"]==mode])),
            } for mode in ("Positive","Negative")},
        }
    return by_tolerance


def evaluate(pos,neg,gatedir,baseline,out):
    meta=json.loads((gatedir/"feasibility.json").read_text())
    quads=[json.loads(s) for s in (gatedir/"frozen_quadruplets.jsonl").read_text().splitlines() if s.strip()]
    if (meta.get("status")!="PHENOLICSDB_INDEPENDENT_CID_METADATA_GATE_V1"
        or meta.get("gate_pass") is not True or meta.get("source_blob_sha1")!=SOURCE_BLOB_SHA1
        or meta.get("coverage",{}).get("complete_quadruplets")!=len(quads)
        or len(quads)!=25):
        raise RuntimeError("original frozen independent PhenolicsDB eligibility was changed")
    for mode,path in (("Positive",pos),("Negative",neg)):
        verify_source(path,mode)
    original=[json.loads(s) for s in baseline.read_text().splitlines() if s.strip()]
    if len(original)!=50 or len({tuple(z[k] for k in KEY) for z in original})!=50:
        raise RuntimeError("original 50-source score rows altered or duplicated")
    baseline_index={tuple(z[k] for k in KEY):z for z in original}
    need=defaultdict(set)
    for r in quads:
        need[r["polarity"]].update(r["source_indices"].values())
    scans={"Positive":load_selected(pos,"Positive",need["Positive"]),
           "Negative":load_selected(neg,"Negative",need["Negative"])}
    outrows=[]
    max_replay_error=0.
    for rec in quads:
        k,pol=rec["inchikey"],rec["polarity"]
        spec={}
        for e in CE_VALUES:
            m,s=scans[pol][rec["source_indices"][str(e)]]
            if m["inchikey"]!=k or m["ce"]!=e or m["mode"]!=pol:
                raise RuntimeError("matched chemical identity changed in original scans")
            spec[e]=s
        for tol in TOLERANCES:
            grid=reference_only_grid(spec[20],spec[80],tolerance_da=tol)
            a,b=grid.low,grid.high
            diagnostics=reference_support_diagnostics(a,b)
            for e in (40,60):
                t=(e-20)/60
                aligned,unmatched=query_mass_on_reference_grid(spec[e],grid,tolerance_da=tol)
                if abs(float(aligned.sum())+unmatched-1)>1e-7:
                    raise RuntimeError("query signal mass is not conserved")
                p=fixed_mechanism_predictions(a,b,t)
                original_quarter=power_mean_distribution(a,b,t,.25)
                original_support=support_only_suppression(a,b,t,power=4)
                original_hybrid=persistent_hellinger_survival(a,b,t)
                if not np.allclose(p["quarter_factored"],original_quarter,atol=1e-12,rtol=1e-12):
                    raise RuntimeError("exact power factorization identity violated")
                if not np.allclose(p["shared_linear_exclusive_fourth"],original_support,atol=1e-12,rtol=1e-12):
                    raise RuntimeError("support-only shared/exclusive identity violated")
                if not np.allclose(p["shared_hellinger_exclusive_fourth"],original_hybrid,atol=1e-12,rtol=1e-12):
                    raise RuntimeError("persistent survival hybrid identity violated")
                pred={
                    "quarter":original_quarter,
                    "hellinger":power_mean_distribution(a,b,t,.5),
                    "support":original_support,"hybrid":original_hybrid,
                }
                scores={name:affinity(aligned,dist) for name,dist in pred.items()}
                if any(not math.isfinite(s) or s<0 or s>1+1e-9 for s in scores.values()):
                    raise RuntimeError("out-of-range predicted affinity")
                key=(k,pol,e)
                if key not in baseline_index:
                    raise RuntimeError("new query not in original metadata-selected cohort")
                if tol==.01:
                    baseline_row=baseline_index[key]
                    for old,new in (("quarter","quarter"),("hellinger","hellinger"),
                                    ("support_only","support")):
                        err=abs(scores[new]-float(baseline_row[old]))
                        max_replay_error=max(max_replay_error,err)
                        if err>1e-10:
                            raise RuntimeError("historical fixed-scorer result disagrees with replay")
                outrows.append({
                    "inchikey":k,"polarity":pol,"target_energy_ev":e,
                    "tolerance_da":tol,"unmatched_target_mass":unmatched,
                    "reference_bin_count":len(a),
                    **diagnostics,**scores,
                })
    if len(outrows)!=len(original)*len(TOLERANCES):
        raise RuntimeError("all four tolerance grids must retain the same 50 queries")
    result={
        "status":"POSTHOC_PHENOLICSDB_EXACT_SUPPORT_FACTOR_AND_MZ_SENSITIVITY",
        "inferential_status":"RETROSPECTIVE: source outcomes observed in PR17 before analysis developed",
        "source_replay_run":39048117682,
        "original_source_blob_sha1":SOURCE_BLOB_SHA1,
        "n_original_queries":len(original),
        "n_original_chemical_identities":len({r["inchikey"] for r in original}),
        "original_replay_max_absolute_error":max_replay_error,
        "method_definition":"same exclusive fourth-power terms for quarter, hybrid, support; change shared-bin alpha only",
        "metadata_selection_or_predictor_fitting":False,
        "interpretation_warning":"Telescoping contrasts involve global normalization and do not identify causal fragmentation processes. Alignment sensitivity and small 15-structure cohort are posthoc exploratory.",
        "by_tolerance":report(outrows),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (out/"all_rows.jsonl").open("w") as f:
        for r in outrows:f.write(json.dumps(r,sort_keys=True)+"\n")
    lines=[
        "# Retrospective PhenolicsDB support vs shared curvature falsification",
        "",
        "Verified exact score replay for original 50-query original source, no refits. "
        f"Maximum original-scorer absolute difference: {max_replay_error:.3e}.",
        "",
        "| Tolerance (Da) | Quarter−Hellinger | Quarter−support | Quarter−hybrid | Hybrid−support | p exact Q−support |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tol,v in result["by_tolerance"].items():
        cs=v["contrasts"]
        get=lambda name:cs[name]["mean_delta"]
        lines.append(f"| {tol} | {get('quarter_vs_hellinger'):+.6f} | "
                     f"{get('quarter_vs_support'):+.6f} | "
                     f"{get('quarter_vs_hybrid'):+.6f} | "
                     f"{get('hybrid_vs_support'):+.6f} | "
                     f"{cs['quarter_vs_support']['exact_structure_signflip']['p_two_sided']:.6f} |")
    lines += ["",
              "**Prior 0.01 Da quarter-vs-Hellinger result remains a prior frozen result; ALL new ablations are posthoc descriptive/falsification only.**",
              "",result["interpretation_warning"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("positive","negative","gate_dir","baseline","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    evaluate(a.positive,a.negative,a.gate_dir,a.baseline,a.out)


if __name__=="__main__":
    main()
