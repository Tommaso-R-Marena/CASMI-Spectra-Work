"""Exploratory leave-ONE-ACQUISITION-ORIGIN-out endpoint-only model selection.

Three origins: MassBank, ExpoLib, and MultiMS2. The latter's CID + EAD
are held out TOGETHER, never treated as separate independent laboratories.
All outcomes were previously observed and motivated this POST-HOC test.
Not a prospective or externally independent validation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from qfd.stats import paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie

SOURCES={
    "MassBank":(2138,"MassBank"),
    "ExpoLib":(1125,"ExpoLib"),
    "MultiMS2_CID":(1781,"MultiMS2"),
    "MultiMS2_EAD_POSITIVE":(694,"MultiMS2"),
}
THRESHOLDS=[0.,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,
            .65,.70,.75,.80,.85,.90,.95,1.0000001]
OLD_ORIGINS={"MassBank":38918658343,"ExpoLib":38964797817,"MultiMS2":38970610885}
NEW_EAD=38985827065

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def read_training_data(features_path,meta_path,ead_rows_path,ead_result_path):
    meta=json.loads(meta_path.read_text())
    if (meta.get("original_runs")!=OLD_ORIGINS or
        meta.get("no_new_scoring_or_training") is not True):
        raise RuntimeError("unrecognized original three-source descriptive feature artifact")
    orig=read_jsonl(features_path)
    out=defaultdict(list)
    for r in orig:
        name={"MassBank":"MassBank","ExpoLib":"ExpoLib","MultiMS2":"MultiMS2_CID"}.get(r["source"])
        if name is None:raise RuntimeError("unknown frozen source")
        out[name].append({
            "bc":float(r["endpoint_bhattacharyya"]),
            "delta_quarter_hellinger":float(r["delta_quarter_hellinger"]),
            "delta_support_hellinger":float(r["delta_quarter_hellinger"]-r["delta_quarter_support"]),
            "formula":str(r["formula"]),
            "inchikey":str(r["inchikey"]),
        })
    ead_info=json.loads(ead_result_path.read_text())
    if (ead_info.get("status")!="FROZEN_EAD_POSITIVE_ONLY_V2_MODALITY_TRANSFER"
        or ead_info.get("original_v1_gate_failed") is not True
        or ead_info.get("all_positive",{}).get("n_queries")!=694):
        raise RuntimeError("the original V1 failed status or EAD V2 score reference changed")
    for r in read_jsonl(ead_rows_path):
        out["MultiMS2_EAD_POSITIVE"].append({
            "bc":float(r["endpoint_bc"]),
            "delta_quarter_hellinger":float(r["quarter"]-r["hellinger"]),
            "delta_support_hellinger":float(r["support"]-r["hellinger"]),
            "formula":str(r["formula"]),
            "inchikey":str(r["inchikey"]),
        })
    for name,(n,_) in SOURCES.items():
        group=out[name]
        if len(group)!=n or any(
            (not (0<=r["bc"]<=1+1e-9)) or
            any(not np.isfinite(r[x]) for x in ("bc","delta_quarter_hellinger","delta_support_hellinger"))
            or not r["formula"] or not r["inchikey"] for r in group):
            raise RuntimeError(f"original source dataset {name} corrupted or incomplete")
    return out

def mean_policy_delta(rows,threshold,baseline="delta_quarter_hellinger"):
    return float(np.mean([r[baseline] if r["bc"]<threshold else 0. for r in rows]))

def predeclared_training_objective(data,training_labs,threshold):
    """Equal origin weight. CID/EAD are each half of the MultiMS2 origin."""
    lab_scores=[]
    for lab in training_labs:
        groups=[name for name,(_,origin) in SOURCES.items() if origin==lab]
        lab_scores.append(float(np.mean([
            mean_policy_delta(data[name],threshold) for name in groups])))
    return float(np.mean(lab_scores))

def choose(data,training_labs):
    candidates=[
        (predeclared_training_objective(data,training_labs,t),t)
        for t in THRESHOLDS
    ]
    # Tie breaks toward HIGHER threshold (quarter), consistently and without query labels.
    return max(candidates,key=lambda pair:(pair[0],pair[1]))

def infer(rows,threshold,baseline,seed):
    xs=np.asarray([r["delta_quarter_hellinger"] if r["bc"]<threshold else 0. for r in rows])
    if baseline=="Hellinger":
        ys=np.zeros_like(xs)
    elif baseline=="Quarter":
        ys=np.asarray([r["delta_quarter_hellinger"] for r in rows])
    elif baseline=="Support":
        ys=np.asarray([r["delta_support_hellinger"] for r in rows])
    else: raise ValueError("unknown baseline")
    f=[r["formula"] for r in rows]
    return {
        "mean_delta":float(np.mean(xs-ys)),
        "win_loss_tie":paired_win_loss_tie(xs,ys,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(xs,ys,f,n_boot=3000,seed=seed),
        "formula_block_signflip_2sided":paired_block_signflip_test(xs,ys,f,n_permutations=10000,seed=seed),
    }

def evaluate(data):
    labs=sorted(set(origin for n,(_,origin) in SOURCES.items()))
    results={}
    trained={}
    for heldout in labs:
        train=[a for a in labs if a!=heldout]
        gain,t=choose(data,train)
        trained[heldout]={"train_origins":train,"selected_endpoint_overlap_threshold":t,
                          "mean_training_origin_balanced_policy_gain_over_Hellinger":gain}
        for name,(n,origin) in SOURCES.items():
            if origin!=heldout:continue
            rows=data[name]
            predicted=[r["delta_quarter_hellinger"] if r["bc"]<t else 0. for r in rows]
            results[name]={
                "n":n,"n_formulas":len({r["formula"] for r in rows}),
                "heldout_entire_origin":origin,
                "selected_training_threshold":t,
                "fraction_chosen_quarter":float(np.mean([r["bc"]<t for r in rows])),
                "BC_mean":float(np.mean([r["bc"] for r in rows])),
                "mean_quarter_vs_hellinger":float(np.mean([r["delta_quarter_hellinger"] for r in rows])),
                "mean_support_vs_hellinger":float(np.mean([r["delta_support_hellinger"] for r in rows])),
                "mean_policy_vs_hellinger":float(np.mean(predicted)),
                "comparisons":{b:infer(rows,t,b,20261016+idx*97)
                               for idx,b in enumerate(("Hellinger","Quarter","Support"))},
            }
    return {
        "status":"POSTHOC_3_ORIGIN_HELDOUT_ENDPOINT_OVERLAP_POLICY",
        "original_three_source_runs":OLD_ORIGINS,
        "original_EAD_V2_run":NEW_EAD,
        "evaluation_class":"retrospective leave-one-acquisition-origin-out, MultiMS2 CID/EAD NEVER separated in train vs test",
        "rule":"if endpoint-only Bhattacharyya overlap < training-selected threshold, use quarter-power; else fixed Hellinger",
        "threshold_grid_predeclared_before_this_TEST_but_after_original_source_outcomes":THRESHOLDS,
        "training_objective":"equal original source weight; half weights for two modes of same MultiMS2 acquisition origin",
        "selected_policy_by_heldout_origin":trained,
        "heldout_score_results":results,
        "limitations":"All four corpus outcomes were previously observed before this analysis. Only three acquisition origins; cannot establish prospective generalization, physics or improved identification.",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--three-source-metadata",type=Path,required=True)
    p.add_argument("--ead-rows",type=Path,required=True)
    p.add_argument("--ead-result",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=read_training_data(a.features,a.three_source_metadata,a.ead_rows,a.ead_result)
    result=evaluate(data)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=[
        "# POSTHOC source-heldout overlap-based geometry selector","",
        "**The entire MultiMS2 CID+EAD source is jointly held out. Outcomes were previously seen. This is exploratory modeling, NOT new untouched validation.**","",
        "| Heldout acquisition | Queries | BC mean | Train-chosen threshold | % quarter selected | Fixed quarter−Hellinger | Support−Hellinger | Heldout policy−Hellinger | Policy−support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in result["heldout_score_results"].items():
        lines.append(
            f"| {name} | {r['n']} | {r['BC_mean']:.4f} | "+
            f"{r['selected_training_threshold']:.2f} | {100*r['fraction_chosen_quarter']:.1f}% | "+
            f"{r['mean_quarter_vs_hellinger']:+.6f} | {r['mean_support_vs_hellinger']:+.6f} | "+
            f"{r['mean_policy_vs_hellinger']:+.6f} | "+
            f"{r['comparisons']['Support']['mean_delta']:+.6f} |")
        for b,v in r["comparisons"].items():
            ci=v["formula_block_bootstrap_95"]
            lines.append(f"- {name} policy−{b}: {v['mean_delta']:+.6f}, "+
                         f"95% formula CI [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], "+
                         f"p={v['formula_block_signflip_2sided']['p_value_two_sided']:.6f}")
    lines.extend(["",result["limitations"],""])
    (a.out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__=="__main__":
    main()
