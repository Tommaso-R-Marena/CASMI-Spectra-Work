"""Frozen MultiMS2 16-eV EAD query-scoring lockbox.

Score only after independent metadata-only gate succeeds. Hybrid method was
frozen before any query peaks were inspected. This evaluates different
fragmentation modality on the *same instrument*, not a new external lab.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

import audit_multims2_third_source_metadata as source
import audit_multims2_ead_metadata as ead
from qfd.power_geometry import power_mean_distribution
from qfd.persistent_survival import persistent_hellinger_survival
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity,
    support_only_suppression, best_hellinger_chord_oracle,
)
from qfd.stats import (
    paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie,
)

METHODS=("hybrid","quarter","support","hellinger","linear","oracle")
GATE_STATUS="MULTIMS2_EAD_12_16_24EV_HEADER_ONLY_GATE_V1"

def read_selected_spectra(path, indices):
    """Read peaks ONLY for source-index records in pre-score eligible manifest."""
    indices=set(indices)
    found={}
    current=-1
    header=None
    mz=[]
    intens=[]
    def flush():
        if current not in indices:
            return
        r,err=ead.eligible(header,current)
        if err is not None:
            raise RuntimeError(f"selected EAD spectrum {current}: {err}")
        if len(mz)<5 or len(intens)!=len(mz):
            raise RuntimeError(f"bad selected EAD scan: {current}")
        if current in found:
            raise RuntimeError(f"duplicate source scan {current}")
        found[current]=(r,SimpleNamespace(mzs=np.asarray(mz,float),
                                         intensities=np.asarray(intens,float)))
    with path.open("r",encoding="utf-8",errors="replace") as fp:
        for raw in fp:
            line=raw.strip()
            if line.upper()=="BEGIN IONS":
                if header is not None:
                    raise RuntimeError("MGF nested")
                current+=1
                header={}
                mz=[]
                intens=[]
            elif line.upper()=="END IONS":
                if header is None:
                    raise RuntimeError("MGF end without begin")
                flush()
                header=None
            elif header is not None and current in indices:
                if "=" in line:
                    key,value=line.split("=",1)
                    key=key.strip().upper()
                    if key in source.METADATA_KEYS:
                        if key in header:
                            raise RuntimeError("duplicate selected header key")
                        header[key]=value.strip()
                else:
                    pieces=line.split()
                    if len(pieces)<2: continue
                    try:
                        m=float(pieces[0]); v=float(pieces[1])
                    except ValueError:
                        continue
                    if math.isfinite(m) and math.isfinite(v) and m>0 and v>0:
                        mz.append(m); intens.append(v)
    if header is not None or set(found)!=indices:
        raise RuntimeError(f"selected EAD spectrum set incomplete: {len(found)}/{len(indices)}")
    return found

def compare(rows,a,b,seed):
    left=np.asarray([r[a] for r in rows],float)
    right=np.asarray([r[b] for r in rows],float)
    blocks=[r["formula"] for r in rows]
    return {
        "mean_delta":float(np.mean(left-right)),
        "win_loss_tie":paired_win_loss_tie(left,right,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            left,right,blocks,n_boot=5000,seed=seed),
        "formula_block_signflip_2sided":paired_block_signflip_test(
            left,right,blocks,n_permutations=20000,seed=seed),
    }

def summary(rows, infer=False):
    if not rows:return None
    r={
        "n_queries":len(rows),
        "n_formulas":len({x["formula"] for x in rows}),
        "n_structures":len({x["inchikey"] for x in rows}),
        "mean_scores":{k:float(np.mean([x[k] for x in rows])) for k in METHODS},
        "mean_endpoint_overlap":float(np.mean([x["endpoint_bc"] for x in rows])),
        "mean_unmatched_query_mass":float(np.mean([x["unmatched_query_mass"] for x in rows])),
        "descriptive_deltas":{
            "hybrid_minus_quarter":float(np.mean([x["hybrid"]-x["quarter"] for x in rows])),
            "hybrid_minus_support":float(np.mean([x["hybrid"]-x["support"] for x in rows])),
            "quarter_minus_hellinger":float(np.mean([x["quarter"]-x["hellinger"] for x in rows])),
            "support_minus_oracle":float(np.mean([x["support"]-x["oracle"] for x in rows])),
        },
    }
    if infer:
        pairings=[
            ("hybrid_vs_quarter","hybrid","quarter"),
            ("hybrid_vs_support","hybrid","support"),
            ("hybrid_vs_hellinger","hybrid","hellinger"),
            ("quarter_vs_hellinger","quarter","hellinger"),
            ("support_vs_hellinger","support","hellinger"),
            ("hybrid_vs_oracle","hybrid","oracle"),
        ]
        r["paired"]={name:compare(rows,a,b,20261016+31*i)
                     for i,(name,a,b) in enumerate(pairings)}
        primary=["hybrid_vs_quarter","hybrid_vs_support"]
        ranked=sorted([(r["paired"][name]["formula_block_signflip_2sided"]["p_value_two_sided"],name)
                       for name in primary])
        adjusted={}
        max_adj=0.
        for i,(p,name) in enumerate(ranked):
            max_adj=max(max_adj,min(1.,(len(primary)-i)*p))
            adjusted[name]=max_adj
        r["holm_two_primary_p_values"]=adjusted
        r["breakthrough_gate_hybrid_better_than_both"] = bool(all(
            r["paired"][name]["mean_delta"]>0 and
            r["paired"][name]["formula_block_bootstrap_95"]["ci_low"]>0 and
            adjusted[name]<.05 for name in primary))
    return r

def score_one(entry,scans):
    identity=("inchikey","formula","ion_mode","adduct","fragmentation","source_dataset_id")
    specs=[]
    for role, energy in (("low",12),("query",16),("high",24)):
        idx=entry[role+"_index"]
        if idx not in scans:
            raise RuntimeError("missing frozen selected index")
        m,s=scans[idx]
        if (any(m[k]!=entry[k] for k in identity) or
            m["energy_eV"]!=energy or entry[role+"_energy_eV"]!=energy):
            raise RuntimeError("metadata/record identity or CE mismatch")
        specs.append(s)
    if abs(float(entry["t"])-1/3)>1e-15:
        raise RuntimeError("frozen EAD energy coordinate changed")
    low,query,high=specs
    grid=reference_only_grid(low,high,tolerance_da=.01)
    q,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=.01)
    a,b=grid.low,grid.high
    t=1/3
    preds={
        "hybrid":persistent_hellinger_survival(a,b,t),
        "quarter":power_mean_distribution(a,b,t,.25),
        "support":support_only_suppression(a,b,t,power=4),
        "hellinger":power_mean_distribution(a,b,t,.5),
        "linear":power_mean_distribution(a,b,t,1.),
    }
    result={name:affinity(q,pred) for name,pred in preds.items()}
    result["oracle"]=best_hellinger_chord_oracle(q,a,b)["affinity"]
    if result["oracle"]+1e-9<result["hellinger"]:
        raise RuntimeError("oracle does not dominate its own path")
    if (abs(float(q.sum())+unmatched-1)>1e-7 or
        any(not math.isfinite(v) or not -1e-9<=v<=1+1e-9 for v in result.values())):
        raise RuntimeError("invalid scored affinity or retained query mass")
    return {**entry,**result,"unmatched_query_mass":unmatched,
            "endpoint_bc":float(np.sum(np.sqrt(a*b))),
            "reference_bins":len(a)}

def evaluate(args):
    gatefile=args.gate_dir/"feasibility.json"
    manifestfile=args.gate_dir/"frozen_triplets.jsonl"
    gate=json.loads(gatefile.read_text())
    manifest=[json.loads(s) for s in manifestfile.read_text().splitlines() if s.strip()]
    if (gate.get("status")!=GATE_STATUS or gate.get("gate_pass") is not True or
        gate.get("16eV_target_peak_positions_or_intensities_read") is not False or
        gate.get("target_scores_or_predictor_outcomes_opened") is not False):
        raise RuntimeError("FROZEN EAD GATE NOT OPEN: do not score")
    if (gate.get("source_sha256")!=ead.SOURCE_SHA256 or
        args.mgf.stat().st_size!=source.UPSTREAM_BYTES or
        source.git_blob_sha1(args.mgf)!=source.UPSTREAM_BLOB_SHA1 or
        source.digest(args.mgf)!=ead.SOURCE_SHA256):
        raise RuntimeError("raw EAD source changed from metadata gate")
    if gate["coverage"]["triplets"]!=len(manifest) or not manifest:
        raise RuntimeError("gate manifest count changed")
    if any(gate["coverage"][k]<n for k,n in ead.THRESHOLDS.items()):
        raise RuntimeError("EAD gate failed numeric threshold")
    if len({r["query_index"] for r in manifest})!=len(manifest):
        raise RuntimeError("duplicate EAD queries")
    needed={r[f"{role}_index"] for r in manifest for role in ("low","query","high")}
    scans=read_selected_spectra(args.mgf,needed)
    rows=[score_one(row,scans) for row in manifest]
    if len(rows)!=len(manifest) or len({r["formula"] for r in rows})!=gate["coverage"]["formula_blocks"]:
        raise RuntimeError("missing original EAD population")
    ion_modes={x["ion_mode"] for x in rows}
    earlier={str(x["inchikey"]) for x in [
        json.loads(line) for line in args.cid_manifest.read_text().splitlines() if line.strip()]}
    if source.digest(args.cid_manifest)!=ead.PINNED_CID_MANIFEST_SHA256:
        raise RuntimeError("original CID manifest changed")
    unrelated=[r for r in rows if r["inchikey"] not in earlier]
    result={
        "status":"FROZEN_EAD_16EV_MODALITY_TRANSFER",
        "scientific_class":"cross-fragmentation EAD prospective held-out-peaks; SAME acquisition laboratory and instrument as previously scored CID",
        "first_source_eligible_population_retained":True,
        "header_gate_sha256":source.digest(gatefile),
        "source_sha256":ead.SOURCE_SHA256,
        "gate_manifest_sha256":source.digest(manifestfile),
        "predictor_fitted_on_EAD_targets":False,
        "model":"persistent shared Hellinger; 4th-power transient fragment suppression; no fitting",
        "energy_coordinate":1/3,
        "all":summary(rows,infer=True),
        "positive":summary([r for r in rows if r["ion_mode"]=="POSITIVE"]),
        "negative":summary([r for r in rows if r["ion_mode"]=="NEGATIVE"]),
        "no_exact_previous_CID_structure_overlap":summary(unrelated,infer=len(unrelated)>=30 and len({r["formula"] for r in unrelated})>=15),
        "source_dataset_hist":dict(Counter(r["source_dataset_id"] for r in rows)),
        "limitations":"different EAD modality, same ZenoTOF source; no new lab or CASMI identification ranking; hypothesis designed after earlier CID outcomes.",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.out/"rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    report=["# EAD 12→16→24 eV target-blind predictive transfer","",
            "**This is a prospectively frozen predictor on a new fragmentation modality, but NOT a new laboratory.**","",
            "| Cohort | n | formulas | hybrid affinity | quarter | support | Hellinger | hybrid−quarter | hybrid−support |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label,s in (("all",result["all"]),("ESI+",result["positive"]),("ESI−",result["negative"]),
                    ("no prior CID identity",result["no_exact_previous_CID_structure_overlap"])):
        if s is None:continue
        p=s["mean_scores"];d=s["descriptive_deltas"]
        report.append(f"| {label} | {s['n_queries']} | {s['n_formulas']} | "
                      f"{p['hybrid']:.6f} | {p['quarter']:.6f} | {p['support']:.6f} | {p['hellinger']:.6f} | "
                      f"{d['hybrid_minus_quarter']:+.6f} | {d['hybrid_minus_support']:+.6f} |")
    for name,stat in result["all"]["paired"].items():
        ci=stat["formula_block_bootstrap_95"]
        report.extend(["",f"- {name}: Δ={stat['mean_delta']:+.6f}; CI "
                       f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]; "
                       f"p={stat['formula_block_signflip_2sided']['p_value_two_sided']:.6f}; "
                       f"wins/losses/ties {stat['win_loss_tie']}."])
    report.extend(["",f"**Predeclared hybrid-beats-both gate:** "
                   f"{result['all']['breakthrough_gate_hybrid_better_than_both']}",
                   f"Holm-adjusted two primary p-values: {result['all']['holm_two_primary_p_values']}",
                   "",result["limitations"],""])
    (args.out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--cid-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    return evaluate(p.parse_args())

if __name__=="__main__":
    main()
