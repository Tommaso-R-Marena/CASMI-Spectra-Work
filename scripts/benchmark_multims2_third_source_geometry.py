"""Third-source MultiMS2 CID transfer, frozen before any third-source scores.

Uses the Stage-1 label-free manifest only if its entire predeclared gate passes.
All predictors are computed from low/high reference spectra, never from target.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity,
    support_only_suppression, best_hellinger_chord_oracle,
)
from qfd.stats import (
    paired_block_bootstrap_delta, paired_block_signflip_test,paired_win_loss_tie,
)
from audit_multims2_third_source_metadata import (
    eligible, digest, git_blob_sha1, UPSTREAM_BYTES, UPSTREAM_BLOB_SHA1,
    THRESHOLDS, UPSTREAM_COMMIT,
)

METHODS=("quarter","hellinger","adaptive_endpoint","support_only","linear","hellinger_oracle")

def load_selected_source_spectra(path,indices):
    """Parse only pinned candidate scans, refuse missing/duplicate endpoints."""
    need=set(indices)
    found={}
    idx=-1
    header=None
    mz=[]
    intensities=[]
    def flush():
        if header is None or idx not in need:
            return
        info,error=eligible(header,idx)
        if error is not None:
            raise RuntimeError(f"selected scan {idx} failed frozen metadata header: {error}")
        if len(mz)<5 or len(mz)!=len(intensities):
            raise RuntimeError(f"selected spectrum {idx} lacks five finite positive peaks")
        if idx in found:
            raise RuntimeError("duplicate selected scan")
        found[idx]=(info, SimpleNamespace(mzs=np.asarray(mz,dtype=float),
                                          intensities=np.asarray(intensities,dtype=float)))
    with Path(path).open("r",encoding="utf-8",errors="replace") as fp:
        for raw in fp:
            line=raw.strip()
            if line.upper()=="BEGIN IONS":
                if header is not None:
                    raise RuntimeError("nested MGF scan")
                idx+=1
                header={}
                mz=[]
                intensities=[]
            elif line.upper()=="END IONS":
                if header is None:
                    raise RuntimeError("MGF END without BEGIN")
                flush()
                header=None
            elif header is not None and idx in need:
                if "=" in line:
                    key,value=line.split("=",1)
                    key=key.strip().upper()
                    if key in {
                        "INCHIAUX","INCHI_AUX","INCHIKEY","FORMULA","IONMODE","ION_MODE",
                        "ADDUCT","FRAGMENTATION_METHOD","COLLISION_ENERGY",
                        "MSLEVEL","SPECTYPE","NUM_PEAKS","NUMPEAKS","DATASET_ID",
                    }:
                        if key in header:
                            raise RuntimeError("duplicate selected header field")
                        header[key]=value.strip()
                else:
                    words=line.split()
                    if len(words)<2:
                        continue
                    try:
                        m=float(words[0]); v=float(words[1])
                    except ValueError:
                        continue
                    if math.isfinite(m) and math.isfinite(v) and m>0 and v>0:
                        mz.append(m); intensities.append(v)
    if header is not None:
        raise RuntimeError("trailing incomplete MGF")
    if set(found)!=need:
        raise RuntimeError(f"selected record inventory mismatch ({len(need)-len(found)} missing)")
    return found

def validated_triplet(row,found):
    identity=("inchikey","formula","adduct","ion_mode","fragmentation","source_dataset_id")
    spectra=[]
    for role,expected_energy in (("low",20),("query",40),("high",60)):
        i=row[role+"_index"]
        if i not in found:
            raise RuntimeError("metadata-selected raw spectrum missing")
        metadata,spectrum=found[i]
        if any(metadata[k]!=row[k] for k in identity):
            raise RuntimeError("source spectrum identity differs from label-free manifest")
        if metadata["energy_eV"]!=expected_energy or row[role+"_energy_eV"]!=expected_energy:
            raise RuntimeError("collision energy changed between selection and scoring")
        spectra.append(spectrum)
    if abs(float(row["t"])-0.5)>1e-15:
        raise RuntimeError("frozen 20/40/60 energy coordinate is not 0.5")
    return spectra

def score_one(row,found):
    low,query,high=validated_triplet(row,found)
    grid=reference_only_grid(low,high,tolerance_da=0.01)
    q,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=0.01)
    if not -1e-12 <= unmatched <= 1+1e-12 or abs(float(q.sum())+unmatched-1)>1e-7:
        raise RuntimeError("unmatched query spectral mass is not conserved")
    p0,p1=grid.low,grid.high
    bc=float(np.sum(np.sqrt(p0*p1)))
    if not -1e-12<=bc<=1+1e-12:
        raise RuntimeError("invalid endpoint Bhattacharyya overlap")
    alpha=.25+.25*np.clip(bc,0,1)
    t=.5
    predictions={
        "quarter":power_mean_distribution(p0,p1,t,.25),
        "hellinger":power_mean_distribution(p0,p1,t,.5),
        "adaptive_endpoint":power_mean_distribution(p0,p1,t,float(alpha)),
        "support_only":support_only_suppression(p0,p1,t,power=4),
        "linear":power_mean_distribution(p0,p1,t,1.),
    }
    scores={name:affinity(q,p) for name,p in predictions.items()}
    oracle=best_hellinger_chord_oracle(q,p0,p1)
    scores["hellinger_oracle"]=oracle["affinity"]
    if scores["hellinger_oracle"]+1e-9<scores["hellinger"]:
        raise RuntimeError("answer-aware Hellinger oracle must dominate its fixed point")
    if any(not math.isfinite(value) or not -1e-9<=value<=1+1e-9 for value in scores.values()):
        raise RuntimeError("nonfinite or unphysical affinity")
    return {**row,**scores,
            "endpoint_bhattacharyya":bc,
            "endpoint_adaptive_alpha":float(alpha),
            "unmatched_target_mass":unmatched,
            "n_reference_bins":len(grid.mz)}

def compare(rows,left,right,seed):
    a=np.asarray([r[left] for r in rows],dtype=float)
    b=np.asarray([r[right] for r in rows],dtype=float)
    blocks=[r["formula"] for r in rows]
    return {
        "mean_paired_affinity_delta":float(np.mean(a-b)),
        "wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,blocks,n_boot=10000,seed=seed),
        "formula_block_signflip_2sided":paired_block_signflip_test(
            a,b,blocks,n_permutations=50000,seed=seed),
    }

def summary(rows, infer=False):
    if not rows:
        return None
    out={
        "n_queries":len(rows),
        "n_structures":len({r["inchikey"] for r in rows}),
        "n_formula_blocks":len({r["formula"] for r in rows}),
        "mean_scores":{m:float(np.mean([r[m] for r in rows])) for m in METHODS},
        "mean_alpha_adaptive":float(np.mean([r["endpoint_adaptive_alpha"] for r in rows])),
        "mean_endpoint_BC":float(np.mean([r["endpoint_bhattacharyya"] for r in rows])),
        "mean_unmatched_target_mass":float(np.mean([r["unmatched_target_mass"] for r in rows])),
        "deltas_descriptive":{
            "quarter_minus_hellinger":float(np.mean([r["quarter"]-r["hellinger"] for r in rows])),
            "adaptive_minus_hellinger":float(np.mean([r["adaptive_endpoint"]-r["hellinger"] for r in rows])),
            "adaptive_minus_quarter":float(np.mean([r["adaptive_endpoint"]-r["quarter"] for r in rows])),
        },
    }
    if infer:
        comps=[
            ("quarter_vs_hellinger","quarter","hellinger"),
            ("adaptive_vs_hellinger","adaptive_endpoint","hellinger"),
            ("adaptive_vs_quarter","adaptive_endpoint","quarter"),
            ("quarter_vs_support","quarter","support_only"),
            ("quarter_vs_oracle","quarter","hellinger_oracle"),
        ]
        out["paired"]={name:compare(rows,left,right,20261016+17*i)
                       for i,(name,left,right) in enumerate(comps)}
        secondary=["adaptive_vs_hellinger","adaptive_vs_quarter"]
        ps=sorted((out["paired"][name]["formula_block_signflip_2sided"]["p_value_two_sided"],name)
                  for name in secondary)
        # Two hypotheses: smallest gets factor 2, larger is max of original larger
        # and adjusted smaller (Holm step-down family-wise control).
        first= min(1.,2.*ps[0][0])
        second= max(first,ps[1][0])
        out["holm_adaptive_p_values"]={ps[0][1]:first,ps[1][1]:second}
        out["classifications"]={
            name:("POSITIVE" if detail["mean_paired_affinity_delta"]>0 and
                detail["formula_block_bootstrap_95"]["ci_low"]>0 and
                (out["holm_adaptive_p_values"].get(name,
                    detail["formula_block_signflip_2sided"]["p_value_two_sided"])<.05)
                else "NOT CONFIRMATORY POSITIVE")
            for name,detail in out["paired"].items()
        }
    return out

def score_source(mgf,metadata_dir,out):
    meta_path=metadata_dir/"feasibility.json"
    manifest_path=metadata_dir/"frozen_triplets.jsonl"
    meta=json.loads(meta_path.read_text())
    manifest=[json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]
    if (meta.get("status")!="MULTIMS2_CID_20_40_60EV_METADATA_ONLY_V1"
        or meta.get("gate_pass") is not True
        or meta.get("outcome_scores_opened") is not False
        or meta.get("target_peak_locations_or_intensities_read") is not False):
        raise RuntimeError("FROZEN THIRD-SOURCE METADATA GATE CLOSED: NO SCORING")
    if (meta.get("upstream_commit")!=UPSTREAM_COMMIT or
        meta.get("upstream_blob_sha1")!=UPSTREAM_BLOB_SHA1 or
        Path(mgf).stat().st_size!=UPSTREAM_BYTES or
        git_blob_sha1(mgf)!=UPSTREAM_BLOB_SHA1 or
        digest(mgf)!=meta.get("raw_sha256")):
        raise RuntimeError("third-source MGF does not match frozen metadata gate")
    coverage=meta["coverage"]
    if any(coverage.get(k)!=len(manifest) if k=="triplets" else False
           for k in ("triplets",)):
        raise RuntimeError("manifest query count differs from gate")
    if any(coverage.get(k)<v for k,v in THRESHOLDS.items()):
        raise RuntimeError("source feasibility below frozen minimum")
    if len({r["query_index"] for r in manifest})!=len(manifest):
        raise RuntimeError("candidate query MS2 scan is reused")
    indices={r[role+"_index"] for r in manifest for role in ("low","query","high")}
    found=load_selected_source_spectra(mgf,indices)
    rows=[score_one(r,found) for r in manifest]
    if len(rows)!=coverage["triplets"] or len({r["inchikey"] for r in rows})!=coverage["structures"]:
        raise RuntimeError("missing or reselected spectrum in scoring run")
    if len({r["formula"] for r in rows})!=coverage["formula_blocks"]:
        raise RuntimeError("formula-block cohort changed")
    result={
        "status":"MULTIMS2_THIRD_SOURCE_FROZEN_20_40_60EV_CID_EFFICACY",
        "original_frozen_source_gate_run":None,
        "source_commit":UPSTREAM_COMMIT,
        "source_mgf_sha256":meta["raw_sha256"],
        "metadata_gate_sha256":digest(meta_path),
        "metadata_manifest_sha256":digest(manifest_path),
        "model_training":False,"adaptive_rule":"alpha=.25+.25*BC(endpoint_low,endpoint_high)",
        "t":0.5,"physical_protocol_eV":"upstream MultiMS2 documentation explicitly specifies CID 20/40/60 eV",
        "all":summary(rows,infer=True),
        "positive_ion":summary([r for r in rows if r["ion_mode"]=="POSITIVE"]),
        "negative_ion":summary([r for r in rows if r["ion_mode"]=="NEGATIVE"]),
        "source_dataset_hist":dict(Counter(r["source_dataset_id"] for r in rows)),
        "limitation":"single ZenoTOF instrument/lab and previously discovered hypotheses; not source-independent laboratory replication or full CASMI molecular-identification test.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    report=[
        "# Third-source MultiMS2 CID 20/40/60 eV frozen geometry transfer","",
        "No model fitting; complete independently eligible source subset; fixed endpoint-only adaptive rule.",
        "",
        "| Cohort | Triplets | Structures | Formulas | quarter BC | Hellinger BC | adaptive BC | Δ quarter−Hellinger | Δ adaptive−Hellinger |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label,group in (("Complete frozen",result["all"]),("ESI+",result["positive_ion"]),("ESI−",result["negative_ion"])):
        if group is None: continue
        report.append(
            f"| {label} | {group['n_queries']} | {group['n_structures']} | {group['n_formula_blocks']} | "
            f"{group['mean_scores']['quarter']:.6f} | {group['mean_scores']['hellinger']:.6f} | "
            f"{group['mean_scores']['adaptive_endpoint']:.6f} | "
            f"{group['deltas_descriptive']['quarter_minus_hellinger']:+.6f} | "
            f"{group['deltas_descriptive']['adaptive_minus_hellinger']:+.6f} |")
    for name,detail in result["all"]["paired"].items():
        ci=detail["formula_block_bootstrap_95"]
        p=detail["formula_block_signflip_2sided"]["p_value_two_sided"]
        report+=["",f"**{name}**: mean {detail['mean_paired_affinity_delta']:+.6f}; "
                f"formula-block bootstrap CI [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]; "
                f"p={p:.6f}; {result['all']['classifications'][name]}."]
    report+=["","Adaptive Holm corrected p: "+json.dumps(result["all"]["holm_adaptive_p_values"]),"",
             result["limitation"],""]
    (out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--metadata-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    score_source(a.mgf,a.metadata_dir,a.out)

if __name__=="__main__":
    main()
