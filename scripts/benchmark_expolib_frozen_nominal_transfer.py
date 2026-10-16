"""Immutable ExpoLib v1 nominal-CE spectral transfer; no parameter training.

Separate independent-acquisition source, nominal-energy units unverified.
The outcome is NOT proof of a universal physical collision-energy law.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import re

import numpy as np

from qfd.power_geometry import power_mean_distribution
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity,
    support_only_suppression, best_hellinger_chord_oracle,
)

EXPECTED = {
    "feasibility": "fbf4a6ab01bdf536dad09edc6dfcb100ee133ada0f4cabddb1d4d6b956f7af42",
    "manifest": "095340d967b177077ce517da7aa332795ce691899cc1320563cce07a76a68560",
    "pos": "aa838492fe11bd068819d3bc872541d7ec727217440dd5d2595af8985fb56959",
    "neg": "b26e11b25c4ee5d24507e5aa9b525540096a4347cfdfbce94d230e4e858fa838",
}
SOURCE_FILES = {"ESI_POS":"pos","ESI_NEG":"neg"}
METHODS=("quarter","hellinger","support_only","linear","hellinger_oracle")
CE=re.compile(r"^\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]$")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def load_selected_spectra(path, indices):
    """Read only the selected record peak data; preserve original raw positive intensities."""
    required=set(indices)
    if not required:
        return {}
    result={}
    current=-1
    header={}
    mz=[]
    y=[]
    in_peaks=False
    def finish():
        if current not in required:
            return
        if len(mz)<5:
            raise ValueError(f"source {path}: selected record {current} has <5 positive finite peaks")
        declared=int(header.get("Num Peaks") or 0)
        if declared<5:
            raise ValueError("selected declared peak count below frozen QC")
        e=CE.fullmatch(header.get("Collision_energy") or "")
        if not e:
            raise ValueError("selected record has ambiguous CE unit/value")
        if current in result:
            raise ValueError("duplicate selected record")
        result[current]={
            "inchikey":(header.get("InChIKey") or "").upper(),
            "formula":header.get("Formula"),
            "adduct":header.get("Precursor_type"),
            "mode":header.get("Ion_mode"),
            "fragmentation":header.get("Method"),
            "energy_nominal":float(e.group(1)),
            "spectrum":SimpleNamespace(mzs=np.array(mz,float),intensities=np.array(y,float)),
        }
    with Path(path).open("r",encoding="utf-8",errors="replace") as fp:
        for line in fp:
            text=line.strip()
            if text.startswith("Name: "):
                finish()
                current+=1
                header={}
                mz=[]
                y=[]
                in_peaks=False
                continue
            if current not in required:
                continue
            if not in_peaks:
                if text.startswith("Num Peaks:"):
                    header["Num Peaks"]=text.split(":",1)[1].strip()
                    in_peaks=True
                elif ":" in text:
                    key,val=text.split(":",1)
                    if key in {"InChIKey","Formula","Precursor_type","Ion_mode",
                               "Method","Collision_energy"}:
                        if key in header:
                            raise ValueError("duplicate selected record metadata header")
                        header[key]=val.strip()
            else:
                fields=text.split()
                if len(fields)!=2:
                    continue
                try:
                    m=float(fields[0]);v=float(fields[1])
                except ValueError:
                    continue
                if math.isfinite(m) and math.isfinite(v) and m>0 and v>0:
                    mz.append(m);y.append(v)
    finish()
    if set(result)!=required:
        raise ValueError(f"selected records incomplete ({len(required)-len(result)} missing)")
    return result

def validate_manifest(feasibility, manifest):
    if (feasibility.get("status")!="EXPOLIB_V1_POSTDISCOVERY_LABEL_FREE_FEASIBILITY"
        or feasibility.get("metadata_feasibility_pass") is not True
        or feasibility.get("physical_eV_unit_validation_pass") is not False
        or feasibility.get("scores_opened") is not False):
        raise RuntimeError("wrong source gate or nominal physical-unit caveat")
    cov=feasibility["coverage"]
    if any(cov.get(k)!=v for k,v in {
        "triplets":1125,"structures":122,"formula_blocks":114,"offcenter_triplets":103}.items()):
        raise RuntimeError("independent source frozen coverage has changed")
    if Counter(x["mode"] for x in manifest)!=Counter({"POSITIVE":587,"NEGATIVE":538}):
        raise RuntimeError("frozen ExpoLib ion-polarity cohort changed")
    if len(manifest)!=1125 or len({(x["source"],x["query_index"]) for x in manifest})!=1125:
        raise RuntimeError("duplicate/missing source query")
    for r in manifest:
        if r["source"] not in SOURCE_FILES or r["mode"] not in {"POSITIVE","NEGATIVE"}:
            raise RuntimeError("unknown source")
        if not 0<float(r["t"])<1:
            raise RuntimeError("invalid nominal interpolation coordinate")

def scoring_spectrum(record,manifest,role):
    index=manifest[role+"_index"]
    if index not in record:
        raise RuntimeError("frozen selected record missing")
    obj=record[index]
    if (obj["inchikey"]!=manifest["inchikey"]
        or obj["adduct"]!=manifest["adduct"] or obj["formula"]!=manifest["formula"]
        or obj["mode"]!=manifest["mode"] or obj["fragmentation"]!=manifest["fragmentation"]
        or obj["energy_nominal"]!=manifest[role+"_energy_nominal"]):
        raise RuntimeError("immutable metadata/spectral record identity mismatch")
    return obj["spectrum"]

def score_triplet(r, selected):
    spectra=selected[r["source"]]
    lo=scoring_spectrum(spectra,r,"low")
    query=scoring_spectrum(spectra,r,"query")
    hi=scoring_spectrum(spectra,r,"high")
    grid=reference_only_grid(lo,hi,tolerance_da=.01)
    q,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=.01)
    t=float(r["t"])
    a,b=grid.low,grid.high
    predictions={
        "quarter":power_mean_distribution(a,b,t,.25),
        "hellinger":power_mean_distribution(a,b,t,.5),
        "support_only":support_only_suppression(a,b,t,power=4),
        "linear":power_mean_distribution(a,b,t,1.),
    }
    scores={name:affinity(q,p) for name,p in predictions.items()}
    scores["hellinger_oracle"]=best_hellinger_chord_oracle(q,a,b)["affinity"]
    if scores["hellinger_oracle"]+1e-9<scores["hellinger"]:
        raise RuntimeError("global oracle was beaten by a point on its own path")
    if any(not np.isfinite(v) or not -1e-9<=v<=1+1e-9 for v in scores.values()):
        raise RuntimeError("invalid scored spectral affinity")
    if not 0<=unmatched<=1+1e-12 or q.sum()+unmatched>1+1e-9 or abs(q.sum()+unmatched-1)>1e-7:
        raise RuntimeError("query intensity mass not conserved")
    return {**r,**scores,"unmatched_target_fraction":unmatched,
            "reference_bins":len(grid.mz)}

def paired(rows,b,seed):
    x=np.array([r["quarter"] for r in rows])
    y=np.array([r[b] for r in rows])
    blocks=[r["formula"] for r in rows]
    return {
        "mean_delta":float(np.mean(x-y)),
        "win_loss_tie":paired_win_loss_tie(x,y,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(x,y,blocks,n_boot=10000,seed=seed),
        "formula_block_signflip_two_sided":paired_block_signflip_test(x,y,blocks,n_permutations=50000,seed=seed),
    }

def summarize(rows,inference=False):
    if not rows:
        return None
    result={
        "queries":len(rows),
        "structures":len({r["inchikey"] for r in rows}),
        "formulas":len({r["formula"] for r in rows}),
        "methods":{name:{"mean_affinity":float(np.mean([r[name] for r in rows])),
                         "median_affinity":float(np.median([r[name] for r in rows]))}
                   for name in METHODS},
        "mean_unmatched_target_fraction":float(np.mean([r["unmatched_target_fraction"] for r in rows])),
        "mean_quarter_minus_hellinger":float(np.mean([r["quarter"]-r["hellinger"] for r in rows])),
        "mean_quarter_minus_support":float(np.mean([r["quarter"]-r["support_only"] for r in rows])),
        "mean_quarter_minus_hellinger_oracle":float(np.mean([r["quarter"]-r["hellinger_oracle"] for r in rows])),
    }
    if inference:
        result["primary_quarter_vs_hellinger"]=paired(rows,"hellinger",20261016)
        result["quarter_vs_support_only"]=paired(rows,"support_only",20261017)
        result["quarter_vs_hellinger_oracle"]=paired(rows,"hellinger_oracle",20261018)
    return result

def evaluate(positive,negative,gate_dir,out):
    fp=gate_dir/"feasibility.json";mp=gate_dir/"nominal_energy_triplets.jsonl"
    if digest(fp)!=EXPECTED["feasibility"] or digest(mp)!=EXPECTED["manifest"]:
        raise RuntimeError("frozen independent-source metadata gate artifacts changed")
    for path,key in ((positive,"pos"),(negative,"neg")):
        if digest(path)!=EXPECTED[key]:
            raise RuntimeError("pinned independent-source raw MSP digest mismatch")
    meta=json.loads(fp.read_text())
    manifest=[json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
    validate_manifest(meta,manifest)
    indexes={}
    for src in SOURCE_FILES:
        s=[r for r in manifest if r["source"]==src]
        indexes[src]={r[role+"_index"] for r in s for role in ("low","query","high")}
    selected={
        "ESI_POS":load_selected_spectra(positive,indexes["ESI_POS"]),
        "ESI_NEG":load_selected_spectra(negative,indexes["ESI_NEG"]),
    }
    rows=[score_triplet(r,selected) for r in manifest]
    if len(rows)!=1125:
        raise RuntimeError("full frozen source scoring did not finish")
    positive_rows=[r for r in rows if r["mode"]=="POSITIVE"]
    negative_rows=[r for r in rows if r["mode"]=="NEGATIVE"]
    offcenter=[r for r in rows if abs(r["t"]-.5)>1e-12]
    result={
        "status":"EXPOLIB_V1_NOMINAL_CE_SOURCE_TRANSFER_OUTCOME",
        "evaluation_class":"INDEPENDENT ACQUISITION SOURCE / NOMINAL CE COORDINATE / EXPLORATORY PHYSICAL UNITS",
        "metadata_feasibility_gate_run":38964190641,
        "energy_unit_verified_eV":False,
        "no_model_fit_or_energy_coordinate_retuning":True,
        "all":summarize(rows,inference=True),
        "positive_ion":summarize(positive_rows),
        "negative_ion":summarize(negative_rows),
        "offcenter_energy":summarize(offcenter),
        "limitations":"MSP reports unitless CE values; cross-source acquisition provenance and other-library overlap not fully audited. Not evidence of universal physical law or CASMI rank improvement.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"rows.jsonl").open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row,sort_keys=True)+"\n")
    lines=["# ExpoLib independent source, nominal collision-energy transfer","",
           "**The collision-energy unit is NOT independently verified eV. This is nominal-coordinate spectral interpolation, not a universal physical law.**","",
           "| Population | n | Formulas | α=.25 mean affinity | Hellinger mean | Δ α=.25−Hellinger | α=.25−support | α=.25−Hellinger oracle |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,cohort in [("All",result["all"]),("ESI positive",result["positive_ion"]),("ESI negative",result["negative_ion"]),("Offcenter nominal CE",result["offcenter_energy"])]:
        lines.append(f"| {name} | {cohort['queries']} | {cohort['formulas']} | "
                     f"{cohort['methods']['quarter']['mean_affinity']:.6f} | "
                     f"{cohort['methods']['hellinger']['mean_affinity']:.6f} | "
                     f"{cohort['mean_quarter_minus_hellinger']:+.6f} | "
                     f"{cohort['mean_quarter_minus_support']:+.6f} | "
                     f"{cohort['mean_quarter_minus_hellinger_oracle']:+.6f} |")
    p=result["all"]["primary_quarter_vs_hellinger"]
    ci=p["formula_block_bootstrap_95"]
    lines.extend(["",
        f"Original frozen primary contrast formula-block bootstrap 95% CI: [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}].",
        f"Two-sided formula-block sign-flip p: {p['formula_block_signflip_two_sided']['p_value_two_sided']:.6f}; win/loss/tie: {p['win_loss_tie']}.",
        "",result["limitations"],""])
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--positive",type=Path,required=True)
    ap.add_argument("--negative",type=Path,required=True)
    ap.add_argument("--gate-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    evaluate(args.positive,args.negative,args.gate_dir,args.out)

if __name__=="__main__":
    main()
