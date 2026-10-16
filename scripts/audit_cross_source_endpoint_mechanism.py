"""Three-source endpoint-only support/geometry mechanism audit, POSTHOC.

Features never read withheld query spectra, target energies (except labels), nor
reference outcomes. Original paired accuracy rows are joined only AFTER all
features are computed and cryptographic provenance is validated.

This audit EXPLAINS already observed scores; it cannot train a deployable policy,
constitute fresh blind validation or erase negative ExpoLib results.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import numpy as np

from qfd.target_blind_geometry import reference_only_grid
from benchmark_massbank_external_energy_geometry import parse_peak_spectrum
from benchmark_expolib_frozen_nominal_transfer import load_selected_spectra

DIGESTS={
    "massbank_rows":"2c973a7b83c1a48f95f923107ee72bb44a2a95f7d7da403f9d6f541d662efc8b",
    "massbank_manifest":"265dfb75e33d2eff3f19945255bddb5db0c0f7f9161593ada80cd2b50ecac9c4",
    "expolib_rows":"f85b67d61be82f62ee1072944f9d300eb2d1d81462eba6ad221566dfaa276695",
    "expolib_manifest":"095340d967b177077ce517da7aa332795ce691899cc1320563cce07a76a68560",
    "multims2_rows":"f97bfaee05e9197cdaa96ad995ab9019c995274820ff57f3b04bd04de12c2be6",
    "massbank_zip":"12c6a922929f9bf2484c46586b6101512a415e44680558883b0630c6e47905fb",
    "expolib_pos":"aa838492fe11bd068819d3bc872541d7ec727217440dd5d2595af8985fb56959",
    "expolib_neg":"b26e11b25c4ee5d24507e5aa9b525540096a4347cfdfbce94d230e4e858fa838",
}
BINS=[0.,.01,.05,.1,.2,.35,.5,.7,1.0000001]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def features(low,high):
    g=reference_only_grid(low,high,tolerance_da=.01)
    a,b=g.low,g.high
    bc=float(np.sum(np.sqrt(a*b)))
    if not -1e-12<=bc<=1+1e-12:
        raise ValueError("endpoint affinity outside unit range")
    overlap=(a>0)&(b>0)
    exclusive_a=(a>0)&(b==0)
    exclusive_b=(b>0)&(a==0)
    return {
        "endpoint_bhattacharyya":bc,
        "endpoint_shared_bins":int(overlap.sum()),
        "endpoint_unique_bins":int((exclusive_a|exclusive_b).sum()),
        "endpoint_exclusive_a_mass":float(a[exclusive_a].sum()),
        "endpoint_exclusive_b_mass":float(b[exclusive_b].sum()),
        "endpoint_min_overlap_mass":float(np.minimum(a,b).sum()),
        "endpoint_bins":int(len(g.mz)),
    }

def massbank_endpoint_features(zip_file,manifest):
    out={}
    with ZipFile(zip_file) as archive:
        present=set(archive.namelist())
        needed={r[k] for r in manifest for k in ("low_path","high_path")}
        if not needed.issubset(present):
            raise RuntimeError("pinned MassBank endpoint accession missing")
        cache={}
        def get(path):
            if path not in cache:
                cache[path]=parse_peak_spectrum(
                    archive.read(path).decode("utf-8",errors="replace"),record_path=path)
            return cache[path]
        for row in manifest:
            qid=row["query_path"]
            if qid in out:
                raise RuntimeError("duplicate MassBank query")
            out[qid]=features(get(row["low_path"]),get(row["high_path"]))
    return out

def expolib_endpoint_features(positive,negative,manifest):
    raw={"ESI_POS":positive,"ESI_NEG":negative}
    selected={}
    for source,path in raw.items():
        rows=[r for r in manifest if r["source"]==source]
        indices={r[k] for r in rows for k in ("low_index","high_index")}
        selected[source]=load_selected_spectra(path,indices)
    output={}
    for r in manifest:
        source=r["source"]
        q=(source,r["query_index"])
        if q in output:
            raise RuntimeError("duplicate ExpoLib query")
        a=selected[source][r["low_index"]]
        b=selected[source][r["high_index"]]
        for record,role in ((a,"low"),(b,"high")):
            if (record["inchikey"]!=r["inchikey"]
                or record["energy_nominal"]!=r[role+"_energy_nominal"]
                or record["formula"]!=r["formula"]):
                raise RuntimeError("ExpoLib endpoint identity mismatch")
        output[q]=features(a["spectrum"],b["spectrum"])
    return output

def bin_label(s):
    ix=int(np.searchsorted(BINS,s,side="right")-1)
    ix=max(0,min(ix,len(BINS)-2))
    return f"[{BINS[ix]:.2f},{BINS[ix+1]:.2f})"

def source_stats(rows):
    vals=np.asarray([r["delta_quarter_hellinger"] for r in rows])
    bcs=np.asarray([r["endpoint_bhattacharyya"] for r in rows])
    x=np.asarray([r["delta_quarter_support"] for r in rows])
    # Report descriptive source-conditional association; do not bootstrap these
    # posthoc BC bins or pretend within-source queries represent laboratories.
    def pearson(a,b):
        if np.std(a)<1e-14 or np.std(b)<1e-14:
            return None
        return float(np.corrcoef(a,b)[0,1])
    grouped=defaultdict(list)
    for r in rows:
        grouped[bin_label(r["endpoint_bhattacharyya"])].append(r)
    bins=[]
    for label,items in sorted(grouped.items()):
        d=np.asarray([r["delta_quarter_hellinger"] for r in items])
        ds=np.asarray([r["delta_quarter_support"] for r in items])
        bins.append({
            "label":label,"n":len(items),
            "mean_endpoint_BC":float(np.mean([r["endpoint_bhattacharyya"] for r in items])),
            "mean_quarter_minus_hellinger":float(d.mean()),
            "mean_quarter_minus_support":float(ds.mean()),
            "insufficient_subgroup_n_for_any_claim":len(items)<20,
        })
    return {
        "n_queries":len(rows),
        "n_structures":len({r["inchikey"] for r in rows}),
        "mean_endpoint_BC":float(bcs.mean()),
        "median_endpoint_BC":float(np.median(bcs)),
        "endpoint_BC_quantiles":{str(v):float(np.quantile(bcs,v)) for v in (.05,.25,.5,.75,.95)},
        "mean_left_exclusive_mass":(
            float(np.mean([r["endpoint_exclusive_a_mass"] for r in rows]))
            if all("endpoint_exclusive_a_mass" in r for r in rows) else None),
        "mean_right_exclusive_mass":(
            float(np.mean([r["endpoint_exclusive_b_mass"] for r in rows]))
            if all("endpoint_exclusive_b_mass" in r for r in rows) else None),
        "mean_quarter_minus_hellinger":float(vals.mean()),
        "mean_quarter_minus_support":float(x.mean()),
        "pearson_delta_vs_endpoint_BC_descriptive":pearson(vals,bcs),
        "endpoint_BC_bins_posthoc":bins,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--massbank-zip",type=Path,required=True)
    p.add_argument("--massbank-artifact",type=Path,required=True)
    p.add_argument("--expolib-positive",type=Path,required=True)
    p.add_argument("--expolib-negative",type=Path,required=True)
    p.add_argument("--expolib-artifact",type=Path,required=True)
    p.add_argument("--expolib-manifest",type=Path,required=True)
    p.add_argument("--multims2-artifact",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    pinned={
        "massbank_rows":a.massbank_artifact/"rows.jsonl",
        "massbank_manifest":a.massbank_artifact/"label_free_triplets.jsonl",
        "expolib_rows":a.expolib_artifact/"rows.jsonl",
        "expolib_manifest":a.expolib_manifest,
        "multims2_rows":a.multims2_artifact/"rows.jsonl",
        "massbank_zip":a.massbank_zip,
        "expolib_pos":a.expolib_positive,
        "expolib_neg":a.expolib_negative,
    }
    for key,path in pinned.items():
        if digest(path)!=DIGESTS[key]:
            raise RuntimeError(f"source artifact changed: {key}")
    mass_manifest=read_rows(pinned["massbank_manifest"])
    exp_manifest=read_rows(pinned["expolib_manifest"])
    mass_rows=read_rows(pinned["massbank_rows"])
    expo_rows=read_rows(pinned["expolib_rows"])
    multi_rows=read_rows(pinned["multims2_rows"])
    if (len(mass_manifest),len(mass_rows),len(exp_manifest),len(expo_rows),len(multi_rows))!=(2138,2138,1125,1125,1781):
        raise RuntimeError("three frozen source cohort sizes changed")
    mb=massbank_endpoint_features(a.massbank_zip,mass_manifest)
    ex=expolib_endpoint_features(a.expolib_positive,a.expolib_negative,exp_manifest)
    outputs={}
    for source,rows in (("MassBank",mass_rows),("ExpoLib",expo_rows),("MultiMS2",multi_rows)):
        features_only=mb if source=="MassBank" else ex if source=="ExpoLib" else None
        annotated=[]
        for r in rows:
            if source=="MassBank":
                key=r["query_path"]
                delta=float(r["alpha_quarter"]-r["hellinger_fixed"])
                support=float(r["alpha_quarter"]-r["support_only"])
            elif source=="ExpoLib":
                key=(r["source"],r["query_index"])
                delta=float(r["quarter"]-r["hellinger"])
                support=float(r["quarter"]-r["support_only"])
            else:
                key=r["query_index"]
                delta=float(r["quarter"]-r["hellinger"])
                support=float(r["quarter"]-r["support_only"])
            f=(features_only[key] if features_only is not None else {
                "endpoint_bhattacharyya":r["endpoint_bhattacharyya"]})
            annotated.append({
                "source":source,"query_id":str(key),"inchikey":r["inchikey"],
                "formula":r["formula"],
                "endpoint_bhattacharyya":f["endpoint_bhattacharyya"],
                "delta_quarter_hellinger":delta,
                "delta_quarter_support":support,
                **{k:v for k,v in f.items() if k!="endpoint_bhattacharyya"},
            })
        if features_only is not None and len(features_only)!=len(rows):
            raise RuntimeError("descriptor coverage differs from frozen source")
        outputs[source]={
            "stats":source_stats(annotated),
            "per_query_descriptors":annotated,
        }
    result={
        "status":"POSTHOC_CROSS_SOURCE_ENDPOINT_GEOMETRY_MECHANISM_DIAGNOSTIC",
        "original_runs":{"MassBank":38918658343,"ExpoLib":38964797817,"MultiMS2":38970610885},
        "hypothesis":"Endpoint-only BC/shared-versus-exclusive support helps explain between-source curvature reversal",
        "data_label":"Retrospective descriptor association. NO new source replication, policy-fitting or confirmatory cutoff testing.",
        "source_stats":{k:v["stats"] for k,v in outputs.items()},
        "no_new_scoring_or_training":True,
        "limitation":"Three acquisition sources, different instrument protocols, collision-energy ranges, compound chemistries and preprocessing. Endpoint BC association is not causal and cannot support a universal model.",
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (a.out/"features.jsonl").open("w") as f:
        for v in outputs.values():
            for row in v["per_query_descriptors"]:
                f.write(json.dumps(row,sort_keys=True)+"\n")
    md=["# Post-hoc endpoint geometry across three measured sources","",
        "**All scores are replayed from original immutable artifacts; endpoint spectra alone provide explanatory features.**","",
        "| Source | Triplets | Mean endpoint BC | Mean quarter-Hellinger Δ | Mean quarter-support Δ | Correlation(BC,Δ) |",
        "|---|---:|---:|---:|---:|---:|"]
    for source,output in outputs.items():
        s=output["stats"]
        md.append(f"| {source} | {s['n_queries']} | {s['mean_endpoint_BC']:.6f} | "
                  f"{s['mean_quarter_minus_hellinger']:+.6f} | {s['mean_quarter_minus_support']:+.6f} | "
                  f"{s['pearson_delta_vs_endpoint_BC_descriptive']!s} |")
    for source,output in outputs.items():
        md+=["",f"## {source} endpoint BC bins","",
             "| Endpoint BC interval | n | Mean BC | quarter−Hellinger | quarter−support |",
             "|---|---:|---:|---:|---:|"]
        for bucket in output["stats"]["endpoint_BC_bins_posthoc"]:
            md.append(f"| {bucket['label']} | {bucket['n']} | {bucket['mean_endpoint_BC']:.6f} | "
                      f"{bucket['mean_quarter_minus_hellinger']:+.6f} | {bucket['mean_quarter_minus_support']:+.6f} |")
    md+=["",result["limitation"],""]
    (a.out/"report.md").write_text("\n".join(md))
    print("\n".join(md))

if __name__=="__main__":
    main()
