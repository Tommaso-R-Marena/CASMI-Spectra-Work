"""Real cross-archive native InChIKey molecular retrieval, first source blind.

Entire GNPS candidate library and MassSpecGym query selection determined by
the earlier original source HEADER ONLY gate. No trainable weight or exponents.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import io
import json
import math
from pathlib import Path

import numpy as np
import zstandard

from audit_harmonized2026_crossprovider_gate import (
    SHA256,MD5,sha256file,meta_only,freeze)
from qfd.stats import (
    paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie)

METHODS=("cosine","hellinger","fragment_overlap","support_composite")
PRIMARY=(("support_vs_cosine","support_composite","cosine"),
         ("support_vs_hellinger","support_composite","hellinger"))
TOLERANCE_DA=.01


def extract_selected_peaks(path:Path, indices:set[int]):
    """Open ONLY the original metadata-frozen GNPS/MassSpecGym spectra."""
    out={}
    with path.open("rb") as file:
        reader=zstandard.ZstdDecompressor().stream_reader(file)
        with io.TextIOWrapper(reader,encoding="utf-8",errors="replace") as f:
            scan=-1;active=False;mass=[];inten=[]
            for raw in f:
                line=raw.strip()
                if line.upper()=="BEGIN IONS":
                    scan+=1;active=scan in indices;mass=[];inten=[]
                elif line.upper()=="END IONS":
                    if active:
                        if not mass:
                            raise RuntimeError("frozen measured spectrum has no peaks")
                        a=np.asarray(mass,float);v=np.asarray(inten,float)
                        idx=np.argsort(a,kind="mergesort")
                        out[scan]=(a[idx],v[idx])
                    active=False
                elif active and line and "=" not in line:
                    a=line.split()
                    if len(a)<2:raise RuntimeError("malformed selected MGF peak")
                    try:m,v=float(a[0]),float(a[1])
                    except ValueError as e:raise RuntimeError("not a numeric selected MGF peak") from e
                    if not(math.isfinite(m) and math.isfinite(v) and m>0 and v>=0):
                        raise RuntimeError("invalid selected MGF mass/intensity")
                    if v>0:mass.append(m);inten.append(v)
    if set(out)!=indices:
        raise RuntimeError("missing metadata selected MGF spectra")
    return out


def scores_ion_spectra(query,reference,tol=TOLERANCE_DA):
    """Deterministic smallest-m/z-error one-to-one matching for ALL scorers.

    Full original query/reference intensity denominators retained, including
    unmatched positive-intensity peaks; no score-specific alignment or fit.
    """
    qm,qi=query;rm,ri=reference
    if not(len(qm) and len(rm) and 0<tol<.5):
        raise ValueError("nonempty peak vectors and valid fixed m/z tolerance required")
    qsum=float(qi.sum());rsum=float(ri.sum())
    if qsum<=0 or rsum<=0:raise ValueError("zero intensity")
    qnorm=qi/qsum;rnorm=ri/rsum
    ed=[]
    for i,m in enumerate(qm):
        left=int(np.searchsorted(rm,m-tol,side="left"))
        right=int(np.searchsorted(rm,m+tol,side="right"))
        for j in range(left,right):
            err=abs(float(rm[j]-m))
            if err<=tol:ed.append((err,j,i))
    ed.sort()
    usedq=set();usedr=set()
    bc=0.;overlap=0.;dot=0.
    for _,j,i in ed:
        if i in usedq or j in usedr:continue
        usedq.add(i);usedr.add(j)
        bc+=math.sqrt(float(qnorm[i]*rnorm[j]))
        overlap+=min(float(qnorm[i]),float(rnorm[j]))
        dot+=float(qi[i]*ri[j])
    c=dot/float(np.linalg.norm(qi)*np.linalg.norm(ri))
    result={"cosine":c,"hellinger":bc,"fragment_overlap":overlap}
    result["support_composite"]=(c+overlap)/2.
    if any(not math.isfinite(v) or not -.000001<=v<=1.000001 for v in result.values()):
        raise RuntimeError("retrieval similarity invalid")
    return {k:float(min(1,max(0,v))) for k,v in result.items()}


def tie_aware_rank(scores,truth):
    if truth not in scores or len(scores)<2:
        raise ValueError("true molecular label and competitor required")
    if any(not math.isfinite(x) for x in scores.values()):
        raise RuntimeError("nonfinite source-based rank score")
    x=scores[truth]
    higher=sum(v>x+1e-12 for v in scores.values())
    tied=sum(abs(v-x)<=1e-12 for v in scores.values())
    if not tied:raise RuntimeError("true key not scored")
    midrank=1+higher+(tied-1)/2
    return {"mrr":1/midrank,"top1_fractional":1/tied if higher==0 else 0.,
            "midrank":midrank,"candidate_count":len(scores)}


def inference(rows,left,right,seed):
    a=np.asarray([x["metrics"][left]["mrr"] for x in rows],float)
    b=np.asarray([x["metrics"][right]["mrr"] for x in rows],float)
    blocks=[x["identity"] for x in rows]
    return {"mean_mrr_delta":float(np.mean(a-b)),
        "paired_wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
        "structure_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,blocks,n_boot=5000,seed=seed),
        "structure_block_signflip":paired_block_signflip_test(
            a,b,blocks,n_permutations=50000,seed=seed)}




def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def evaluate(source:Path,gate_dir:Path,out:Path):
    original=json.loads((gate_dir/"feasibility.json").read_text())
    original_reference=read_rows(gate_dir/"reference_manifest.jsonl")
    original_queries=read_rows(gate_dir/"hard_queries.jsonl")
    if (original.get("status")!="HARMONIZED2026_CROSS_PROVIDER_NATIVE_HEADER_GATE_V1"
        or original.get("gate_pass") is not True
        or original.get("source_sha256")!=SHA256 or original.get("source_md5")!=MD5
        or original.get("query_peak_locations_or_intensities_opened") is not False
        or original.get("gnps_reference_peaks_opened") is not False
        or len(original_queries)!=original.get("coverage",{}).get("hard_queries",0)):
        raise RuntimeError("original metadata-only gate not passed; NO peaks authorized")
    sha256file(source)
    # REPLAY native header-only gate first. Missing or changed checksum,
    # identity, candidate or query selection MUST ABORT before peak parsing.
    rows,meta=meta_only(source)
    refs,queries,cov,passed=freeze(rows)
    if (not passed or refs!=original_reference or queries!=original_queries
        or cov!=original["coverage"] or meta!=original["raw_header_audit"]):
        raise RuntimeError("frozen source native identity/candidate pool mismatch")
    queries=original_queries
    if not queries:raise RuntimeError("no genuinely competing chemical identities")
    needed={r["index"] for r in queries}
    needed.update(j for r in queries for j in r["selected_reference_indices"])
    spectra=extract_selected_peaks(source,needed)
    reference_by_index={r["index"]:r for r in original_reference}
    scores=[]
    duplicate_signal=0
    for query in queries:
        obs=spectra[query["index"]]
        pool=query["selected_reference_indices"]
        per_method={m:{} for m in METHODS}
        actual_ids=set()
        same_exact=False
        for idx in pool:
            if idx not in reference_by_index:
                raise RuntimeError("candidate does not belong to original GNPS fixed inventory")
            native=reference_by_index[idx]
            if (native["source"]!="GNPS" or
                native["formula"]!=query["formula"] or
                native["mode"]!=query["mode"] or
                abs(native["precursor_mz"]-query["precursor_mz"])>
                  max(.02,20e-6*query["precursor_mz"])+1e-12):
                raise RuntimeError("reference candidate native formula, polarity or mass drifted")
            label=native["inchikey"];actual_ids.add(label)
            ref=spectra[idx]
            if (np.array_equal(obs[0],ref[0]) and np.array_equal(
                obs[1]/obs[1].sum(),ref[1]/ref[1].sum())):
                same_exact=True
            v=scores_ion_spectra(obs,ref)
            for method in METHODS:
                per_method[method][label]=max(per_method[method].get(label,-1),v[method])
        if (len(actual_ids)!=query["candidate_structures"]
            or len(actual_ids)<2 or query["inchikey"] not in actual_ids):
            raise RuntimeError("exact original hard-candidate pool was altered")
        if same_exact:duplicate_signal+=1
        scores.append({"identity":query["inchikey"],"formula":query["formula"],
            "polarity":query["mode"],"query_index":query["index"],
            "candidate_count":len(actual_ids),
            "exact_identical_query_reference_signal":same_exact,
            "metrics":{m:tie_aware_rank(per_method[m],query["inchikey"])
                       for m in METHODS}})
    comparisons={name:inference(scores,left,right,20261017+j*301)
                 for j,(name,left,right) in enumerate(PRIMARY)}
    arr=sorted((c["structure_block_signflip"]["p_value_two_sided"],name)
               for name,c in comparisons.items())
    prior=0.
    for i,(p,name) in enumerate(arr):
        prior=max(prior,min(1.,(len(arr)-i)*p))
        comparisons[name]["holm_p"]=prior
    statistical=all(
        c["mean_mrr_delta"]>0 and
        c["structure_block_bootstrap_95"]["ci_low"]>0 and
        c["holm_p"]<.05 for c in comparisons.values())
    scientifically_qualified=statistical and duplicate_signal==0
    result={
        "status":"FIRST_LOOK_HARMONIZED2026_GNPS_TO_MSG_SAME_FORMULA_IDENTIFICATION",
        "source_doi":"10.5281/zenodo.20042904",
        "source_sha256":SHA256,"no_train_or_tuning_on_new_source":True,
        "query_source":"MassSpecGym","reference_source":"GNPS",
        "study_scope":"cross-archive same-native-formula, same ion mode and precursor-matched structure ID, NOT new-lab acquisition or energy interpolation",
        "n_hard_queries":len(scores),
        "n_distinct_structure_blocks":len({r["identity"] for r in scores}),
        "n_distinct_formulas":len({r["formula"] for r in scores}),
        "exact_identical_query_reference_spectra":duplicate_signal,
        "mean_mrr":{m:float(np.mean([r["metrics"][m]["mrr"] for r in scores]))
                    for m in METHODS},
        "fractional_top1":{m:float(np.mean([r["metrics"][m]["top1_fractional"]
                                          for r in scores])) for m in METHODS},
        "comparisons":comparisons,
        "predeclared_joint_mrr_test_passed":statistical,
        "qualified_for_further_external_validation":scientifically_qualified,
        "by_polarity":{pol:{
            "n_queries":sum(x["polarity"]==pol for x in scores),
            "mrr":{m:float(np.mean([x["metrics"][m]["mrr"] for x in scores if x["polarity"]==pol]))
                   for m in METHODS},
        } for pol in ("positive","negative")},
        "limitations":"Computationally harmonized identities not independently curated; cross-library provenance not proof of independent physical acquisition; max two reference spectra per mass bucket; hypotheses selected after earlier unrelated sources, no universal collision-energy geometry result.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\\n")
    with (out/"query_metrics.jsonl").open("w") as f:
        for row in scores:f.write(json.dumps(row,sort_keys=True)+"\\n")
    lines=[
      "# GNPS → MassSpecGym independent source-archive structure ranking",
      "",
      f"**Hard same-formula queries:** {len(scores)}; distinct structures {result['n_distinct_structure_blocks']}; exact duplicate query/reference spectra {duplicate_signal}.",
      "",
      "| Method | MRR | Fractional top1 |",
      "|---|---:|---:|",
    ]
    for m in METHODS:
        lines.append(f"| {m} | {result['mean_mrr'][m]:.6f} | {result['fractional_top1'][m]:.6f} |")
    lines+=[""]
    for name,c in comparisons.items():
        ci=c["structure_block_bootstrap_95"]
        lines.append(f"- {name}: mean ΔMRR {c['mean_mrr_delta']:+.6f}, "
            f"95% chemical block CI [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], "
            f"Holm p {c['holm_p']:.6f}, wins/loss/ties {c['paired_wins_losses_ties']}")
    lines+=["",f"**Joint statistical success:** {statistical}.",
            f"**No exact-duplicate contamination and joint success:** {scientifically_qualified}.",
            "",result["limitations"],""]
    (out/"report.md").write_text("\\n".join(lines))
    print("\\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zst",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.zst,a.gate_dir,a.out)


if __name__=="__main__":main()
