"""Frozen single-energy SPLISUM external molecular retrieval. No fitting.

Do not run unless the original header-only MD5-verified gate passes. This is
REAL query/reference candidate identification, not energy interpolation.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_splisum_header_gate import (
    MD5, MINIMA, check, headers_only, file_md5)
from qfd.stats import (
    paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie)

METHODS=("cosine","hellinger","fragment_overlap","support_composite")
PRIMARY=(("support_vs_cosine","support_composite","cosine"),
         ("support_vs_hellinger","support_composite","hellinger"))
TOLERANCE_DA=.01


def peaks_for_selected(path:Path,needed:set[int],metadata:list[dict]):
    """Stream only selected reference/query fragment peaks in their native MGF.

    The source and cohort are frozen before this parser is called.
    """
    spectra={};scan=-1;active=False;masses=[];intensities=[]
    for raw in path.open("r",encoding="utf-8",errors="replace"):
        line=raw.strip()
        if line.upper()=="BEGIN IONS":
            scan+=1;active=scan in needed;masses=[];intensities=[]
        elif line.upper()=="END IONS":
            if active:
                if not masses:
                    raise RuntimeError("empty eligible measured spectrum")
                mz=np.asarray(masses,float);inten=np.asarray(intensities,float)
                order=np.argsort(mz,kind="mergesort")
                spectra[scan]=(mz[order],inten[order])
            active=False
        elif active and line and "=" not in line:
            values=line.split()
            if len(values)<2:raise RuntimeError("malformed source fragment row")
            try:m,i=float(values[0]),float(values[1])
            except ValueError as exc:raise RuntimeError("malformed numeric peak") from exc
            if not(math.isfinite(m) and math.isfinite(i) and m>0 and i>=0):
                raise RuntimeError("invalid source intensity/mass")
            if i>0:masses.append(m);intensities.append(i)
    if set(spectra)!=needed:
        raise RuntimeError("original header-selected scan inventory incomplete")
    return spectra


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


def evaluate(files:dict[str,Path],gate_dir:Path,out:Path):
    old=json.loads((gate_dir/"feasibility.json").read_text())
    frozen=[json.loads(s) for s in (gate_dir/"eligible_queries.jsonl").read_text().splitlines() if s.strip()]
    if (old.get("status")!="SPLISUM_2026_FIFTH_ORIGIN_HEADER_ONLY_GATE_V1"
        or old.get("gate_pass") is not True or old.get("source_checksums_md5")!=MD5
        or old.get("query_fragment_mass_values_or_intensities_opened") is not False
        or old.get("score_model_selected_using_this_source") is not False
        or len(frozen)!=old.get("coverage",{}).get("evaluable_queries")):
        raise RuntimeError("scientific gate not passed; no access to spectral fragment peaks")
    # Recapture full original metadata and compare with baseline BEFORE scores.
    # This expensive pass protects against source or manifest substitution.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        now=check(files,Path(d))
        replay=[json.loads(s) for s in (Path(d)/"eligible_queries.jsonl").read_text().splitlines() if s.strip()]
    if now!=old or replay!=frozen:
        raise RuntimeError("original labels, source bytes, or metadata eligibility drifted")
    use=[q for q in frozen if q["target_candidate_structures"]>=2]
    if len(use)<MINIMA["isomer_or_isobar_queries"]:
        raise RuntimeError("no qualifying authentic competitor tasks")
    allsource={}
    for k,p in files.items():
        md,_=headers_only(p,"positive" if k.endswith("pos") else "negative")
        allsource[k]=md
    # Populate *every* native mass-eligible labeled target reference.
    target_mz={}
    target_data={}
    for suffix in ("pos","neg"):
        key="t"+suffix
        md=allsource[key]
        target_data[key]=sorted([x for x in md if x["label"] and x["precursor_mz"]],
                         key=lambda x:(x["precursor_mz"],x["index"]))
        target_mz[key]=[x["precursor_mz"] for x in target_data[key]]
    q_scans={}
    t_scans={}
    for suffix in ("pos","neg"):
        qkey="q"+suffix;tkey="t"+suffix
        needed_query={r["query_index"] for r in use if r["query_source"]==qkey}
        needed_targets=set()
        for query in use:
            if query["query_source"]!=qkey:continue
            mass=query["precursor_mz"]
            tol=max(.02,20e-6*mass)
            mzs=target_mz[tkey]
            lo=bisect.bisect_left(mzs,mass-tol)
            hi=bisect.bisect_right(mzs,mass+tol)
            needed_targets.update(r["index"] for r in target_data[tkey][lo:hi])
        q_scans[qkey]=peaks_for_selected(files[qkey],needed_query,allsource[qkey])
        t_scans[tkey]=peaks_for_selected(files[tkey],needed_targets,allsource[tkey])
    results=[]
    duplicate_spectra=0
    for query in use:
        qkey=query["query_source"];tkey="t"+qkey[1:]
        qspec=q_scans[qkey][query["query_index"]]
        mass=query["precursor_mz"];tol=max(.02,20e-6*mass)
        mzs=target_mz[tkey]
        lo=bisect.bisect_left(mzs,mass-tol)
        hi=bisect.bisect_right(mzs,mass+tol)
        pool=target_data[tkey][lo:hi]
        labels={r["label"] for r in pool}
        if query["identity"] not in labels or len(labels)!=query["target_candidate_structures"]:
            raise RuntimeError("frozen label-blind precursor candidate pool changed")
        candidates={method:{} for method in METHODS}
        exact_same_scan=False
        for candidate in pool:
            ref=t_scans[tkey][candidate["index"]]
            label=candidate["label"]
            score=scores_ion_spectra(qspec,ref)
            if (np.array_equal(qspec[0],ref[0]) and np.array_equal(
                qspec[1]/qspec[1].sum(),ref[1]/ref[1].sum())):
                exact_same_scan=True
            for method in METHODS:
                candidates[method][label]=max(
                    candidates[method].get(label,-1.),score[method])
        metrics={name:tie_aware_rank(candidates[name],query["identity"])
                 for name in METHODS}
        if exact_same_scan:duplicate_spectra+=1
        results.append({"identity":query["identity"],
                        "query_index":query["query_index"],"polarity":query["polarity"],
                        "candidate_count":len(labels),
                        "exact_duplicate_in_target_library":exact_same_scan,
                        "metrics":metrics})
    if len(results)!=len(use):
        raise RuntimeError("not all preidentified competitor queries scored")
    comps={name:inference(results,left,right,20261017+i*311)
           for i,(name,left,right) in enumerate(PRIMARY)}
    porder=sorted((v["structure_block_signflip"]["p_value_two_sided"],name)
                  for name,v in comps.items())
    monotone=0.
    for j,(p,name) in enumerate(porder):
        monotone=max(monotone,min(1.,(len(porder)-j)*p))
        comps[name]["holm_p"]=monotone
    confirmed=all(
        v["mean_mrr_delta"]>0
        and v["structure_block_bootstrap_95"]["ci_low"]>0
        and v["holm_p"]<.05 for v in comps.values())
    result={"status":"SPLISUM_INDEPENDENT_SINGLE_ENERGY_STRUCTURE_RETRIEVAL",
        "source_doi":"10.5281/zenodo.19834896",
        "scientific_success":confirmed,
        "same_energy_mgf_identity_retrieval_not_multi_energy_interpolation":True,
        "no_model_fitting_to_source":True,
        "n_queries_competitor_pools":len(results),
        "n_identity_blocks":len({r["identity"] for r in results}),
        "n_exact_duplicate_query_reference_spectra":duplicate_spectra,
        "original_gate_eligible_queries":len(frozen),
        "mean_mrr":{m:float(np.mean([r["metrics"][m]["mrr"] for r in results]))
                    for m in METHODS},
        "fractional_top1":{m:float(np.mean([r["metrics"][m]["top1_fractional"] for r in results]))
                          for m in METHODS},
        "comparisons":comps,
        "limitations":"Same collision energy throughout, exact raw chemical identity matching only, candidate pools are precursor/polarity constrained, multiple target replicates max pooled; any exact duplicate spectra weaken independent measurement validity.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (out/"query_metrics.jsonl").open("w") as f:
        for row in results:f.write(json.dumps(row,sort_keys=True)+"\n")
    lines=["# Independent SPLISUM single-energy structure retrieval",
           "",
           f"Hard competitor query spectra: {len(results)}; chemical blocks: {result['n_identity_blocks']}.",
           f"Exact identical query/target spectra: {duplicate_spectra}.",
           "","| Fixed method | MRR | fractional Top1 |","|---|---:|---:|"]
    for m in METHODS:
        lines.append(f"| {m} | {result['mean_mrr'][m]:.6f} | {result['fractional_top1'][m]:.6f} |")
    for name,c in comps.items():
        ci=c["structure_block_bootstrap_95"]
        lines.append(f"- {name}: ΔMRR {c['mean_mrr_delta']:+.6f}, "
            f"95% molecule-block CI [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], "
            f"Holm p={c['holm_p']:.6f}")
    lines += ["",f"Frozen joint two-baseline success: **{confirmed}**.",
              "",result["limitations"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    for k in MD5:p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate({k:getattr(a,k) for k in MD5},a.gate_dir,a.out)


if __name__=="__main__":
    main()
