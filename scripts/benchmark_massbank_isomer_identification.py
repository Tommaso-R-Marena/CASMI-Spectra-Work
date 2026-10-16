"""Frozen post-hoc exact-formula, exact-instrument MassBank isomer-ranking audit.

Selected using metadata alone. Interpolators were fixed on prior MSnLib
discovery. Each candidate prediction uses only its own lower/high references.
This is NOT an untouched new source or a full CASMI candidate generation test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from qfd.power_geometry import power_mean_distribution
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity, support_only_suppression,
)
from benchmark_massbank_external_energy_geometry import parse_peak_spectrum

ARCHIVE_SHA = "12c6a922929f9bf2484c46586b6101512a415e44680558883b0630c6e47905fb"
ROWS_SHA = "2c973a7b83c1a48f95f923107ee72bb44a2a95f7d7da403f9d6f541d662efc8b"
MANIFEST_SHA = "265dfb75e33d2eff3f19945255bddb5db0c0f7f9161593ada80cd2b50ecac9c4"
METHODS = ("alpha_quarter", "hellinger_fixed", "linear", "support_only")
POOL_FIELDS = ("formula", "ion_mode", "adduct", "contributor",
               "instrument", "fragmentation", "query_eV")
EXPECTED = {"pools": 105, "queries": 252, "formula_blocks": 44,
            "strict_nonoverlap_pools": 94, "strict_nonoverlap_queries": 230,
            "strict_nonoverlap_formulas": 40}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def select_pools(manifest):
    """No spectral peaks, intensity, rankings, or methods are consulted."""
    grouped = defaultdict(list)
    paths = set()
    for r in manifest:
        if r["query_path"] in paths:
            raise ValueError("duplicate query accession in frozen manifest")
        paths.add(r["query_path"])
        grouped[tuple(r[field] for field in POOL_FIELDS)].append(r)
    pools = {}
    discarded_stereo_groups = 0
    for key, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        identities = [r["inchikey"][:14] for r in rows]
        if len(set(identities)) != len(identities):
            discarded_stereo_groups += 1
            continue
        pools[key] = sorted(rows, key=lambda r:r["inchikey"][:14])
    return pools, discarded_stereo_groups

def check_pools(pools, prior):
    if len(pools) != EXPECTED["pools"]:
        raise ValueError("frozen number of isomer pools changed")
    cohort = [r for v in pools.values() for r in v]
    if len(cohort) != EXPECTED["queries"] or len({r["formula"] for r in cohort}) != EXPECTED["formula_blocks"]:
        raise ValueError("frozen isomer challenge counts changed")
    no_prior = {k: v for k, v in pools.items()
                if all(r["inchikey"].upper() not in prior for r in v)}
    pure = [r for v in no_prior.values() for r in v]
    if (len(no_prior) != EXPECTED["strict_nonoverlap_pools"] or
        len(pure) != EXPECTED["strict_nonoverlap_queries"] or
        len({r["formula"] for r in pure}) != EXPECTED["strict_nonoverlap_formulas"]):
        raise ValueError("frozen structure-nonoverlap challenge counts changed")
    if len({r["query_path"] for r in cohort}) != len(cohort):
        raise ValueError("query reused in multiple identification pools")
    return no_prior

def validate_artifacts(efficacy_dir, prior_inventory):
    if sha256(efficacy_dir / "rows.jsonl") != ROWS_SHA:
        raise ValueError("frozen original paired-score artifact has changed")
    if sha256(efficacy_dir / "label_free_triplets.jsonl") != MANIFEST_SHA:
        raise ValueError("frozen label-free manifest has changed")
    prior = json.loads(prior_inventory.read_text())
    if (prior.get("status") != "PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC" or
        set(prior.get("sources",{})) != {"mcebio_neg","mcescaf_pos","enammol_pos","enamdisc_neg"}):
        raise ValueError("invalid or incomplete prior library structure inventory")
    keys = prior["all_prior_evaluated_full_inchikeys"]
    if keys != sorted(set(keys)):
        raise ValueError("unsorted/duplicated prior exact structure keys")
    manifest = read_jsonl(efficacy_dir / "label_free_triplets.jsonl")
    scored = read_jsonl(efficacy_dir / "rows.jsonl")
    if len(manifest) != len(scored) or len(manifest) != 2138:
        raise ValueError("original MassBank count changed")
    by_query = {r["query_path"]:r for r in scored}
    if len(by_query) != len(scored):
        raise ValueError("original MassBank duplicate accession")
    for r in manifest:
        row=by_query.get(r["query_path"])
        if row is None or any(row[k]!=v for k,v in r.items()):
            raise ValueError("original scored target does not match frozen metadata")
    return manifest, by_query, set(keys)

def candidate_prediction(grid, t):
    return {
        "alpha_quarter":power_mean_distribution(grid.low,grid.high,t,0.25),
        "hellinger_fixed":power_mean_distribution(grid.low,grid.high,t,0.5),
        "linear":power_mean_distribution(grid.low,grid.high,t,1.0),
        "support_only":support_only_suppression(grid.low,grid.high,t,power=4),
    }

def rank_pool(pool, spectra, archived, *, tolerance=0.01):
    """Precompute all candidate references; only then read withheld targets."""
    predictions = []
    for candidate in pool:
        grid=reference_only_grid(
            spectra(candidate["low_path"]),spectra(candidate["high_path"]),
            tolerance_da=tolerance)
        predictions.append((candidate["inchikey"][:14],grid,
                            candidate_prediction(grid,float(candidate["t"]))))
    out=[]
    for truth in pool:
        q = spectra(truth["query_path"])
        scores={method:[] for method in METHODS}
        for key,grid,method_predictions in predictions:
            aligned,unmatched = query_mass_on_reference_grid(q,grid,tolerance_da=tolerance)
            if not -1e-12<=unmatched<=1+1e-12:
                raise ValueError("unmatched target mass invalid")
            for method in METHODS:
                score=affinity(aligned,method_predictions[method])
                if not np.isfinite(score) or not -1e-9<=score<=1+1e-9:
                    raise ValueError("invalid candidate affinity")
                scores[method].append((key,float(score)))
        correct=truth["inchikey"][:14]
        query_rec={"query_path":truth["query_path"],"inchikey":truth["inchikey"],
                   "formula":truth["formula"],"contributor":truth["contributor"],
                   "instrument_type":truth["instrument_type"],
                   "instrument":truth["instrument"],"n_candidates":len(pool),
                   "correct_key14":correct}
        original=archived[truth["query_path"]]
        for method in METHODS:
            ranked=sorted(scores[method],key=lambda item:(-item[1],item[0]))
            hits=[i+1 for i,(key,_) in enumerate(ranked) if key==correct]
            if len(hits)!=1:
                raise ValueError("correct connectivity absent or duplicated")
            query_rec[method+"_rank"]=hits[0]
            query_rec[method+"_rr"]=1.0/hits[0]
            own=dict(scores[method])[correct]
            wrong=max(s for key,s in scores[method] if key != correct)
            query_rec[method+"_margin"]=float(own-wrong)
            if method in ("alpha_quarter","hellinger_fixed","support_only"):
                if abs(own-float(original[method]))>1e-9:
                    raise ValueError("own-identity affinity differs from frozen original result")
        out.append(query_rec)
    return out

def compare(rows,left,right,seed):
    a=np.asarray([r[left+"_rr"] for r in rows])
    b=np.asarray([r[right+"_rr"] for r in rows])
    blocks=[r["formula"] for r in rows]
    d=a-b
    return {
        "mean_paired_rr_delta":float(d.mean()),
        "wins":int((d>1e-12).sum()),"losses":int((d<-1e-12).sum()),
        "ties":int((abs(d)<=1e-12).sum()),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,blocks,n_boot=10000,seed=seed),
        "formula_block_signflip_two_sided":paired_block_signflip_test(
            a,b,blocks,n_permutations=50000,seed=seed),
    }

def summarize(rows, *, inference):
    if not rows:
        return None
    cands=np.asarray([r["n_candidates"] for r in rows])
    mrr_random=float(np.mean([sum(1/j for j in range(1,k+1))/k for k in cands]))
    output={
        "queries":len(rows),"structures":len({r["inchikey"][:14] for r in rows}),
        "formulas":len({r["formula"] for r in rows}),
        "sources":len({r["contributor"] for r in rows}),
        "candidate_count_distribution":dict(sorted(Counter(map(int,cands)).items())),
        "random_ranking_expected_top1":float(np.mean(1/cands)),
        "random_ranking_expected_mrr":mrr_random,
        "methods":{},
    }
    for method in METHODS:
        rr=np.asarray([r[method+"_rr"] for r in rows])
        ranks=np.asarray([r[method+"_rank"] for r in rows])
        margins=np.asarray([r[method+"_margin"] for r in rows])
        output["methods"][method]={
            "mrr":float(rr.mean()),"top1":float((ranks==1).mean()),
            "mean_correct_minus_best_wrong_affinity":float(margins.mean()),
            "median_correct_minus_best_wrong_affinity":float(np.median(margins)),
        }
    if inference:
        output["paired_comparisons"]={
            "quarter_minus_hellinger":compare(rows,"alpha_quarter","hellinger_fixed",20261016),
            "quarter_minus_support":compare(rows,"alpha_quarter","support_only",20261017),
            "quarter_minus_linear":compare(rows,"alpha_quarter","linear",20261018),
        }
    else:
        a=np.asarray([r["alpha_quarter_rr"]-r["hellinger_fixed_rr"] for r in rows])
        output["quarter_minus_hellinger_mrr"]=float(a.mean())
    return output

def evaluate(massbank_zip,efficacy_dir,prior_inventory,out):
    manifest,archived,prior=validate_artifacts(efficacy_dir,prior_inventory)
    pools, discarded=select_pools(manifest)
    no_prior=check_pools(pools,prior)
    if sha256(massbank_zip)!=ARCHIVE_SHA:
        raise ValueError("MassBank archive differs from original source")
    rows=[]
    with ZipFile(massbank_zip) as z:
        namelist=set(z.namelist())
        ref_paths={p for pool in pools.values() for r in pool for p in (r["low_path"],r["high_path"])}
        q_paths={r["query_path"] for pool in pools.values() for r in pool}
        if not ref_paths.issubset(namelist) or not q_paths.issubset(namelist):
            raise ValueError("required raw MassBank accession missing")
        cache={}
        def spectrum(path):
            if path not in cache:
                cache[path]=parse_peak_spectrum(z.read(path).decode("utf-8",errors="replace"),record_path=path)
            return cache[path]
        for pool in pools.values():
            rows.extend(rank_pool(pool,spectrum,archived))
    if len(rows)!=EXPECTED["queries"]:
        raise ValueError("scored isomer query count changed")
    nonoverlap_q={r["query_path"] for v in no_prior.values() for r in v}
    pure=[r for r in rows if r["query_path"] in nonoverlap_q]
    if len(pure)!=EXPECTED["strict_nonoverlap_queries"]:
        raise ValueError("nonoverlap final ranking count changed")
    by_source=defaultdict(list)
    for r in rows:
        by_source[(r["contributor"],r["instrument_type"])].append(r)
    results={
        "status":"POSTHOC_MASSBANK_2026_03_FORMULA_ISOMER_IDENTIFICATION",
        "science_class":"retrospective, same MassBank source; NOT external acquisition replication",
        "original_efficacy_run":38918658343,
        "original_source_overlap_run":38920155068,
        "immutable_scorer":"reference_only_grid + one_to_one_query_matching + power_mean_distribution",
        "scoring_input_archive_sha256":ARCHIVE_SHA,
        "discarded_stereo_ambiguous_groups":discarded,
        "all":summarize(rows,inference=True),
        "all_candidates_no_prior_msnlib_overlap":summarize(pure,inference=True),
        "source_instrument_descriptive":[
            {"contributor":k[0],"instrument_type":k[1],**summarize(v,inference=False)}
            for k,v in sorted(by_source.items())],
        "limitations":"tiny exact-formula candidate pools; same source previously used for affinity validation; historical datasets may contain acquisition duplicates; not official CASMI 2026 MRR@25.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(results,indent=2)+"\n")
    with (out/"rows.jsonl").open("w") as fp:
        for r in sorted(rows,key=lambda x:x["query_path"]):
            fp.write(json.dumps(r,sort_keys=True)+"\n")
    lines=[
        "# MassBank post-hoc exact-formula isomer identification", "",
        "**Not a new externally blinded or independent dataset.**", "",
        "| Population | n | Formula blocks | Random MRR | Quarter MRR | Hellinger MRR | Support MRR | Linear MRR | Δ quarter−Hellinger |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name,v in [("All eligible isomer challenges",results["all"]),("No prior evaluated MSnLib structure in candidate pool",results["all_candidates_no_prior_msnlib_overlap"])]:
        lines.append(f"| {name} | {v['queries']} | {v['formulas']} | {v['random_ranking_expected_mrr']:.6f} | "
                     f"{v['methods']['alpha_quarter']['mrr']:.6f} | {v['methods']['hellinger_fixed']['mrr']:.6f} | "
                     f"{v['methods']['support_only']['mrr']:.6f} | {v['methods']['linear']['mrr']:.6f} | "
                     f"{v['paired_comparisons']['quarter_minus_hellinger']['mean_paired_rr_delta']:+.6f} |")
        ci=v["paired_comparisons"]["quarter_minus_hellinger"]["formula_block_bootstrap_95"]
        test=v["paired_comparisons"]["quarter_minus_hellinger"]["formula_block_signflip_two_sided"]
        lines.append(f"\n{name}: quarter−Hellinger MRR 95% formula bootstrap [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}], two-sided signflip p={test['p_value_two_sided']:.6f}.")
    lines.extend(["","## Full contributor/instrument transparency",
                  "| Source | Instrument type | n | Quarter MRR | Hellinger MRR | Δ MRR |",
                  "|---|---|---:|---:|---:|---:|"])
    for v in results["source_instrument_descriptive"]:
        lines.append(f"| {v['contributor']} | {v['instrument_type']} | {v['queries']} | "
                     f"{v['methods']['alpha_quarter']['mrr']:.6f} | {v['methods']['hellinger_fixed']['mrr']:.6f} | "
                     f"{v['quarter_minus_hellinger_mrr']:+.6f} |")
    lines.extend(["",results["limitations"],""])
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--massbank-zip",type=Path,required=True)
    p.add_argument("--efficacy-dir",type=Path,required=True)
    p.add_argument("--prior-inventory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    evaluate(args.massbank_zip,args.efficacy_dir,args.prior_inventory,args.out)

if __name__=="__main__":
    main()
