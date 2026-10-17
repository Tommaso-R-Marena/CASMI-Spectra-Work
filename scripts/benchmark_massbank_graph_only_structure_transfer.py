"""MassBank 2026.03 graph-only molecular structure transfer: no candidate spectra.

The *same frozen* RDKit graph signature model from the forward GNPS/MSG
experiment is evaluated on ALL 252 original exact-formula isomer queries,
including candidates without usable graph signatures. The source originally
appeared in previous MassBank spectral-geometry studies: retrospective
cross-archive transfer, NOT a never-seen confirmatory external study.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_massbank_graph_native_headers import (
    STATUS, evaluate as recheck_prescore,
)
from benchmark_massbank_isomer_identification import (
    ARCHIVE_SHA, validate_artifacts,select_pools,check_pools,sha256,
)
from benchmark_massbank_external_energy_geometry import parse_peak_spectrum
from qfd.graph_fragment_signatures import (
    signatures,discriminative_graph_scores,
)
from qfd.stats import paired_block_bootstrap_delta


def readjsonl(path:Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_label_null(scores):
    """Expected reciprocal of midrank under every possible true-label slot."""
    values=np.asarray(list(scores.values()),float)
    if len(values)<2 or not np.all(np.isfinite(values)):
        raise ValueError("requires two finite candidate chemical scores")
    rr=[]
    for s in values:
        above=int(np.sum(values>s+1e-12))
        tie=int(np.sum(np.abs(values-s)<=1e-12))
        rr.append(1./(1+above+(tie-1)/2))
    return float(np.mean(rr))


def tieaware(scores,key):
    """Same rank policy used in original GNPS graph experiment."""
    if key not in scores or len(scores)<2:
        raise ValueError("true original key missing")
    vals=np.asarray(list(scores.values()),float)
    truth=float(scores[key])
    better=int(np.sum(vals>truth+1e-12))
    ties=int(np.sum(np.abs(vals-truth)<=1e-12))
    return 1./(1+better+(ties-1)/2), (1./ties if better==0 else 0.)


def compare(rows):
    a=np.asarray([r["graph_mrr"] for r in rows],float)
    b=np.asarray([r["exact_random_label_null_mrr"] for r in rows],float)
    return paired_block_bootstrap_delta(
        a,b,[r["formula"] for r in rows],n_boot=10000,seed=20261017)


def summarize(rows):
    if not rows:
        return None
    stat=compare(rows)
    return {
        "n_queries":len(rows),
        "n_formulas":len({r["formula"] for r in rows}),
        "n_full_structure_truths":len({r["inchikey"] for r in rows}),
        "mean_graph_mrr":float(np.mean([r["graph_mrr"] for r in rows])),
        "mean_exact_random_label_null_mrr":float(np.mean([
            r["exact_random_label_null_mrr"] for r in rows])),
        "mean_graph_top1_fractional":float(np.mean([
            r["graph_top1_fractional"] for r in rows])),
        "mean_uniform_label_top1_fractional":float(np.mean([
            r["uniform_label_top1_fractional"] for r in rows])),
        "formula_block_bootstrap_95":stat,
        "exploratory_positive_delta_and_ci":bool(stat["delta"]>0 and stat["ci_low"]>0),
    }


def evaluate(source:Path,efficacy:Path,prior_inventory:Path,locked:Path,out:Path):
    originally=json.loads((locked/"feasibility.json").read_text())
    locked_rows=readjsonl(locked/"candidate_native_graph_headers.jsonl")
    if (
        originally.get("status")!=STATUS
        or originally.get("gate_pass") is not True
        or originally.get("source_sha256")!=ARCHIVE_SHA
        or originally.get("unchanged_original_queries")!=252
        or originally.get("unchanged_original_formula_blocks")!=44
        or originally.get("n_queries_in_pools_with_all_graph_structures_usable")!=220
        or originally.get("n_formulas_in_pools_with_all_graph_structures_usable")!=38
        or originally.get("experimental_target_peak_lines_opened") is not False
        or originally.get("experimental_reference_peak_lines_opened") is not False
        or originally.get("query_smiles_never_read") is not True
    ):
        raise RuntimeError("pre-score MassBank graph eligibility failed; NO NEW PEAKS OPENED")
    if sha256(source)!=ARCHIVE_SHA:
        raise RuntimeError("MassBank source archive identity changed")

    # Independent source HEADER-ONLY reconstruction BEFORE reading any query
    # peaks. This refreezes every label, source accession and graph status.
    replay=out/"locked_metadata_replay"
    now=recheck_prescore(source,efficacy,prior_inventory,replay)
    current_rows=readjsonl(replay/"candidate_native_graph_headers.jsonl")
    if now!=originally or current_rows!=locked_rows:
        raise RuntimeError("original graph metadata and source archive drifted")
    manifest,old_scores,prior=validate_artifacts(efficacy,prior_inventory)
    pools,discarded=select_pools(manifest)
    clean=check_pools(pools,prior)
    if len(pools)!=105 or len(locked_rows)!=105:
        raise RuntimeError("MassBank native hard candidate pool count drifted")
    by_pool={tuple(row["pool"]):row for row in locked_rows}
    eligible_query_paths={
        x["query_path"] for pool in clean.values() for x in pool
    }
    outrows=[]
    chemistry=Counter()
    with ZipFile(source) as archive:
        all_names=set(archive.namelist())
        paths={q["query_path"] for pool in pools.values() for q in pool}
        if not paths.issubset(all_names):
            raise RuntimeError("missing frozen withheld measured query accession")
        # Intentionally no candidate["low_path"] or ["high_path"] peak loads:
        # only measured query paths can be opened as a spectrum here.
        for key,pool in sorted(pools.items()):
            locked_pool=by_pool.get(tuple(key))
            if locked_pool is None:
                raise RuntimeError("locked formula/isomer group missing")
            candidates=locked_pool["candidates"]
            if ([q["inchikey"][:14] for q in pool]
                !=[x["key14"] for x in candidates]):
                raise RuntimeError("native candidate structural identity order changed")
            masses={}
            for entry in candidates:
                label=entry["key14"]
                status=entry["status"]
                chemistry[status]+=1
                if status=="OK":
                    graph,new_status,num=signatures(
                        entry["smiles"],entry["formula"])
                    if (new_status!="OK" or len(graph)!=entry["n_fragment_masses"]
                        or num!=entry["n_eligible_breaks"]):
                        raise RuntimeError("frozen RDKit graph definitions changed")
                    masses[label]=graph
                else:
                    masses[label]=tuple()
            if len(masses)!=len(pool):
                raise RuntimeError("distinct connectivity candidates not preserved")
            for target in pool:
                raw=archive.read(target["query_path"]).decode(
                    "utf-8",errors="replace")
                query=parse_peak_spectrum(raw,record_path=target["query_path"])
                model,diag=discriminative_graph_scores(
                    (query.mzs,query.intensities),masses)
                label=target["inchikey"][:14]
                reciprocal,top1=tieaware(model,label)
                null=exact_label_null(model)
                outrows.append({
                    "query_path":target["query_path"],
                    "inchikey":target["inchikey"],
                    "formula":target["formula"],
                    "ion_mode":target["ion_mode"],
                    "contributor":target["contributor"],
                    "instrument_type":target["instrument_type"],
                    "n_candidates":len(model),
                    "graph_mrr":float(reciprocal),
                    "graph_top1_fractional":float(top1),
                    "exact_random_label_null_mrr":null,
                    "uniform_label_top1_fractional":1./len(model),
                    "graph_truth_score":model[label],
                    "graph_explained_intensity_fraction":
                        diag["fraction_of_query_intensity_explained_by_any"],
                    "no_prior_msnlib_structures_in_pool":
                        target["query_path"] in eligible_query_paths,
                })
    if (len(outrows)!=252 or len({r["query_path"] for r in outrows})!=252
        or len({r["formula"] for r in outrows})!=44):
        raise RuntimeError("original exact 252-query denominator not preserved")
    no_prior=[r for r in outrows if r["no_prior_msnlib_structures_in_pool"]]
    if len(no_prior)!=230 or len({r["formula"] for r in no_prior})!=40:
        raise RuntimeError("original 230/40 nonoverlap sensitivity cannot be replayed")
    total=summarize(outrows)
    clean_stat=summarize(no_prior)
    by_mode={
        mode:summarize([r for r in outrows if r["ion_mode"]==mode])
        for mode in ("POSITIVE","NEGATIVE")
    }
    outcome={
        "status":"MASSBANK_GRAPH_ONLY_NATIVE_STRUCTURE_ID_SECOND_SOURCE_V1",
        "source_sha256":ARCHIVE_SHA,
        "source_doi":"10.5281/zenodo.19590880",
        "historical_massbank_identification_source_previously_used":True,
        "prescore_graph_native_gate_run_id":39060743234,
        "original_GNPS_MSG_graph_only_run":39060157608,
        "frozen_model_original_git_blob":"45616998a79f0a8451abb55ff56628ff527f3586",
        "native_source": "MassBank 2026.03",
        "all_original_252":total,
        "no_prior_msnlib_structures_230":clean_stat,
        "by_ion_mode":by_mode,
        "status_counts_per_pool_candidate":dict(chemistry),
        "candidate_reference_peak_arrays_used_by_ranker":False,
        "query_peak_spectra_used_by_ranker":True,
        "invalid_graph_candidates_preserved":True,
        "original_252_query_pool_candidate_sets_preserved":True,
        "qualified_confirmatory_independent_replication":False,
        "limitations":"MassBank source and exact answer identities were already evaluated "
            "before this fixed graph-only test; some underlying acquisitions may "
            "overlap other public libraries. Original isomer pools contain measured "
            "candidate spectra by construction, although the graph-only scorer "
            "does not consult them. No unseen candidate database or official "
            "CASMI 2026 Class2/Class3/MRR@25 experiment.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n")
    with (out/"per_query_ranking.jsonl").open("w") as fp:
        for row in sorted(outrows,key=lambda x:x["query_path"]):
            fp.write(json.dumps(row,sort_keys=True)+"\n")
    lines=[
        "# MassBank independent archive graph-only molecular-identification transfer",
        "",
        "**Historical source already evaluated, but graph model was frozen before this MassBank graph-only scoring.**",
        "| Source cohort | n | Formula blocks | Graph-only MRR | Exact random-label MRR | ΔMRR | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label,x in (
        ("All frozen isomer queries",total),
        ("No prior evaluated MSnLib structures",clean_stat)
    ):
        ci=x["formula_block_bootstrap_95"]
        lines.append(f"| {label} | {x['n_queries']} | {x['n_formulas']} | "
                     f"{x['mean_graph_mrr']:.6f} | {x['mean_exact_random_label_null_mrr']:.6f} | "
                     f"{ci['delta']:+.6f} | [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}] |")
    lines+=["",
            f"Full-cohort exploratory improvement gate: **{total['exploratory_positive_delta_and_ci']}**.",
            f"Graph-only top1 {total['mean_graph_top1_fractional']:.6f} vs exact uniform-label null {total['mean_uniform_label_top1_fractional']:.6f}.",
            "",
            "Unmodified native chemistry candidate statuses:"]
    for k,v in sorted(chemistry.items()):
        lines.append(f"- {k}: {v}")
    lines+=["", "Native ion mode (descriptive):"]
    for mode,x in by_mode.items():
        if x is not None:
            lines.append(f"- {mode}: n={x['n_queries']}, graph MRR {x['mean_graph_mrr']:.6f}, null {x['mean_exact_random_label_null_mrr']:.6f}")
    lines+=["",outcome["limitations"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return outcome


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--efficacy",type=Path,required=True)
    p.add_argument("--prior-inventory",type=Path,required=True)
    p.add_argument("--locked",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    evaluate(args.source,args.efficacy,args.prior_inventory,args.locked,args.out)
