"""Exact graph-only structural identification signal without reference spectra.

An intentionally LIMITED proxy for spectrum-unknown candidate discrimination.
The candidate set is still conditioned on GNPS having reference spectra.
GNPS peak arrays are never parsed by this experiment; only GNPS headers
with native molecular SMILES/formulas are used for model features.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from audit_harmonized2026_crossprovider_gate import (
    SHA256, MD5, sha256file, meta_only, freeze,
)
from benchmark_harmonized2026_crossprovider import (
    read_rows, extract_selected_peaks, tie_aware_rank,
)
from benchmark_graph_conditioned_fragments_4398 import source_smiles
from qfd.graph_fragment_signatures import (
    signatures, discriminative_graph_scores,
)
from qfd.stats import paired_block_bootstrap_delta


def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_random_label_expected_rank(scores):
    """Exact equal-probability structural label permutation expectation.

    The K distinct candidate scores are FIXED. Under null, any candidate
    identity is equally likely to be the answer. We enumerate all K possible
    answer slots and their original deterministic tie-aware reciprocal ranks.
    """
    if len(scores)<2:
        raise ValueError("candidate pool requires >=2 full structures")
    if len(set(scores))!=len(scores):
        raise ValueError("structural candidate labels cannot repeat")
    arr=np.asarray(list(scores.values()),float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite graph scores")
    values=[]
    for score in arr:
        higher=np.count_nonzero(arr>score+1e-12)
        ties=np.count_nonzero(np.abs(arr-score)<=1e-12)
        if ties<1:
            raise RuntimeError("invalid tie count")
        values.append(1. / (1+float(higher)+(int(ties)-1)/2))
    return float(np.mean(values))


def formula_cluster_delta(rows):
    graph=np.asarray([x["graph_only_mrr"] for x in rows])
    neutral=np.asarray([x["uniform_label_null_mrr"] for x in rows])
    blocks=[x["formula"] for x in rows]
    return paired_block_bootstrap_delta(
        graph,neutral,blocks,n_boot=10000,seed=20261017)


def evaluate(source:Path,v1_dir:Path,v2_dir:Path,original_graph:Path,out:Path):
    v1=json.loads((v1_dir/"feasibility.json").read_text())
    v2=json.loads((v2_dir/"feasibility.json").read_text())
    originally=json.loads((original_graph/"result.json").read_text())
    historical=read_rows(original_graph/"query_metrics.jsonl")
    refs=read_rows(v1_dir/"reference_manifest.jsonl")
    queries=read_rows(v1_dir/"hard_queries.jsonl")
    if (
        v1.get("status")!="HARMONIZED2026_CROSS_PROVIDER_NATIVE_HEADER_GATE_V1"
        or v1.get("gate_pass") is not False
        or v1.get("source_sha256")!=SHA256
        or v1.get("source_md5")!=MD5
        or v1.get("query_peak_locations_or_intensities_opened") is not False
        or v1.get("gnps_reference_peaks_opened") is not False
        or v2.get("status")!="HARMONIZED2026_POSITIVE_ONLY_NATIVE_COHORT_V2"
        or v2.get("gate_pass") is not True
        or v2.get("original_v1_two_polarity_scientific_pass") is not False
        or v2.get("source_query_spectra_opened") is not False
        or v2.get("original_v1_feasibility_sha256")!=sha(v1_dir/"feasibility.json")
        or v2.get("original_v1_reference_sha256")!=sha(v1_dir/"reference_manifest.jsonl")
        or v2.get("original_v1_queries_sha256")!=sha(v1_dir/"hard_queries.jsonl")
        or originally.get("status")!="POSTHOC_GNPS_MSG_CHEMICAL_GRAPH_FRAGMENT_RANKING_V1"
        or originally.get("source_sha256")!=SHA256
        or originally.get("retrospective_fixed_primary_passed") is not True
        or originally.get("n_original_queries")!=4398
        or len(queries)!=4398
        or len(historical)!=4398
        or len(refs)!=119863
    ):
        raise RuntimeError("prescore source selection and original graph results required")
    original_rows={(r["identity"],r["query_index"]):r for r in historical}
    if len(original_rows)!=4398:
        raise RuntimeError("original candidate output duplicated")
    sha256file(source)
    native,header_summary=meta_only(source)
    refs_now,query_now,coverage,old_two_polarity_passed=freeze(native)
    if (old_two_polarity_passed or refs_now!=refs or query_now!=queries
        or header_summary!=v1["raw_header_audit"] or coverage!=v1["coverage"]):
        raise RuntimeError("source native labels or candidates changed")
    reference_by_idx={x["index"]:x for x in refs}
    query_ids={q["index"] for q in queries}
    candidate_indices={i for q in queries for i in q["selected_reference_indices"]}
    # THIS IS THE KEY EXPERIMENTAL ABLATION: only target/query spectral peaks
    # are opened; selected GNPS candidate reference spectra are NEVER passed to
    # extract_selected_peaks or a spectral similarity algorithm.
    measured_queries=extract_selected_peaks(source,query_ids)
    candidate_smiles=source_smiles(source,candidate_indices,reference_by_idx)
    representatives={}
    for scan in sorted(candidate_indices):
        r=reference_by_idx[scan]
        key=(r["inchikey"],r["formula"],r["mode"])
        if key not in representatives:
            representatives[key]=candidate_smiles[scan]
    total_status=Counter()
    outputs=[]
    sum_explained=0.
    for q in queries:
        prev=original_rows.get((q["inchikey"],q["index"]))
        if prev is None or q["mode"]!="positive" or prev["formula"]!=q["formula"]:
            raise RuntimeError("original forward eligible query missing or unvalidated polarity")
        fixed_candidates={}
        for i in q["selected_reference_indices"]:
            r=reference_by_idx[i]
            if (r["source"]!="GNPS" or r["formula"]!=q["formula"]
                or r["mode"]!=q["mode"]
                or abs(r["precursor_mz"]-q["precursor_mz"])>
                  max(.02,20e-6*q["precursor_mz"])+1e-12):
                raise RuntimeError("native metadata identity/precursor candidate drift")
            fixed_candidates[r["inchikey"]]=(r["inchikey"],q["formula"],q["mode"])
        if len(fixed_candidates)!=q["candidate_structures"] or q["inchikey"] not in fixed_candidates:
            raise RuntimeError("candidate size or true source structure missing")
        fragments={}
        for label,native_key in sorted(fixed_candidates.items()):
            masses,status,cut_count=signatures(
                representatives[native_key],q["formula"])
            total_status[status]+=1
            fragments[label]=masses
        prediction,info=discriminative_graph_scores(
            measured_queries[q["index"]],fragments)
        if abs(prediction[q["inchikey"]]-prev["graph_true_structure_score"])>1e-12:
            raise RuntimeError("original candidate graph model did not replay")
        true_score=tie_aware_rank(prediction,q["inchikey"])
        expected=exact_random_label_expected_rank(prediction)
        k=len(prediction)
        sum_explained+=info["fraction_of_query_intensity_explained_by_any"]
        outputs.append({
            "query_index":q["index"],"identity":q["inchikey"],"formula":q["formula"],
            "candidate_count":k,"graph_only_mrr":true_score["mrr"],
            "graph_only_fractional_top1":true_score["top1_fractional"],
            "uniform_label_null_mrr":expected,
            "uniform_label_null_fractional_top1":1./k,
            "native_reference_candidate_spectra_opened":False,
            "true_graph_score":prediction[q["inchikey"]],
            "any_graph_explained_query_intensity":
                info["fraction_of_query_intensity_explained_by_any"],
        })
    if len(outputs)!=4398 or len({r["formula"] for r in outputs})!=2254:
        raise RuntimeError("scored incomplete originally eligible data")
    ci=formula_cluster_delta(outputs)
    mean_graph=float(np.mean([r["graph_only_mrr"] for r in outputs]))
    mean_null=float(np.mean([r["uniform_label_null_mrr"] for r in outputs]))
    delta=mean_graph-mean_null
    if abs(delta-ci["delta"])>1e-12:
        raise RuntimeError("formula block comparison denominator drift")
    passed=bool(delta>0 and ci["ci_low"]>0)
    summary={
        "status":"GRAPH_ONLY_REFERENCE_SPECTRUM_ABLATION_EXPLORATORY_V1",
        "source_sha256":SHA256,
        "original_graph_run":39058512753,
        "n_queries":len(outputs),
        "n_formula_blocks":len({r["formula"] for r in outputs}),
        "only_measured_query_peaks_opened":True,
        "experimental_candidate_reference_peaks_opened":False,
        "GNPS_reference_SMILES_headers_used":True,
        "MassSpecGym_query_SMILES_read":False,
        "candidate_pool_requires_GNPS_reference_spectra_at_metadata_stage":True,
        "not_genuine_CASMI_spectrum_novel_class2_evaluation":True,
        "model":"original fixed candidate graph cuts and rarity without Hellinger",
        "mean_graph_only_mrr":mean_graph,
        "mean_uniform_label_exact_null_mrr":mean_null,
        "mean_mrr_delta_vs_exact_null":delta,
        "mean_graph_only_fractional_top1":float(np.mean([
            r["graph_only_fractional_top1"] for r in outputs])),
        "mean_uniform_label_fractional_top1":float(np.mean([
            r["uniform_label_null_fractional_top1"] for r in outputs])),
        "formula_cluster_bootstrap_95":ci,
        "retrospective_exploratory_criterion_met":passed,
        "n_candidate_graph_status_occurrences":dict(total_status),
        "mean_fraction_query_intensity_explained_by_any_graph":
            sum_explained/len(outputs),
        "limitations":"Previously examined archive, candidate enumeration conditioned on "
            "having existing GNPS reference spectra, original graph hypothesis posthoc, "
            "all positive-ion, full-key truth rather than official tautomer-canonical "
            "InChIKey14, not independent acquisition or CASMI MRR@25.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (out/"per_query_no_reference_ranks.jsonl").open("w") as f:
        for r in outputs:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    report=[
        "# Graph-only chemical identification: GNPS REFERENCE SPECTRA ABLATED",
        "",
        "Existing GNPS candidate spectra were never opened as peak data; native candidate SMILES came only from GNPS headers. This is NOT a true spectrum-novel structural candidate pool.",
        f"All original {len(outputs)} measured MassSpecGym queries, {summary['n_formula_blocks']} formula groups.",
        "",
        "| Model | Mean reciprocal rank | Fractional top1 |",
        "|---|---:|---:|",
        f"| Untrained graph fragments only | {mean_graph:.6f} | {summary['mean_graph_only_fractional_top1']:.6f} |",
        f"| Exact random-identity assignment on same candidate scores | {mean_null:.6f} | {summary['mean_uniform_label_fractional_top1']:.6f} |",
        "",
        f"Descriptive ΔMRR vs exact conditional permutation null: {delta:+.6f}, "
        f"chemical formula-block bootstrap 95% CI "
        f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}].",
        f"Exploratory positive criterion: {passed}.",
        "",
        "Native graph success/attrition:",
    ]
    for status,n in sorted(total_status.items()):
        report.append(f"- {status}: {n}")
    report+=["",summary["limitations"],""]
    (out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--v1-dir",type=Path,required=True)
    p.add_argument("--v2-dir",type=Path,required=True)
    p.add_argument("--original-graph",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.source,a.v1_dir,a.v2_dir,a.original_graph,a.out)
