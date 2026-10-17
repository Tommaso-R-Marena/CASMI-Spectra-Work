"""Reference-only formula-conditional fragment rarity on prior real GNPS/MSG cohort.

The earlier source's outcomes are already public. This is exploratory rather
than blind replication. Formula-group inference guards dependent isomer pools.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np

from audit_harmonized2026_crossprovider_gate import SHA256
from benchmark_harmonized2026_crossprovider import (
    evaluate as replay_original, read_rows, extract_selected_peaks,
    scores_ion_spectra, tie_aware_rank)
from qfd.stats import (
    paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie)

TOL = .01
BLEND = .5


def molecular_documents(indices, references, spectra):
    by_identity = defaultdict(list)
    for i in indices:
        if references[i]["source"] != "GNPS":
            raise RuntimeError("not frozen GNPS reference")
        by_identity[references[i]["inchikey"]].append(spectra[i][0])
    if len(by_identity) < 2:
        raise RuntimeError("not a molecular hard-negative query")
    return [np.unique(np.concatenate(v)) for _, v in sorted(by_identity.items())]


def ref_idf(masses, documents, tol=TOL):
    """Only GNPS reference presence; each molecule counted once despite scans."""
    x = np.asarray(masses, float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("valid masses required")
    if not documents or not 0 < tol < .5:
        raise ValueError("valid fixed tolerance and distinct documents required")
    df = np.zeros(len(x), dtype=int)
    for doc in documents:
        d = np.asarray(doc, float)
        if d.ndim != 1 or not len(d) or not np.all(np.diff(d) >= 0):
            raise ValueError("sorted nonempty reference masses required")
        j = np.searchsorted(d, x)
        left = np.clip(j-1, 0, len(d)-1)
        right = np.clip(j, 0, len(d)-1)
        df += ((np.abs(d[left]-x) <= tol) | (np.abs(d[right]-x) <= tol))
    return 1. + np.log((len(documents)+1.)/(df+1.))


def weighted_hellinger(query, ref, documents):
    qm, qi = query
    rm, ri = ref
    qv = qi * ref_idf(qm, documents)
    rv = ri * ref_idf(rm, documents)
    if (np.any(qv < 0) or np.any(rv < 0)
        or not np.all(np.isfinite(qv)) or not np.all(np.isfinite(rv))
        or qv.sum() <= 0 or rv.sum() <= 0):
        raise ValueError("invalid weighted intensities")
    qp = qv/qv.sum()
    rp = rv/rv.sum()
    edges = []
    for i,m in enumerate(qm):
        lo = int(np.searchsorted(rm, m-TOL, side="left"))
        hi = int(np.searchsorted(rm, m+TOL, side="right"))
        edges += [(abs(float(rm[j]-m)),j,i) for j in range(lo,hi)
                  if abs(float(rm[j]-m)) <= TOL]
    edges.sort()
    used_q, used_r = set(), set()
    affinity = 0.
    for _,j,i in edges:
        if i not in used_q and j not in used_r:
            affinity += math.sqrt(float(qp[i]*rp[j]))
            used_q.add(i)
            used_r.add(j)
    if not -1e-9 <= affinity <= 1+1e-9:
        raise RuntimeError("affinity not in [0,1]")
    return max(0.,min(1.,affinity))


def grouped_comparison(rows, left, right="hellinger", seed=20261017):
    a = np.asarray([r["metrics"][left]["mrr"] for r in rows])
    b = np.asarray([r["metrics"][right]["mrr"] for r in rows])
    blocks = [r["formula"] for r in rows]
    return {
        "delta_mrr": float((a-b).mean()),
        "formula_bootstrap_95": paired_block_bootstrap_delta(
            a,b,blocks,n_boot=5000,seed=seed),
        "formula_signflip": paired_block_signflip_test(
            a,b,blocks,n_permutations=50000,seed=seed),
        "wins_losses_ties": paired_win_loss_tie(a,b),
    }


def evaluate(source:Path, gate:Path, v2:Path, out:Path):
    # Rerun the complete original V1 failed scientific gate, positive-only
    # V2 freeze and SHA verified source/eligibility replay, BEFORE peaks.
    prior = replay_original(source,gate,v2,out/"original_baseline_replay")
    if (prior["source_sha256"] != SHA256
        or prior["n_hard_queries"] != 4398
        or prior["n_distinct_formulas"] != 2254
        or prior["exact_identical_query_reference_spectra"] != 0
        or prior["predeclared_joint_mrr_test_passed"] is not False
        or abs(prior["mean_mrr"]["hellinger"]-.811962)>2e-6
        or abs(prior["mean_mrr"]["cosine"]-.808541)>2e-6):
        raise RuntimeError("frozen negative original control drifted")
    queries = read_rows(gate/"hard_queries.jsonl")
    references = {r["index"]:r for r in
                  read_rows(gate/"reference_manifest.jsonl")}
    previous = read_rows(out/"original_baseline_replay"/"query_metrics.jsonl")
    if len(queries)!=len(previous):
        raise RuntimeError("missing historical query rank")
    selected = {q["index"] for q in queries}
    selected.update(idx for q in queries for idx in q["selected_reference_indices"])
    spectra = extract_selected_peaks(source,selected)
    outcome = []
    for q, old in zip(queries,previous):
        if (q["index"] != old["query_index"] or q["inchikey"] != old["identity"]
            or q["formula"] != old["formula"]):
            raise RuntimeError("original target rank drifted")
        pool = q["selected_reference_indices"]
        docs = molecular_documents(pool,references,spectra)
        if len(docs)!=q["candidate_structures"]:
            raise RuntimeError("molecular reference pool changed")
        rarity,blended = {},{}
        for idx in pool:
            label=references[idx]["inchikey"]
            new=weighted_hellinger(spectra[q["index"]],spectra[idx],docs)
            hell=scores_ion_spectra(spectra[q["index"]],spectra[idx])["hellinger"]
            score=(1.-BLEND)*hell+BLEND*new
            rarity[label]=max(rarity.get(label,-1.),new)
            blended[label]=max(blended.get(label,-1.),score)
        outcome.append({
            "identity":q["inchikey"],"formula":q["formula"],
            "query_index":q["index"],"candidate_count":len(docs),
            "metrics":{
                **old["metrics"],
                "rarity":tie_aware_rank(rarity,q["inchikey"]),
                "rarity_blend":tie_aware_rank(blended,q["inchikey"]),
            },
        })
    if len(outcome)!=4398 or len({r["identity"] for r in outcome})!=4398:
        raise RuntimeError("duplicate or missing true full InChIKey")
    methods = ("cosine","hellinger","fragment_overlap",
               "support_composite","rarity","rarity_blend")
    primary=grouped_comparison(outcome,"rarity_blend")
    ci=primary["formula_bootstrap_95"]
    p=primary["formula_signflip"]["p_value_two_sided"]
    success=bool(primary["delta_mrr"]>0 and ci["ci_low"]>0 and p<.05)
    summary={
        "status":"EXPLORATORY_REFERENCE_ONLY_FORMULA_RARITY_AFTER_BASELINE_V1",
        "source_sha256":SHA256,
        "source_doi":"10.5281/zenodo.20042904",
        "prior_control_actions_run":39056094950,
        "original_composite_hypothesis_failed":True,
        "historical_source_and_targets_reused":True,
        "query_peaks_not_used_for_reference_document_frequency":True,
        "fixed_molecular_document_idf":"1+ln((N+1)/(df+1)), m/z tolerance 0.01Da",
        "fixed_mixture_blend":BLEND,
        "n_queries":len(outcome),
        "n_formula_blocks":len({r["formula"] for r in outcome}),
        "mean_mrr":{m:float(np.mean([r["metrics"][m]["mrr"] for r in outcome]))
                    for m in methods},
        "fractional_top1":{m:float(np.mean(
            [r["metrics"][m]["top1_fractional"] for r in outcome])) for m in methods},
        "predeclared_exploratory_primary":primary,
        "descriptive_rarity_vs_hellinger":grouped_comparison(
            outcome,"rarity",seed=20261018),
        "descriptive_blend_vs_cosine":grouped_comparison(
            outcome,"rarity_blend","cosine",seed=20261019),
        "exploratory_criterion_passed":success,
        "independent_replication_or_breakthrough":False,
        "limitations":"Reused prior source and outcomes, no independent acquisition proof, "
                      "computationally harmonized identities not curator-verified, "
                      "positive-ion only, no CASMI 2026 blind test or EAD physics.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    with (out/"query_ranks.jsonl").open("w") as f:
        for row in outcome:f.write(json.dumps(row,sort_keys=True)+"\n")
    report=["# GNPS→MassSpecGym formula reference rarity: exploratory",
            "",f"{len(outcome)} hard queries; {summary['n_formula_blocks']} formula blocks.",
            "Previously reported Hellinger and support-composite rankings are immutable.",
            "", "| Method | MRR | Fractional top1 |",
            "|---|---:|---:|"]
    for method in methods:
        report.append(f"| {method} | {summary['mean_mrr'][method]:.6f} | "
                      f"{summary['fractional_top1'][method]:.6f} |")
    report += ["", f"Fixed rarity blend minus Hellinger MRR: {primary['delta_mrr']:+.6f}, "
               f"formula bootstrap [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}], "
               f"two-sided formula signflip p={p:.6f}, "
               f"wins/losses/ties {primary['wins_losses_ties']}.",
               f"**Exploratory success criterion: {success}**.",
               "",summary["limitations"],""]
    (out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zst",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--positive-v2-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.zst,a.gate_dir,a.positive_v2_dir,a.out)
