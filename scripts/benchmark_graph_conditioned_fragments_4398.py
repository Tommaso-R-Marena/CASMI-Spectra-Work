"""Untrained chemical-graph fragment plausibility versus frozen 4398 Hellinger.

Explicit RETROSPECTIVE test on already seen source. Native reference SMILES
only, all original formula/polarity/mass candidate pools preserved. A
chemistry-mismatched structure contributes score 0, NEVER exclusion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import zstandard

from audit_harmonized2026_crossprovider_gate import (
    SHA256,sha256file,meta_only,freeze)
from benchmark_harmonized2026_crossprovider import (
    read_rows,extract_selected_peaks,scores_ion_spectra,tie_aware_rank)
from qfd.graph_fragment_signatures import signatures,discriminative_graph_scores
from qfd.stats import (
    paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie)

MIX_H=.9
MIX_GRAPH=.1
ORIGINAL_RUN=39056094950


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_smiles(source:Path,needed:set[int],original_reference):
    """Only reference GNPS native structural annotations, NEVER query labels."""
    out={}
    with source.open("rb") as file:
        reader=zstandard.ZstdDecompressor().stream_reader(file)
        with io.TextIOWrapper(reader,encoding="utf-8",errors="replace") as f:
            scan=-1;active=False;meta={}
            for raw in f:
                line=raw.strip()
                if line.upper()=="BEGIN IONS":
                    scan+=1
                    active=scan in needed
                    meta={}
                elif line.upper()=="END IONS":
                    if active:
                        original=original_reference[scan]
                        if (meta.get("SOURCE_DATASET")!="GNPS" or
                            meta.get("INCHIKEY")!=original["inchikey"] or
                            meta.get("FORMULA")!=original["formula"]):
                            raise RuntimeError("native GNPS candidate identity/formula mismatch")
                        out[scan]=meta.get("SMILES","")
                    active=False
                elif active and "=" in line:
                    k,v=line.split("=",1)
                    k=k.strip().upper()
                    if k in ("SMILES","SOURCE_DATASET","INCHIKEY","FORMULA"):
                        if k in meta:raise RuntimeError("duplicate native candidate header")
                        meta[k]=v.strip()
    if set(out)!=needed:
        raise RuntimeError("missing originally selected reference structure annotation")
    return out


def infer(rows):
    lhs=np.asarray([r["graph_mrr"] for r in rows],float)
    rhs=np.asarray([r["hellinger_mrr"] for r in rows],float)
    blocks=[r["formula"] for r in rows]
    return {
        "mean_mrr_delta":float(np.mean(lhs-rhs)),
        "wins_losses_ties":paired_win_loss_tie(lhs,rhs,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            lhs,rhs,blocks,n_boot=10000,seed=20261017),
        "formula_block_signflip_2sided":paired_block_signflip_test(
            lhs,rhs,blocks,n_permutations=50000,seed=20261017)
    }


def evaluate(source:Path,v1_dir:Path,v2_dir:Path,prior_dir:Path,out:Path):
    v1=json.loads((v1_dir/"feasibility.json").read_text())
    v2=json.loads((v2_dir/"feasibility.json").read_text())
    prior=json.loads((prior_dir/"results.json").read_text())
    previous=read_rows(prior_dir/"query_metrics.jsonl")
    ref=read_rows(v1_dir/"reference_manifest.jsonl")
    query=read_rows(v1_dir/"hard_queries.jsonl")
    if (
        v1.get("gate_pass") is not False or v1.get("source_sha256")!=SHA256
        or v1.get("query_peak_locations_or_intensities_opened") is not False
        or v1.get("coverage",{}).get("hard_positive")!=4398
        or v1.get("coverage",{}).get("hard_negative",0)!=0
        or v2.get("status")!="HARMONIZED2026_POSITIVE_ONLY_NATIVE_COHORT_V2"
        or v2.get("gate_pass") is not True
        or v2.get("original_v1_two_polarity_scientific_pass") is not False
        or v2.get("source_query_spectra_opened") is not False
        or v2.get("original_v1_reference_sha256")!=digest(v1_dir/"reference_manifest.jsonl")
        or v2.get("original_v1_queries_sha256")!=digest(v1_dir/"hard_queries.jsonl")
        or prior.get("status")!="FIRST_LOOK_HARMONIZED2026_GNPS_TO_MSG_SAME_FORMULA_IDENTIFICATION"
        or prior.get("source_sha256")!=SHA256
        or prior.get("n_hard_queries")!=4398
        or prior.get("n_distinct_formulas")!=2254
        or prior.get("predeclared_joint_mrr_test_passed") is not False
        or len(query)!=4398 or len(previous)!=4398 or len(ref)!=119863
    ):
        raise RuntimeError("unaltered original science gates and historical identification null required")
    historical={(r["identity"],r["query_index"]):r for r in previous}
    if len(historical)!=4398:
        raise RuntimeError("original query truth list nonunique")
    sha256file(source)
    all_meta,raw=meta_only(source)
    refs_now,queries_now,coverage,both_polarities_passed=freeze(all_meta)
    if (both_polarities_passed or refs_now!=ref or queries_now!=query or
        coverage!=v1["coverage"] or raw!=v1["raw_header_audit"]):
        raise RuntimeError("original chemically full-key metadata gate not identical")
    ref_by_scan={r["index"]:r for r in ref}
    needed_query={r["index"] for r in query}
    needed_reference={s for r in query for s in r["selected_reference_indices"]}
    spectra=extract_selected_peaks(source,needed_query|needed_reference)
    smiles=source_smiles(source,needed_reference,ref_by_scan)
    # Resolve native GNPS SMILES for a FULL molecule from its deterministic
    # LOWEST source scan index, without opening query chemical annotations.
    representative={}
    for scan in sorted(needed_reference):
        native=ref_by_scan[scan]
        key=(native["inchikey"],native["formula"],native["mode"])
        if key not in representative:
            representative[key]=smiles[scan]
    rows=[]
    chemical_outcomes=Counter()
    sum_intensity_explained=0.
    for q in query:
        original=historical.get((q["inchikey"],q["index"]))
        if original is None:
            raise RuntimeError("one of original 4398 heldout queries absent")
        obs=spectra[q["index"]]
        hellinger_by_identity={}
        for i in q["selected_reference_indices"]:
            native=ref_by_scan[i]
            if (
                native["source"]!="GNPS"
                or native["formula"]!=q["formula"]
                or native["mode"]!="positive"
                or abs(native["precursor_mz"]-q["precursor_mz"])>
                  max(.02,20e-6*q["precursor_mz"])+1e-12
            ):
                raise RuntimeError("frozen reference chemistry or precursor gate inconsistent")
            label=native["inchikey"]
            value=scores_ion_spectra(obs,spectra[i])["hellinger"]
            hellinger_by_identity[label]=max(hellinger_by_identity.get(label,-1.),value)
        if (len(hellinger_by_identity)!=q["candidate_structures"] or
            q["inchikey"] not in hellinger_by_identity or
            len(hellinger_by_identity)<2):
            raise RuntimeError("full metadata-selected same-formula structure inventory changed")
        old_rank=tie_aware_rank(hellinger_by_identity,q["inchikey"])
        for field in ("mrr","top1_fractional","midrank","candidate_count"):
            if abs(old_rank[field]-original["metrics"]["hellinger"][field])>1e-12:
                raise RuntimeError("historical Hellinger ranking fails exact replay")
        candidate_features={}
        for identity in sorted(hellinger_by_identity):
            sm=representative[(identity,q["formula"],"positive")]
            masses,status,nbonds=signatures(sm,q["formula"])
            chemical_outcomes[status]+=1
            candidate_features[identity]=masses
        graph,diag=discriminative_graph_scores(obs,candidate_features)
        sum_intensity_explained+=diag["fraction_of_query_intensity_explained_by_any"]
        final={k:MIX_H*hellinger_by_identity[k]+MIX_GRAPH*graph[k]
               for k in hellinger_by_identity}
        new_rank=tie_aware_rank(final,q["inchikey"])
        rows.append({
            "identity":q["inchikey"],"formula":q["formula"],
            "query_index":q["index"],"candidate_count":len(hellinger_by_identity),
            "hellinger_mrr":old_rank["mrr"],
            "graph_mrr":new_rank["mrr"],
            "hellinger_top1":old_rank["top1_fractional"],
            "graph_top1":new_rank["top1_fractional"],
            "graph_true_structure_score":graph[q["inchikey"]],
            "fraction_of_query_intensity_any_graph_explained":
                diag["fraction_of_query_intensity_explained_by_any"],
        })
    if len(rows)!=4398 or len({r["formula"] for r in rows})!=2254:
        raise RuntimeError("original data denominator has changed")
    baseline_mrr=float(np.mean([r["hellinger_mrr"] for r in rows]))
    if abs(baseline_mrr-prior["mean_mrr"]["hellinger"])>1e-12:
        raise RuntimeError("historical whole-cohort Hellinger MRR not reproduced")
    test=infer(rows)
    ci=test["formula_block_bootstrap_95"]
    statistical=(test["mean_mrr_delta"]>0 and ci["ci_low"]>0 and
                 test["formula_block_signflip_2sided"]["p_value_two_sided"]<.05)
    result={
        "status":"POSTHOC_GNPS_MSG_CHEMICAL_GRAPH_FRAGMENT_RANKING_V1",
        "scientific_status":"RETROSPECTIVE: original 4398-query spectral outcomes already published",
        "source_sha256":SHA256,"original_firstlook_run":ORIGINAL_RUN,
        "original_two_polarity_scientific_gate_failed":True,
        "original_positive_only_pre_score_gate_passed":True,
        "n_original_queries":len(rows),
        "n_unique_structure_truths":len({r["identity"] for r in rows}),
        "n_native_formula_blocks":len({r["formula"] for r in rows}),
        "reference_structure_source":"GNPS SMILES checked against GNPS FORMULA",
        "query_structure_used_as_feature":False,
        "native_chemical_fallback":"zero graph term, no candidate exclusion",
        "mix_hellinger":MIX_H,"mix_graph":MIX_GRAPH,
        "original_historical_hellinger_mrr":baseline_mrr,
        "graph_conditioned_mrr":float(np.mean([r["graph_mrr"] for r in rows])),
        "original_hellinger_fractional_top1":float(np.mean([r["hellinger_top1"] for r in rows])),
        "graph_conditioned_fractional_top1":float(np.mean([r["graph_top1"] for r in rows])),
        "total_candidate_graph_feature_statuses":dict(chemical_outcomes),
        "mean_query_intensity_explained_by_any_structure_fragment":
            sum_intensity_explained/len(rows),
        "retrospective_formula_block_inference":test,
        "retrospective_fixed_primary_passed":bool(statistical),
        "limitation":"Heuristic one/two nonring bond cuts plus parent-state H corrections. Not mechanistically complete; uncalibrated ion formulas, no independent new-lab validation or negative-ion evidence.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (out/"query_metrics.jsonl").open("w") as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
    lines=[
      "# Fixed chemistry-conditioned graph fragment score: original 4398-query replay",
      "",
      "**RETROSPECTIVE** already-seen GNPS/MassSpecGym positive-ion queries.",
      f"Original {len(rows)} Hellinger MRR and fractional top1 values reproduced exactly.",
      "",
      "| Model | MRR | Fractional top1 |",
      "|---|---:|---:|",
      f"| Original Hellinger | {result['original_historical_hellinger_mrr']:.6f} | {result['original_hellinger_fractional_top1']:.6f} |",
      f"| Fixed 0.90 Hellinger + 0.10 graph | {result['graph_conditioned_mrr']:.6f} | {result['graph_conditioned_fractional_top1']:.6f} |",
      "",
      f"MRR delta {test['mean_mrr_delta']:+.6f}, formula-cluster 95% CI "
      f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], formula-block two-sided "
      f"signflip p={test['formula_block_signflip_2sided']['p_value_two_sided']:.6f}.",
      f"Retrospective fixed primary result: **{statistical}**.",
      "",
      "Native GNPS chemistry validation status counts, counting one candidate occurrence per query:",
    ]
    for k,v in sorted(chemical_outcomes.items()):
        lines.append(f"- {k}: {v}")
    lines+=["",f"Mean query intensity explained by any fixed graph signature: "
            f"{result['mean_query_intensity_explained_by_any_structure_fragment']:.6f}",
            "",result["limitation"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("source","v1_dir","v2_dir","prior_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    evaluate(a.source,a.v1_dir,a.v2_dir,a.prior_dir,a.out)

if __name__=="__main__":main()
