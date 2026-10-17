"""Locked reverse direction of graph-conditioned MS/MS identity retrieval.

Metadata only GNPS-query/MassSpecGym-reference cohort must be frozen and
scientifically PASSED in its separate, earlier GitHub Actions run. There is
no reweighting, retraining, positive-result rescue, or label-conditioned
candidate pruning. Chemical-graph model is byte-for-byte imported unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import zstandard

from audit_harmonized2026_crossprovider_gate import (
    SHA256, MD5, sha256file, meta_only,
)
from audit_reverse_gnps_msg_graph_gate import freeze_reverse, STATUS
from benchmark_harmonized2026_crossprovider import (
    read_rows, extract_selected_peaks, scores_ion_spectra, tie_aware_rank,
)
from qfd.graph_fragment_signatures import (
    signatures, discriminative_graph_scores,
)
from qfd.stats import (
    paired_block_bootstrap_delta, paired_block_signflip_test,
    paired_win_loss_tie,
)

MIX_HELLINGER=.90
MIX_GRAPH=.10
BASE_FORWARD_RUN=39058512753
SUPPORTED_ION_SHIFTS_ORIGINAL_POSITIVE_ONLY=True


def query_reference_signal_equal(a,b):
    if len(a[0])!=len(b[0]):return False
    if not np.array_equal(a[0],b[0]):return False
    return bool(np.array_equal(a[1]/a[1].sum(),b[1]/b[1].sum()))


def reference_smiles(source:Path, wanted:set[int], reference_by_scan):
    """Read native MassSpecGym reference SMILES only; not GNPS query SMILES."""
    values={}
    with source.open("rb") as file:
        reader=zstandard.ZstdDecompressor().stream_reader(file)
        with io.TextIOWrapper(reader,encoding="utf-8",errors="replace") as f:
            current=-1
            active=False
            header={}
            for raw in f:
                line=raw.strip()
                if line.upper()=="BEGIN IONS":
                    current+=1
                    active=current in wanted
                    header={}
                elif line.upper()=="END IONS":
                    if active:
                        native=reference_by_scan[current]
                        tok=re.sub(r"[^A-Z0-9]","",
                                   header.get("SOURCE_DATASET","").upper())
                        if (tok!="MASSSPECGYM"
                            or header.get("INCHIKEY","").upper()!=native["inchikey"]
                            or header.get("FORMULA","")!=native["formula"]):
                            raise RuntimeError("MassSpecGym selected candidate identity mismatch")
                        values[current]=header.get("SMILES","")
                    active=False
                elif active and "=" in line:
                    key,value=line.split("=",1)
                    key=key.strip().upper()
                    if key in ("SMILES","INCHIKEY","FORMULA","SOURCE_DATASET"):
                        if key in header:
                            raise RuntimeError("duplicate required native reference header")
                        header[key]=value.strip()
    if set(values)!=wanted:
        raise RuntimeError("source missing selected reference SMILES metadata")
    return values


def manifest_digest(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_inference(rows):
    a=np.asarray([r["graph_mrr"] for r in rows],float)
    b=np.asarray([r["hellinger_mrr"] for r in rows],float)
    labels=[r["formula"] for r in rows]
    return {
        "mean_mrr_delta":float(np.mean(a-b)),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,labels,n_boot=10000,seed=20261017),
        "formula_block_signflip":paired_block_signflip_test(
            a,b,labels,n_permutations=50000,seed=20261017),
        "wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
    }


def evaluate(source:Path,locked:Path,out:Path):
    gate=json.loads((locked/"feasibility.json").read_text())
    ref=read_rows(locked/"reference_manifest.jsonl")
    queries=read_rows(locked/"hard_queries.jsonl")
    if (
        gate.get("status")!=STATUS
        or gate.get("gate_pass") is not True
        or gate.get("source_sha256")!=SHA256
        or gate.get("source_md5")!=MD5
        or gate.get("reference_source")!="MassSpecGym"
        or gate.get("query_source")!="GNPS"
        or gate.get("query_peak_positions_or_intensities_read") is not False
        or gate.get("reference_peak_positions_or_intensities_read") is not False
        or gate.get("query_smiles_read") is not False
        or len(queries)!=gate.get("coverage",{}).get("hard_queries",0)
        or len(queries)<100
        or len({q["inchikey"] for q in queries})<50
        or len({q["formula"] for q in queries})<20
    ):
        raise RuntimeError("reverse provider prescore scientific gate not passed; NO PEAKS")
    sha256file(source)
    all_header_rows,header_audit=meta_only(source)
    now_refs,now_queries,coverage,passed=freeze_reverse(all_header_rows)
    if (
        not passed or ref!=now_refs or queries!=now_queries
        or coverage!=gate["coverage"]
        or header_audit!=gate["native_header_audit"]
    ):
        raise RuntimeError("prescore selected molecules / candidate pool drifted")
    ref_by_idx={r["index"]:r for r in ref}
    wanted_ref={s for q in queries for s in q["selected_reference_indices"]}
    wanted_query={q["index"] for q in queries}
    spectra=extract_selected_peaks(source,wanted_ref|wanted_query)
    smiles=reference_smiles(source,wanted_ref,ref_by_idx)
    representative={}
    for i in sorted(wanted_ref):
        row=ref_by_idx[i]
        label=(row["inchikey"],row["formula"],row["mode"])
        if label not in representative:
            representative[label]=smiles[i]

    score_rows=[]
    graph_status=Counter()
    matched_fraction_total=0.
    same_signal_count=0
    for q in queries:
        query_spectrum=spectra[q["index"]]
        hel={}
        same_signal=False
        for i in q["selected_reference_indices"]:
            native=ref_by_idx[i]
            if (native["source"]!="MASSSPECGYM"
                or native["mode"]!=q["mode"]
                or native["formula"]!=q["formula"]
                or abs(native["precursor_mz"]-q["precursor_mz"])>
                     max(.02,20e-6*q["precursor_mz"])+1e-12):
                raise RuntimeError("candidate violates original metadata freeze")
            label=native["inchikey"]
            val=scores_ion_spectra(query_spectrum,spectra[i])["hellinger"]
            hel[label]=max(hel.get(label,-1.),val)
            same_signal|=query_reference_signal_equal(query_spectrum,spectra[i])
        if len(hel)!=q["candidate_structures"] or q["inchikey"] not in hel:
            raise RuntimeError("candidate pool changed after native metadata lockbox")
        same_signal_count+=int(same_signal)
        candidates={}
        for label in sorted(hel):
            sm=representative[(label,q["formula"],q["mode"])]
            possible,status,num_cuts=signatures(sm,q["formula"])
            graph_status[status]+=1
            candidates[label]=possible
        plausibility,diagnostics=discriminative_graph_scores(
            query_spectrum,candidates)
        matched_fraction_total+=diagnostics[
            "fraction_of_query_intensity_explained_by_any"]
        combined={label:MIX_HELLINGER*hel[label]+MIX_GRAPH*plausibility[label]
                  for label in hel}
        original=tie_aware_rank(hel,q["inchikey"])
        transfer=tie_aware_rank(combined,q["inchikey"])
        score_rows.append({
            "query_index":q["index"],"identity":q["inchikey"],
            "formula":q["formula"],"mode":q["mode"],
            "candidate_count":len(hel),
            "reference_exact_identical_signal":same_signal,
            "hellinger_mrr":original["mrr"],
            "hellinger_top1":original["top1_fractional"],
            "graph_mrr":transfer["mrr"],
            "graph_top1":transfer["top1_fractional"],
            "true_graph_score":plausibility[q["inchikey"]],
            "any_candidate_fragment_coverage":
                diagnostics["fraction_of_query_intensity_explained_by_any"],
        })
    if len(score_rows)!=len(queries):
        raise RuntimeError("scoring did not preserve EVERY frozen query")
    test=block_inference(score_rows)
    ci=test["formula_block_bootstrap_95"]
    p=test["formula_block_signflip"]["p_value_two_sided"]
    passed_science=bool(
        test["mean_mrr_delta"]>0 and ci["ci_low"]>0 and p<.05
        and same_signal_count==0
    )
    by_mode={}
    for mode in ("positive","negative"):
        subgroup=[r for r in score_rows if r["mode"]==mode]
        by_mode[mode]={
            "n_queries":len(subgroup),
            "mrr_hellinger":float(np.mean([r["hellinger_mrr"] for r in subgroup]))
                   if subgroup else None,
            "mrr_graph":float(np.mean([r["graph_mrr"] for r in subgroup]))
                   if subgroup else None,
        }
    report={
        "status":"REVERSE_PROVIDER_GNPS_QUERY_MSG_LIBRARY_GRAPH_TRANSFER_SCORED_V1",
        "source_sha256":SHA256,
        "prescore_gate_artifact_feasibility_sha256":
            manifest_digest(locked/"feasibility.json"),
        "prescore_reference_manifest_sha256":
            manifest_digest(locked/"reference_manifest.jsonl"),
        "prescore_query_manifest_sha256":
            manifest_digest(locked/"hard_queries.jsonl"),
        "original_forward_graph_model_workflow":BASE_FORWARD_RUN,
        "graph_model_code_unchanged_from_forward_v1":True,
        "reference_provider":"MassSpecGym",
        "query_provider":"GNPS",
        "n_queries":len(score_rows),
        "n_structure_labels":len({r["identity"] for r in score_rows}),
        "n_formula_blocks":len({r["formula"] for r in score_rows}),
        "same_exact_signal_query_reference_count":same_signal_count,
        "mrr_hellinger":float(np.mean([r["hellinger_mrr"] for r in score_rows])),
        "mrr_graph":float(np.mean([r["graph_mrr"] for r in score_rows])),
        "fractional_top1_hellinger":float(np.mean(
            [r["hellinger_top1"] for r in score_rows])),
        "fractional_top1_graph":float(np.mean(
            [r["graph_top1"] for r in score_rows])),
        "mean_query_intensity_explained_by_any_graph":
            matched_fraction_total/len(score_rows),
        "graph_chemistry_status_occurrences":dict(graph_status),
        "by_mode":by_mode,
        "fixed_score_weights":{"hellinger":MIX_HELLINGER,"graph":MIX_GRAPH},
        "primary_formula_block_comparison":test,
        "qualified_same_archive_directional_transfer":passed_science,
        "fully_independent_laboratory_replication":False,
        "models_negative_mode_without_negative_specific_ion_chemistry":True,
        "limitations":"Same source as forward experiment, overlapping original chemical "
            "identities and possible correlated physical acquisitions; positive-mode "
            "graph ion shifts used even on negative mode if eligible, without negative "
            "ion calibration. Does not simulate official CASMI de novo novelty classes.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (out/"query_metrics.jsonl").open("w") as stream:
        for row in score_rows:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    lines=["# Frozen inverse-provider graph-conditioned molecular ranking",
           "",
           "GNPS measured queries / MassSpecGym reference candidates; **same combined archive**, NOT independent acquisition.",
           f"All {len(score_rows)} native-source hard queries and {report['n_formula_blocks']} formula groups retained.",
           "",
           "| Method | MRR | Fractional top1 |",
           "|---|---:|---:|",
           f"| Hellinger | {report['mrr_hellinger']:.6f} | {report['fractional_top1_hellinger']:.6f} |",
           f"| Fixed 90% Hellinger + 10% graph | {report['mrr_graph']:.6f} | {report['fractional_top1_graph']:.6f} |",
           "",
           f"Delta MRR {test['mean_mrr_delta']:+.6f}; formula block 95% bootstrap "
           f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]; two-sided p={p:.6f}; "
           f"wins/losses/ties {test['wins_losses_ties']}.",
           f"Exact matching query/reference signal count: {same_signal_count}.",
           f"**Directional transfer criterion: {passed_science}**.",
           "",
           "Per-native-polarity scientific coverage:"]
    for mode,details in by_mode.items():
        lines.append(f"- {mode}: {details}")
    lines+=["","Graph native reference chemistry statuses:"]
    for k,v in sorted(graph_status.items()):
        lines.append(f"- {k}: {v}")
    lines+=["",report["limitations"],""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--locked",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.source,a.locked,a.out)
