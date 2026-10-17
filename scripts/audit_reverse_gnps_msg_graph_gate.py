"""Reverse archive provider validation: GNPS queries / MassSpecGym references.

This stage MUST NOT inspect measured fragment peaks, prediction outcomes or
true-query SMILES. Independent deterministic metadata selection fixed before
the graph model is allowed to access any query spectrum. Two archive directions
are NOT equivalent to independent physical laboratory acquisition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_harmonized2026_crossprovider_gate import (
    SHA256, MD5, sha256file, stablehash, meta_only,
)

STATUS="REVERSE_PROVIDER_NATIVE_HEADER_LOCKBOX_20261017_V1"
MIN_QUERIES=100
MIN_STRUCTURES=50
MIN_FORMULAS=20
FIXED_MAX_PER_PRECURSOR_BUCKET=2


def selection_key(row,salt):
    # Whitelist native metadata fields: never hash arbitrary peak/answer fields.
    safe={k:row[k] for k in ("index","source","inchikey","formula",
                            "mode","precursor_mz")}
    return stablehash(safe,salt)


def freeze_reverse(rows):
    # Source and selection rules are flipped relative to the already scored
    # GNPS-reference / MassSpecGym-query cohort, before any peaks are read.
    refs=[r for r in rows if r["source"]=="MASSSPECGYM"]
    queries=[r for r in rows if r["source"]=="GNPS"]
    by_bucket=defaultdict(list)
    for r in refs:
        bucket=round(r["precursor_mz"]*100)
        k=(r["inchikey"],r["formula"],r["mode"],bucket)
        by_bucket[k].append(r)
    selected_refs=[]
    for k,members in sorted(by_bucket.items()):
        selected_refs.extend(sorted(
            members,key=lambda r:selection_key(r,"reverse-MSG-ref-two-prescore-20261017")
        )[:FIXED_MAX_PER_PRECURSOR_BUCKET])
    selected_refs.sort(key=lambda r:(r["index"]))
    ref_by_competition=defaultdict(list)
    for r in selected_refs:
        ref_by_competition[(r["formula"],r["mode"])].append(r)
    for pool in ref_by_competition.values():
        pool.sort(key=lambda r:(r["precursor_mz"],r["index"]))
    original_query_groups=defaultdict(list)
    for r in queries:
        original_query_groups[(r["inchikey"],r["formula"],r["mode"])].append(r)
    selected_queries=[]
    for _,members in sorted(original_query_groups.items()):
        selected_queries.append(min(
            members,key=lambda r:selection_key(r,"reverse-GNPS-query-one-prescore-20261017")))
    eligible=[]
    before_hard=0
    for q in sorted(selected_queries,key=lambda x:(x["mode"],x["inchikey"],x["index"])):
        pool=ref_by_competition.get((q["formula"],q["mode"]),[])
        tol=max(.02,20e-6*q["precursor_mz"])
        candidates=[r for r in pool if abs(r["precursor_mz"]-q["precursor_mz"])<=tol]
        ids={r["inchikey"] for r in candidates}
        if q["inchikey"] not in ids:continue
        before_hard+=1
        if len(ids)<2:continue
        eligible.append({
            **q,"candidate_structures":len(ids),
            "selected_reference_indices":[r["index"] for r in candidates],
        })
    eligible.sort(key=lambda x:(x["mode"],x["inchikey"],x["index"]))
    info={
        "total_source_metadata_rows":len(rows),
        "raw_msg_reference_records":len(refs),
        "raw_gnps_query_records":len(queries),
        "selected_msg_reference_scans":len(selected_refs),
        "gnps_one_query_per_full_identity_formula_mode":len(selected_queries),
        "cross_provider_exact_structure_query_overlap":before_hard,
        "hard_queries":len(eligible),
        "hard_query_positive":sum(r["mode"]=="positive" for r in eligible),
        "hard_query_negative":sum(r["mode"]=="negative" for r in eligible),
        "hard_structure_truths":len({r["inchikey"] for r in eligible}),
        "distinct_native_formulas":len({r["formula"] for r in eligible}),
        "hard_reference_candidate_scans":sum(
            len(q["selected_reference_indices"]) for q in eligible),
    }
    pass_gate=(info["hard_queries"]>=MIN_QUERIES
               and info["hard_structure_truths"]>=MIN_STRUCTURES
               and info["distinct_native_formulas"]>=MIN_FORMULAS)
    return selected_refs,eligible,info,pass_gate


def evaluate(source:Path,out:Path):
    original_sha=sha256file(source)
    rows,header_audit=meta_only(source)
    references,queries,coverage,passed=freeze_reverse(rows)
    out.mkdir(parents=True,exist_ok=True)
    result={
        "status":STATUS,
        "gate_pass":bool(passed),
        "source_sha256":original_sha,
        "source_md5":MD5,
        "source_doi":"10.5281/zenodo.20042904",
        "reference_source":"MassSpecGym",
        "query_source":"GNPS",
        "native_header_audit":header_audit,
        "coverage":coverage,
        "prespecified_gate_minimum":{
            "hard_queries":MIN_QUERIES,
            "hard_structure_truths":MIN_STRUCTURES,
            "distinct_native_formulas":MIN_FORMULAS,
        },
        "query_peak_positions_or_intensities_read":False,
        "reference_peak_positions_or_intensities_read":False,
        "query_smiles_read":False,
        "selection_rule":"one SHA-select GNPS query/full InChIKey/formula/mode; up to two SHA-selected MSG reference scans per 0.01Da precursor bucket; native full InChIKey/formula/mode, same precursor <=max(.02Da,20ppm), hard pool >=2",
        "is_cross_archive_direction_only_not_independent_new_acquisition":True,
        "not_training_or_parameter_fitting":True,
        "retrospective_context":"Prior forward GNPS reference to MSG query tested same source; reverse target ranks not previously published.",
    }
    (out/"feasibility.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    for name,items in (
        ("reference_manifest.jsonl",references),
        ("hard_queries.jsonl",queries),
    ):
        with (out/name).open("w") as f:
            for row in items:f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps({"status":STATUS,"scientific_gate_pass":passed,
                      "coverage":coverage},sort_keys=True))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    evaluate(args.source,args.out)


if __name__=="__main__":
    main()
