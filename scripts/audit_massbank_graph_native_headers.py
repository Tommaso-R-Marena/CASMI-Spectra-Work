"""MassBank 2026.03 graph chemistry prescore gate: STRICT REFERENCE HEADERS ONLY.

Uses the exact archived 252-query/105-isomer-pool candidate policy from
the original published MassBank identification study, before inspecting
ANY withheld target peak. Never alter the original candidate population.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from benchmark_massbank_isomer_identification import (
    ARCHIVE_SHA, validate_artifacts, select_pools, check_pools, sha256,
)
from qfd.graph_fragment_signatures import signatures

STATUS="MASSBANK_202603_GRAPH_ONLY_PRESCORE_NATIVE_HEADER_GATE_V1"
MIN_VALID_QUERIES=50
MIN_VALID_FORMULAS=15


def reference_header(archive:ZipFile,path:str):
    """Stop at peak-table delimiter; no numeric MS/MS peak line is read."""
    keys={}
    with archive.open(path,"r") as stream:
        for raw in stream:
            text=raw.decode("utf-8",errors="replace").strip()
            if text.startswith("PK$PEAK:"):
                break
            if text.startswith("CH$SMILES:"):
                keys.setdefault("SMILES",[]).append(text.split(":",1)[1].strip())
            elif text.startswith("CH$FORMULA:"):
                keys.setdefault("FORMULA",[]).append(text.split(":",1)[1].strip())
            elif text.startswith("CH$LINK: INCHIKEY "):
                keys.setdefault("INCHIKEY",[]).append(
                    text.removeprefix("CH$LINK: INCHIKEY ").strip().upper())
    result={}
    for name,values in keys.items():
        result[name]=values[0] if len(values)==1 else None
    return result


def chemical_candidate(candidate,low,high):
    """Fail invalid chemistry CLOSED to graph score zero, not candidate removal."""
    formula=candidate["formula"]
    full_key=candidate["inchikey"].upper()
    for h in (low,high):
        if h.get("INCHIKEY")!=full_key or h.get("FORMULA")!=formula:
            raise RuntimeError("native MassBank header contradicts original locked structure")
    a,b=low.get("SMILES"),high.get("SMILES")
    if not a or not b:
        return {"status":"MISSING_NATIVE_REFERENCE_SMILES","smiles":"",
                "n_fragment_masses":0,"n_eligible_breaks":0}
    if a!=b:
        return {"status":"CONTRADICTORY_NATIVE_REFERENCE_SMILES","smiles":"",
                "n_fragment_masses":0,"n_eligible_breaks":0}
    masses,status,n_breaks=signatures(a,formula)
    return {"status":status,"smiles":a if status=="OK" else "",
            "n_fragment_masses":len(masses),"n_eligible_breaks":n_breaks}


def evaluate(source:Path,efficacy:Path,prior_inventory:Path,out:Path):
    manifest,old_scores,prior=validate_artifacts(efficacy,prior_inventory)
    pools,discarded=select_pools(manifest)
    untouched=check_pools(pools,prior)
    if sha256(source)!=ARCHIVE_SHA:
        raise RuntimeError("MassBank 2026.03 source differs from frozen archive")
    group_rows=[]
    status_counts=Counter()
    query_usable=0
    formula_usable=set()
    candidate_count=0
    with ZipFile(source) as archive:
        available=set(archive.namelist())
        required={path for pool in pools.values() for c in pool
                  for path in (c["low_path"],c["high_path"])}
        if not required.issubset(available):
            raise RuntimeError("missing locked source candidate reference file")
        header_cache={}
        def header(path):
            if path not in header_cache:
                header_cache[path]=reference_header(archive,path)
            return header_cache[path]
        for key,pool in sorted(pools.items()):
            candidates=[]
            for c in pool:
                x=chemical_candidate(c,header(c["low_path"]),header(c["high_path"]))
                status_counts[x["status"]]+=1
                candidate_count+=1
                candidates.append({
                    "key14":c["inchikey"][:14],
                    "full_inchikey":c["inchikey"],
                    "formula":c["formula"],
                    "low_path":c["low_path"],
                    "high_path":c["high_path"],
                    **x,
                })
            all_valid=all(c["status"]=="OK" for c in candidates)
            if all_valid:
                query_usable+=len(pool)
                formula_usable.add(pool[0]["formula"])
            group_rows.append({
                "pool":list(key),
                "query_paths":[c["query_path"] for c in pool],
                "candidates":candidates,
                "all_candidate_graphs_usable":all_valid,
            })
    n_groups=sum(bool(group["all_candidate_graphs_usable"]) for group in group_rows)
    pass_gate=query_usable>=MIN_VALID_QUERIES and len(formula_usable)>=MIN_VALID_FORMULAS
    result={
        "status":STATUS,
        "gate_pass":bool(pass_gate),
        "source_sha256":ARCHIVE_SHA,
        "source_doi":"10.5281/zenodo.19590880",
        "unchanged_original_pools":len(pools),
        "unchanged_original_queries":sum(len(pool) for pool in pools.values()),
        "unchanged_original_formula_blocks":len({r["formula"] for p in pools.values() for r in p}),
        "n_original_no_prior_msnlib_structure_pools":len(untouched),
        "n_pools_with_all_graph_structures_usable":n_groups,
        "n_queries_in_pools_with_all_graph_structures_usable":query_usable,
        "n_formulas_in_pools_with_all_graph_structures_usable":len(formula_usable),
        "n_reference_candidate_structure_instances":candidate_count,
        "native_smiles_status_counts":dict(status_counts),
        "prespecified_minimum":{
            "all_graph_usable_queries":MIN_VALID_QUERIES,
            "distinct_formulas":MIN_VALID_FORMULAS,
        },
        "original_prior_efficacy_artifact_sha256":sha256(efficacy/"rows.jsonl"),
        "original_frozen_manifest_sha256":sha256(efficacy/"label_free_triplets.jsonl"),
        "original_prior_inventory_sha256":sha256(prior_inventory),
        "native_reference_headers_only":True,
        "experimental_target_peak_lines_opened":False,
        "experimental_reference_peak_lines_opened":False,
        "query_smiles_never_read":True,
        "no_exclusion_from_original_252_queries":True,
        "frozen_original_model_blob":"45616998a79f0a8451abb55ff56628ff527f3586",
        "limitation":"Source previously evaluated for energy spectra and identification; "
            "candidate pools conditioned on measured reference spectra and may "
            "overlap external GNPS data. Metadata feasibility is NOT model efficacy.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"feasibility.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    with (out/"candidate_native_graph_headers.jsonl").open("w") as f:
        for row in group_rows:f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--efficacy",type=Path,required=True)
    p.add_argument("--prior-inventory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.source,a.efficacy,a.prior_inventory,a.out)
