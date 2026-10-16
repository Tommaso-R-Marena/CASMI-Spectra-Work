"""Posthoc MultiMS2 alternate replicate selection, keeping original 1781 groups.

Original frozen source used first eligible file-position record at each energy.
This tests the deterministic LAST eligible scan from the same fixed structure,
formula/adduct/polarity/dataset/energy group. Never selects by target scores.

DO NOT interpret this as a new acquisition-source replication or allow it to
replace the original primary mean. The source contains >14k duplicate slots.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

import audit_multims2_third_source_metadata as source
from audit_multims2_third_source_metadata_v2 import NEW_EXACT_CID_ENERGY
source.ENERGY=NEW_EXACT_CID_ENERGY

from benchmark_multims2_third_source_geometry import (
    load_selected_source_spectra,score_one,summary
)
from qfd.stats import paired_block_bootstrap_delta,paired_block_signflip_test

SOURCE_SHA="4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"
MANIFEST_SHA="5345f532ded9baa4134d38b840a6e05a1b1f85b3cf19923569ea80d96f46289d"
ORIGINAL_ROWS_SHA="f97bfaee05e9197cdaa96ad995ab9019c995274820ff57f3b04bd04de12c2be6"
ORIGINAL_RESULT_SHA="1693afcad9da7103875d9333bb0e43980ac222f77e5b25c106c1d458b2fcaef8"
METADATA_GATE_SHA="941648b8653cf2ea258f357793e61f1da8162886a2c833d13239dcc714ca8377"
IDENTITY=("inchikey","formula","adduct","ion_mode","fragmentation","source_dataset_id")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def groupkey(r):
    return tuple(r[k] for k in IDENTITY)

def last_replicate_manifest(records,original_manifest):
    """Pure metadata selection; must reproduce original FIRST slots too."""
    original={groupkey(r):r for r in original_manifest}
    if len(original)!=len(original_manifest):
        raise RuntimeError("original frozen source contains duplicate groups")
    encountered=defaultdict(lambda:defaultdict(list))
    for r in records:
        key=groupkey(r)
        if key in original:
            encountered[key][r["energy_eV"]].append(r["record_index"])
    other=[]
    changes=Counter()
    for key,row in original.items():
        slots=encountered.get(key)
        if slots is None or set(slots)!={20,40,60}:
            raise RuntimeError("alternate replicate source group lacks 20/40/60 records")
        first=[min(slots[e]) for e in (20,40,60)]
        old=[row[role+"_index"] for role in ("low","query","high")]
        if first!=old:
            raise RuntimeError("source scanner cannot reproduce frozen FIRST-record selection")
        last=[max(slots[e]) for e in (20,40,60)]
        new=dict(row)
        for role,new_i,old_i in zip(("low","query","high"),last,old):
            new[role+"_index"]=new_i
            if new_i!=old_i:
                changes[role]+=1
        if any(a!=b for a,b in zip(last,old)):
            changes["groups_with_any_alternate_scan"]+=1
        other.append(new)
    if len(other)!=len(original_manifest) or len({r["query_index"] for r in other})!=len(other):
        raise RuntimeError("changed original 1781-query group population")
    return other,dict(changes)

def scan_relevant_headers(mgf,original_manifest):
    wanted={groupkey(r) for r in original_manifest}
    records=[]
    for i,h in source.stream_headers(mgf):
        r,why=source.eligible(h,i)
        if r is not None and groupkey(r) in wanted:
            records.append(r)
    return records

def evaluate(mgf,metadata_dir,score_dir,out):
    paths={
        "mgf":mgf,
        "manifest":metadata_dir/"frozen_triplets.jsonl",
        "original_gate":metadata_dir/"feasibility.json",
        "rows":score_dir/"rows.jsonl",
        "original_score":score_dir/"result.json",
    }
    checks={
        "mgf":SOURCE_SHA,"manifest":MANIFEST_SHA,
        "original_gate":METADATA_GATE_SHA,"rows":ORIGINAL_ROWS_SHA,
        "original_score":ORIGINAL_RESULT_SHA,
    }
    for k,path in paths.items():
        if digest(path)!=checks[k]:
            raise RuntimeError("source artifact or original primary outcome digest changed: "+k)
    before=read_rows(paths["manifest"])
    original_rows=read_rows(paths["rows"])
    old_scoring=json.loads(paths["original_score"].read_text())
    if len(before)!=1781 or len(original_rows)!=1781 or old_scoring["all"]["n_queries"]!=1781:
        raise RuntimeError("original frozen independent cohort differs")
    alt,changes=last_replicate_manifest(scan_relevant_headers(mgf,before),before)
    indices={r[role+"_index"] for r in alt for role in ("low","query","high")}
    available=load_selected_source_spectra(mgf,indices)
    alternative=[score_one(r,available) for r in alt]
    if len(alternative)!=1781:
        raise RuntimeError("alternate-run sample count changed")
    orig={groupkey(r):r for r in original_rows}
    x=np.array([r["quarter"]-r["hellinger"] for r in alternative])
    y=np.array([orig[groupkey(r)]["quarter"]-orig[groupkey(r)]["hellinger"]
                for r in alternative])
    blocks=[r["formula"] for r in alternative]
    result={
        "status":"POSTHOC_MULTIMS2_FIRST_VS_LAST_ELIGIBLE_REPLICATE_SENSITIVITY",
        "scientific_class":"retrospective within-source alternate acquisition replicate; not independent third/fourth source",
        "source_upstream_git_blob_sha1":source.UPSTREAM_BLOB_SHA1,
        "original_primary_scorer_run":38970610885,
        "original_header_gate_run":38970318925,
        "frozen_original_population_kept":True,
        "n_groups":len(alt),
        "changed_scans_by_energy":changes,
        "original_mean_quarter_minus_hellinger":float(y.mean()),
        "alternate_last_mean_quarter_minus_hellinger":float(x.mean()),
        "alternate_last":summary(alternative,infer=True),
        "mean_last_minus_original_delta":float(np.mean(x-y)),
        "paired_formula_block_bootstrap_last_minus_first":paired_block_bootstrap_delta(
            x,y,blocks,n_boot=10000,seed=20261016),
        "paired_formula_block_signflip_last_minus_first":paired_block_signflip_test(
            x,y,blocks,n_permutations=50000,seed=20261016),
        "limitations":"within-pinned-source alternate scan selection is posthoc; no new instruments, labs or chemical structures. Last-file scan may be from a different technical replicate. Do not interpret it as an independent validation or claim if sign changes.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"rows_last_selection.jsonl").open("w") as fp:
        for row in alternative:
            fp.write(json.dumps(row,sort_keys=True)+"\n")
    ci=result["alternate_last"]["paired"]["quarter_vs_hellinger"]["formula_block_bootstrap_95"]
    report=[
        "# Posthoc MultiMS2 20/40/60 eV deterministic alternate replicate robustness","",
        "**Existing 1,781 groups, original first-file choice unchanged and retained.**","",
        f"Groups with any changed scan: {changes.get('groups_with_any_alternate_scan',0)} / 1781.",
        f"Changed low/query/high records: {changes.get('low',0)}/{changes.get('query',0)}/{changes.get('high',0)}.",
        f"Original first-selection quarter−Hellinger mean: {result['original_mean_quarter_minus_hellinger']:+.6f}.",
        f"Alternate last-selection quarter−Hellinger mean: {result['alternate_last_mean_quarter_minus_hellinger']:+.6f}.",
        f"Alternate last-selection 95% formula-block CI: [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}].",
        f"Last minus first mean paired delta: {result['mean_last_minus_original_delta']:+.6f}.",
        "",result["limitations"],"",
    ]
    (out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--metadata-dir",type=Path,required=True)
    p.add_argument("--original-score-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.mgf,a.metadata_dir,a.original_score_dir,a.out)

if __name__=="__main__":
    main()
