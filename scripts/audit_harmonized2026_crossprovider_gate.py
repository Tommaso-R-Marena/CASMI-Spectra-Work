"""2026 harmonized cross-provider MGF: checksum-gated HEADERS ONLY.

No fragment m/z or intensities are parsed or counted. All candidate and
query cohorts are fixed using native formula/precursor/polarity/InChIKey.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re

import zstandard as zstd

SHA256="aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e"
MD5="e2b792e384a9c9f577b17daad6c4247d"
IK=re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
MINIMUM={"cross_provider_structures":200,"hard_queries":100,
         "hard_positive":20,"hard_negative":20}


def sha256file(path):
    h=hashlib.sha256();md5=hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1<<20),b""):
            h.update(block);md5.update(block)
    if h.hexdigest()!=SHA256 or md5.hexdigest()!=MD5:
        raise RuntimeError("original harmonized v3 Zenodo source changed")
    return h.hexdigest()


def stablehash(row,salt):
    payload=(salt+"|"+json.dumps(row,sort_keys=True,separators=(",",":"))).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_head(header,index):
    source=re.sub(r"[^A-Z0-9]","",header.get("SOURCE_DATASET","").upper())
    if source not in ("GNPS","MASSSPECGYM"):return None
    ik=header.get("INCHIKEY","").strip().upper()
    formula=header.get("FORMULA","").strip()
    mode=header.get("IONMODE","").strip().lower()
    try:mass=float(header.get("PEPMASS","").split()[0])
    except(IndexError,ValueError):return None
    if (not IK.fullmatch(ik) or not formula or not math.isfinite(mass)
        or mass<=0 or mode not in ("positive","negative")):return None
    return {"index":index,"source":source,"inchikey":ik,"formula":formula,
            "mode":mode,"precursor_mz":mass,
            "splash":header.get("SPLASH",""),
            "instrument":header.get("SOURCE_INSTRUMENT","")}


def iter_headers(source):
    """Native MGF complete headers; fragment peak lines IGNORED as strings."""
    with source.open("rb") as compressed:
        reader=zstd.ZstdDecompressor().stream_reader(compressed)
        with io.TextIOWrapper(reader,encoding="utf-8",errors="replace") as f:
            header=None;index=-1
            for raw in f:
                line=raw.strip()
                if line.upper()=="BEGIN IONS":
                    if header is not None:raise RuntimeError("nested MGF ions")
                    index+=1;header={}
                elif line.upper()=="END IONS":
                    if header is None:raise RuntimeError("missing BEGIN IONS")
                    yield index,header
                    header=None
                elif header is not None and "=" in line:
                    k,v=line.split("=",1)
                    k=k.strip().upper()
                    if re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}",k):
                        if k in header:
                            raise RuntimeError("duplicate native MGF header "+k)
                        header[k]=v.strip()
                # All numeric peak lines silently ignored
            if header is not None:raise RuntimeError("truncated zstd native MGF")


def meta_only(path):
    counts=Counter();native=Counter();usable=[];n=0
    for i,h in iter_headers(path):
        n+=1
        rawsrc=re.sub(r"[^A-Z0-9]","",h.get("SOURCE_DATASET","").upper())
        native[rawsrc]+=1
        row=parse_head(h,i)
        if row is not None:usable.append(row);counts[row["source"]]+=1
    if n!=443905:
        raise RuntimeError(f"source count not matching published 443905 ({n})")
    return usable,{"raw_total":n,"raw_source_tokens":dict(native),
                   "usable_header_counts_by_source":dict(counts),
                   "usable_total":len(usable)}


def candidate_key(r):
    return (r["formula"],r["mode"])


def structure_key(r):
    return (r["inchikey"],r["formula"],r["mode"])


def freeze(rows):
    reference=[r for r in rows if r["source"]=="GNPS"]
    query=[r for r in rows if r["source"]=="MASSSPECGYM"]
    # Reference selection before query eligibility. Preserve distinct
    # adduct masses with a predeclared precision bucket of 0.01 Da.
    by_reference=defaultdict(list)
    for r in reference:
        bucket=round(r["precursor_mz"]*100)
        by_reference[(r["inchikey"],r["formula"],r["mode"],bucket)].append(r)
    chosen_ref=[]
    for k,members in sorted(by_reference.items()):
        chosen_ref.extend(sorted(
            members,key=lambda r:stablehash(r,"gnps-ref-two-v1"))[:2])
    target_groups=defaultdict(list)
    for r in chosen_ref:target_groups[candidate_key(r)].append(r)
    for key in target_groups:
        target_groups[key].sort(key=lambda r:(r["precursor_mz"],r["index"]))
    # Choose one MassSpecGym query before knowing if ground truth has any
    # reference match. This prevents advantageous replicate selection.
    query_groups=defaultdict(list)
    for r in query:query_groups[structure_key(r)].append(r)
    chosen_query=[min(members,key=lambda r:stablehash(r,"msg-query-one-v1"))
                  for members in query_groups.values()]
    chosen_query.sort(key=lambda r:(r["mode"],r["formula"],r["inchikey"]))
    accepted=[];coverage=Counter();full_overlap=set()
    for r in chosen_query:
        key=candidate_key(r)
        group=target_groups.get(key,[])
        qmass=r["precursor_mz"]
        tol=max(.02,20e-6*qmass)
        pool=[v for v in group if abs(v["precursor_mz"]-qmass)<=tol]
        ids={v["inchikey"] for v in pool}
        if r["inchikey"] in ids:
            full_overlap.add(r["inchikey"])
            coverage["cross_provider_queries"]+=1
            if len(ids)>=2:
                coverage["hard_queries"]+=1
                coverage["hard_positive" if r["mode"]=="positive" else "hard_negative"]+=1
                accepted.append({
                    **r, "candidate_structures":len(ids),
                    "selected_reference_indices":[v["index"] for v in pool],
                })
    accepted.sort(key=lambda r:(r["mode"],r["inchikey"],r["index"]))
    info={
        **dict(coverage),
        "cross_provider_structures":len(full_overlap),
        "source_gnps_header_records":len(reference),
        "source_msg_header_records":len(query),
        "frozen_gnps_reference_scans":len(chosen_ref),
        "one_query_per_full_identity_formula_mode":len(chosen_query),
        "distinct_hard_query_structures":len({r["inchikey"] for r in accepted}),
        "hard_candidate_scan_count":sum(len(r["selected_reference_indices"]) for r in accepted),
    }
    passed=all(info.get(k,0)>=v for k,v in MINIMUM.items())
    return chosen_ref,accepted,info,passed


def evaluate(source:Path,out:Path):
    digest=sha256file(source)
    rows,raw=meta_only(source)
    ref,query,coverage,passed=freeze(rows)
    out.mkdir(parents=True,exist_ok=True)
    result={
        "status":"HARMONIZED2026_CROSS_PROVIDER_NATIVE_HEADER_GATE_V1",
        "gate_pass":passed,"source_doi":"10.5281/zenodo.20042904",
        "source_sha256":digest,"source_md5":MD5,
        "coverage":coverage,"raw_header_audit":raw,"minimum":MINIMUM,
        "query_peak_locations_or_intensities_opened":False,
        "gnps_reference_peaks_opened":False,
        "metadata_rule":"full InChIKey, formula, native polarity; mass max(.02,20ppm); one MSG query hashed per identity, max 2 GNPS reference per .01-Da native precursor bucket",
        "study_type":"cross-library GNPS vs MassSpecGym, not independent physical acquisition and not collision energy interpolation",
        "native_inchikey_annotation_is_computationally_harmonized":True,
    }
    (out/"feasibility.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    for filename,items in (("reference_manifest.jsonl",ref),("hard_queries.jsonl",query)):
        with (out/filename).open("w") as f:
            for item in items:f.write(json.dumps(item,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zst",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(a.zst,a.out),indent=2))


if __name__=="__main__":main()
