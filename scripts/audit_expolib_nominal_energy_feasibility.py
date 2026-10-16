"""ExpoLib v1 two-polarity metadata-only independent-source feasibility gate.

Never parses peak m/z or intensities. Explicitly distinguishes nominal numeric
collision-energy coordinate from unverified physically calibrated eV.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from qfd.msnlib import load_msnlib_jsonl

KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
CE = re.compile(r"^\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]$")

EXPECTED_MD5 = {
    "ESI_POS": "c08dcc4dc3c11eef188dff1eb1f2bc1d",
    "ESI_NEG": "3a7544d71bf9ee3564243095b3134d3d",
}
DISCOVERY_MD5 = "806f33c9c60af74a0aa21fb66aeb84f3"
GATES = {"triplets":100,"structures":50,"formula_blocks":30,"offcenter_triplets":30}

def hashfile(path, kind="sha256"):
    h=hashlib.new(kind)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def parse_headers(path, source):
    """Only inspect recognized header keys; ignore every spectral peak line."""
    records=[]
    header={}
    current=-1
    def flush():
        if not header:
            return
        row=accept(header,source,current)
        if row is not None:
            records.append(row)
    with Path(path).open("r",encoding="utf-8",errors="replace") as stream:
        for line in stream:
            line=line.strip()
            if line.startswith("Name: "):
                flush()
                current+=1
                header={"Name":line[6:].strip()}
            elif ":" in line and current>=0:
                key,value=line.split(":",1)
                if key in {"InChIKey","Formula","Precursor_type","Spectrum_type",
                           "Method","Collision_energy","Ion_mode","Num Peaks","charge"}:
                    if key in header:
                        header[key]=None  # duplicate header is invalid, fail closed
                    else:
                        header[key]=value.strip()
    flush()
    return records,current+1

def accept(h,source,index):
    key=(h.get("InChIKey") or "").upper()
    m=CE.fullmatch(h.get("Collision_energy") or "")
    try:
        n=int(h.get("Num Peaks") or 0)
        ms=int(h.get("Spectrum_type") or 0)
    except (ValueError,TypeError):
        return None
    if (not KEY.fullmatch(key) or not m or not 0<float(m.group(1))<1000
        or not h.get("Formula") or not h.get("Precursor_type")
        or h.get("Method","").upper() not in {"CID","HCD"}
        or h.get("Ion_mode","").upper() not in {"POSITIVE","NEGATIVE"}
        or ms!=2 or n<5):
        return None
    mode=h["Ion_mode"].upper()
    if (source=="ESI_POS" and mode!="POSITIVE") or (source=="ESI_NEG" and mode!="NEGATIVE"):
        return None
    return {
        "inchikey":key,"formula":h["Formula"],
        "adduct":h["Precursor_type"],"mode":mode,
        "fragmentation":h["Method"].upper(),"energy_nominal":float(m.group(1)),
        "record_index":index,"source":source,"n_peaks_declared":n,
    }

def select(rows,discovery):
    """Deterministically choose earliest record at repeated nominal energy."""
    groups=defaultdict(list)
    excluded_records=0
    for r in rows:
        if r["inchikey"] in discovery:
            excluded_records+=1
            continue
        key=(r["inchikey"],r["adduct"],r["mode"],r["fragmentation"],r["source"])
        groups[key].append(r)
    trips=[]
    slots_discarded=0
    for group,records in sorted(groups.items()):
        slots={}
        for r in sorted(records,key=lambda x:x["record_index"]):
            e=r["energy_nominal"]
            if e in slots:
                slots_discarded+=1
            else:
                slots[e]=r
        es=sorted(slots)
        for i in range(1,len(es)-1):
            low,mid,high=slots[es[i-1]],slots[es[i]],slots[es[i+1]]
            t=(es[i]-es[i-1])/(es[i+1]-es[i-1])
            if not 0<t<1:
                raise ValueError("invalid interior nominal-energy fraction")
            trips.append({
                "inchikey":mid["inchikey"],"formula":mid["formula"],
                "source":mid["source"],"mode":mid["mode"],
                "adduct":mid["adduct"],"fragmentation":mid["fragmentation"],
                "low_index":low["record_index"],"query_index":mid["record_index"],
                "high_index":high["record_index"],
                "low_energy_nominal":es[i-1],"query_energy_nominal":es[i],
                "high_energy_nominal":es[i+1],"t":t,
            })
    return trips,{"excluded_discovery_records":excluded_records,
                  "duplicate_nominal_energy_records_discarded_earliest_wins":slots_discarded}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--positive",type=Path,required=True)
    ap.add_argument("--negative",type=Path,required=True)
    ap.add_argument("--discovery",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    paths={"ESI_POS":a.positive,"ESI_NEG":a.negative}
    digests={}
    for source,path in paths.items():
        md5=hashfile(path,"md5")
        if md5!=EXPECTED_MD5[source]:
            raise RuntimeError(f"source {source} md5 mismatch")
        digests[source]={"md5":md5,"sha256":hashfile(path)}
    if hashfile(a.discovery,"md5")!=DISCOVERY_MD5:
        raise RuntimeError("discovery MD5 mismatch")
    discovery={r.full_inchikey.upper() for r in load_msnlib_jsonl(a.discovery)}
    all_rows=[]
    total=0
    by_source={}
    for source,path in paths.items():
        parsed,n=parse_headers(path,source)
        by_source[source]={"records":n,"metadata_pass":len(parsed)}
        total+=n
        all_rows.extend(parsed)
    manifest,exclusion=select(all_rows,discovery)
    coverage={
        "triplets":len(manifest),
        "structures":len({x["inchikey"] for x in manifest}),
        "formula_blocks":len({x["formula"] for x in manifest}),
        "offcenter_triplets":sum(abs(x["t"]-.5)>1e-12 for x in manifest),
        "polarity":dict(Counter(x["mode"] for x in manifest)),
        "nominal_energy_hist":dict(Counter(str(x["query_energy_nominal"]) for x in manifest).most_common()),
    }
    passed=all(coverage[k]>=v for k,v in GATES.items())
    result={
        "status":"EXPOLIB_V1_POSTDISCOVERY_LABEL_FREE_FEASIBILITY",
        "source_zenodo_record":"18186811",
        "source_institution":"University of Vienna / Global Exposomics and Biomonitoring Laboratory",
        "source_input_hashes":digests,"discovery_md5":DISCOVERY_MD5,
        "total_msp_records":total,"headers_by_source":by_source,
        "accepted_metadata_records":len(all_rows),
        "selection":"full InChIKey, formula, adduct, MS2 CID/HCD, nominal one-element CE, earliest duplicate at each CE, at least five declared peaks",
        "coverage":coverage,"minimum_gate":GATES,
        "metadata_feasibility_pass":passed,
        "physical_eV_unit_validation_pass":False,
        "reason_physical_validation_closed":"MSP headers give unqualified numeric collision_energy (e.g. [40.0]), not verified explicit eV; source instrument/acquisition metadata not fully harmonized",
        "feasibility_exclusions":exclusion,
        "scores_opened":False,
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
    with (a.out/"nominal_energy_triplets.jsonl").open("w") as stream:
        for r in sorted(manifest,key=lambda x:(x["source"],x["inchikey"],x["adduct"],x["query_energy_nominal"],x["query_index"])):
            stream.write(json.dumps(r,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
