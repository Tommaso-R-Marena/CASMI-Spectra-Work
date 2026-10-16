"""MultiMS2 v1: frozen, *header-only* independent CID 20/40/60 eV gate.

This module never parses peak m/z or intensities, even for rejected records.
No prediction scores or query spectra are accessed at the gate stage.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re

from qfd.msnlib import load_msnlib_jsonl

UPSTREAM_COMMIT = "659bd9b608408a75ecbb4b03295ed4c03ba52539"
UPSTREAM_BLOB_SHA1 = "5ca1b01d1c6c5b2930658a6cd32ee2659070b49d"
UPSTREAM_BYTES = 77183510
DISCOVERY_MD5 = "806f33c9c60af74a0aa21fb66aeb84f3"
EXACT_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
ENERGY = re.compile(r"^\s*(20|40|60)(?:\.0+)?(?:\s*eV)?\s*$",re.IGNORECASE)
THRESHOLDS = {"triplets":100,"structures":80,"formula_blocks":60,
              "positive_triplets":20,"negative_triplets":20}
METADATA_KEYS={
    "INCHIAUX","INCHI_AUX","INCHIKEY","FORMULA","IONMODE","ION_MODE",
    "ADDUCT","FRAGMENTATION_METHOD","COLLISION_ENERGY","MSLEVEL",
    "SPECTYPE","NUM_PEAKS","NUMPEAKS","DATASET_ID","DESCRIPTION","ACQUISITION",
}

def digest(path,algorithm="sha256"):
    h=hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def git_blob_sha1(path):
    size=Path(path).stat().st_size
    h=hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def stream_headers(path):
    """Yield header dict + file order; numeric spectral rows are not inspected."""
    current=None
    count=0
    with Path(path).open("r",encoding="utf-8",errors="replace") as fp:
        for raw in fp:
            line=raw.strip()
            if line.upper()=="BEGIN IONS":
                if current is not None:
                    raise ValueError("nested MGF entry")
                current={}
            elif line.upper()=="END IONS":
                if current is None:
                    raise ValueError("MGF end without beginning")
                yield count,current
                count+=1
                current=None
            elif current is not None and "=" in line:
                key,value=line.split("=",1)
                key=key.strip().upper()
                if key in METADATA_KEYS:
                    if key in current:
                        current[key]=None
                    else:
                        current[key]=value.strip()
    if current is not None:
        raise ValueError("unterminated MGF entry")

def normalize_full_key(value):
    s=(value or "").strip().upper()
    if s.startswith("INCHIKEY="):
        s=s.split("=",1)[1].strip()
    if not EXACT_KEY.fullmatch(s):
        return None
    return s

def eligible(header,index):
    """Returns (record,attrition_code); no dependence on peak intensity."""
    def get(*names):
        values=[header[k] for k in names if k in header]
        if len(values)!=1:
            return None
        return values[0]
    if (header.get("MSLEVEL") not in (None,"2","MS2")):
        return None,"not_ms2"
    if (header.get("FRAGMENTATION_METHOD") or "").upper()!="CID":
        return None,"not_cid"
    energy_text=header.get("COLLISION_ENERGY") or ""
    energy=ENERGY.fullmatch(energy_text)
    if energy is None:
        return None,"energy_not_fixed_20_40_60_eV"
    key=normalize_full_key(get("INCHIAUX","INCHI_AUX","INCHIKEY"))
    if key is None:
        return None,"bad_full_inchikey"
    formula=header.get("FORMULA")
    if not formula:
        return None,"missing_formula"
    mode=(get("IONMODE","ION_MODE") or "").upper()
    if mode not in ("POSITIVE","NEGATIVE"):
        return None,"missing_or_unknown_polarity"
    adduct=header.get("ADDUCT")
    if not adduct:
        return None,"missing_adduct"
    peaks=get("NUM_PEAKS","NUMPEAKS")
    try:
        n=int(peaks)
    except (ValueError,TypeError):
        n=0
    if n<5:
        return None,"insufficient_declared_peaks"
    source=header.get("DATASET_ID") or ""
    # Avoid unspecified compound-name differences across CE in grouping.
    result={
       "record_index":int(index),"inchikey":key,"formula":formula,
       "ion_mode":mode,"adduct":adduct,
       "fragmentation":"CID","energy_eV":int(energy.group(1)),
       "source_dataset_id":str(source),
       "declared_positive_peak_minimum":int(n),
    }
    return result,None

def select_triplets(records,blocked):
    """Pure metadata operation: deduplicate by file position, not fidelity."""
    groups=defaultdict(list)
    excluded=Counter()
    for r in records:
        identity=r["inchikey"]
        owners=[owner for owner,keys in blocked.items() if identity in keys]
        if owners:
            for owner in owners:
                excluded[owner]+=1
            continue
        grouping=(r["inchikey"],r["formula"],r["adduct"],r["ion_mode"],
                  r["fragmentation"],r["source_dataset_id"])
        groups[grouping].append(r)
    manifest=[]
    duplicates=0
    missing_energy=Counter()
    for key,rows in sorted(groups.items()):
        slots={}
        for row in sorted(rows,key=lambda r:r["record_index"]):
            en=row["energy_eV"]
            if en in slots:
                duplicates+=1
            else:
                slots[en]=row
        if set(slots)!={20,40,60}:
            missing_energy[",".join(str(x) for x in sorted(slots))]+=1
            continue
        low,mid,high=(slots[e] for e in (20,40,60))
        manifest.append({
            "inchikey":mid["inchikey"],"formula":mid["formula"],
            "adduct":mid["adduct"],"ion_mode":mid["ion_mode"],
            "fragmentation":"CID","source_dataset_id":mid["source_dataset_id"],
            "low_index":low["record_index"],"query_index":mid["record_index"],
            "high_index":high["record_index"],
            "low_energy_eV":20,"query_energy_eV":40,"high_energy_eV":60,
            "t":0.5,
        })
    manifest.sort(key=lambda r:(r["inchikey"],r["adduct"],r["ion_mode"],r["source_dataset_id"],r["query_index"]))
    return manifest,{
        "blocked_records_by_source":dict(sorted(excluded.items())),
        "duplicate_energy_slots_discarded_first_in_file_wins":duplicates,
        "incomplete_energy_group_hist":dict(sorted(missing_energy.items())),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--discovery",type=Path,required=True)
    p.add_argument("--prior-inventory",type=Path,required=True)
    p.add_argument("--massbank-manifest",type=Path,required=True)
    p.add_argument("--expolib-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.mgf.stat().st_size!=UPSTREAM_BYTES or git_blob_sha1(a.mgf)!=UPSTREAM_BLOB_SHA1:
        raise RuntimeError("upstream file byte length or Git blob SHA changed")
    if digest(a.discovery,"md5")!=DISCOVERY_MD5:
        raise RuntimeError("original MCEBIO+ discovery file digest mismatch")
    inventory=json.loads(a.prior_inventory.read_text())
    if (inventory.get("status")!="PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC" or
        set(inventory.get("sources",{}))!={"mcebio_neg","mcescaf_pos","enammol_pos","enamdisc_neg"}):
        raise ValueError("prior library source inventory incomplete")
    oldkeys=inventory["all_prior_evaluated_full_inchikeys"]
    if oldkeys!=sorted(set(oldkeys)):
        raise ValueError("prior evaluated identity inventory not canonical")
    massbank=[json.loads(s) for s in a.massbank_manifest.read_text().splitlines() if s.strip()]
    expolib=[json.loads(s) for s in a.expolib_manifest.read_text().splitlines() if s.strip()]
    if len(massbank)!=2138 or len(expolib)!=1125:
        raise ValueError("original MassBank / ExpoLib manifests changed")
    blocked={
        "mcebio_original_discovery":{r.full_inchikey.upper() for r in load_msnlib_jsonl(a.discovery)},
        "four_prior_evaluated_MSnLib":set(oldkeys),
        "massbank_2026_03_eligible":{r["inchikey"].upper() for r in massbank},
        "expolib_v1_eligible":{r["inchikey"].upper() for r in expolib},
    }
    total=0
    reasons=Counter()
    records=[]
    example_keys=Counter()
    for i,h in stream_headers(a.mgf):
        total+=1
        if i<50:
            example_keys.update(h.keys())
        r,why=eligible(h,i)
        if r is None:
            reasons[why]+=1
        else:
            records.append(r)
    manifest,selection=select_triplets(records,blocked)
    coverage={
        "triplets":len(manifest),
        "structures":len({r["inchikey"] for r in manifest}),
        "formula_blocks":len({r["formula"] for r in manifest}),
        "positive_triplets":sum(r["ion_mode"]=="POSITIVE" for r in manifest),
        "negative_triplets":sum(r["ion_mode"]=="NEGATIVE" for r in manifest),
        "source_datasets":dict(Counter(r["source_dataset_id"] for r in manifest)),
    }
    passed=all(coverage[k]>=minimum for k,minimum in THRESHOLDS.items())
    result={
        "status":"MULTIMS2_CID_20_40_60EV_METADATA_ONLY_V1",
        "upstream_commit":UPSTREAM_COMMIT,"upstream_blob_sha1":UPSTREAM_BLOB_SHA1,
        "upstream_size_bytes":UPSTREAM_BYTES,"raw_sha256":digest(a.mgf),
        "source":"Zamboni-Lab MultiMS2, ZenoTOF 7600, published fixed 20/40/60 eV CID",
        "license":"CC0 1.0 source data",
        "total_source_spectra_records":total,"valid_cid_header_records":len(records),
        "header_attrition":dict(reasons),"first_50_header_key_presence":dict(example_keys),
        "exclusions_and_dedup":selection,
        "coverage":coverage,"frozen_minimum_thresholds":THRESHOLDS,
        "gate_pass":passed,
        "target_peak_locations_or_intensities_read":False,
        "outcome_scores_opened":False,
        "scientific_limit":"independent instrument acquisition source, not evidence of independently sampled laboratories or universal physical interpolation law",
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
    with (a.out/"frozen_triplets.jsonl").open("w") as f:
        for item in manifest:
            f.write(json.dumps(item,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
