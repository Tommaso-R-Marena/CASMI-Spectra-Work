"""PRE-SCORE header-only EAD 12/16/24-eV feasibility lockbox for MultiMS2.

The numeric peak rows are ignored. This creates a deterministic identity
and source-file-position manifest BEFORE any 16-eV target peak scoring.
Published EAD energy labels are source-defined; no cross-method physical
kinetics or independence of the lab/instrument is inferred.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from qfd.msnlib import load_msnlib_jsonl
import audit_multims2_third_source_metadata as cid
from audit_multims2_third_source_metadata_v2 import SOURCE_SHA256

ENERGY=re.compile(r"^\[(12|16|24)\.0\]$")
THRESHOLDS={"triplets":30,"structures":20,"formula_blocks":15,
            "positive_triplets":5,"negative_triplets":5}
PINNED_EAD_HEADERS={"[12.0]":2484,"[16.0]":2760,"[24.0]":2811}
PINNED_RAW_SPECTRA=43728
PINNED_CID_MANIFEST_SHA256="5345f532ded9baa4134d38b840a6e05a1b1f85b3cf19923569ea80d96f46289d"
PINNED_PRIOR_KEYS_SHA256="5520ecb3f134290953fd5ad63b97cc1561dd39c2ea658aeaf4a90b2be348d07a"
PINNED_MB_MANIFEST_SHA256="265dfb75e33d2eff3f19945255bddb5db0c0f7f9161593ada80cd2b50ecac9c4"
PINNED_EXPO_MANIFEST_SHA256="095340d967b177077ce517da7aa332795ce691899cc1320563cce07a76a68560"

def eligible(header,index):
    """EAD-only counterpart to the frozen CID eligibility logic."""
    def exactly_one(*names):
        values=[header[n] for n in names if n in header]
        return values[0] if len(values)==1 else None
    if header.get("MSLEVEL") not in (None,"2","MS2"):
        return None,"not_ms2"
    if (header.get("FRAGMENTATION_METHOD") or "").upper()!="EAD":
        return None,"not_ead"
    m=ENERGY.fullmatch(header.get("COLLISION_ENERGY") or "")
    if not m:
        return None,"energy_not_exact_ead"
    key=cid.normalize_full_key(exactly_one("INCHIAUX","INCHI_AUX","INCHIKEY"))
    if key is None:
        return None,"bad_full_inchikey"
    formula=header.get("FORMULA")
    if not formula:
        return None,"missing_formula"
    mode=(exactly_one("IONMODE","ION_MODE") or "").upper()
    if mode not in ("POSITIVE","NEGATIVE"):
        return None,"missing_or_unknown_polarity"
    adduct=header.get("ADDUCT")
    if not adduct:
        return None,"missing_adduct"
    peaks=exactly_one("NUM_PEAKS","NUMPEAKS")
    try:
        n=int(peaks)
    except (ValueError,TypeError):
        n=0
    if n<5:
        return None,"insufficient_declared_peaks"
    return {
        "record_index":int(index),"inchikey":key,"formula":formula,
        "ion_mode":mode,"adduct":adduct,"fragmentation":"EAD",
        "energy_eV":int(m.group(1)),
        "source_dataset_id":str(header.get("DATASET_ID") or ""),
        "declared_positive_peak_minimum":n,
    },None

def build_manifest(records,blocked):
    groups=defaultdict(list)
    excluded=Counter()
    for r in records:
        owners=[name for name,keys in blocked.items() if r["inchikey"] in keys]
        if owners:
            excluded.update(owners)
            continue
        key=(r["inchikey"],r["formula"],r["adduct"],r["ion_mode"],
             r["fragmentation"],r["source_dataset_id"])
        groups[key].append(r)
    manifest=[]
    duplicates=0
    incomplete=Counter()
    for group,rows in sorted(groups.items()):
        slots={}
        for r in sorted(rows,key=lambda x:x["record_index"]):
            en=r["energy_eV"]
            if en in slots:
                duplicates+=1
            else:
                slots[en]=r
        if set(slots)!={12,16,24}:
            incomplete[",".join(map(str,sorted(slots)))]+=1
            continue
        low,mid,high=[slots[e] for e in (12,16,24)]
        manifest.append({
            "inchikey":mid["inchikey"],"formula":mid["formula"],
            "ion_mode":mid["ion_mode"],"adduct":mid["adduct"],
            "fragmentation":"EAD","source_dataset_id":mid["source_dataset_id"],
            "low_index":low["record_index"],"query_index":mid["record_index"],
            "high_index":high["record_index"],
            "low_energy_eV":12,"query_energy_eV":16,"high_energy_eV":24,
            "t":1/3,
        })
    manifest.sort(key=lambda r:(r["inchikey"],r["adduct"],r["ion_mode"],
                                r["source_dataset_id"],r["query_index"]))
    return manifest,{
        "prior_excluded_records_by_source":dict(sorted(excluded.items())),
        "duplicate_slots_discarded_first_scan_only":duplicates,
        "incomplete_triplet_groups":dict(sorted(incomplete.items())),
    }

def read_rows(path,checksum):
    if cid.digest(path)!=checksum:
        raise RuntimeError(f"PRIOR SOURCE HASH MISMATCH: {path}")
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def evaluate(args):
    if (args.mgf.stat().st_size!=cid.UPSTREAM_BYTES or
        cid.git_blob_sha1(args.mgf)!=cid.UPSTREAM_BLOB_SHA1 or
        cid.digest(args.mgf)!=SOURCE_SHA256):
        raise RuntimeError("MultiMS2 pinned raw file changed")
    if cid.digest(args.discovery,"md5")!=cid.DISCOVERY_MD5:
        raise RuntimeError("original discovery identity file hash changed")
    if cid.digest(args.prior_inventory)!=PINNED_PRIOR_KEYS_SHA256:
        raise RuntimeError("prior MSnLib inventory hash changed")
    prior=json.loads(args.prior_inventory.read_text())
    if prior.get("status")!="PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC":
        raise RuntimeError("prior identity inventory not authentic")
    oldkeys=prior["all_prior_evaluated_full_inchikeys"]
    if oldkeys!=sorted(set(oldkeys)):
        raise RuntimeError("prior keys not unique and sorted")
    mb=read_rows(args.massbank_manifest,PINNED_MB_MANIFEST_SHA256)
    expo=read_rows(args.expolib_manifest,PINNED_EXPO_MANIFEST_SHA256)
    cid_triples=read_rows(args.cid_manifest,PINNED_CID_MANIFEST_SHA256)
    if len(mb)!=2138 or len(expo)!=1125 or len(cid_triples)!=1781:
        raise RuntimeError("previous evaluated population sizes changed")
    blocked={
        "original_discovery_MCEBIO_positive":{
            r.full_inchikey.upper() for r in load_msnlib_jsonl(args.discovery)},
        "prior_evaluated_four_MSnLib":set(oldkeys),
        "MassBank_original_eligible":{r["inchikey"].upper() for r in mb},
        "ExpoLib_original_eligible":{r["inchikey"].upper() for r in expo},
    }
    counts=Counter()
    reasons=Counter()
    records=[]
    energy_counts=Counter()
    for i,h in cid.stream_headers(args.mgf):
        counts["total"]+=1
        if (h.get("FRAGMENTATION_METHOD") or "").upper()=="EAD":
            energy_counts[str(h.get("COLLISION_ENERGY"))]+=1
        r,why=eligible(h,i)
        if r is None:
            reasons[why]+=1
        else:
            records.append(r)
    if counts["total"]!=PINNED_RAW_SPECTRA or dict(energy_counts)!=PINNED_EAD_HEADERS:
        raise RuntimeError("EAD original metadata header inventory changed")
    manifest,attrition=build_manifest(records,blocked)
    previous_cid={r["inchikey"] for r in cid_triples}
    overlap=sum(r["inchikey"] in previous_cid for r in manifest)
    cid_disjoint=[r for r in manifest if r["inchikey"] not in previous_cid]
    coverage={
        "triplets":len(manifest),
        "structures":len({r["inchikey"] for r in manifest}),
        "formula_blocks":len({r["formula"] for r in manifest}),
        "positive_triplets":sum(r["ion_mode"]=="POSITIVE" for r in manifest),
        "negative_triplets":sum(r["ion_mode"]=="NEGATIVE" for r in manifest),
        "previous_CID_evaluated_identity_overlap_triplets":overlap,
        "no_previous_CID_identity_overlap_triplets":len(cid_disjoint),
        "no_previous_CID_identity_overlap_formulas":len({r["formula"] for r in cid_disjoint}),
        "source_dataset_id_counts":dict(Counter(r["source_dataset_id"] for r in manifest)),
    }
    passes=all(coverage[k]>=n for k,n in THRESHOLDS.items())
    result={
        "status":"MULTIMS2_EAD_12_16_24EV_HEADER_ONLY_GATE_V1",
        "source_blob_sha1":cid.UPSTREAM_BLOB_SHA1,
        "source_sha256":SOURCE_SHA256,
        "source_commit":cid.UPSTREAM_COMMIT,
        "source_data_license":"CC0",
        "modality":"EAD, electron-activated dissociation, same MultiMS2 laboratory/instrument as already scored CID",
        "t":1/3,
        "raw_spectra_count":counts["total"],
        "ead_energy_header_counts":dict(energy_counts),
        "eligible_ead_records_before_identity_exclusion":len(records),
        "header_attrition":dict(sorted(reasons.items())),
        "selection":attrition,
        "coverage":coverage,
        "metadata_only_thresholds":THRESHOLDS,
        "gate_pass":passes,
        "16eV_target_peak_positions_or_intensities_read":False,
        "target_scores_or_predictor_outcomes_opened":False,
        "primary_population":"complete metadata-filtered and identity-excluded EAD cohort; no outcome-conditioned exclusions",
        "limitations":"EAD is a distinct fragmentation modality but SAME instrument/laboratory and overlapping chemicals are possible; not independent acquisition-source replication; no molecule-identification conclusion.",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"feasibility.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (args.out/"frozen_triplets.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser()
    for k in ("mgf","discovery","prior_inventory","massbank_manifest",
              "expolib_manifest","cid_manifest","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    return evaluate(p.parse_args())

if __name__=="__main__":
    main()
