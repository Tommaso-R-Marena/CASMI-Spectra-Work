"""Post-failed-v1 MultiMS2 header-only collision-energy representation audit.

This stage does not change or rerun the v1 gate and never opens peak lines.
The exact same pinned 77MB Git blob is used. This is *protocol repair research*
and cannot authorize efficacy scoring without a separately frozen new v2 gate.
"""
from __future__ import annotations
from collections import Counter
import argparse
import json
from pathlib import Path
import re

from audit_multims2_third_source_metadata import (
    digest,git_blob_sha1,UPSTREAM_BYTES,UPSTREAM_BLOB_SHA1,stream_headers,
)

def inspect(path):
    data={
        "total_mgf_records":0,
        "fragmentation_method_counts":Counter(),
        "collision_energy_raw_all":Counter(),
        "collision_energy_raw_cid":Counter(),
        "collision_energy_raw_ead":Counter(),
        "ion_modes_in_cid":Counter(),
        "key_format_prefixes_cid":Counter(),
        "example_cid_headers":[],
    }
    for i,h in stream_headers(path):
        data["total_mgf_records"]+=1
        frag=str(h.get("FRAGMENTATION_METHOD","<absent>"))
        ce=str(h.get("COLLISION_ENERGY","<absent>"))
        data["fragmentation_method_counts"][frag]+=1
        data["collision_energy_raw_all"][ce]+=1
        if frag.upper()=="CID":
            data["collision_energy_raw_cid"][ce]+=1
            data["ion_modes_in_cid"][str(h.get("IONMODE","<absent>"))]+=1
            ink=str(h.get("INCHIAUX","<absent>"))
            data["key_format_prefixes_cid"][ink.split("=",1)[0] if "=" in ink else "<full-value>"]+=1
            if len(data["example_cid_headers"])<3:
                data["example_cid_headers"].append({
                    "index":i,
                    "collision_energy":ce,
                    "formula":h.get("FORMULA"),
                    "identity":h.get("INCHIAUX"),
                    "mslevel":h.get("MSLEVEL"),
                    "ionmode":h.get("IONMODE"),
                    "num_peaks_declared":h.get("NUM_PEAKS"),
                    "dataset_id":h.get("DATASET_ID"),
                    "adduct":h.get("ADDUCT"),
                })
        elif frag.upper()=="EAD":
            data["collision_energy_raw_ead"][ce]+=1
    result={
        "status":"MULTIMS2_EXACT_BLOB_V1_FAILURE_HEADER_REPRESENTATION_AUDIT",
        "purpose":"Explain why original frozen gate failed. No efficacy; no parser changes.",
        "upstream_file_sha256":digest(path),
        "total_mgf_records":data["total_mgf_records"],
        "fragmentation_method_counts":dict(data["fragmentation_method_counts"]),
        "collision_energy_raw_all":dict(data["collision_energy_raw_all"]),
        "collision_energy_raw_cid":dict(data["collision_energy_raw_cid"]),
        "collision_energy_raw_ead":dict(data["collision_energy_raw_ead"]),
        "ion_modes_in_cid":dict(data["ion_modes_in_cid"]),
        "key_format_prefixes_cid":dict(data["key_format_prefixes_cid"]),
        "example_cid_headers":data["example_cid_headers"],
        "target_mz_or_intensity_values_read":False,
        "no_scoring_allowed":True,
    }
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.mgf.stat().st_size!=UPSTREAM_BYTES or git_blob_sha1(a.mgf)!=UPSTREAM_BLOB_SHA1:
        raise RuntimeError("upstream source changed since original v1 protocol")
    r=inspect(a.mgf)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"metadata_diagnostic.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,indent=2))

if __name__=="__main__":
    main()
