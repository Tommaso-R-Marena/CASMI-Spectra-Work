"""MultiMS2 v2 metadata-only parsing of exact published bracketed CID energies.

v1 was archived as failed; header-only diagnostic independently observed
[20.0]/[40.0]/[60.0]. No spectrum peaks/results were consulted.
The frozen v1 exclusions and all thresholds are reused byte for byte.
"""
from __future__ import annotations
from collections import Counter
import argparse
import json
from pathlib import Path
import re

import audit_multims2_third_source_metadata as original

NEW_EXACT_CID_ENERGY = re.compile(r"^\[(20|40|60)\.0\]$")
OBSERVED_CID = {"[20.0]":9538,"[40.0]":13089,"[60.0]":13046}
OBSERVED_EAD = {"[12.0]":2484,"[16.0]":2760,"[24.0]":2811}
SOURCE_SHA256 = "4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"

def verify_previous_header_diagnostic(mgf):
    if (Path(mgf).stat().st_size!=original.UPSTREAM_BYTES
        or original.git_blob_sha1(mgf)!=original.UPSTREAM_BLOB_SHA1
        or original.digest(mgf)!=SOURCE_SHA256):
        raise RuntimeError("v2 input differs from original v1 header-audited source")
    cid=Counter()
    ead=Counter()
    for _,hdr in original.stream_headers(mgf):
        f=(hdr.get("FRAGMENTATION_METHOD") or "").upper()
        v=str(hdr.get("COLLISION_ENERGY"))
        if f=="CID":
            cid[v]+=1
        elif f=="EAD":
            ead[v]+=1
        else:
            raise RuntimeError("unanticipated fragmentation family in the pinned source")
    if dict(cid)!=OBSERVED_CID or dict(ead)!=OBSERVED_EAD:
        raise RuntimeError("frozen header-only CE representation inventory changed")
    return {"CID":dict(cid),"EAD":dict(ead)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--discovery",type=Path,required=True)
    p.add_argument("--prior-inventory",type=Path,required=True)
    p.add_argument("--massbank-manifest",type=Path,required=True)
    p.add_argument("--expolib-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    inventory=verify_previous_header_diagnostic(a.mgf)
    # Only permitted v2 parser modification: replace the v1 CE regex in the
    # imported module. Its original eligible(), select_triplets(), blocked
    # identities, and all feasibility thresholds are unchanged.
    original.ENERGY=NEW_EXACT_CID_ENERGY
    # Reuse the v1 immutable scientific selection after the CE syntax repair.
    # It reads args from the same command line (identical CLI).
    original.main()
    target=a.out/"feasibility.json"
    result=json.loads(target.read_text())
    if result.get("status")!="MULTIMS2_CID_20_40_60EV_METADATA_ONLY_V1":
        raise RuntimeError("unexpected v1 source meta status")
    if result["total_source_spectra_records"]!=43728:
        raise RuntimeError("metadata source record count changed")
    result["status"]="MULTIMS2_CID_20_40_60EV_METADATA_ONLY_V2_BRACKETED_CE"
    result["metadata_parser_protocol"]="V2: only [20.0], [40.0], [60.0] CE header values accepted; all v1 criteria unmodified"
    result["original_v1_failed_metadata_run"]=38969490208
    result["v1_failure_diagnostic_run"]=38969970105
    result["verified_original_header_inventory"]=inventory
    result["target_peak_locations_or_intensities_read"]=False
    result["outcome_scores_opened"]=False
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"v2_status":result["status"],"gate_pass":result["gate_pass"],
                      "coverage":result["coverage"]},indent=2))

if __name__=="__main__":
    main()
