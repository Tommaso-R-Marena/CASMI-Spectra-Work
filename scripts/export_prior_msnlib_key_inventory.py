"""Identity-only inventory for retrospective MassBank overlap sensitivity.

This code never fits or scores spectra. Exactly four previously evaluated
MSnLib libraries must match their archived cryptographic digests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_MD5 = {
    "mcebio_neg": "f0b8490cd619c9ea7fdea8de95393c28",
    "mcescaf_pos": "3ba52914a2f27239c5278c7d96888ecf",
    "enammol_pos": "ed6db189b4b5e1fc6a8cf3df211dbbe8",
    "enamdisc_neg": "ad9c6e971a1902c091aeb4d555ca8db6",
}

def inventory(libraries):
    if set(libraries) != set(EXPECTED_MD5):
        raise ValueError("exactly four original MSnLib evaluation cohorts are required")
    union = set()
    sources = {}
    for cohort in sorted(libraries):
        keys = {str(x).upper() for x in libraries[cohort]}
        union.update(keys)
        sources[cohort] = {"n_distinct_full_inchikeys": len(keys)}
    return {
        "status": "PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC",
        "scope": "Four previously scored MSnLib cohorts; MCEBIO+ discovery separately excluded by original MassBank eligibility gate",
        "sources": sources,
        "all_prior_evaluated_full_inchikeys": sorted(union),
        "spectra_scored": False,
        "frozen_model_changed": False,
    }

def main():
    from qfd.msnlib import load_msnlib_jsonl
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True, help="cohort=path, all four")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    paths = {}
    for arg in args.source:
        if "=" not in arg:
            raise ValueError("source must use cohort=path")
        cohort, path = arg.split("=", 1)
        if cohort not in EXPECTED_MD5 or cohort in paths:
            raise ValueError("unknown or duplicate cohort")
        paths[cohort] = Path(path)
    if set(paths) != set(EXPECTED_MD5):
        raise ValueError("incomplete four-cohort source inventory")
    libraries, provenance = {}, {}
    for cohort, path in paths.items():
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 != EXPECTED_MD5[cohort]:
            raise ValueError(f"pinned MD5 mismatch in {cohort}")
        libraries[cohort] = {r.full_inchikey.upper() for r in load_msnlib_jsonl(path)}
        provenance[cohort] = {
            "filename": path.name,
            "md5": md5,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    out = inventory(libraries)
    for cohort in out["sources"]:
        out["sources"][cohort].update(provenance[cohort])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "status": out["status"],
        "n_union_full_keys": len(out["all_prior_evaluated_full_inchikeys"]),
        "sources": {c: s["n_distinct_full_inchikeys"] for c,s in out["sources"].items()},
    }, indent=2))

if __name__ == "__main__":
    main()
