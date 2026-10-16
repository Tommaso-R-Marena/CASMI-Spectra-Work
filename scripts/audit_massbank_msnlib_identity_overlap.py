"""Post-release metadata overlap audit of the frozen MassBank triplet population.

Never filters or re-ranks any MassBank observation. Exact full InChIKey is a
structure-identity overlap diagnostic, *not* proof of shared acquisition.
This analysis is POST-HOC to the frozen MassBank efficacy result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl

PREVIOUSLY_EVALUATED = (
    "mcebio_neg", "mcescaf_pos", "enammol_pos", "enamdisc_neg",
)


def audit_overlap(manifest, library_keys, *, discovery_keys):
    """Analyze fixed, unfiltered query manifest against external identity inventories."""
    if not manifest:
        raise ValueError("empty manifest")
    if set(library_keys) != set(PREVIOUSLY_EVALUATED):
        raise ValueError("must audit all four previously evaluated MSnLib cohorts")
    if len({r["query_path"] for r in manifest}) != len(manifest):
        raise ValueError("duplicate MassBank query accession in manifest")
    all_keys=set().union(*(set(x) for x in library_keys.values()))
    if any(r["inchikey"].upper() in discovery_keys for r in manifest):
        raise ValueError("discovery source leaked through the original gate")

    structures={r["inchikey"].upper() for r in manifest}
    overlap=structures & all_keys
    query_overlap=[r for r in manifest if r["inchikey"].upper() in overlap]
    source_groups=defaultdict(list)
    for row in manifest:
        source_groups[(row["contributor"],row["instrument_type"])].append(row)
    by_cohort={}
    for cohort in PREVIOUSLY_EVALUATED:
        keys=set(library_keys[cohort])
        shared=structures & keys
        by_cohort[cohort]={
            "library_unique_full_inchikeys":len(keys),
            "shared_massbank_structures":len(shared),
            "shared_massbank_query_triplets":sum(r["inchikey"].upper() in shared for r in manifest),
        }
    groups=[]
    for (source,inst),rows in sorted(source_groups.items()):
        count=sum(r["inchikey"].upper() in overlap for r in rows)
        groups.append({
            "contributor":source,"instrument_type":inst,
            "n_query_triplets":len(rows),
            "n_unique_structures":len({r["inchikey"] for r in rows}),
            "previous_msnlib_overlap_queries":count,
            "previous_msnlib_overlap_structures":len({r["inchikey"] for r in rows if r["inchikey"].upper() in overlap}),
            "previous_msnlib_nonoverlap_queries":len(rows)-count,
        })
    return {
        "status":"POSTHOC_MASSBANK_FULL_INCHIKEY_SOURCE_OVERLAP",
        "n_query_triplets":len(manifest),
        "n_structures":len(structures),
        "discovery_excluded_structures_reappearing":len(structures & discovery_keys),
        "n_overlapping_prior_msnlib_structures":len(overlap),
        "n_nonoverlapping_prior_msnlib_structures":len(structures-overlap),
        "n_overlapping_prior_msnlib_queries":len(query_overlap),
        "n_nonoverlapping_prior_msnlib_queries":len(manifest)-len(query_overlap),
        "by_previous_evaluated_cohort":by_cohort,
        "by_massbank_contributor_instrument":groups,
        "model_training_or_selection":False,
        "retrospective_filtering_of_primary_scores":False,
        "limitations":"Full-InChIKey overlap identifies same chemical structures, not identical acquisition; non-overlap does not prove independence of instrumentation, preprocessing, other databases, or chemical families.",
    }


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as fp:
        for chunk in iter(lambda:fp.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feasibility-dir",type=Path,required=True)
    ap.add_argument("--discovery",type=Path,required=True)
    ap.add_argument("--evaluated",action="append",required=True,
                    help="cohort=path: one argument for each of the four previously evaluated cohorts")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    gate=json.loads((args.feasibility_dir/"feasibility.json").read_text())
    if gate.get("status")!="EXTERNAL_MASSBANK_2026_03_FEASIBILITY_ONLY" or gate.get("gate_pass") is not True:
        raise ValueError("frozen metadata feasibility gate must have passed")
    manifest=[json.loads(x) for x in
              (args.feasibility_dir/"label_free_triplets.jsonl").read_text().splitlines()
              if x.strip()]
    # Exact frozen cohort, independently verified in scoring run 38918658343.
    expected={"query_triplets":2138,"structures":715,"formula_blocks":613,
              "offcenter_triplets":259,"non_orbitrap_triplets":1853,
              "non_orbitrap_structures":647}
    if any(gate["coverage"].get(k)!=v for k,v in expected.items()):
        raise ValueError("post-hoc provenance audit input cohort changed")
    if len(manifest)!=gate["coverage"]["query_triplets"]:
        raise ValueError("manifest count mismatch")
    if sha256(args.discovery)!=gate["discovery_msnlib_sha256"]:
        raise ValueError("discovery source SHA256 mismatch")
    discovery={r.full_inchikey.upper() for r in load_msnlib_jsonl(args.discovery)}
    libraries={}
    file_info={}
    for arg in args.evaluated:
        if "=" not in arg:
            raise ValueError("invalid evaluated source argument")
        cohort,raw=arg.split("=",1)
        if cohort in libraries or cohort not in PREVIOUSLY_EVALUATED:
            raise ValueError("unknown/duplicate evaluation cohort")
        path=Path(raw)
        libraries[cohort]={r.full_inchikey.upper() for r in load_msnlib_jsonl(path)}
        file_info[cohort]={"filename":path.name,"sha256":sha256(path)}
    result=audit_overlap(manifest,libraries,discovery_keys=discovery)
    result["frozen_massbank_source_sha256"]=gate["massbank_archive_sha256"]
    result["discovery_source_sha256"]=gate["discovery_msnlib_sha256"]
    result["previous_msnlib_sources"]=file_info
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"source_overlap.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
