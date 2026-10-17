"""Positive-only metadata retention after the documented V1 two-polarity fail.

NO spectral fragment lines, locations, or intensities opened. This does not
rewrite the original V1 as success. Before first-ever outcome scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from audit_harmonized2026_crossprovider_gate import SHA256,MD5,IK

MINIMA={"positive_hard_queries":3000,"full_structures":3000,
        "distinct_formulas":500}
ORIGINAL_RUN=39055589643
EXPECTED=4398


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def evaluate(v1_dir:Path,out:Path):
    f=v1_dir/"feasibility.json"
    q=v1_dir/"hard_queries.jsonl"
    r=v1_dir/"reference_manifest.jsonl"
    old=json.loads(f.read_text())
    query=rows(q);refs=rows(r)
    counts=old.get("coverage",{})
    if (old.get("status")!="HARMONIZED2026_CROSS_PROVIDER_NATIVE_HEADER_GATE_V1"
        or old.get("gate_pass") is not False
        or old.get("source_sha256")!=SHA256 or old.get("source_md5")!=MD5
        or old.get("query_peak_locations_or_intensities_opened") is not False
        or old.get("gnps_reference_peaks_opened") is not False
        or counts.get("hard_negative",0)!=0
        or counts.get("hard_positive")!=EXPECTED
        or counts.get("hard_queries")!=EXPECTED
        or counts.get("cross_provider_structures")!=6237
        or len(query)!=EXPECTED or len(refs)!=119863):
        raise RuntimeError("failed native metadata-only V1 original cohort missing or altered")
    by_idx={x["index"]:x for x in refs}
    if len(by_idx)!=len(refs):raise RuntimeError("duplicate frozen GNPS reference index")
    for qrow in query:
        if (qrow["mode"]!="positive" or qrow["source"]!="MASSSPECGYM"
            or not IK.fullmatch(qrow["inchikey"])
            or qrow["candidate_structures"]<2
            or qrow["index"] in by_idx or
            len(qrow["selected_reference_indices"])<2):
            raise RuntimeError("original source positive-only cohort altered")
        pool=[by_idx[i] for i in qrow["selected_reference_indices"]]
        if (qrow["inchikey"] not in {x["inchikey"] for x in pool}
            or qrow["candidate_structures"]!=len({x["inchikey"] for x in pool})):
            raise RuntimeError("source candidate identities no longer match frozen query")
    ids={x["inchikey"] for x in query}
    formulas={x["formula"] for x in query}
    info={
        "positive_hard_queries":len(query),
        "full_structures":len(ids),
        "distinct_formulas":len(formulas),
        "original_v1_negative_hard_queries":0,
        "original_v1_matching_structures_total":counts["cross_provider_structures"],
        "retained_fraction_of_v1_positive_hard_queries":1.,
    }
    passed=all(info[k]>=v for k,v in MINIMA.items())
    result={
        "status":"HARMONIZED2026_POSITIVE_ONLY_NATIVE_COHORT_V2",
        "gate_pass":passed,
        "source_sha256":SHA256,"source_md5":MD5,
        "original_v1_workflow_run":ORIGINAL_RUN,
        "original_v1_two_polarity_scientific_pass":False,
        "original_v1_feasibility_sha256":digest(f),
        "original_v1_queries_sha256":digest(q),
        "original_v1_reference_sha256":digest(r),
        "coverage":info,"minimum":MINIMA,
        "source_query_spectra_opened":False,
        "source_candidate_spectra_opened":False,
        "analysis_class":"V2 positive-only protocol after V1 header-only failure but before any target/ref spectral scores",
        "preserves_all_original_positive_queries":True,
        "limitation":"ZERO negative-ion candidate tasks; no two-polarity generality, no independent physical lab provenance.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"feasibility.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v1-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(a.v1_dir,a.out),indent=2))

if __name__=="__main__":main()
