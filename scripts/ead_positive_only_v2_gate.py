"""EAD positive-ion V2: protocol change AFTER V1 negative-ion metadata failure,
BEFORE any EAD target peak/scoring. Retain ALL V1 eligible positive-ion queries.
No re-selection of spectra, predictors, m/z tolerance or outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

OLD_STATUS="MULTIMS2_EAD_12_16_24EV_HEADER_ONLY_GATE_V1"
STATUS="EAD_POSITIVE_ONLY_METADATA_RETENTION_V2"
MINIMA={"triplets":100,"structures":80,"formula_blocks":60}
EXPECTED_ALL=695
EXPECTED_POSITIVE=694
EXPECTED_NEGATIVE=1
SOURCE_SHA="4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(original_dir,out):
    f=original_dir/"feasibility.json"
    m=original_dir/"frozen_triplets.jsonl"
    old=json.loads(f.read_text())
    records=[json.loads(x) for x in m.read_text().splitlines() if x.strip()]
    cov=old["coverage"]
    if (old.get("status")!=OLD_STATUS or old.get("gate_pass") is not False or
        old.get("16eV_target_peak_positions_or_intensities_read") is not False or
        old.get("target_scores_or_predictor_outcomes_opened") is not False or
        old.get("source_sha256")!=SOURCE_SHA or
        len(records)!=EXPECTED_ALL or cov.get("triplets")!=EXPECTED_ALL or
        cov.get("positive_triplets")!=EXPECTED_POSITIVE or
        cov.get("negative_triplets")!=EXPECTED_NEGATIVE):
        raise RuntimeError("V1 scientifically failed header gate is missing or altered")
    if len({r["query_index"] for r in records})!=len(records):
        raise RuntimeError("duplicate original EAD scan")
    selected=[r for r in records if r["ion_mode"]=="POSITIVE"]
    rejected=[r for r in records if r["ion_mode"]=="NEGATIVE"]
    if len(selected)!=EXPECTED_POSITIVE or len(rejected)!=EXPECTED_NEGATIVE:
        raise RuntimeError("polarity partition differs from original gate")
    if any(r.get("fragmentation")!="EAD" or r.get("t")!=1/3 or
           (r.get("low_energy_eV"),r.get("query_energy_eV"),r.get("high_energy_eV"))!=(12,16,24)
           for r in selected):
        raise RuntimeError("unexpected EAD energy or interpolation coordinate")
    info={
        "triplets":len(selected),
        "structures":len({r["inchikey"] for r in selected}),
        "formula_blocks":len({r["formula"] for r in selected}),
        "positive_triplets":len(selected),
        "negative_triplets":0,
        "excluded_original_negative_singleton":len(rejected),
        "source_dataset_ids":dict(Counter(r["source_dataset_id"] for r in selected)),
    }
    passed=all(info[k]>=minimum for k,minimum in MINIMA.items())
    result={
        "status":STATUS,
        "source_sha256":SOURCE_SHA,
        "gate_pass":passed,
        "pre_score_meta_protocol":"V2: positive-ion ONLY, fixed after metadata-only V1 failure, before outcomes",
        "original_failed_v1_run":38985385615,
        "original_failed_v1_gate_pass":False,
        "original_v1_gate_sha256":sha(f),
        "original_v1_manifest_sha256":sha(m),
        "original_v1_cohort_triplets":len(records),
        "selection_rule":"retain EVERY original V1 metadata-eligible POSITIVE mode triplet, exclude only 1 negative singleton",
        "coverage":info,
        "new_predeclared_minima":MINIMA,
        "original_v1_peak_intensity_scored":False,
        "new_v2_target_intensity_opened":False,
        "new_v2_outcome_scores_opened":False,
        "limitations":"Cannot support negative-ion EAD conclusions. Same source/lab as MultiMS2 CID, previous CID outcome known, V1 feasibility failed, positive-only protocol is declared AFTER header eligibility but BEFORE EAD target intensities.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"feasibility.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (out/"frozen_triplets.jsonl").open("w") as file:
        for r in selected:
            file.write(json.dumps(r,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.original_dir,a.out)

if __name__=="__main__":
    main()
