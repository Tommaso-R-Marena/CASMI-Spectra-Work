"""Metadata-only locked exact-isomer CID/EAD retrieval cohort.

Reads ONLY previously frozen query-index manifests and source provenance, not
any peaks, scores or outcomes. The CID/EAD overlap was already investigated in
prior experiments, so this is an exploratory follow-on, NOT a pristine lockbox.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

CID_SHA = "5345f532ded9baa4134d38b840a6e05a1b1f85b3cf19923569ea80d96f46289d"
SOURCE_SHA = "4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"
MIN_GROUPS = 8
MIN_QUERIES = 20
JOIN_FIELDS = ("inchikey", "formula", "adduct", "ion_mode", "source_dataset_id")
RANK_FIELDS = ("formula", "adduct", "ion_mode", "source_dataset_id")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def unique(records: list[dict], modality: str) -> dict[tuple, dict]:
    result = {}
    used = set()
    for r in records:
        if r.get("fragmentation") != modality:
            raise RuntimeError("unexpected fragmentation in " + modality)
        key = tuple(str(r[field]) for field in JOIN_FIELDS)
        if key in result:
            raise RuntimeError("ambiguous identity/adduct/dataset duplicate")
        indices = [r[f"{role}_index"] for role in ("low", "query", "high")]
        if len(set(indices)) != 3 or any(i in used for i in indices):
            raise RuntimeError("duplicate or reused source scan in " + modality)
        used.update(indices)
        if not r["formula"] or not r["inchikey"] or not r["adduct"] or not r["ion_mode"]:
            raise RuntimeError("incomplete chemical identity")
        result[key] = r
    return result


def build(cid_records: list[dict], ead_records: list[dict]) -> tuple[list[dict], dict]:
    cid, ead = unique(cid_records, "CID"), unique(ead_records, "EAD")
    candidates = []
    for key in sorted(cid.keys() & ead.keys()):
        c, e = cid[key], ead[key]
        if c["ion_mode"] != "POSITIVE" or e["ion_mode"] != "POSITIVE":
            raise RuntimeError("V2 is exclusively positive-ion EAD")
        if ((c["low_energy_eV"], c["query_energy_eV"], c["high_energy_eV"]) != (20, 40, 60)
            or (e["low_energy_eV"], e["query_energy_eV"], e["high_energy_eV"]) != (12, 16, 24)
            or abs(float(c["t"]) - .5) > 1e-14
            or abs(float(e["t"]) - 1/3) > 1e-14):
            raise RuntimeError("unrecognized source energy coordinates")
        candidates.append({
            **dict(zip(JOIN_FIELDS, key)),
            "cid_indices": {role: c[role + "_index"] for role in ("low", "query", "high")},
            "ead_indices": {role: e[role + "_index"] for role in ("low", "query", "high")},
        })
    by_group = defaultdict(list)
    for c in candidates:
        by_group[tuple(c[f] for f in RANK_FIELDS)].append(c)
    groups = []
    for group_key, members in sorted(by_group.items()):
        # Only exact-formula, same-adduct, same-polarity and same-source isomers;
        # distinct full InChIKeys are never collapsed to 14-character scaffolds.
        if len(members) < 2:
            continue
        group_id = "|".join(group_key)
        for m in members:
            m["group_id"] = group_id
            m["candidate_count"] = len(members)
        groups.extend(sorted(members, key=lambda x: x["inchikey"]))
    n_groups = sum(len(m) >= 2 for m in by_group.values())
    coverage = {
        "original_cid": len(cid_records),
        "original_ead_positive": len(ead_records),
        "exact_identity_matched": len(candidates),
        "same_formula_isomer_groups": n_groups,
        "eligible_isomer_queries": len(groups),
        "eligible_distinct_full_inchikeys": len({r["inchikey"] for r in groups}),
        "distinct_formulas": len({r["formula"] for r in groups}),
        "all_matched_singletons_excluded_by_label_free_rule": len(candidates) - len(groups),
    }
    return groups, coverage


def evaluate(cid_manifest: Path, ead_manifest: Path, ead_gate: Path, out: Path) -> dict:
    if sha(cid_manifest) != CID_SHA:
        raise RuntimeError("original CID metadata manifest SHA changed")
    v2 = json.loads(ead_gate.read_text())
    if (v2.get("status") != "EAD_POSITIVE_ONLY_METADATA_RETENTION_V2"
        or v2.get("gate_pass") is not True
        or v2.get("original_failed_v1_gate_pass") is not False
        or v2.get("new_v2_target_intensity_opened") is not False
        or v2.get("new_v2_outcome_scores_opened") is not False
        or v2.get("source_sha256") != SOURCE_SHA):
        raise RuntimeError("unaltered EAD metadata-only V2 is required")
    cid, ead = rows(cid_manifest), rows(ead_manifest)
    if len(cid) != 1781 or len(ead) != 694 or v2["coverage"]["triplets"] != len(ead):
        raise RuntimeError("manifest count differs from frozen original cohorts")
    groups, coverage = build(cid, ead)
    passed = coverage["same_formula_isomer_groups"] >= MIN_GROUPS and coverage["eligible_isomer_queries"] >= MIN_QUERIES
    result = {
        "status": "MULTIMS2_DUAL_MODAL_ISOMER_IDENTIFICATION_METADATA_V1",
        "gate_pass": passed,
        "thresholds": {"isomer_groups": MIN_GROUPS, "isomer_queries": MIN_QUERIES},
        "coverage": coverage,
        "cid_manifest_sha256": sha(cid_manifest),
        "ead_manifest_sha256": sha(ead_manifest),
        "ead_v2_gate_sha256": sha(ead_gate),
        "source_sha256": SOURCE_SHA,
        "target_peak_positions_or_intensities_opened": False,
        "candidate_selection_uses_outcome_scores": False,
        "analysis_status": "EXPLORATORY: earlier MultiMS2 interpolation outcomes and CID/EAD reversal were already known",
        "scientific_gate_failure_means": "Do not open target peaks or report an identification efficacy estimate.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "feasibility.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (out / "eligible_isomer_queries.jsonl").open("w") as f:
        for r in groups:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    for flag in ("cid_manifest", "ead_manifest", "ead_gate", "out"):
        p.add_argument("--" + flag.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    result = evaluate(a.cid_manifest, a.ead_manifest, a.ead_gate, a.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
