from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REFERENCE_ENERGIES = (10.0, 20.0, 40.0)
QUERY_ENERGY = 30.0
ENERGY_TOL = 1e-9


@dataclass(frozen=True)
class Record:
    path: str
    accession: str
    date: str
    formula: str
    inchikey: str
    connectivity_key: str
    instrument: str
    instrument_type: str
    ms_type: str
    ion_mode: str
    precursor_type: str
    collision_energy_ev: float


def _value(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _parse_energy(lines: list[str]) -> float | None:
    prefix = "AC$MASS_SPECTROMETRY: COLLISION_ENERGY"
    for line in lines:
        if line.startswith(prefix):
            m = re.search(r"([-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:eV)?", line[len(prefix):], re.I)
            if m:
                return float(m.group(1))
    return None


def _parse_inchikey(lines: list[str]) -> str:
    prefix = "CH$LINK: INCHIKEY"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip().split()[0]
    return ""


def parse_record(path: Path, root: Path) -> Record | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    energy = _parse_energy(lines)
    inchikey = _parse_inchikey(lines)
    formula = _value(lines, "CH$FORMULA:")
    accession = _value(lines, "ACCESSION:")
    if energy is None or not inchikey or not formula or not accession:
        return None
    return Record(
        path=str(path.relative_to(root)),
        accession=accession,
        date=_value(lines, "DATE:"),
        formula=formula,
        inchikey=inchikey,
        connectivity_key=inchikey.split("-")[0],
        instrument=_value(lines, "AC$INSTRUMENT:"),
        instrument_type=_value(lines, "AC$INSTRUMENT_TYPE:"),
        ms_type=_value(lines, "AC$MASS_SPECTROMETRY: MS_TYPE"),
        ion_mode=_value(lines, "AC$MASS_SPECTROMETRY: ION_MODE"),
        precursor_type=_value(lines, "MS$FOCUSED_ION: PRECURSOR_TYPE"),
        collision_energy_ev=float(energy),
    )


def _is_energy(x: float, target: float) -> bool:
    return abs(float(x) - float(target)) <= ENERGY_TOL


def _tree_sha256(paths: list[Path], root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda q: str(q.relative_to(root))):
        rel = str(p.relative_to(root)).replace("\\", "/")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    xs = sorted(values)
    def q(p: float) -> float:
        if len(xs) == 1:
            return float(xs[0])
        z = p * (len(xs) - 1)
        lo = int(math.floor(z))
        hi = int(math.ceil(z))
        if lo == hi:
            return float(xs[lo])
        a = z - lo
        return float((1.0 - a) * xs[lo] + a * xs[hi])
    return {
        "min": int(xs[0]),
        "q25": q(0.25),
        "median": q(0.50),
        "q75": q(0.75),
        "max": int(xs[-1]),
        "mean": float(sum(xs) / len(xs)),
    }


def build_audit(root: Path, *, dataset_commit: str, baseline_tag: str) -> tuple[dict, list[dict]]:
    paths = sorted(root.rglob("*.txt"))
    parsed = [r for p in paths if (r := parse_record(p, root)) is not None]

    scoped = [
        r for r in parsed
        if r.ms_type.upper() == "MS2"
        and r.ion_mode.upper() == "POSITIVE"
        and r.precursor_type.replace(" ", "") == "[M+H]+"
    ]

    by_structure: dict[str, list[Record]] = defaultdict(list)
    for r in scoped:
        by_structure[r.connectivity_key].append(r)

    complete_refs: dict[str, dict[float, Record]] = {}
    query_record: dict[str, Record] = {}
    structure_formula: dict[str, str] = {}
    structure_inchikey: dict[str, str] = {}

    for key, rows in by_structure.items():
        formulas = sorted(set(r.formula for r in rows))
        if len(formulas) != 1:
            continue
        structure_formula[key] = formulas[0]
        structure_inchikey[key] = sorted(r.inchikey for r in rows)[0]
        refs: dict[float, Record] = {}
        for e in REFERENCE_ENERGIES:
            candidates = sorted((r for r in rows if _is_energy(r.collision_energy_ev, e)), key=lambda r: r.accession)
            if candidates:
                refs[e] = candidates[0]
        if len(refs) == len(REFERENCE_ENERGIES):
            complete_refs[key] = refs
        qs = sorted((r for r in rows if _is_energy(r.collision_energy_ev, QUERY_ENERGY)), key=lambda r: r.accession)
        if qs:
            query_record[key] = qs[0]

    by_formula_candidates: dict[str, list[str]] = defaultdict(list)
    for key in complete_refs:
        by_formula_candidates[structure_formula[key]].append(key)
    for formula in by_formula_candidates:
        by_formula_candidates[formula] = sorted(set(by_formula_candidates[formula]))

    manifest: list[dict] = []
    for truth_key in sorted(set(complete_refs) & set(query_record)):
        formula = structure_formula[truth_key]
        candidates = by_formula_candidates[formula]
        if len(candidates) < 2:
            continue
        q = query_record[truth_key]
        refs = complete_refs[truth_key]
        manifest.append({
            "formula": formula,
            "truth_connectivity_key": truth_key,
            "truth_inchikey": structure_inchikey[truth_key],
            "query_accession": q.accession,
            "query_energy_ev": QUERY_ENERGY,
            "candidate_count": len(candidates),
            "candidate_connectivity_keys": ";".join(candidates),
            "truth_ref_10_accession": refs[10.0].accession,
            "truth_ref_20_accession": refs[20.0].accession,
            "truth_ref_40_accession": refs[40.0].accession,
            "instrument": q.instrument,
            "instrument_type": q.instrument_type,
            "date": q.date,
        })

    pool_sizes = [int(r["candidate_count"]) for r in manifest]
    formulas = [str(r["formula"]) for r in manifest]
    dates = [r.date for r in scoped if r.date]
    energies = Counter(float(r.collision_energy_ev) for r in scoped)

    summary = {
        "status": "MASSBANK_MUI_LABEL_FREE_FEASIBILITY_AUDIT",
        "dataset": {
            "source": "MassBank/MassBank-data MUI directory",
            "dataset_commit": dataset_commit,
            "baseline_tag_without_MUI": baseline_tag,
            "tree_sha256": _tree_sha256(paths, root),
            "record_files": len(paths),
            "parsed_records": len(parsed),
            "scoped_positive_mh_ms2_records": len(scoped),
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
        },
        "predeclared_design": {
            "structure_identity": "first 14 characters of InChIKey (2D connectivity block)",
            "candidate_pool": "all distinct MUI structures with the same molecular formula and complete 10/20/40 eV references",
            "reference_energies_ev": list(REFERENCE_ENERGIES),
            "heldout_query_energy_ev": QUERY_ENERGY,
            "ion_scope": "MS2, POSITIVE, [M+H]+",
            "primary_cohort": "every eligible query with >=2 same-formula candidate structures",
            "prespecified_difficulty_strata": [">=2", ">=5", ">=10", ">=20"],
            "no_ranking_scores_computed": True,
        },
        "coverage": {
            "unique_scoped_2d_structures": len(by_structure),
            "structures_with_complete_10_20_40_refs": len(complete_refs),
            "structures_with_30ev_query": len(query_record),
            "eligible_ambiguous_queries": len(manifest),
            "eligible_formula_groups": len(set(formulas)),
            "pool_size": _quantiles(pool_sizes),
            "queries_by_min_pool_size": {
                "ge_2": sum(n >= 2 for n in pool_sizes),
                "ge_5": sum(n >= 5 for n in pool_sizes),
                "ge_10": sum(n >= 10 for n in pool_sizes),
                "ge_20": sum(n >= 20 for n in pool_sizes),
            },
            "formula_groups_by_min_pool_size": {
                "ge_2": len({r["formula"] for r in manifest if int(r["candidate_count"]) >= 2}),
                "ge_5": len({r["formula"] for r in manifest if int(r["candidate_count"]) >= 5}),
                "ge_10": len({r["formula"] for r in manifest if int(r["candidate_count"]) >= 10}),
                "ge_20": len({r["formula"] for r in manifest if int(r["candidate_count"]) >= 20}),
            },
            "collision_energy_record_counts": {str(k): int(v) for k, v in sorted(energies.items())},
        },
        "guardrails": {
            "feasibility_only": True,
            "truth_labels_used_only_to_define which structure owns a heldout query; no score or rank is computed": True,
            "selector_parameters_remain_frozen": {
                "overlap_threshold": 0.8,
                "protected_k": 1,
                "overlap_k": 10,
                "head_survival_slack": 1,
                "tail_rule": "minimax rank aggregation",
            },
        },
    }
    return summary, manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Label-free feasibility audit for the frozen v5.3 MassBank MUI lockbox")
    p.add_argument("mui_root", type=Path)
    p.add_argument("--dataset-commit", required=True)
    p.add_argument("--baseline-tag", default="2026.03")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    args = p.parse_args()

    summary, manifest = build_audit(
        args.mui_root,
        dataset_commit=args.dataset_commit,
        baseline_tag=args.baseline_tag,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(manifest[0].keys()) if manifest else [
        "formula", "truth_connectivity_key", "truth_inchikey", "query_accession",
        "query_energy_ev", "candidate_count", "candidate_connectivity_keys",
        "truth_ref_10_accession", "truth_ref_20_accession", "truth_ref_40_accession",
        "instrument", "instrument_type", "date",
    ]
    with args.out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
