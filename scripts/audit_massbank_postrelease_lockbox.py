from __future__ import annotations

import argparse
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
    formula: str
    inchikey: str
    connectivity_key: str
    ms_type: str
    ion_mode: str
    precursor_type: str
    collision_energy_ev: float
    instrument: str
    instrument_type: str
    date: str


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
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    energy = _parse_energy(lines)
    inchikey = _parse_inchikey(lines)
    formula = _value(lines, "CH$FORMULA:")
    accession = _value(lines, "ACCESSION:")
    if energy is None or not inchikey or not formula or not accession:
        return None
    return Record(
        path=str(path.relative_to(root)).replace("\\", "/"),
        accession=accession,
        formula=formula,
        inchikey=inchikey,
        connectivity_key=inchikey.split("-")[0],
        ms_type=_value(lines, "AC$MASS_SPECTROMETRY: MS_TYPE"),
        ion_mode=_value(lines, "AC$MASS_SPECTROMETRY: ION_MODE"),
        precursor_type=_value(lines, "MS$FOCUSED_ION: PRECURSOR_TYPE"),
        collision_energy_ev=float(energy),
        instrument=_value(lines, "AC$INSTRUMENT:"),
        instrument_type=_value(lines, "AC$INSTRUMENT_TYPE:"),
        date=_value(lines, "DATE:"),
    )


def _in_scope(r: Record) -> bool:
    return r.ms_type.upper() == "MS2" and r.ion_mode.upper() == "POSITIVE" and r.precursor_type.replace(" ", "") == "[M+H]+"


def _is_energy(x: float, target: float) -> bool:
    return abs(float(x) - float(target)) <= ENERGY_TOL


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    xs = sorted(values)
    def q(p: float) -> float:
        if len(xs) == 1:
            return float(xs[0])
        z = p * (len(xs) - 1); lo = int(math.floor(z)); hi = int(math.ceil(z))
        if lo == hi:
            return float(xs[lo])
        a = z - lo
        return float((1-a)*xs[lo] + a*xs[hi])
    return {"min": xs[0], "q25": q(.25), "median": q(.5), "q75": q(.75), "max": xs[-1], "mean": float(sum(xs)/len(xs))}


def _tree_hash(paths: list[Path], root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: str(p.relative_to(root))):
        rel = str(p.relative_to(root)).replace("\\", "/")
        h.update(rel.encode()); h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode()); h.update(b"\n")
    return h.hexdigest()


def build_audit(current_root: Path, baseline_root: Path, added_paths: set[str], *, current_sha: str, baseline_sha: str) -> tuple[dict, list[dict]]:
    current_paths = sorted(current_root.rglob("*.txt"))
    baseline_paths = sorted(baseline_root.rglob("*.txt"))
    current_records = [r for p in current_paths if (r := parse_record(p, current_root)) is not None and _in_scope(r)]
    baseline_records = [r for p in baseline_paths if (r := parse_record(p, baseline_root)) is not None and _in_scope(r)]
    baseline_structures = {r.connectivity_key for r in baseline_records}

    # Strict temporal query provenance: only paths classified A(dded) by git with rename detection enabled.
    added_records = [r for r in current_records if r.path in added_paths]
    new_query_records = [r for r in added_records if _is_energy(r.collision_energy_ev, QUERY_ENERGY)]

    # Current pinned candidate library; references never use the held-out 30 eV query energy.
    by_structure: dict[str, list[Record]] = defaultdict(list)
    for r in current_records:
        by_structure[r.connectivity_key].append(r)

    complete_refs: dict[str, dict[float, Record]] = {}
    structure_formula: dict[str, str] = {}
    for key, rows in by_structure.items():
        formulas = sorted(set(r.formula for r in rows))
        if len(formulas) != 1:
            continue
        structure_formula[key] = formulas[0]
        refs: dict[float, Record] = {}
        for e in REFERENCE_ENERGIES:
            cands = sorted((r for r in rows if _is_energy(r.collision_energy_ev, e)), key=lambda r: (r.accession, r.path))
            if cands:
                refs[e] = cands[0]
        if len(refs) == len(REFERENCE_ENERGIES):
            complete_refs[key] = refs

    by_formula: dict[str, list[str]] = defaultdict(list)
    for key in complete_refs:
        by_formula[structure_formula[key]].append(key)
    for formula in by_formula:
        by_formula[formula] = sorted(set(by_formula[formula]))

    query_by_structure: dict[str, Record] = {}
    for r in sorted(new_query_records, key=lambda r: (r.connectivity_key, r.accession, r.path)):
        query_by_structure.setdefault(r.connectivity_key, r)

    manifest: list[dict] = []
    for key, q in sorted(query_by_structure.items()):
        if key not in complete_refs or key not in structure_formula:
            continue
        formula = structure_formula[key]
        pool = by_formula[formula]
        if len(pool) < 2:
            continue
        refs = complete_refs[key]
        manifest.append({
            "formula": formula,
            "truth_connectivity_key": key,
            "query_accession": q.accession,
            "query_path": q.path,
            "query_energy_ev": QUERY_ENERGY,
            "structure_novel_at_baseline": key not in baseline_structures,
            "candidate_count": len(pool),
            "candidate_connectivity_keys": ";".join(pool),
            "truth_ref_10_path": refs[10.0].path,
            "truth_ref_20_path": refs[20.0].path,
            "truth_ref_40_path": refs[40.0].path,
            "instrument": q.instrument,
            "instrument_type": q.instrument_type,
            "date": q.date,
        })

    def cohort(rows: list[dict]) -> dict:
        pools = [int(r["candidate_count"]) for r in rows]
        return {
            "queries": len(rows),
            "formula_blocks": len({r["formula"] for r in rows}),
            "pool_size": _quantiles(pools),
            "queries_by_min_pool_size": {k: sum(n >= t for n in pools) for k, t in (("ge_2",2),("ge_5",5),("ge_10",10),("ge_20",20))},
            "formula_blocks_by_min_pool_size": {k: len({r["formula"] for r in rows if int(r["candidate_count"]) >= t}) for k, t in (("ge_2",2),("ge_5",5),("ge_10",10),("ge_20",20))},
        }

    novel_rows = [r for r in manifest if bool(r["structure_novel_at_baseline"])]
    directory_counts = Counter(r.path.split("/", 1)[0] for r in added_records)
    query_directory_counts = Counter(r["query_path"].split("/", 1)[0] for r in manifest)
    summary = {
        "status": "MASSBANK_POSTRELEASE_LABEL_FREE_FEASIBILITY_AUDIT",
        "dataset": {
            "baseline_sha": baseline_sha,
            "current_sha": current_sha,
            "baseline_txt_files": len(baseline_paths),
            "current_txt_files": len(current_paths),
            "strict_git_added_txt_paths": len(added_paths),
            "current_tree_sha256": _tree_hash(current_paths, current_root),
            "baseline_tree_sha256": _tree_hash(baseline_paths, baseline_root),
            "in_scope_current_records": len(current_records),
            "in_scope_baseline_records": len(baseline_records),
            "strict_git_added_in_scope_records": len(added_records),
            "added_directory_counts": dict(sorted(directory_counts.items())),
        },
        "design": {
            "query_rule": "one deterministic 30 eV query per 2D structure from paths classified git-added after baseline",
            "candidate_rule": "all current-pinned 2D structures sharing exact molecular formula with complete 10/20/40 eV references",
            "reference_energies_ev": list(REFERENCE_ENERGIES),
            "query_energy_ev": QUERY_ENERGY,
            "scope": "MS2, POSITIVE, [M+H]+",
            "structure_identity": "first InChIKey block (2D connectivity)",
            "ranking_scores_computed": False,
            "hierarchy": ["novel_structure", "added_spectrum"],
        },
        "coverage": {
            "current_complete_reference_structures": len(complete_refs),
            "new_30ev_query_structures": len(query_by_structure),
            "eligible_novel_structure": cohort(novel_rows),
            "eligible_added_spectrum": cohort(manifest),
            "eligible_query_directory_counts": dict(sorted(query_directory_counts.items())),
        },
        "guardrails": {
            "label_free": True,
            "selector_frozen": {
                "frozen_parent_sha": "0006c7f81c7dfac53da48690ca0fe627108b9767",
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
    p = argparse.ArgumentParser()
    p.add_argument("current_root", type=Path)
    p.add_argument("baseline_root", type=Path)
    p.add_argument("--added-paths-file", type=Path, required=True)
    p.add_argument("--current-sha", required=True)
    p.add_argument("--baseline-sha", required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    args = p.parse_args()
    added_paths = {line.strip().replace("\\", "/") for line in args.added_paths_file.read_text().splitlines() if line.strip().endswith(".txt")}
    summary, manifest = build_audit(args.current_root, args.baseline_root, added_paths, current_sha=args.current_sha, baseline_sha=args.baseline_sha)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    import csv
    fields = list(manifest[0]) if manifest else ["formula","truth_connectivity_key","query_accession","query_path","query_energy_ev","structure_novel_at_baseline","candidate_count","candidate_connectivity_keys","truth_ref_10_path","truth_ref_20_path","truth_ref_40_path","instrument","instrument_type","date"]
    with args.out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
