from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENERGIES = (10, 20, 40)


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name!r} under {root}, found {len(hits)}")
    return hits[0]


def _quantiles(values):
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return {}
    return {
        "min": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit CFM candidate coverage for the untouched NIST20 Outlier benchmark")
    p.add_argument("nist_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    spec = pd.read_pickle(_find_one(args.nist_root, "spec_df.pkl"))
    cand = pd.read_pickle(_find_one(args.nist_root, "cand_df.pkl"))
    cfm_path = _find_one(args.cfm_root, "pcasmi_spec_df.pkl")
    cfm = pd.read_pickle(cfm_path)

    required_spec = {"spec_id", "mol_id", "prec_type"}
    required_cand = {"query_mol_id", "candidate_mol_id"}
    required_cfm = {"mol_id", "ace", "prec_type"}
    if not required_spec.issubset(spec.columns):
        raise RuntimeError(f"spec_df missing {sorted(required_spec - set(spec.columns))}")
    if not required_cand.issubset(cand.columns):
        raise RuntimeError(f"cand_df missing {sorted(required_cand - set(cand.columns))}")
    if not required_cfm.issubset(cfm.columns):
        raise RuntimeError(f"CFM table missing {sorted(required_cfm - set(cfm.columns))}")

    pos = spec[spec["prec_type"] == "[M+H]+"].copy()
    cfm_pos = cfm[(cfm["prec_type"] == "[M+H]+") & cfm["ace"].isin(ENERGIES)].copy()
    energy_sets = cfm_pos.groupby("mol_id")["ace"].agg(lambda x: tuple(sorted(set(int(v) for v in x))))
    complete = set(int(mid) for mid, es in energy_sets.items() if tuple(es) == ENERGIES)

    by_query = cand.groupby("query_mol_id")["candidate_mol_id"].agg(
        lambda s: tuple(sorted(set(int(x) for x in s)))
    )
    per = []
    for r in pos.itertuples(index=False):
        truth = int(r.mol_id)
        original = tuple(int(x) for x in by_query.get(truth, ()))
        covered = sum(int(x in complete) for x in original)
        frac = covered / len(original) if original else 0.0
        per.append({
            "spec_id": int(r.spec_id),
            "truth": truth,
            "candidate_count": int(len(original)),
            "covered": int(covered),
            "coverage_fraction": float(frac),
            "truth_complete": bool(truth in complete),
        })

    truth_ids = set(int(x) for x in pos["mol_id"])
    candidate_ids = set(int(x) for x in cand["candidate_mol_id"])
    fractions = [x["coverage_fraction"] for x in per]
    counts = [x["candidate_count"] for x in per]
    thresholds = {
        str(t): int(sum(x["truth_complete"] and x["coverage_fraction"] >= t - 1e-15 for x in per))
        for t in (1.0, 0.99, 0.95, 0.90)
    }

    summary = {
        "label": "NIST20 Outlier / MassFormer pCASMI",
        "prediction_file": cfm_path.name,
        "protocol": "positive [M+H]+; complete CFM means exactly 10/20/40 eV; no ranking outcomes computed",
        "nist20": {
            "all_spectra": int(len(spec)),
            "positive_mh_spectra": int(len(pos)),
            "unique_positive_truths": int(len(truth_ids)),
            "candidate_links": int(len(cand)),
            "candidate_molecules": int(len(candidate_ids)),
        },
        "cfm": {
            "rows_total": int(len(cfm)),
            "positive_supported_energy_rows": int(len(cfm_pos)),
            "complete_molecules": int(len(complete)),
            "candidate_molecule_coverage": float(len(candidate_ids & complete) / len(candidate_ids)) if candidate_ids else 0.0,
        },
        "truth_complete_fraction": float(sum(t in complete for t in truth_ids) / len(truth_ids)) if truth_ids else 0.0,
        "threshold_counts": thresholds,
        "coverage_quantiles": _quantiles(fractions),
        "candidate_count_quantiles": _quantiles(counts),
        "per_spectrum": per,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "label", "prediction_file", "nist20", "cfm", "truth_complete_fraction", "threshold_counts", "coverage_quantiles", "candidate_count_quantiles"
    )}, indent=2))


if __name__ == "__main__":
    main()
