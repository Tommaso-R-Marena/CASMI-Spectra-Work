from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name!r} under {root}, found {len(hits)}")
    return hits[0]


def _quantiles(values):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {}
    return {
        "min": float(arr.min()),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-ranking CFM coverage audit for the NIST20 Outlier (formerly pCASMI) benchmark"
    )
    p.add_argument("nist_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    spec = pd.read_pickle(_find_one(args.nist_root, "spec_df.pkl"))
    cand = pd.read_pickle(_find_one(args.nist_root, "cand_df.pkl"))
    cfm = pd.read_pickle(_find_one(args.cfm_root, "pcasmi_spec_df.pkl"))

    required_cfm = {"mol_id", "ace", "prec_type", "peaks"}
    required_cand = {"candidate_mol_id", "query_mol_id"}
    required_spec = {"spec_id", "mol_id", "prec_type"}
    if not required_cfm.issubset(cfm.columns):
        raise RuntimeError(f"CFM dataframe missing columns: {sorted(required_cfm - set(cfm.columns))}")
    if not required_cand.issubset(cand.columns):
        raise RuntimeError(f"candidate dataframe missing columns: {sorted(required_cand - set(cand.columns))}")
    if not required_spec.issubset(spec.columns):
        raise RuntimeError(f"spectrum dataframe missing columns: {sorted(required_spec - set(spec.columns))}")

    cfm_pos = cfm[cfm["prec_type"] == "[M+H]+"].copy()
    energy_sets = cfm_pos.groupby("mol_id")["ace"].agg(
        lambda x: tuple(sorted(set(int(v) for v in x)))
    )
    complete_ids = {int(mid) for mid, es in energy_sets.items() if es == (10, 20, 40)}
    any_ids = {int(x) for x in cfm_pos["mol_id"].unique()}

    by_query = cand.groupby("query_mol_id")["candidate_mol_id"].agg(
        lambda s: tuple(sorted(set(int(x) for x in s)))
    )
    positive = spec[spec["prec_type"] == "[M+H]+"].copy()

    per_spectrum = []
    for row in positive.itertuples(index=False):
        truth = int(row.mol_id)
        candidates = tuple(int(x) for x in by_query.get(truth, ()))
        covered = sum(c in complete_ids for c in candidates)
        n = len(candidates)
        frac = covered / n if n else 0.0
        per_spectrum.append({
            "spec_id": int(row.spec_id),
            "query_mol_id": truth,
            "candidate_count": n,
            "covered_candidate_count": covered,
            "candidate_coverage_fraction": float(frac),
            "truth_has_any_cfm": truth in any_ids,
            "truth_has_complete_10_20_40": truth in complete_ids,
            "full_candidate_coverage": bool(n > 0 and covered == n),
        })

    # Repeated spectra for a truth must have the same candidate-pool coverage.
    per_truth: dict[int, dict] = {}
    for r in per_spectrum:
        qid = int(r["query_mol_id"])
        if qid not in per_truth:
            per_truth[qid] = dict(r)
        else:
            for key in (
                "candidate_count",
                "covered_candidate_count",
                "candidate_coverage_fraction",
                "truth_has_complete_10_20_40",
            ):
                if per_truth[qid][key] != r[key]:
                    raise RuntimeError(f"inconsistent repeated-query coverage for mol_id={qid}")

    thresholds = (1.0, 0.999, 0.99, 0.95, 0.90)
    threshold_counts = {
        str(t): int(sum(
            r["truth_has_complete_10_20_40"] and r["candidate_coverage_fraction"] >= t
            for r in per_spectrum
        ))
        for t in thresholds
    }
    all_candidate_ids = {int(x) for x in cand["candidate_mol_id"].unique()}
    ratios = [r["candidate_coverage_fraction"] for r in per_spectrum]
    counts = [r["candidate_count"] for r in per_spectrum]

    summary = {
        "scope": "NIST20 Outlier (formerly pCASMI) positive [M+H]+ queries against the published MassFormer CFM pcasmi prediction table",
        "dataset_role": "coverage/schema audit only; no ranking outcomes computed",
        "policy": {
            "complete_prediction_definition": "candidate has exactly 10, 20, and 40 eV CFM spectra in pcasmi_spec_df.pkl",
            "missing_candidates_never_silently_removed_from_end_to_end_claim": True,
            "future_primary_threshold_must_be_frozen_before_any_nist20_ranking": True,
        },
        "nist20": {
            "all_spectra": int(len(spec)),
            "positive_mh_spectra": int(len(positive)),
            "positive_mh_unique_truth_molecules": int(positive["mol_id"].nunique()),
            "candidate_links_total": int(len(cand)),
            "candidate_molecules_total": int(cand["candidate_mol_id"].nunique()),
        },
        "cfm": {
            "positive_rows": int(len(cfm_pos)),
            "positive_molecules_any_prediction": int(len(any_ids)),
            "positive_molecules_complete_10_20_40": int(len(complete_ids)),
            "original_candidate_molecules_with_complete_predictions": int(len(all_candidate_ids & complete_ids)),
            "original_candidate_molecule_coverage_fraction": (
                float(len(all_candidate_ids & complete_ids) / len(all_candidate_ids))
                if all_candidate_ids else 0.0
            ),
        },
        "positive_spectrum_coverage": {
            "truth_complete_count": int(sum(r["truth_has_complete_10_20_40"] for r in per_spectrum)),
            "truth_complete_fraction": (
                float(np.mean([r["truth_has_complete_10_20_40"] for r in per_spectrum]))
                if per_spectrum else 0.0
            ),
            "counts_truth_complete_at_candidate_coverage_threshold": threshold_counts,
            "candidate_coverage_fraction_quantiles": _quantiles(ratios),
            "candidate_count_quantiles": _quantiles(counts),
        },
        "positive_unique_truth_coverage": {
            "n_truths": int(len(per_truth)),
            "truth_complete_count": int(sum(r["truth_has_complete_10_20_40"] for r in per_truth.values())),
            "full_candidate_coverage_and_truth_complete_count": int(sum(
                r["truth_has_complete_10_20_40"] and r["full_candidate_coverage"]
                for r in per_truth.values()
            )),
        },
        "per_spectrum": per_spectrum,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_spectrum"}, indent=2))


if __name__ == "__main__":
    main()
