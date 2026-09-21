from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_casmi2022_consensus_lockbox import _paired, _risk, _summarize


def _subset_report(rows: list[dict], seed_offset: int) -> dict:
    methods = ("static", "expert", "gated", "permuted_control")
    if not rows:
        return {"n": 0, "metrics": {m: {"n": 0} for m in methods}, "risk": {}, "inference": None}
    return {
        "n": int(len(rows)),
        "n_truth_blocks": int(len({int(r["query_mol_id"]) for r in rows})),
        "metrics": {m: _summarize(rows, m) for m in methods},
        "risk": {m: _risk(rows, m) for m in ("expert", "gated", "permuted_control")},
        "inference": {"gated_vs_static": _paired(rows, "gated", seed_offset)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Append predeclared CASMI-2022 unseen-structure sensitivities to a completed lockbox summary")
    p.add_argument("summary", type=Path)
    p.add_argument("per_query", type=Path)
    p.add_argument("overlap_csv", type=Path)
    p.add_argument("overlap_summary", type=Path)
    args = p.parse_args()

    summary = json.loads(args.summary.read_text())
    overlap_summary = json.loads(args.overlap_summary.read_text())
    q = pd.read_csv(args.per_query)
    o = pd.read_csv(args.overlap_csv)
    required = {"query_mol_id", "exact_training_overlap", "connectivity_training_overlap"}
    if not required.issubset(o.columns):
        raise RuntimeError(f"overlap CSV missing columns: {sorted(required - set(o.columns))}")
    if o["query_mol_id"].duplicated().any():
        raise RuntimeError("overlap audit has duplicate query_mol_id rows")

    merged = q.merge(
        o[["query_mol_id", "exact_training_overlap", "connectivity_training_overlap"]],
        on="query_mol_id",
        how="left",
        validate="many_to_one",
    )
    if merged[["exact_training_overlap", "connectivity_training_overlap"]].isna().any().any():
        missing = merged.loc[merged["exact_training_overlap"].isna(), "query_mol_id"].unique().tolist()
        raise RuntimeError(f"scored lockbox truths missing from overlap audit: {missing[:10]}")

    exact_unseen = merged[~merged["exact_training_overlap"].astype(bool)].to_dict("records")
    connectivity_unseen = merged[~merged["connectivity_training_overlap"].astype(bool)].to_dict("records")
    exact_overlap = merged[merged["exact_training_overlap"].astype(bool)].to_dict("records")
    connectivity_overlap = merged[merged["connectivity_training_overlap"].astype(bool)].to_dict("records")

    summary["identity_overlap_sensitivity"] = {
        "status": "predeclared secondary sensitivity; labels computed before ranking outcomes; primary replication gate unchanged",
        "overlap_audit": overlap_summary,
        "exact_unseen_truths": _subset_report(exact_unseen, 5000),
        "connectivity_unseen_truths": _subset_report(connectivity_unseen, 6000),
        "exact_overlap_truths_descriptive": _subset_report(exact_overlap, 7000),
        "connectivity_overlap_truths_descriptive": _subset_report(connectivity_overlap, 8000),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({
        "audit_counts": overlap_summary["counts"],
        "exact_unseen": summary["identity_overlap_sensitivity"]["exact_unseen_truths"],
        "connectivity_unseen": summary["identity_overlap_sensitivity"]["connectivity_unseen_truths"],
    }, indent=2))


if __name__ == "__main__":
    main()
