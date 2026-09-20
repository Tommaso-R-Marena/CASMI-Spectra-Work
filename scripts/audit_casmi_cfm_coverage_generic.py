from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name!r} under {root}, found {len(hits)}")
    return hits[0]


def quantiles(values):
    a = np.asarray(list(values), float)
    if not len(a):
        return {}
    return {k: float(v) for k, v in {
        "min": a.min(), "q05": np.quantile(a, .05), "q25": np.quantile(a, .25),
        "median": np.quantile(a, .5), "q75": np.quantile(a, .75),
        "q95": np.quantile(a, .95), "max": a.max(), "mean": a.mean(),
    }.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("casmi_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("--cfm-file", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    spec = pd.read_pickle(find_one(args.casmi_root, "spec_df.pkl"))
    cand = pd.read_pickle(find_one(args.casmi_root, "cand_df.pkl"))
    cfm = pd.read_pickle(find_one(args.cfm_root, args.cfm_file))
    if not {"candidate_mol_id", "query_mol_id"}.issubset(cand.columns):
        raise RuntimeError("candidate table schema mismatch")
    if not {"mol_id", "ace", "prec_type", "peaks"}.issubset(cfm.columns):
        raise RuntimeError("CFM table schema mismatch")

    cfm = cfm[cfm["prec_type"] == "[M+H]+"].copy()
    energy_sets = cfm.groupby("mol_id")["ace"].agg(lambda x: tuple(sorted(set(int(v) for v in x))))
    complete = {int(mid) for mid, es in energy_sets.items() if es == (10, 20, 40)}
    by_query = cand.groupby("query_mol_id")["candidate_mol_id"].agg(lambda s: tuple(sorted(set(int(x) for x in s))))
    pos = spec[spec["prec_type"] == "[M+H]+"].copy()

    rows = []
    for r in pos.itertuples(index=False):
        truth = int(r.mol_id)
        cands = tuple(int(x) for x in by_query.get(truth, ()))
        covered = sum(x in complete for x in cands)
        rows.append({
            "spec_id": int(r.spec_id), "truth": truth, "candidate_count": len(cands),
            "covered": covered, "coverage_fraction": covered / len(cands) if cands else 0.0,
            "truth_complete": truth in complete,
        })

    thresholds = (1.0, .99, .95, .90)
    all_candidates = {int(x) for x in cand["candidate_mol_id"].unique()}
    summary = {
        "label": args.label,
        "prediction_file": args.cfm_file,
        "protocol": "positive [M+H]+; complete CFM means exactly 10/20/40 eV",
        "casmi": {"all_spectra": int(len(spec)), "positive_mh_spectra": int(len(pos)),
                  "unique_positive_truths": int(pos["mol_id"].nunique()),
                  "candidate_links": int(len(cand)), "candidate_molecules": int(cand["candidate_mol_id"].nunique())},
        "cfm": {"rows": int(len(cfm)), "complete_molecules": int(len(complete)),
                "candidate_molecule_coverage": float(len(all_candidates & complete) / len(all_candidates)) if all_candidates else 0.0},
        "truth_complete_fraction": float(np.mean([r["truth_complete"] for r in rows])) if rows else 0.0,
        "threshold_counts": {str(t): int(sum(r["truth_complete"] and r["coverage_fraction"] >= t for r in rows)) for t in thresholds},
        "coverage_quantiles": quantiles(r["coverage_fraction"] for r in rows),
        "candidate_count_quantiles": quantiles(r["candidate_count"] for r in rows),
        "per_spectrum": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_spectrum"}, indent=2))


if __name__ == "__main__":
    main()
