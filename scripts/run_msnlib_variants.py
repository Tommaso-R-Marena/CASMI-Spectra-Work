from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_variants import evaluate_exact_condition_variants, evaluate_merged_variants
from qfd.stats import paired_bootstrap_delta, paired_win_loss_tie


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _paired(rows, methods, baseline, *, n_boot, seed, tie_aware=False):
    out = {}
    if not rows:
        return out
    prefix = "rr_tie_" if tie_aware else "rr_"
    base = [float(r[f"{prefix}{baseline}"]) for r in rows]
    for method in methods:
        if method == baseline:
            continue
        vals = [float(r[f"{prefix}{method}"]) for r in rows]
        out[method] = {
            **paired_bootstrap_delta(vals, base, n_boot=n_boot, seed=seed),
            **paired_win_loss_tie(vals, base),
        }
    return out


def _write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: _json_safe(v) for k, v in row.items()})


def main():
    p = argparse.ArgumentParser(description="Open MSnLib censor-aware and mixture-geometry variants")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results/msnlib_variants"))
    p.add_argument("--hard-static-threshold", type=float, default=0.8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    trajectory = evaluate_exact_condition_variants(records, hard_static_threshold=args.hard_static_threshold)
    merged = evaluate_merged_variants(records, hard_static_threshold=args.hard_static_threshold)

    trows = trajectory.pop("rows")
    mrows = merged.pop("rows")
    tmethods = tuple(trajectory["methods"])
    mmethods = tuple(merged["methods"])
    thard = [r for r in trows if r["hard_static"]]
    mhard = [r for r in mrows if r["hard_static"]]

    trajectory["paired_vs_joint_mean_deterministic"] = _paired(
        trows, tmethods, "joint_mean", n_boot=args.bootstrap, seed=args.seed,
    )
    trajectory["hard_paired_vs_joint_mean_deterministic"] = _paired(
        thard, tmethods, "joint_mean", n_boot=args.bootstrap, seed=args.seed,
    )
    trajectory["paired_vs_joint_mean_tieaware"] = _paired(
        trows, tmethods, "joint_mean", n_boot=args.bootstrap, seed=args.seed, tie_aware=True,
    )
    trajectory["hard_paired_vs_joint_mean_tieaware"] = _paired(
        thard, tmethods, "joint_mean", n_boot=args.bootstrap, seed=args.seed, tie_aware=True,
    )

    merged["paired_vs_static_deterministic"] = _paired(
        mrows, mmethods, "static_best", n_boot=args.bootstrap, seed=args.seed,
    )
    merged["hard_paired_vs_static_deterministic"] = _paired(
        mhard, mmethods, "static_best", n_boot=args.bootstrap, seed=args.seed,
    )
    merged["paired_vs_static_tieaware"] = _paired(
        mrows, mmethods, "static_best", n_boot=args.bootstrap, seed=args.seed, tie_aware=True,
    )
    merged["hard_paired_vs_static_tieaware"] = _paired(
        mhard, mmethods, "static_best", n_boot=args.bootstrap, seed=args.seed, tie_aware=True,
    )

    summary = {
        "dataset_records": len(records),
        "hard_static_threshold": args.hard_static_threshold,
        "primary_ranking_policy": "tie-aware expected reciprocal rank and fractional top-k credit; deterministic ranks retained only for diagnostics",
        "trajectory_exact_condition": trajectory,
        "merged_energy_variants": merged,
        "interpretation_guardrail": (
            "These are Class-1/representation-consistency experiments on one released MSnLib library. "
            "Weights are predeclared, not selected from the reported outcomes. They are not CASMI hidden-test results."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.out_dir / "trajectory_per_query.csv", trows)
    _write_csv(args.out_dir / "merged_per_query.csv", mrows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
