from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from qfd.msnlib import read_msnlib_jsonl
from qfd.msnlib_variants import evaluate_exact_condition_variants, evaluate_merged_variants
from qfd.stats import paired_bootstrap_delta, paired_win_loss_tie


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _paired(rows, methods, baseline, *, n_boot, seed):
    out = {}
    if not rows:
        return out
    base = [float(r[f"rr_{baseline}"]) for r in rows]
    for method in methods:
        if method == baseline:
            continue
        vals = [float(r[f"rr_{method}"]) for r in rows]
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

    records = list(read_msnlib_jsonl(args.jsonl))
    trajectory = evaluate_exact_condition_variants(records, hard_static_threshold=args.hard_static_threshold)
    merged = evaluate_merged_variants(records, hard_static_threshold=args.hard_static_threshold)

    trows = trajectory.pop("rows")
    mrows = merged.pop("rows")
    tmethods = tuple(trajectory["methods"])
    mmethods = tuple(merged["methods"])

    trajectory["paired_vs_joint_mean"] = _paired(
        trows, tmethods, "joint_mean", n_boot=args.bootstrap, seed=args.seed,
    )
    thard = [r for r in trows if r["hard_static"]]
    trajectory["hard_paired_vs_joint_mean"] = _paired(
        thard, tmethods, "joint_mean", n_boot=args.bootstrap, seed=args.seed,
    )
    merged["paired_vs_static"] = _paired(
        mrows, mmethods, "static_best", n_boot=args.bootstrap, seed=args.seed,
    )
    mhard = [r for r in mrows if r["hard_static"]]
    merged["hard_paired_vs_static"] = _paired(
        mhard, mmethods, "static_best", n_boot=args.bootstrap, seed=args.seed,
    )

    summary = {
        "dataset_records": len(records),
        "hard_static_threshold": args.hard_static_threshold,
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
