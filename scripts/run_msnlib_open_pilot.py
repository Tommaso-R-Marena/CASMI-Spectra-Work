from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import evaluate_merged_convex, evaluate_same_energy_trajectory
from qfd.stats import paired_bootstrap_delta, paired_win_loss_tie


def _comparison(rows, challenger: str, baseline: str, *, n_boot: int, seed: int):
    if not rows:
        return None
    a = [r[f"rr_{challenger}"] for r in rows]
    b = [r[f"rr_{baseline}"] for r in rows]
    return {
        **paired_bootstrap_delta(a, b, n_boot=n_boot, seed=seed),
        **paired_win_loss_tie(a, b),
    }


def _strip_rows(result):
    return {k: v for k, v in result.items() if k != "rows"}


def _write_rows(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("energies"), list):
                out["energies"] = ";".join(map(str, out["energies"]))
            w.writerow(out)


def main():
    p = argparse.ArgumentParser(description="Run QFD/merged-energy pilots on an open MSnLib JSONL spectral library")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results/msnlib_open"))
    p.add_argument("--hard-static-threshold", type=float, default=0.8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    trajectory = evaluate_same_energy_trajectory(records, hard_static_threshold=args.hard_static_threshold)
    merged = evaluate_merged_convex(records, hard_static_threshold=args.hard_static_threshold)

    trajectory_summary = _strip_rows(trajectory)
    trajectory_summary["paired_vs_static"] = {
        "joint_energy": _comparison(trajectory["rows"], "joint_energy", "static_best", n_boot=args.bootstrap, seed=args.seed),
        "two_energy_qfd": _comparison(trajectory["rows"], "two_energy_qfd", "static_best", n_boot=args.bootstrap, seed=args.seed),
    }
    hard_t = [r for r in trajectory["rows"] if r["hard_static"]]
    trajectory_summary["hard_paired_vs_static"] = {
        "joint_energy": _comparison(hard_t, "joint_energy", "static_best", n_boot=args.bootstrap, seed=args.seed),
        "two_energy_qfd": _comparison(hard_t, "two_energy_qfd", "static_best", n_boot=args.bootstrap, seed=args.seed),
    }

    merged_summary = _strip_rows(merged)
    merged_summary["paired_vs_static"] = {
        "merged_convex": _comparison(merged["rows"], "merged_convex", "static_best", n_boot=args.bootstrap, seed=args.seed),
    }
    hard_m = [r for r in merged["rows"] if r["hard_static"]]
    merged_summary["hard_paired_vs_static"] = {
        "merged_convex": _comparison(hard_m, "merged_convex", "static_best", n_boot=args.bootstrap, seed=args.seed),
    }

    report = {
        "dataset_records": len(records),
        "hard_static_threshold": args.hard_static_threshold,
        "trajectory": trajectory_summary,
        "merged_energy": merged_summary,
        "interpretation_guardrail": "These are open MSnLib Class-1/mixture-consistency experiments. They are not CASMI hidden-test results and do not establish de novo structure elucidation.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_rows(args.out_dir / "trajectory_per_query.csv", trajectory["rows"])
    _write_rows(args.out_dir / "merged_per_query.csv", merged["rows"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
