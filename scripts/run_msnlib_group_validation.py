from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_variants import evaluate_exact_condition_variants, evaluate_merged_variants
from qfd.stats import (
    paired_block_bootstrap_delta,
    paired_win_loss_tie,
    stable_group_is_validation,
)


SPLIT_SALT = "msnlib-group-validation-v1"
PRIMARY_TRAJECTORY = "joint_mean"
PRIMARY_MERGED = "hybrid_static_convex_100"


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _assign_split(rows, *, validation_fraction: float):
    out = []
    for row in rows:
        copied = dict(row)
        copied["validation_group"] = row["formula"]
        copied["split"] = (
            "validation"
            if stable_group_is_validation(
                row["formula"],
                validation_fraction=validation_fraction,
                salt=SPLIT_SALT,
            )
            else "discovery"
        )
        out.append(copied)
    return out


def _summary(rows, methods):
    result = {
        "n_queries": len(rows),
        "n_formula_blocks": len({r["formula"] for r in rows}),
    }
    if not rows:
        return result
    for method in methods:
        result[f"mrr25_tieaware_{method}"] = float(
            np.mean([float(r[f"rr_tie_{method}"]) for r in rows])
        )
        result[f"top1_tieaware_{method}"] = float(
            np.mean([float(r[f"top1_tie_{method}"]) for r in rows])
        )
        result[f"top5_tieaware_{method}"] = float(
            np.mean([float(r[f"top5_tie_{method}"]) for r in rows])
        )
    return result


def _block_comparison(rows, method, baseline, *, n_boot: int, seed: int):
    if not rows:
        return {}
    blocks = [r["formula"] for r in rows]
    rr_method = [float(r[f"rr_tie_{method}"]) for r in rows]
    rr_base = [float(r[f"rr_tie_{baseline}"]) for r in rows]
    top1_method = [float(r[f"top1_tie_{method}"]) for r in rows]
    top1_base = [float(r[f"top1_tie_{baseline}"]) for r in rows]
    return {
        "method": method,
        "baseline": baseline,
        "rr": {
            **paired_block_bootstrap_delta(
                rr_method, rr_base, blocks, n_boot=n_boot, seed=seed
            ),
            **paired_win_loss_tie(rr_method, rr_base),
        },
        "top1": paired_block_bootstrap_delta(
            top1_method, top1_base, blocks, n_boot=n_boot, seed=seed + 1
        ),
    }


def _formula_effects(rows, method, baseline):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["formula"]].append(
            float(r[f"rr_tie_{method}"]) - float(r[f"rr_tie_{baseline}"])
        )
    effects = {formula: float(np.mean(vals)) for formula, vals in grouped.items()}
    if not effects:
        return {"n_formula_blocks": 0, "effects": {}}
    vals = np.asarray(list(effects.values()), dtype=float)
    atol = 1e-12
    return {
        "n_formula_blocks": len(effects),
        "positive_blocks": int(np.sum(vals > atol)),
        "negative_blocks": int(np.sum(vals < -atol)),
        "zero_blocks": int(np.sum(np.abs(vals) <= atol)),
        "mean_formula_delta": float(vals.mean()),
        "median_formula_delta": float(np.median(vals)),
        "effects": dict(sorted(effects.items())),
    }


def _section(rows, methods, primary, baseline, *, n_boot, seed):
    discovery = [r for r in rows if r["split"] == "discovery"]
    validation = [r for r in rows if r["split"] == "validation"]
    hard = [r for r in rows if r["hard_static"]]
    hard_discovery = [r for r in discovery if r["hard_static"]]
    hard_validation = [r for r in validation if r["hard_static"]]
    return {
        "all": _summary(rows, methods),
        "discovery": _summary(discovery, methods),
        "validation": _summary(validation, methods),
        "hard_all": _summary(hard, methods),
        "hard_discovery": _summary(hard_discovery, methods),
        "hard_validation": _summary(hard_validation, methods),
        "primary_method": primary,
        "baseline": baseline,
        "primary_block_bootstrap_all": _block_comparison(
            rows, primary, baseline, n_boot=n_boot, seed=seed
        ),
        "primary_block_bootstrap_validation": _block_comparison(
            validation, primary, baseline, n_boot=n_boot, seed=seed + 10
        ),
        "primary_block_bootstrap_hard_validation": _block_comparison(
            hard_validation, primary, baseline, n_boot=n_boot, seed=seed + 20
        ),
        "primary_formula_effects_validation": _formula_effects(
            validation, primary, baseline
        ),
        "primary_formula_effects_hard_validation": _formula_effects(
            hard_validation, primary, baseline
        ),
    }


def _write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(v) for k, v in row.items()})


def main():
    p = argparse.ArgumentParser(
        description="Formula-block bootstrap and frozen formula-held-out validation for open MSnLib"
    )
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results/msnlib_group_validation"))
    p.add_argument("--hard-static-threshold", type=float, default=0.8)
    p.add_argument("--validation-fraction", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=271828)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    trajectory = evaluate_exact_condition_variants(
        records, hard_static_threshold=args.hard_static_threshold
    )
    merged = evaluate_merged_variants(
        records, hard_static_threshold=args.hard_static_threshold
    )

    trows = _assign_split(trajectory["rows"], validation_fraction=args.validation_fraction)
    mrows = _assign_split(merged["rows"], validation_fraction=args.validation_fraction)
    tmethods = tuple(trajectory["methods"])
    mmethods = tuple(merged["methods"])

    t_discovery = {r["formula"] for r in trows if r["split"] == "discovery"}
    t_validation = {r["formula"] for r in trows if r["split"] == "validation"}
    m_discovery = {r["formula"] for r in mrows if r["split"] == "discovery"}
    m_validation = {r["formula"] for r in mrows if r["split"] == "validation"}
    if t_discovery & t_validation or m_discovery & m_validation:
        raise RuntimeError("formula leakage across discovery/validation split")

    summary = {
        "dataset_records": len(records),
        "protocol": {
            "split_unit": "molecular formula (all adducts and queries for a formula stay together)",
            "block_bootstrap_unit": "molecular formula",
            "validation_fraction": args.validation_fraction,
            "split_salt": SPLIT_SALT,
            "split_algorithm": "SHA-256(salt + unit-separator + formula), first 64 bits",
            "primary_ranking_metric": "tie-aware MRR@25",
            "primary_trajectory_method_frozen_before_validation": PRIMARY_TRAJECTORY,
            "primary_trajectory_baseline": "static_best",
            "primary_merged_method_frozen_before_validation": PRIMARY_MERGED,
            "primary_merged_baseline": "static_best",
            "hard_subset_rule": f"best wrong static similarity >= {args.hard_static_threshold}",
            "selection_guardrail": (
                "The validation split, split salt, primary methods, baselines, and weights are fixed in code. "
                "Validation outcomes must not be used to retune this run."
            ),
        },
        "trajectory_exact_condition": _section(
            trows,
            tmethods,
            PRIMARY_TRAJECTORY,
            "static_best",
            n_boot=args.bootstrap,
            seed=args.seed,
        ),
        "merged_energy": _section(
            mrows,
            mmethods,
            PRIMARY_MERGED,
            "static_best",
            n_boot=args.bootstrap,
            seed=args.seed + 1000,
        ),
        "interpretation_guardrail": (
            "This removes query-level pseudoreplication from uncertainty estimates and provides a frozen, "
            "formula-disjoint internal validation split. It is still an internal validation on one released "
            "MSnLib acquisition family, not an independent external library or CASMI hidden-test result."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.out_dir / "trajectory_per_query.csv", trows)
    _write_csv(args.out_dir / "merged_per_query.csv", mrows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
