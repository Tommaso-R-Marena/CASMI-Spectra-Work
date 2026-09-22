from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.protected_tail_v53 import protected_tail_v53_scores
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tie_metrics(scores: np.ndarray, truth: np.ndarray, cutoff: int = 25) -> dict[str, float | int]:
    s = np.asarray(scores, dtype=float)
    t = np.asarray(truth, dtype=bool)
    truth_idx = np.flatnonzero(t)
    if truth_idx.size != 1:
        raise ValueError(f"expected exactly one truth candidate, found {truth_idx.size}")
    if s.ndim != 1 or s.size == 0 or np.any(~np.isfinite(s)):
        raise ValueError("finite nonempty score vector required")
    truth_score = float(s[int(truth_idx[0])])
    better = int(np.sum(s > truth_score))
    tied = int(np.sum(s == truth_score))
    ranks = np.arange(better + 1, better + tied + 1, dtype=float)
    return {
        "rr": float(np.mean(np.where(ranks <= cutoff, 1.0 / ranks, 0.0))),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top25": float(np.mean(ranks <= 25)),
        "expected_rank": float(np.mean(ranks)),
        "rank_min": int(better + 1),
        "rank_max": int(better + tied),
        "tie_size": int(tied),
    }


def _tie_topk_credit_vector(scores: np.ndarray, k: int) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    sorted_s = s[order]
    out = np.empty(s.size, dtype=float)
    start = 0
    while start < s.size:
        end = start + 1
        while end < s.size and sorted_s[end] == sorted_s[start]:
            end += 1
        tied = end - start
        credited = max(0, min(end, int(k)) - start)
        out[order[start:end]] = float(credited / tied)
        start = end
    return out


def _aggregate(metrics: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not metrics:
        return {"n_queries": 0}
    return {
        "n_queries": int(len(metrics)),
        "mrr25": float(np.mean([float(x["rr"]) for x in metrics])),
        "top1": float(np.mean([float(x["top1"]) for x in metrics])),
        "top5": float(np.mean([float(x["top5"]) for x in metrics])),
        "top25": float(np.mean([float(x["top25"]) for x in metrics])),
        "median_expected_rank": float(np.median([float(x["expected_rank"]) for x in metrics])),
        "truth_tied_queries": int(sum(int(x["tie_size"]) > 1 for x in metrics)),
    }


def _inference(
    alt: np.ndarray,
    base: np.ndarray,
    blocks: list[str],
    *,
    seed: int,
) -> dict[str, object]:
    if len(alt) == 0:
        return {"n_queries": 0, "n_formula_blocks": 0}
    return {
        "n_queries": int(len(alt)),
        "n_formula_blocks": int(len(set(blocks))),
        "mean_delta_mrr25": float(np.mean(alt - base)),
        "win_loss_tie": paired_win_loss_tie(alt, base, atol=1e-15),
        "formula_block_bootstrap_95": paired_block_bootstrap_delta(
            alt,
            base,
            blocks,
            n_boot=10000,
            seed=seed,
            alpha=0.05,
        ),
        "formula_block_signflip_two_sided": paired_block_signflip_test(
            alt,
            base,
            blocks,
            seed=seed,
            n_permutations=200000,
            exact_max_blocks=20,
            atol=1e-15,
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen prospective 30 eV heldout-energy lockbox evaluation.")
    p.add_argument("candidate_features", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "discovery", "all"], default="validation")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--overlap-threshold", type=float, default=0.8)
    p.add_argument("--protected-k", type=int, default=1)
    p.add_argument("--overlap-k", type=int, default=10)
    p.add_argument("--head-survival-slack", type=int, default=1)
    args = p.parse_args()

    df = pd.read_csv(args.candidate_features)
    required = {
        "spec_id",
        "formula",
        "split",
        "candidate_id",
        "is_truth",
        "candidate_count",
        "static_best",
        "expert_score",
        "equal_mix_control",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candidate feature table missing columns: {missing}")
    if args.split != "all":
        df = df.loc[df["split"].eq(args.split)].copy()
    if df.empty:
        raise ValueError("selected heldout-energy cohort is empty")

    methods = ("static", "interpolation", "equal_mix", "v53")
    per_method: dict[str, list[dict[str, float | int]]] = {m: [] for m in methods}
    query_rows: list[dict[str, object]] = []
    protected_invariant_violations = 0

    for spec_id, g in df.groupby("spec_id", sort=False):
        static = g["static_best"].to_numpy(dtype=float)
        expert = g["expert_score"].to_numpy(dtype=float)
        control = g["equal_mix_control"].to_numpy(dtype=float)
        truth = g["is_truth"].to_numpy(dtype=bool)
        if np.any(~np.isfinite(static)) or np.any(~np.isfinite(expert)) or np.any(~np.isfinite(control)):
            raise ValueError(f"nonfinite score in query {spec_id}")

        v53, overlap, used, protected, head_survives, max_rank = protected_tail_v53_scores(
            static,
            expert,
            overlap_threshold=args.overlap_threshold,
            protected_k=args.protected_k,
            overlap_k=args.overlap_k,
            head_survival_slack=args.head_survival_slack,
        )

        sm = _tie_metrics(static, truth)
        im = _tie_metrics(expert, truth)
        cm = _tie_metrics(control, truth)
        vm = _tie_metrics(v53, truth)
        per_method["static"].append(sm)
        per_method["interpolation"].append(im)
        per_method["equal_mix"].append(cm)
        per_method["v53"].append(vm)

        before = _tie_topk_credit_vector(static, args.protected_k)
        after = _tie_topk_credit_vector(v53, args.protected_k)
        protected_invariant_violations += int(np.sum(before != after))
        if args.protected_k == 1 and float(sm["top1"]) != float(vm["top1"]):
            raise RuntimeError(f"v5.3 changed tie-aware Top-1 on {spec_id}")

        formula_vals = g["formula"].astype(str).unique()
        count_vals = g["candidate_count"].astype(int).unique()
        split_vals = g["split"].astype(str).unique()
        if len(formula_vals) != 1 or len(count_vals) != 1 or len(split_vals) != 1:
            raise ValueError(f"query metadata inconsistent for {spec_id}")
        query_rows.append({
            "spec_id": str(spec_id),
            "formula": str(formula_vals[0]),
            "split": str(split_vals[0]),
            "candidate_count": int(count_vals[0]),
            "overlap": float(overlap),
            "used_v53": bool(used),
            "head_survives": bool(head_survives),
            "max_protected_expert_rank": int(max_rank),
            "protected_count": int(np.sum(protected)),
            "rr_static": float(sm["rr"]),
            "rr_interpolation": float(im["rr"]),
            "rr_equal_mix": float(cm["rr"]),
            "rr_v53": float(vm["rr"]),
            "top1_static": float(sm["top1"]),
            "top1_interpolation": float(im["top1"]),
            "top1_equal_mix": float(cm["top1"]),
            "top1_v53": float(vm["top1"]),
        })

    if protected_invariant_violations:
        raise RuntimeError(f"protected-prefix invariant failure count={protected_invariant_violations}")

    rr = {
        key: np.asarray([float(x["rr"]) for x in per_method[key]], dtype=float)
        for key in methods
    }
    formulas = [str(x["formula"]) for x in query_rows]
    counts = np.asarray([int(x["candidate_count"]) for x in query_rows], dtype=int)

    def subset_inference(mask: np.ndarray) -> dict[str, object]:
        idx = np.flatnonzero(mask)
        bl = [formulas[int(i)] for i in idx]
        return {
            "v53_vs_static": _inference(rr["v53"][idx], rr["static"][idx], bl, seed=args.seed),
            "interpolation_vs_static": _inference(rr["interpolation"][idx], rr["static"][idx], bl, seed=args.seed + 1),
            "interpolation_vs_equal_mix": _inference(rr["interpolation"][idx], rr["equal_mix"][idx], bl, seed=args.seed + 2),
            "equal_mix_vs_static": _inference(rr["equal_mix"][idx], rr["static"][idx], bl, seed=args.seed + 3),
        }

    result = {
        "status": "MSNLIB_HELDOUT30_PROSPECTIVE_LOCKBOX",
        "evaluation_split": args.split,
        "source": {
            "candidate_features": str(args.candidate_features),
            "sha256": _sha256(args.candidate_features),
            "queries": int(len(query_rows)),
            "candidate_rows": int(len(df)),
            "formula_blocks": int(len(set(formulas))),
        },
        "frozen_protocol": {
            "query_energy_ev": 30.0,
            "reference_energies_ev": [20.0, 60.0],
            "static_baseline": "best endpoint Bhattacharyya affinity",
            "trajectory_expert": "energy-constrained linear interpolation affinity, 75% 20 eV + 25% 60 eV",
            "energy_blind_control": "50/50 endpoint mixture affinity",
            "selector": {
                "overlap_threshold": float(args.overlap_threshold),
                "protected_k": int(args.protected_k),
                "overlap_k": int(args.overlap_k),
                "head_survival_slack": int(args.head_survival_slack),
                "tail_rule": "minimax(max(tie-aware static rank, tie-aware expert rank))",
            },
        },
        "primary_endpoint": "formula-blocked query-weighted delta tie-aware MRR@25, frozen v5.3 minus static, validation formulas only",
        "secondary_endpoints": [
            "interpolation minus static",
            "interpolation minus energy-blind 50/50 mixture",
            "equal-mix control minus static",
        ],
        "invariants": {
            "protected_topk_candidatewise_violations": int(protected_invariant_violations),
            "top1_exactly_invariant_v53": bool(all(
                float(a["top1"]) == float(b["top1"])
                for a, b in zip(per_method["static"], per_method["v53"])
            )),
        },
        "metrics": {m: _aggregate(per_method[m]) for m in methods},
        "selector_usage": {
            "used_queries": int(sum(bool(x["used_v53"]) for x in query_rows)),
            "used_fraction": float(np.mean([bool(x["used_v53"]) for x in query_rows])),
            "mean_overlap": float(np.mean([float(x["overlap"]) for x in query_rows])),
            "mean_protected_count": float(np.mean([int(x["protected_count"]) for x in query_rows])),
        },
        "inference": {
            "all_selected_queries": subset_inference(np.ones(len(query_rows), dtype=bool)),
            "candidate_pool_ge_5": subset_inference(counts >= 5),
            "candidate_pool_ge_10": subset_inference(counts >= 10),
        },
        "query_diagnostics": query_rows,
        "guardrail": (
            "The validation split, target/reference energies, interpolation law, controls, selector parameters, "
            "primary endpoint, formula-block inference, and pool-size sensitivities were fixed before validation "
            "ranking outcomes were opened."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    printable = {k: v for k, v in result.items() if k != "query_diagnostics"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
