from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.consensus_ranker import protected_tail_consensus_scores


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tie_metrics(scores: np.ndarray, truth: np.ndarray, cutoff: int) -> dict[str, float]:
    truth_idx = np.flatnonzero(np.asarray(truth, dtype=bool))
    if truth_idx.size != 1:
        raise ValueError(f"expected exactly one truth candidate, found {truth_idx.size}")
    s = np.asarray(scores, dtype=float)
    truth_score = float(s[int(truth_idx[0])])
    better = int(np.sum(s > truth_score))
    tied = int(np.sum(s == truth_score))
    ranks = np.arange(better + 1, better + tied + 1, dtype=float)
    return {
        "rr": float(np.mean(np.where(ranks <= cutoff, 1.0 / ranks, 0.0))),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "expected_rank": float(np.mean(ranks)),
        "rank_min": int(better + 1),
        "rank_max": int(better + tied),
        "tie_size": int(tied),
    }


def _tie_topk_credit_vector(scores: np.ndarray, k: int) -> np.ndarray:
    """Tie-aware Top-k credit for every possible truth in O(n log n)."""
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
        # One-indexed occupied ranks are start+1,...,end. The fraction at or
        # above k is exactly the tie-aware Top-k credit for this score group.
        credited = max(0, min(end, int(k)) - start)
        out[order[start:end]] = float(credited / tied)
        start = end
    return out


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mrr": float(np.mean([r["rr"] for r in rows])),
        "top1": float(np.mean([r["top1"] for r in rows])),
        "top5": float(np.mean([r["top5"] for r in rows])),
        "median_expected_rank": float(np.median([r["expected_rank"] for r in rows])),
    }


def _bootstrap_delta(delta: np.ndarray, *, seed: int, n_boot: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = int(delta.size)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = float(np.mean(delta[rng.integers(0, n, size=n)]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [float(lo), float(hi)]


def _signflip_p(delta: np.ndarray, *, seed: int, n_perm: int) -> float:
    nz = np.asarray(delta[np.abs(delta) > 0], dtype=float)
    if nz.size == 0:
        return 1.0
    observed = abs(float(np.mean(delta)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=nz.size)
        # Preserve the original denominator: zero-delta queries remain zeros.
        stat = abs(float(np.sum(nz * signs) / delta.size))
        extreme += int(stat >= observed - 1e-15)
    return float((extreme + 1) / (n_perm + 1))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate the label-free protected-head / expert-tail reranker on an existing candidate-feature artifact."
    )
    p.add_argument("candidate_features", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overlap-threshold", type=float, default=0.8)
    p.add_argument("--protected-k", type=int, default=1)
    p.add_argument("--overlap-k", type=int, default=10)
    p.add_argument("--rr-cutoff", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--signflips", type=int, default=100000)
    p.add_argument(
        "--role",
        default="development_audit",
        choices=["development_audit", "external_to_rule_audit", "untouched_confirmation"],
    )
    args = p.parse_args()

    df = pd.read_csv(args.candidate_features)
    required = {"spec_id", "is_truth", "static_best", "expert_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candidate feature table missing columns: {missing}")

    static_rows: list[dict[str, float]] = []
    protected_rows: list[dict[str, float]] = []
    query_rows: list[dict[str, float]] = []
    invariant_violations = 0

    for spec_id, g in df.groupby("spec_id", sort=False):
        static = g["static_best"].to_numpy(dtype=float)
        expert = g["expert_score"].to_numpy(dtype=float)
        truth = g["is_truth"].to_numpy(dtype=bool)
        chosen, overlap, used, protected = protected_tail_consensus_scores(
            static,
            expert,
            overlap_threshold=args.overlap_threshold,
            protected_k=args.protected_k,
            overlap_k=args.overlap_k,
        )
        sm = _tie_metrics(static, truth, args.rr_cutoff)
        pm = _tie_metrics(chosen, truth, args.rr_cutoff)
        static_rows.append(sm)
        protected_rows.append(pm)

        # Audit the theorem on every real candidate in this pool without using
        # labels: tie-aware Top-protected_k credit must be candidatewise equal.
        before_credit = _tie_topk_credit_vector(static, args.protected_k)
        after_credit = _tie_topk_credit_vector(chosen, args.protected_k)
        invariant_violations += int(np.sum(before_credit != after_credit))

        query_rows.append({
            "spec_id": int(spec_id),
            "overlap": float(overlap),
            "used_expert_tail": bool(used),
            "protected_count": int(np.sum(protected)),
            "rr_static": sm["rr"],
            "rr_protected": pm["rr"],
            "top1_static": sm["top1"],
            "top1_protected": pm["top1"],
            "top5_static": sm["top5"],
            "top5_protected": pm["top5"],
            "expected_rank_static": sm["expected_rank"],
            "expected_rank_protected": pm["expected_rank"],
        })

    static_summary = _mean_metrics(static_rows)
    protected_summary = _mean_metrics(protected_rows)
    delta_rr = np.asarray(
        [p["rr"] - s["rr"] for s, p in zip(static_rows, protected_rows)], dtype=float
    )
    delta_top1 = np.asarray(
        [p["top1"] - s["top1"] for s, p in zip(static_rows, protected_rows)], dtype=float
    )
    delta_top5 = np.asarray(
        [p["top5"] - s["top5"] for s, p in zip(static_rows, protected_rows)], dtype=float
    )

    if invariant_violations:
        raise RuntimeError(f"protected-prefix invariant failed {invariant_violations} times")
    if args.protected_k == 1 and not np.all(delta_top1 == 0):
        raise RuntimeError("Top-1 changed despite protected_k=1")

    result = {
        "status": "PROTECTED_TAIL_V5_ANALYSIS",
        "evaluation_role": args.role,
        "source": {
            "candidate_features": str(args.candidate_features),
            "sha256": _sha256(args.candidate_features),
            "queries": int(len(query_rows)),
            "candidates": int(len(df)),
        },
        "rule": {
            "overlap_threshold": float(args.overlap_threshold),
            "protected_k": int(args.protected_k),
            "overlap_k": int(args.overlap_k),
            "rr_cutoff": int(args.rr_cutoff),
            "cutoff_ties_included": True,
            "score_combination": "lexicographic protected static prefix, expert-only tail",
        },
        "invariant_audit": {
            "violations": int(invariant_violations),
            "top1_exactly_invariant_when_protected_k_1": bool(
                args.protected_k != 1 or np.all(delta_top1 == 0)
            ),
        },
        "static": static_summary,
        "protected_tail": protected_summary,
        "delta": {
            "mrr": float(np.mean(delta_rr)),
            "top1": float(np.mean(delta_top1)),
            "top5": float(np.mean(delta_top5)),
            "wins": int(np.sum(delta_rr > 0)),
            "losses": int(np.sum(delta_rr < 0)),
            "ties": int(np.sum(delta_rr == 0)),
            "bootstrap_95pct_mrr": _bootstrap_delta(
                delta_rr, seed=args.seed, n_boot=args.bootstrap
            ),
            "signflip_p_two_sided": _signflip_p(
                delta_rr, seed=args.seed + 1, n_perm=args.signflips
            ),
        },
        "usage": {
            "used_queries": int(sum(r["used_expert_tail"] for r in query_rows)),
            "fraction": float(np.mean([r["used_expert_tail"] for r in query_rows])),
            "mean_protected_count": float(np.mean([r["protected_count"] for r in query_rows])),
        },
        "guardrail": (
            "The protected-prefix theorem guarantees only the predeclared protected Top-k metric. "
            "Any MRR/Top-5 improvement is empirical. Development or previously inspected datasets "
            "must not be relabeled as untouched confirmation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
