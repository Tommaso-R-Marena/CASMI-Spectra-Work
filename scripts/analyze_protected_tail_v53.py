from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.protected_tail_v52 import protected_tail_v52_scores
from qfd.protected_tail_v53 import protected_tail_v53_scores


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
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


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mrr25": float(np.mean([r["rr"] for r in rows])),
        "top1": float(np.mean([r["top1"] for r in rows])),
        "top5": float(np.mean([r["top5"] for r in rows])),
        "top25": float(np.mean([r["top25"] for r in rows])),
        "median_expected_rank": float(np.median([r["expected_rank"] for r in rows])),
    }


def _comparison(base: list[dict[str, float]], alt: list[dict[str, float]]) -> dict[str, float | int]:
    drr = np.asarray([a["rr"] - b["rr"] for b, a in zip(base, alt)], dtype=float)
    return {
        "delta_mrr25": float(np.mean(drr)),
        "delta_top1": float(np.mean([a["top1"] - b["top1"] for b, a in zip(base, alt)])),
        "delta_top5": float(np.mean([a["top5"] - b["top5"] for b, a in zip(base, alt)])),
        "delta_top25": float(np.mean([a["top25"] - b["top25"] for b, a in zip(base, alt)])),
        "wins": int(np.sum(drr > 1e-15)),
        "losses": int(np.sum(drr < -1e-15)),
        "ties": int(np.sum(np.abs(drr) <= 1e-15)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen protected-tail v5.2/v5.3 external replay.")
    p.add_argument("candidate_features", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--role", default="external_to_selector_audit")
    p.add_argument("--overlap-threshold", type=float, default=0.8)
    p.add_argument("--protected-k", type=int, default=1)
    p.add_argument("--overlap-k", type=int, default=10)
    p.add_argument("--head-survival-slack", type=int, default=1)
    p.add_argument("--rr-cutoff", type=int, default=25)
    p.add_argument("--control-seed", type=int, default=20260921)
    p.add_argument(
        "--guardrail",
        default=(
            "Selection parameters were frozen before this replay. Interpret this result according to "
            "the dataset-specific provenance recorded by the calling workflow."
        ),
    )
    args = p.parse_args()

    df = pd.read_csv(args.candidate_features)
    required = {"spec_id", "is_truth", "static_best", "expert_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candidate feature table missing columns: {missing}")

    static_rows: list[dict[str, float]] = []
    v52_rows: list[dict[str, float]] = []
    v53_rows: list[dict[str, float]] = []
    perm_rows: list[dict[str, float]] = []
    query_rows: list[dict[str, object]] = []
    invariant_violations_v52 = 0
    invariant_violations_v53 = 0

    for qidx, (spec_id, g) in enumerate(df.groupby("spec_id", sort=False)):
        static = g["static_best"].to_numpy(dtype=float)
        expert = g["expert_score"].to_numpy(dtype=float)
        truth = g["is_truth"].to_numpy(dtype=bool)

        v52, overlap52, used52, protected52, survives52, max_rank52 = protected_tail_v52_scores(
            static,
            expert,
            overlap_threshold=args.overlap_threshold,
            protected_k=args.protected_k,
            overlap_k=args.overlap_k,
            head_survival_slack=args.head_survival_slack,
        )
        v53, overlap53, used53, protected53, survives53, max_rank53 = protected_tail_v53_scores(
            static,
            expert,
            overlap_threshold=args.overlap_threshold,
            protected_k=args.protected_k,
            overlap_k=args.overlap_k,
            head_survival_slack=args.head_survival_slack,
        )

        rng = np.random.default_rng(args.control_seed + qidx)
        perm_expert = expert[rng.permutation(expert.size)]
        perm, _, used_perm, _, _, _ = protected_tail_v53_scores(
            static,
            perm_expert,
            overlap_threshold=args.overlap_threshold,
            protected_k=args.protected_k,
            overlap_k=args.overlap_k,
            head_survival_slack=args.head_survival_slack,
        )

        sm = _tie_metrics(static, truth, args.rr_cutoff)
        m52 = _tie_metrics(v52, truth, args.rr_cutoff)
        m53 = _tie_metrics(v53, truth, args.rr_cutoff)
        mp = _tie_metrics(perm, truth, args.rr_cutoff)
        static_rows.append(sm)
        v52_rows.append(m52)
        v53_rows.append(m53)
        perm_rows.append(mp)

        before = _tie_topk_credit_vector(static, args.protected_k)
        invariant_violations_v52 += int(np.sum(before != _tie_topk_credit_vector(v52, args.protected_k)))
        invariant_violations_v53 += int(np.sum(before != _tie_topk_credit_vector(v53, args.protected_k)))

        if not np.isclose(overlap52, overlap53, rtol=0.0, atol=1e-15):
            raise RuntimeError("v5.2/v5.3 guard overlap mismatch")
        if not np.array_equal(protected52, protected53):
            raise RuntimeError("v5.2/v5.3 protected-head mismatch")
        if survives52 != survives53 or max_rank52 != max_rank53:
            raise RuntimeError("v5.2/v5.3 head-survival mismatch")
        if used52 != used53:
            raise RuntimeError("v5.2/v5.3 eligibility mismatch")

        query_rows.append({
            "spec_id": str(spec_id),
            "candidate_count": int(len(g)),
            "overlap": float(overlap53),
            "head_survives": bool(survives53),
            "max_protected_expert_rank": int(max_rank53),
            "used_v53": bool(used53),
            "used_permuted_control": bool(used_perm),
            "protected_count": int(np.sum(protected53)),
            "rr_static": sm["rr"],
            "rr_v52": m52["rr"],
            "rr_v53": m53["rr"],
            "rr_permuted_control": mp["rr"],
            "top5_static": sm["top5"],
            "top5_v52": m52["top5"],
            "top5_v53": m53["top5"],
        })

    if invariant_violations_v52 or invariant_violations_v53:
        raise RuntimeError(
            f"protected-prefix invariant failure: v52={invariant_violations_v52}, v53={invariant_violations_v53}"
        )
    if args.protected_k == 1:
        if any(a["top1"] != b["top1"] for a, b in zip(static_rows, v52_rows)):
            raise RuntimeError("v5.2 changed Top-1 despite protected_k=1")
        if any(a["top1"] != b["top1"] for a, b in zip(static_rows, v53_rows)):
            raise RuntimeError("v5.3 changed Top-1 despite protected_k=1")

    result = {
        "status": "PROTECTED_TAIL_V53_EXTERNAL_REPLAY",
        "evaluation_role": args.role,
        "source": {
            "candidate_features": str(args.candidate_features),
            "sha256": _sha256(args.candidate_features),
            "queries": int(len(query_rows)),
            "candidates": int(len(df)),
        },
        "frozen_rule": {
            "overlap_threshold": float(args.overlap_threshold),
            "protected_k": int(args.protected_k),
            "overlap_k": int(args.overlap_k),
            "head_survival_slack": int(args.head_survival_slack),
            "rr_cutoff": int(args.rr_cutoff),
            "v52_tail": "expert-only",
            "v53_tail": "minimax(max(tie-aware static rank, tie-aware expert rank))",
        },
        "invariants": {
            "v52_candidatewise_protected_topk_violations": int(invariant_violations_v52),
            "v53_candidatewise_protected_topk_violations": int(invariant_violations_v53),
            "top1_exactly_invariant": bool(args.protected_k != 1 or all(
                s["top1"] == m["top1"] for s, m in zip(static_rows, v53_rows)
            )),
        },
        "static": _summary(static_rows),
        "v52": _summary(v52_rows),
        "v53": _summary(v53_rows),
        "permuted_expert_control_v53": _summary(perm_rows),
        "v52_vs_static": _comparison(static_rows, v52_rows),
        "v53_vs_static": _comparison(static_rows, v53_rows),
        "permuted_control_vs_static": _comparison(static_rows, perm_rows),
        "v53_vs_v52": _comparison(v52_rows, v53_rows),
        "usage": {
            "eligible_queries": int(sum(bool(r["used_v53"]) for r in query_rows)),
            "eligible_fraction": float(np.mean([bool(r["used_v53"]) for r in query_rows])),
            "permuted_control_eligible_queries": int(sum(bool(r["used_permuted_control"]) for r in query_rows)),
            "mean_protected_count": float(np.mean([int(r["protected_count"]) for r in query_rows])),
        },
        "query_diagnostics": query_rows,
        "guardrail": str(args.guardrail),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "query_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
