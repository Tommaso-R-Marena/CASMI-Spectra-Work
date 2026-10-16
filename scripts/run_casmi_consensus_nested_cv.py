from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.consensus_ranker import (
    CONSENSUS_FEATURE_NAMES,
    conditional_softmax_fit,
    consensus_feature_matrix,
    deterministic_fold,
    selective_consensus_scores,
)
from qfd.metrics import tie_aware_truth_metrics
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test


PRIMARY_COVERAGE = 0.95
L2 = 1.0
TOPK = 10
OUTER_FOLDS = 5
INNER_FOLDS = 4
OUTER_SALT = "consensus-outer-v3"
INNER_SALT_PREFIX = "consensus-inner"
THRESHOLDS = tuple([i / 10 for i in range(11)] + [1.1])
N_BOOT = 100_000
N_SIGNFLIP = 500_000
BOOT_SEED = 20260920
SIGNFLIP_SEED = 20260921


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _query_metric(candidate_ids, scores, truth):
    m = tie_aware_truth_metrics(
        [str(x) for x in candidate_ids],
        [float(x) for x in scores],
        str(truth),
        k=25,
    )
    return {
        "rr": float(m["rr_at_k"]),
        "top1": float(m["top1_credit"]),
        "top5": float(m["top5_credit"]),
        "expected_rank": float(m["expected_rank"]),
        "rank_min": int(m["rank_min"]),
        "rank_max": int(m["rank_max"]),
    }


def _summary(rows, prefix):
    if not rows:
        return {"n": 0}
    return {
        "n": int(len(rows)),
        "mrr25": float(np.mean([r[f"rr_{prefix}"] for r in rows])),
        "top1": float(np.mean([r[f"top1_{prefix}"] for r in rows])),
        "top5": float(np.mean([r[f"top5_{prefix}"] for r in rows])),
        "median_expected_rank": float(np.median([r[f"expected_rank_{prefix}"] for r in rows])),
    }


def _risk(rows, method="gated"):
    catastrophes = int(sum(
        r["expected_rank_static"] <= 3.0 and r[f"expected_rank_{method}"] > 25.0
        for r in rows
    ))
    return {
        "rr_wins": int(sum(r[f"rr_{method}"] > r["rr_static"] + 1e-15 for r in rows)),
        "rr_losses": int(sum(r[f"rr_{method}"] < r["rr_static"] - 1e-15 for r in rows)),
        "rr_ties": int(sum(abs(r[f"rr_{method}"] - r["rr_static"]) <= 1e-15 for r in rows)),
        "delta_mrr": float(np.mean([r[f"rr_{method}"] - r["rr_static"] for r in rows])),
        "delta_top1": float(np.mean([r[f"top1_{method}"] - r["top1_static"] for r in rows])),
        "delta_top5": float(np.mean([r[f"top5_{method}"] - r["top5_static"] for r in rows])),
        "catastrophic_static_top3_to_below25": catastrophes,
    }


def _build_groups(candidate_df: pd.DataFrame, query_ids: set[int]):
    groups = {}
    required = {
        "query_mol_id", "candidate_mol_id", "is_truth", "cos10", "cos20",
        "cos40", "static_best", "ordered_path_score",
    }
    missing = required - set(candidate_df.columns)
    if missing:
        raise RuntimeError(f"candidate feature cache missing columns: {sorted(missing)}")

    for qid, g in candidate_df[candidate_df["query_mol_id"].isin(query_ids)].groupby(
        "query_mol_id", sort=True
    ):
        g = g.reset_index(drop=True)
        truth_rows = np.flatnonzero(g["is_truth"].to_numpy(dtype=int) == 1)
        if len(truth_rows) != 1:
            raise RuntimeError(f"query {qid}: expected exactly one truth, got {len(truth_rows)}")
        X = consensus_feature_matrix(
            g["cos10"], g["cos20"], g["cos40"], g["ordered_path_score"]
        )
        static = g["static_best"].to_numpy(dtype=float)
        recomputed = np.max(g[["cos10", "cos20", "cos40"]].to_numpy(dtype=float), axis=1)
        if not np.allclose(static, recomputed, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"query {qid}: static cache disagrees with max energy cosine")
        groups[int(qid)] = {
            "candidate_ids": g["candidate_mol_id"].to_numpy(dtype=int),
            "truth": int(qid),
            "truth_index": int(truth_rows[0]),
            "X": X,
            "static": static,
        }
    absent = sorted(query_ids - set(groups))
    if absent:
        raise RuntimeError(f"primary queries missing from candidate cache: {absent[:10]}")
    return groups


def _fit(groups, train_ids):
    fit = conditional_softmax_fit(
        [(groups[q]["X"], groups[q]["truth_index"]) for q in train_ids],
        l2=L2,
    )
    if not fit.success or not np.all(np.isfinite(fit.weights)):
        raise RuntimeError("conditional ranker failed to converge")
    return fit


def _row_for_scores(groups, qid, expert, gated, overlap, used):
    g = groups[qid]
    ms = _query_metric(g["candidate_ids"], g["static"], g["truth"])
    me = _query_metric(g["candidate_ids"], expert, g["truth"])
    mg = _query_metric(g["candidate_ids"], gated, g["truth"])
    out = {
        "query_mol_id": int(qid),
        "candidate_count": int(len(g["candidate_ids"])),
        "gate_overlap_top10": float(overlap),
        "gate_used_expert": bool(used),
    }
    for name, m in (("static", ms), ("expert", me), ("gated", mg)):
        for k, v in m.items():
            out[f"{k}_{name}"] = v
    return out


def _inner_select_threshold(groups, train_ids, outer_fold):
    inner_expert = {}
    for inner_fold in range(INNER_FOLDS):
        tr = [
            q for q in train_ids
            if deterministic_fold(
                q, salt=f"{INNER_SALT_PREFIX}-{outer_fold}", n_folds=INNER_FOLDS
            ) != inner_fold
        ]
        te = [
            q for q in train_ids
            if deterministic_fold(
                q, salt=f"{INNER_SALT_PREFIX}-{outer_fold}", n_folds=INNER_FOLDS
            ) == inner_fold
        ]
        if not te:
            continue
        if not tr:
            raise RuntimeError("empty inner training fold")
        fit = _fit(groups, tr)
        for q in te:
            inner_expert[q] = groups[q]["X"] @ fit.weights
    if set(inner_expert) != set(train_ids):
        raise RuntimeError("inner cross-validation did not predict every outer-training query")

    static_rows = []
    for q in train_ids:
        static = groups[q]["static"]
        m = _query_metric(groups[q]["candidate_ids"], static, groups[q]["truth"])
        static_rows.append(m)
    static_top1 = float(np.mean([m["top1"] for m in static_rows]))

    candidates = []
    for threshold in THRESHOLDS:
        rows = []
        for q in train_ids:
            gated, overlap, used = selective_consensus_scores(
                groups[q]["static"],
                inner_expert[q],
                overlap_threshold=threshold,
                k=TOPK,
            )
            rows.append(_row_for_scores(groups, q, inner_expert[q], gated, overlap, used))
        mrr = float(np.mean([r["rr_gated"] for r in rows]))
        top1 = float(np.mean([r["top1_gated"] for r in rows]))
        top5 = float(np.mean([r["top5_gated"] for r in rows]))
        catastrophes = int(sum(
            r["expected_rank_static"] <= 3.0 and r["expected_rank_gated"] > 25.0
            for r in rows
        ))
        feasible = bool(top1 >= static_top1 - 1e-12 and catastrophes == 0)
        candidates.append({
            "threshold": float(threshold),
            "feasible": feasible,
            "mrr25": mrr,
            "top1": top1,
            "top5": top5,
            "catastrophes": catastrophes,
            "expert_fraction": float(np.mean([r["gate_used_expert"] for r in rows])),
        })

    feasible = [x for x in candidates if x["feasible"]]
    if not feasible:
        raise RuntimeError("no feasible reliability threshold, including static fallback")
    selected = max(
        feasible,
        key=lambda x: (x["mrr25"], x["top5"], -x["threshold"]),
    )
    return selected, candidates


def main():
    p = argparse.ArgumentParser(
        description="Nested-CV validation and freezing of CASMI energy-consensus ranker v2"
    )
    p.add_argument("candidate_features", type=Path)
    p.add_argument("per_query", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    candidates = pd.read_csv(args.candidate_features)
    per_query = pd.read_csv(args.per_query)
    primary = per_query[per_query["candidate_coverage_fraction"] >= PRIMARY_COVERAGE].copy()
    query_ids = set(int(x) for x in primary["query_mol_id"])
    if len(query_ids) != len(primary):
        raise RuntimeError("primary set must contain one spectrum per truth molecule")
    groups = _build_groups(candidates, query_ids)
    qids = sorted(groups)

    outer_rows = []
    outer_selection = []
    for outer_fold in range(OUTER_FOLDS):
        train = [
            q for q in qids
            if deterministic_fold(q, salt=OUTER_SALT, n_folds=OUTER_FOLDS) != outer_fold
        ]
        test = [
            q for q in qids
            if deterministic_fold(q, salt=OUTER_SALT, n_folds=OUTER_FOLDS) == outer_fold
        ]
        selected, threshold_table = _inner_select_threshold(groups, train, outer_fold)
        fit = _fit(groups, train)
        outer_selection.append({
            "outer_fold": int(outer_fold),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "selected_threshold": float(selected["threshold"]),
            "inner_selected": selected,
            "inner_threshold_table": threshold_table,
            "weights": [float(x) for x in fit.weights],
        })
        for q in test:
            expert = groups[q]["X"] @ fit.weights
            gated, overlap, used = selective_consensus_scores(
                groups[q]["static"],
                expert,
                overlap_threshold=selected["threshold"],
                k=TOPK,
            )
            row = _row_for_scores(groups, q, expert, gated, overlap, used)
            row["outer_fold"] = int(outer_fold)
            row["selected_threshold"] = float(selected["threshold"])
            outer_rows.append(row)

    if {r["query_mol_id"] for r in outer_rows} != set(qids):
        raise RuntimeError("outer CV did not produce exactly one prediction per query")

    static_rr = [r["rr_static"] for r in outer_rows]
    gated_rr = [r["rr_gated"] for r in outer_rows]
    blocks = [str(r["query_mol_id"]) for r in outer_rows]
    bootstrap = paired_block_bootstrap_delta(
        gated_rr, static_rr, blocks, n_boot=N_BOOT, seed=BOOT_SEED
    )
    signflip = paired_block_signflip_test(
        gated_rr,
        static_rr,
        blocks,
        n_permutations=N_SIGNFLIP,
        seed=SIGNFLIP_SEED,
        exact_max_blocks=20,
    )

    selected_thresholds = [x["selected_threshold"] for x in outer_selection]
    frozen_threshold = float(np.median(selected_thresholds))
    final_fit = _fit(groups, qids)

    static_summary = _summary(outer_rows, "static")
    expert_summary = _summary(outer_rows, "expert")
    gated_summary = _summary(outer_rows, "gated")
    risk = _risk(outer_rows, "gated")
    freeze_gate = {
        "nested_delta_mrr_positive": risk["delta_mrr"] > 0.0,
        "bootstrap_95pct_lower_bound_positive": float(bootstrap["ci_low"]) > 0.0,
        "two_sided_signflip_p_lt_0_01": float(signflip["p_value_two_sided"]) < 0.01,
        "top1_noninferior": risk["delta_top1"] >= -1e-12,
        "zero_static_top3_to_below25_catastrophes": (
            risk["catastrophic_static_top3_to_below25"] == 0
        ),
        "threshold_stability_range_le_0_2": (
            max(selected_thresholds) - min(selected_thresholds) <= 0.2 + 1e-12
        ),
    }
    freeze_gate["pass"] = bool(all(freeze_gate.values()))

    model = {
        "model_name": "CASMI energy-consensus conditional ranker v2",
        "status": "FROZEN_FOR_CASMI2022_IF_GATE_PASSES",
        "feature_names": list(CONSENSUS_FEATURE_NAMES),
        "feature_definition": {
            "z_static_best": "within-query z-score of max(cos10, cos20, cos40)",
            "z_ordered_path": "within-query z-score of ordered 10->20->40 trajectory score",
            "z_neg_energy_spread": "within-query z-score of -std(cos10, cos20, cos40)",
        },
        "training_loss": "query-balanced conditional softmax",
        "l2": L2,
        "weights": [float(x) for x in final_fit.weights],
        "gate": {
            "type": "top-k static/expert set overlap",
            "top_k": TOPK,
            "overlap_threshold": frozen_threshold,
            "fallback": "static_best",
        },
        "training_queries": [int(q) for q in qids],
        "training_query_count": int(len(qids)),
        "primary_training_coverage_threshold": PRIMARY_COVERAGE,
        "candidate_features_sha256": _sha256(args.candidate_features),
        "per_query_sha256": _sha256(args.per_query),
        "freeze_gate_passed": bool(freeze_gate["pass"]),
        "casmi2022_policy": (
            "Apply these exact features, weights, gate, preprocessing, and candidate policy "
            "once to CASMI 2022. No CASMI 2022 outcome may be used to alter this model."
        ),
    }

    summary = {
        "protocol": {
            "dataset_role": "CASMI 2016 development only; CASMI 2022 remains external lockbox",
            "primary_coverage_threshold": PRIMARY_COVERAGE,
            "n_queries": len(qids),
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "outer_salt": OUTER_SALT,
            "inner_salt_prefix": INNER_SALT_PREFIX,
            "l2": L2,
            "top_k": TOPK,
            "threshold_grid": list(THRESHOLDS),
            "threshold_selection": (
                "inner out-of-fold: require Top-1 noninferiority and zero static-top3->below25 "
                "catastrophes; maximize MRR@25, then Top-5, then smaller threshold"
            ),
            "frozen_threshold_rule": "median of five independently selected outer-fold thresholds",
            "no_casmi2022_outcomes_seen": True,
        },
        "nested_oof": {
            "static": static_summary,
            "expert": expert_summary,
            "gated": gated_summary,
            "risk": risk,
            "bootstrap": bootstrap,
            "signflip": signflip,
            "outer_selection": outer_selection,
        },
        "frozen_threshold": frozen_threshold,
        "final_fit_weights": [float(x) for x in final_fit.weights],
        "freeze_gate": freeze_gate,
        "interpretation_guardrail": (
            "Nested CASMI 2016 results are development evidence, not independent external "
            "generalization. A general empirical claim requires the frozen model to replicate "
            "on CASMI 2022 without modification."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "frozen_model.json").write_text(json.dumps(model, indent=2) + "\n")
    pd.DataFrame(outer_rows).to_csv(args.out_dir / "nested_oof_predictions.csv", index=False)
    print(json.dumps({
        "freeze_gate": freeze_gate,
        "static": static_summary,
        "gated": gated_summary,
        "risk": risk,
        "bootstrap": bootstrap,
        "signflip": signflip,
        "frozen_threshold": frozen_threshold,
        "weights": model["weights"],
    }, indent=2))


if __name__ == "__main__":
    main()
