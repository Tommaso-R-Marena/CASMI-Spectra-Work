from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.consensus_ranker import consensus_feature_matrix, selective_consensus_scores
from run_casmi2022_consensus_lockbox import _metric, _paired, _risk, _summarize


ORDER_AUDIT_SEED = 20260925
ORDER_AUDIT_REPS = 100


def _scores_from_candidate_frame(g: pd.DataFrame, weights: np.ndarray) -> dict[str, np.ndarray]:
    c10 = g["cos10"].to_numpy(float)
    c20 = g["cos20"].to_numpy(float)
    c40 = g["cos40"].to_numpy(float)
    path = g["ordered_path_score"].to_numpy(float)
    C = np.column_stack([c10, c20, c40])
    X = consensus_feature_matrix(c10, c20, c40, path)

    # Frozen expert and fixed-weight component ablations.  No weights are refit.
    full = X @ weights
    no_path = X @ np.array([weights[0], 0.0, weights[2]], dtype=float)
    no_spread = X @ np.array([weights[0], weights[1], 0.0], dtype=float)
    static_component = X @ np.array([weights[0], 0.0, 0.0], dtype=float)

    return {
        "static": np.max(C, axis=1),
        "expert": full,
        "no_path": no_path,
        "no_spread": no_spread,
        "static_component": static_component,
        "cos10": c10,
        "cos20": c20,
        "cos40": c40,
        "mean_energy_cosine": np.mean(C, axis=1),
        "min_energy_cosine": np.min(C, axis=1),
        "neg_energy_spread": -np.std(C, axis=1),
    }


def _gate(static: np.ndarray, expert: np.ndarray, top_k: int, threshold: float):
    return selective_consensus_scores(
        static,
        expert,
        overlap_threshold=threshold,
        k=top_k,
    )


def _gate_order_audit(
    candidate_ids: np.ndarray,
    static: np.ndarray,
    expert: np.ndarray,
    *,
    top_k: int,
    threshold: float,
    spec_id: int,
) -> dict:
    _, canonical_overlap, canonical_used = _gate(static, expert, top_k, threshold)
    rng = np.random.default_rng(ORDER_AUDIT_SEED + int(spec_id))
    used_values = []
    overlaps = []
    for _ in range(ORDER_AUDIT_REPS):
        p = rng.permutation(len(candidate_ids))
        _, ov, used = _gate(static[p], expert[p], top_k, threshold)
        used_values.append(bool(used))
        overlaps.append(float(ov))
    return {
        "canonical_overlap": float(canonical_overlap),
        "canonical_used_expert": bool(canonical_used),
        "n_permutations": ORDER_AUDIT_REPS,
        "gate_decision_invariant": bool(all(x == canonical_used for x in used_values)),
        "overlap_invariant": bool(all(abs(x - canonical_overlap) <= 1e-15 for x in overlaps)),
        "overlap_min": float(min(overlaps)),
        "overlap_max": float(max(overlaps)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-lockbox fixed-protocol robustness analyses from persisted CASMI-2022 candidate features"
    )
    p.add_argument("candidate_features", type=Path)
    p.add_argument("primary_summary", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    primary = json.loads(args.primary_summary.read_text())
    model = json.loads(args.frozen_model.read_text())
    weights = np.asarray(model["weights"], dtype=float)
    top_k = int(model["gate"]["top_k"])
    threshold = float(model["gate"]["overlap_threshold"])

    cf = pd.read_csv(args.candidate_features)
    required = {
        "spec_id", "query_mol_id", "candidate_mol_id", "is_truth",
        "cos10", "cos20", "cos40", "static_best", "ordered_path_score", "expert_score",
    }
    if not required.issubset(cf.columns):
        raise RuntimeError(f"candidate feature table missing: {sorted(required - set(cf.columns))}")

    method_names = [
        "static", "expert", "gated",
        "no_path_gated", "no_spread_gated", "static_component_gated",
        "cos10", "cos20", "cos40", "mean_energy_cosine", "min_energy_cosine",
        "neg_energy_spread",
    ]
    rows: list[dict] = []
    order_audits: list[dict] = []

    for spec_id, g0 in cf.groupby("spec_id", sort=True):
        g = g0.sort_values("candidate_mol_id", kind="stable").reset_index(drop=True)
        truths = g.loc[g["is_truth"].astype(int) == 1, "candidate_mol_id"].astype(int).unique()
        if len(truths) != 1:
            raise RuntimeError(f"spec {spec_id}: expected one truth candidate, found {len(truths)}")
        truth = int(truths[0])
        qids = g["query_mol_id"].astype(int).unique()
        if len(qids) != 1 or int(qids[0]) != truth:
            raise RuntimeError(f"spec {spec_id}: inconsistent query/truth IDs")
        ids = g["candidate_mol_id"].to_numpy(int)
        scores = _scores_from_candidate_frame(g, weights)

        # Reconstruct the frozen expert exactly before running any secondary analysis.
        if not np.allclose(scores["static"], g["static_best"].to_numpy(float), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"spec {spec_id}: static reconstruction mismatch")
        if not np.allclose(scores["expert"], g["expert_score"].to_numpy(float), rtol=0.0, atol=1e-10):
            raise RuntimeError(f"spec {spec_id}: expert reconstruction mismatch")

        gated, _, _ = _gate(scores["static"], scores["expert"], top_k, threshold)
        no_path_gated, _, _ = _gate(scores["static"], scores["no_path"], top_k, threshold)
        no_spread_gated, _, _ = _gate(scores["static"], scores["no_spread"], top_k, threshold)
        static_component_gated, _, _ = _gate(scores["static"], scores["static_component"], top_k, threshold)
        score_sets = {
            "static": scores["static"],
            "expert": scores["expert"],
            "gated": gated,
            "no_path_gated": no_path_gated,
            "no_spread_gated": no_spread_gated,
            "static_component_gated": static_component_gated,
            "cos10": scores["cos10"],
            "cos20": scores["cos20"],
            "cos40": scores["cos40"],
            "mean_energy_cosine": scores["mean_energy_cosine"],
            "min_energy_cosine": scores["min_energy_cosine"],
            "neg_energy_spread": scores["neg_energy_spread"],
        }
        out = {"spec_id": int(spec_id), "query_mol_id": truth}
        for name, s in score_sets.items():
            m, _ = _metric(ids.tolist(), np.asarray(s, dtype=float).tolist(), truth)
            for k, v in m.items():
                out[f"{k}_{name}"] = v
        rows.append(out)

        oa = _gate_order_audit(
            ids,
            scores["static"],
            scores["expert"],
            top_k=top_k,
            threshold=threshold,
            spec_id=int(spec_id),
        )
        oa.update({"spec_id": int(spec_id), "query_mol_id": truth})
        order_audits.append(oa)

    if not rows:
        raise RuntimeError("no candidate-feature query groups found")

    # Verify the persisted primary result is exactly reconstructable from the artifact.
    primary_gated = primary["primary_ge_0_99"]["metrics"]["gated"]
    reconstructed_gated = _summarize(rows, "gated")
    for key in ("mrr25", "top1", "top5", "median_expected_rank"):
        if not np.isclose(float(primary_gated[key]), float(reconstructed_gated[key]), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"primary reconstruction mismatch for {key}")

    metrics = {name: _summarize(rows, name) for name in method_names}
    risks = {
        name: _risk(rows, name)
        for name in method_names
        if name != "static"
    }
    inference = {
        name: _paired(rows, name, 10000 + 1000 * i)
        for i, name in enumerate(method_names)
        if name != "static"
    }

    order_df = pd.DataFrame(order_audits)
    result = {
        "status": "post-lockbox robustness analysis; no parameter fitting or primary-gate changes",
        "primary_result_reconstructed_exactly": True,
        "frozen_weights": [float(x) for x in weights],
        "frozen_gate": {"top_k": top_k, "overlap_threshold": threshold},
        "n_queries": int(len(rows)),
        "n_truth_blocks": int(len({r["query_mol_id"] for r in rows})),
        "metrics": metrics,
        "risk_vs_static": risks,
        "paired_inference_vs_static": inference,
        "gate_order_invariance": {
            "permutations_per_query": ORDER_AUDIT_REPS,
            "queries_with_decision_instability": int((~order_df["gate_decision_invariant"]).sum()),
            "queries_with_overlap_instability": int((~order_df["overlap_invariant"]).sum()),
            "all_gate_decisions_invariant": bool(order_df["gate_decision_invariant"].all()),
            "all_overlap_values_invariant": bool(order_df["overlap_invariant"].all()),
            "details": order_audits,
        },
        "interpretation": {
            "no_path_gated": "same frozen static/spread weights with ordered-path weight set to zero; no refit",
            "no_spread_gated": "same frozen static/path weights with energy-consistency weight set to zero; no refit",
            "static_component_gated": "frozen consensus representation retaining only its static component; no refit",
            "single_energy_and_simple_consensus": "parameter-free descriptive baselines on the identical candidate pools",
            "gate_order_audit": "100 fixed-seed candidate-order permutations per query; detects exact-tie dependence in the top-k reliability gate",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "n_queries": result["n_queries"],
        "primary_reconstructed": result["primary_result_reconstructed_exactly"],
        "gated": metrics["gated"],
        "no_path_gated": metrics["no_path_gated"],
        "no_spread_gated": metrics["no_spread_gated"],
        "mean_energy_cosine": metrics["mean_energy_cosine"],
        "gate_order_invariance": result["gate_order_invariance"],
    }, indent=2))


if __name__ == "__main__":
    main()
