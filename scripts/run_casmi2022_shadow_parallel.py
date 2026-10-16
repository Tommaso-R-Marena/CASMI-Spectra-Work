from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.consensus_ranker import CONSENSUS_FEATURE_NAMES, consensus_feature_matrix, selective_consensus_scores
from qfd.spectra import matched_cosine
from run_casmi2016_cfm_transfer import ENERGIES, TOLERANCE_DA, _clean_raw, _find_one, _pessimistic_missing_bounds, _sqrt_view
from run_casmi2016_trajectory_dev import _path_score
from run_casmi2022_consensus_lockbox import (
    CONTROL_SEED,
    PRIMARY_COVERAGE,
    _load_energy_frames,
    _metric,
    _paired,
    _risk,
    _summarize,
)


_G_ENERGY_FRAMES = None
_G_WEIGHTS = None
_G_TOP_K = None
_G_THRESHOLD = None


def _score_one(task):
    spec_id, truth, peaks, prec_mz, original, supported = task
    energy_frames = _G_ENERGY_FRAMES
    weights = _G_WEIGHTS
    top_k = _G_TOP_K
    threshold = _G_THRESHOLD
    if energy_frames is None or weights is None:
        raise RuntimeError("worker globals were not initialized")

    coverage = len(supported) / len(original)
    q_raw = _clean_raw(peaks, prec_mz, 0.0)
    q_sqrt = _sqrt_view(q_raw)
    ids = list(supported)
    energy_slices = {
        e: list(energy_frames[e].loc[ids].itertuples(index=False))
        for e in ENERGIES
    }
    static_scores = []
    path_scores = []
    energy_cosines = []
    for i, cid in enumerate(ids):
        refs_raw = []
        cos = []
        for e in ENERGIES:
            r = energy_slices[e][i]
            raw = _clean_raw(r.peaks, r.prec_mz, e)
            refs_raw.append(raw)
            sr = _sqrt_view(raw)
            cos.append(float(matched_cosine(
                q_sqrt.mzs,
                q_sqrt.intensities,
                sr.mzs,
                sr.intensities,
                tolerance_da=TOLERANCE_DA,
            )))
        path, _, _ = _path_score(q_raw, refs_raw)
        if not math.isfinite(path):
            raise RuntimeError(f"non-finite ordered-path score for spec={spec_id}, candidate={cid}")
        static_scores.append(float(max(cos)))
        path_scores.append(float(path))
        energy_cosines.append(cos)

    C = np.asarray(energy_cosines, dtype=float)
    X = consensus_feature_matrix(C[:, 0], C[:, 1], C[:, 2], path_scores)
    expert = X @ weights
    gated, overlap, used = selective_consensus_scores(
        static_scores, expert, overlap_threshold=threshold, k=top_k
    )
    rng = np.random.default_rng(CONTROL_SEED + int(spec_id))
    perm_expert = expert[rng.permutation(len(expert))]
    control, control_overlap, control_used = selective_consensus_scores(
        static_scores, perm_expert, overlap_threshold=threshold, k=top_k
    )

    score_sets = {
        "static": np.asarray(static_scores, dtype=float),
        "expert": np.asarray(expert, dtype=float),
        "gated": np.asarray(gated, dtype=float),
        "permuted_control": np.asarray(control, dtype=float),
    }
    missing = len(original) - len(ids)
    out = {
        "spec_id": int(spec_id),
        "query_mol_id": int(truth),
        "candidate_count_original": len(original),
        "candidate_count": len(ids),
        "missing_candidate_count": missing,
        "candidate_coverage_fraction": float(coverage),
        "gate_overlap_top10": float(overlap),
        "gate_used_expert": bool(used),
        "control_overlap_top10": float(control_overlap),
        "control_used_expert": bool(control_used),
    }
    for name, scores in score_sets.items():
        metric, raw_metric = _metric(ids, scores, truth)
        pess = _pessimistic_missing_bounds(raw_metric, missing)
        for k, v in metric.items():
            out[f"{k}_{name}"] = v
        out[f"rr_pessimistic_{name}"] = float(pess["rr"])
        out[f"top1_pessimistic_{name}"] = float(pess["top1"])
        out[f"top5_pessimistic_{name}"] = float(pess["top5"])
    print(
        f"shadow spec={spec_id} candidates={len(ids)} overlap={overlap:.2f} expert={used} "
        f"static_rank={out['expected_rank_static']:.1f} gated_rank={out['expected_rank_gated']:.1f}",
        flush=True,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel metrics-only replication of the frozen CASMI-2022 lockbox scorer")
    p.add_argument("casmi_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    model = json.loads(args.frozen_model.read_text())
    if not bool(model.get("freeze_gate_passed", False)):
        raise RuntimeError("frozen development gate did not pass")
    if tuple(model.get("feature_names", ())) != CONSENSUS_FEATURE_NAMES:
        raise RuntimeError("frozen feature schema mismatch")
    weights = np.asarray(model["weights"], dtype=float)
    top_k = int(model["gate"]["top_k"])
    threshold = float(model["gate"]["overlap_threshold"])

    spec = pd.read_pickle(_find_one(args.casmi_root, "spec_df.pkl"))
    cand = pd.read_pickle(_find_one(args.casmi_root, "cand_df.pkl"))
    spec = spec[spec["prec_type"] == "[M+H]+"].copy()
    by_query = cand.groupby("query_mol_id")["candidate_mol_id"].agg(
        lambda s: tuple(sorted(set(int(x) for x in s)))
    )
    energy_frames, complete_ids = _load_energy_frames(_find_one(args.cfm_root, "casmi22_spec_df.pkl"))

    tasks = []
    excluded = {"no_candidate_list": 0, "truth_not_predicted": 0, "coverage_below_0_99": 0}
    for qrow in spec.itertuples(index=False):
        truth = int(qrow.mol_id)
        original = tuple(int(x) for x in by_query.get(truth, ()))
        if not original:
            excluded["no_candidate_list"] += 1
            continue
        supported = tuple(c for c in original if c in complete_ids)
        if truth not in complete_ids:
            excluded["truth_not_predicted"] += 1
            continue
        if len(supported) / len(original) < PRIMARY_COVERAGE:
            excluded["coverage_below_0_99"] += 1
            continue
        tasks.append((int(qrow.spec_id), truth, qrow.peaks, qrow.prec_mz, original, supported))

    if not tasks:
        raise RuntimeError("no eligible lockbox tasks")

    global _G_ENERGY_FRAMES, _G_WEIGHTS, _G_TOP_K, _G_THRESHOLD
    _G_ENERGY_FRAMES = energy_frames
    _G_WEIGHTS = weights
    _G_TOP_K = top_k
    _G_THRESHOLD = threshold

    max_workers = max(1, min(4, int(os.cpu_count() or 1)))
    workers = max_workers if args.workers <= 0 else max(1, min(int(args.workers), max_workers))
    ctx = mp.get_context("fork")
    print(f"shadow scoring {len(tasks)} spectra with {workers} fork workers", flush=True)
    with ctx.Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_score_one, tasks, chunksize=1))
    rows.sort(key=lambda r: int(r["spec_id"]))

    methods = ("static", "expert", "gated", "permuted_control")
    metrics = {m: _summarize(rows, m) for m in methods}
    risks = {m: _risk(rows, m) for m in ("expert", "gated", "permuted_control")}
    inference = {
        "gated_vs_static": _paired(rows, "gated", 0),
        "expert_vs_static": _paired(rows, "expert", 1000),
        "permuted_control_vs_static": _paired(rows, "permuted_control", 2000),
    }
    full_rows = [r for r in rows if r["candidate_coverage_fraction"] >= 1.0 - 1e-15]
    full_metrics = {m: _summarize(full_rows, m) for m in methods}
    full_inference = _paired(full_rows, "gated", 3000) if full_rows else None

    gated_risk = risks["gated"]
    primary_boot = inference["gated_vs_static"]["bootstrap"]
    primary_sign = inference["gated_vs_static"]["signflip"]
    control_delta = risks["permuted_control"]["delta_mrr"]
    replication_gate = {
        "delta_mrr_positive": gated_risk["delta_mrr"] > 0.0,
        "bootstrap_95pct_lower_bound_positive": float(primary_boot["ci_low"]) > 0.0,
        "two_sided_signflip_p_lt_0_05": float(primary_sign["p_value_two_sided"]) < 0.05,
        "top1_noninferior": gated_risk["delta_top1"] >= -1e-12,
        "zero_static_top3_to_below25_catastrophes": gated_risk["catastrophic_static_top3_to_below25"] == 0,
        "real_delta_mrr_exceeds_permuted_control": gated_risk["delta_mrr"] > control_delta,
    }
    replication_gate["pass"] = bool(all(replication_gate.values()))

    result = {
        "status": "noncanonical parallel metrics-only exact-protocol replication; frozen before CASMI-2022 outcomes and not eligible for tuning",
        "workers": workers,
        "coverage": {
            "positive_mh_spectra": int(len(spec)),
            "primary_scored_spectra": int(len(rows)),
            "primary_truth_blocks": int(len(set(r["query_mol_id"] for r in rows))),
            "excluded": excluded,
            "full_coverage_scored_spectra": int(len(full_rows)),
        },
        "primary_ge_0_99": {
            "metrics": metrics,
            "risk": risks,
            "inference": inference,
            "expert_usage_fraction": float(np.mean([r["gate_used_expert"] for r in rows])),
            "replication_gate": replication_gate,
        },
        "full_coverage_sensitivity": {
            "metrics": full_metrics,
            "gated_vs_static_inference": full_inference,
        },
        "per_query": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "coverage": result["coverage"],
        "static": metrics["static"],
        "gated": metrics["gated"],
        "risk": gated_risk,
        "inference": inference["gated_vs_static"],
        "control": risks["permuted_control"],
        "expert_usage_fraction": result["primary_ge_0_99"]["expert_usage_fraction"],
        "replication_gate": replication_gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
