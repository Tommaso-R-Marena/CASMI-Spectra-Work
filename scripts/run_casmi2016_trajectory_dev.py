from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.dynamic_variants import candidate_zscores
from qfd.merged import _cluster_grid
from qfd.trajectory_geometry import ordered_path_projection
from qfd.spectra import matched_cosine

from run_casmi2016_cfm_transfer import (
    BASELINE,
    ENERGIES,
    PRIMARY_MIN_COVERAGE,
    SCORED_MIN_COVERAGE,
    TOLERANCE_DA,
    _clean_raw,
    _find_one,
    _paired_inference,
    _pessimistic_missing_bounds,
    _sqrt_view,
    _summarize,
    _truth_metrics,
)


SPLIT_SALT = "casmi2016-trajectory-v1"
METHODS = (
    BASELINE,
    "ordered_path",
    "hybrid_path_linear_050",
    "hybrid_path_tanh_050",
    "hybrid_path_tanh_025_guard_top1",
    "hybrid_path_tanh_050_guard_top1",
)
# This order is an explicit simplicity/safety tie-breaker only; performance is
# compared first. The static baseline is a legitimate fallback if geometry does
# not improve the development subset.
SELECTION_TIEBREAK_ORDER = (
    BASELINE,
    "hybrid_path_tanh_025_guard_top1",
    "hybrid_path_tanh_050_guard_top1",
    "hybrid_path_tanh_050",
    "hybrid_path_linear_050",
    "ordered_path",
)


def _split(query_mol_id: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{int(query_mol_id)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "development" if bucket < 7 else "internal_validation"


def _path_score(query_raw, refs_raw):
    _, M = _cluster_grid([query_raw] + refs_raw, TOLERANCE_DA)
    M = np.maximum(np.asarray(M, float), 0.0)
    if M.shape[1] != 4:
        return float("nan"), -1, float("nan")
    sums = M.sum(axis=0)
    if np.any(sums <= 0):
        return float("nan"), -1, float("nan")
    P = M / sums[None, :]
    d, seg, t = ordered_path_projection(P[:, 0], P[:, 1:])
    return -float(d), int(seg), float(t)


def _z(values, ids):
    m = candidate_zscores({str(k): float(v) for k, v in zip(ids, values)})
    return np.asarray([m[str(k)] for k in ids], dtype=float)


def _guard_static_top_tie(static_scores, fused_scores):
    """Keep the exact static top-tie set at rank 1, rerank everything else.

    This makes Top-1 credit exactly non-inferior to the static baseline by
    construction while still allowing geometry to reorder all non-leading
    candidates. It is intentionally conservative because CASMI 2016 showed
    catastrophic demotions of otherwise excellent static winners.
    """
    s = np.asarray(static_scores, dtype=float)
    f = np.asarray(fused_scores, dtype=float).copy()
    top = np.isclose(s, np.max(s), rtol=0.0, atol=1e-12)
    guard_value = float(np.max(f[np.isfinite(f)])) + 10.0 if np.any(np.isfinite(f)) else 10.0
    f[top] = guard_value
    return f


def _method_scores(candidate_ids, static_scores, path_scores):
    zs = _z(static_scores, candidate_ids)
    zp = _z(path_scores, candidate_ids)
    linear_050 = zs + 0.5 * zp
    tanh_050 = zs + 0.5 * np.tanh(zp)
    tanh_025 = zs + 0.25 * np.tanh(zp)
    return {
        BASELINE: list(map(float, static_scores)),
        "ordered_path": list(map(float, path_scores)),
        "hybrid_path_linear_050": linear_050.tolist(),
        "hybrid_path_tanh_050": tanh_050.tolist(),
        "hybrid_path_tanh_025_guard_top1": _guard_static_top_tie(static_scores, tanh_025).tolist(),
        "hybrid_path_tanh_050_guard_top1": _guard_static_top_tie(static_scores, tanh_050).tolist(),
    }


def _risk_summary(rows, method):
    if not rows:
        return {}
    dr = np.asarray([r[f"expected_rank_{method}"] - r[f"expected_rank_{BASELINE}"] for r in rows], float)
    drr = np.asarray([r[f"rr_{method}"] - r[f"rr_{BASELINE}"] for r in rows], float)
    return {
        "rr_wins": int(np.sum(drr > 1e-15)),
        "rr_losses": int(np.sum(drr < -1e-15)),
        "rr_ties": int(np.sum(np.abs(drr) <= 1e-15)),
        "mean_delta_rr": float(np.mean(drr)),
        "median_rank_change_new_minus_static": float(np.median(dr)),
        "rank_change_q05": float(np.quantile(dr, 0.05)),
        "rank_change_q95": float(np.quantile(dr, 0.95)),
        "catastrophic_demotions_static_top3_to_below25": int(sum(
            r[f"expected_rank_{BASELINE}"] <= 3.0 and r[f"expected_rank_{method}"] > 25.0
            for r in rows
        )),
        "catastrophic_demotions_static_top5_to_below25": int(sum(
            r[f"expected_rank_{BASELINE}"] <= 5.0 and r[f"expected_rank_{method}"] > 25.0
            for r in rows
        )),
        "top1_delta": float(np.mean([r[f"top1_{method}"] - r[f"top1_{BASELINE}"] for r in rows])),
        "top5_delta": float(np.mean([r[f"top5_{method}"] - r[f"top5_{BASELINE}"] for r in rows])),
    }


def _select_on_development(rows):
    metrics = _summarize(rows, METHODS)
    order = {m: i for i, m in enumerate(SELECTION_TIEBREAK_ORDER)}
    ranked = sorted(
        METHODS,
        key=lambda m: (
            -metrics[f"mrr25_tieaware_{m}"],
            -metrics[f"top5_tieaware_{m}"],
            order[m],
        ),
    )
    return ranked[0], ranked, metrics


def main():
    import argparse

    p = argparse.ArgumentParser(description="Pre-registered CASMI 2016 development test of ordered collision-energy trajectory geometry")
    p.add_argument("casmi_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=271828)
    args = p.parse_args()

    spec = pd.read_pickle(_find_one(args.casmi_root, "spec_df.pkl"))
    cand = pd.read_pickle(_find_one(args.casmi_root, "cand_df.pkl"))
    cfm = pd.read_pickle(_find_one(args.cfm_root, "casmi_spec_df.pkl"))
    spec = spec[spec["prec_type"] == "[M+H]+"].copy()
    cfm = cfm[cfm["prec_type"] == "[M+H]+"].copy()

    by_query = cand.groupby("query_mol_id")["candidate_mol_id"].agg(lambda s: tuple(sorted(set(int(x) for x in s))))
    cfm_groups = {}
    for mid, g in cfm.groupby("mol_id", sort=False):
        rows_by_e = {}
        for row in g.itertuples(index=False):
            e = int(row.ace)
            if e in ENERGIES:
                if e in rows_by_e:
                    raise RuntimeError(f"duplicate CFM prediction for mol_id={mid}, energy={e}")
                rows_by_e[e] = row
        if tuple(sorted(rows_by_e)) == ENERGIES:
            cfm_groups[int(mid)] = rows_by_e

    eligible = []
    coverage_rows = []
    excluded = {"no_candidate_list": 0, "truth_not_predicted": 0, "coverage_below_90pct": 0}
    for row in spec.itertuples(index=False):
        truth = int(row.mol_id)
        original = tuple(int(x) for x in by_query.get(truth, ()))
        if not original:
            excluded["no_candidate_list"] += 1
            continue
        supported = tuple(c for c in original if c in cfm_groups)
        frac = len(supported) / len(original)
        coverage_rows.append((int(row.spec_id), truth, len(original), len(supported), frac))
        if truth not in cfm_groups:
            excluded["truth_not_predicted"] += 1
            continue
        if frac < SCORED_MIN_COVERAGE:
            excluded["coverage_below_90pct"] += 1
            continue
        eligible.append((row, original, supported, frac))

    result_rows = []
    candidate_rows = []
    for qi, (qrow, original, candidate_ids, coverage_fraction) in enumerate(eligible):
        q_raw = _clean_raw(qrow.peaks, qrow.prec_mz, 0.0)
        q_sqrt = _sqrt_view(q_raw)
        static_scores, path_scores = [], []
        local_diag = []
        for cid in candidate_ids:
            refs_raw, refs_sqrt = [], []
            energy_cos = []
            for e in ENERGIES:
                r = cfm_groups[cid][e]
                raw = _clean_raw(r.peaks, r.prec_mz, e)
                refs_raw.append(raw)
                sr = _sqrt_view(raw)
                refs_sqrt.append(sr)
                energy_cos.append(float(matched_cosine(
                    q_sqrt.mzs, q_sqrt.intensities, sr.mzs, sr.intensities,
                    tolerance_da=TOLERANCE_DA,
                )))
            static = max(energy_cos)
            path, seg, t = _path_score(q_raw, refs_raw)
            if not math.isfinite(path):
                path = -math.inf
            static_scores.append(float(static))
            path_scores.append(float(path))
            local_diag.append((cid, energy_cos, seg, t))

        score_sets = _method_scores(candidate_ids, static_scores, path_scores)
        missing = len(original) - len(candidate_ids)
        out = {
            "spec_id": int(qrow.spec_id),
            "query_mol_id": int(qrow.mol_id),
            "split": _split(int(qrow.mol_id)),
            "candidate_count_original": len(original),
            "candidate_count": len(candidate_ids),
            "missing_candidate_count": missing,
            "candidate_coverage_fraction": float(coverage_fraction),
        }
        for method, values in score_sets.items():
            m = _truth_metrics(list(candidate_ids), list(values), int(qrow.mol_id))
            pess = _pessimistic_missing_bounds(m, missing)
            out[f"rank_min_{method}"] = m["rank_min"]
            out[f"rank_max_{method}"] = m["rank_max"]
            out[f"expected_rank_{method}"] = m["expected_rank"]
            out[f"tie_size_{method}"] = m["tie_size"]
            out[f"rr_{method}"] = m["rr_at_k"]
            out[f"top1_{method}"] = m["top1_credit"]
            out[f"top5_{method}"] = m["top5_credit"]
            out[f"rr_pessimistic_{method}"] = pess["rr"]
            out[f"top1_pessimistic_{method}"] = pess["top1"]
            out[f"top5_pessimistic_{method}"] = pess["top5"]
        result_rows.append(out)

        # Cache raw candidate features so later diagnostics do not require a
        # second 1.1-GB CFM download or another expensive spectral pass.
        for idx, (cid, energy_cos, seg, t) in enumerate(local_diag):
            candidate_rows.append({
                "spec_id": int(qrow.spec_id),
                "query_mol_id": int(qrow.mol_id),
                "split": out["split"],
                "candidate_mol_id": int(cid),
                "is_truth": int(cid == int(qrow.mol_id)),
                "cos10": energy_cos[0],
                "cos20": energy_cos[1],
                "cos40": energy_cos[2],
                "static_best": static_scores[idx],
                "ordered_path_score": path_scores[idx],
                "path_segment": seg,
                "path_t": t,
            })
        print(
            f"[{qi + 1}/{len(eligible)}] spec={qrow.spec_id} split={out['split']} "
            f"coverage={coverage_fraction:.3f} supported={len(candidate_ids)}/{len(original)}",
            flush=True,
        )

    primary = [r for r in result_rows if r["candidate_coverage_fraction"] >= PRIMARY_MIN_COVERAGE]
    dev = [r for r in primary if r["split"] == "development"]
    val = [r for r in primary if r["split"] == "internal_validation"]
    selected, dev_ranking, dev_metrics = _select_on_development(dev)
    val_metrics = _summarize(val, METHODS)

    selected_val_delta = float(val_metrics[f"mrr25_tieaware_{selected}"] - val_metrics[f"mrr25_tieaware_{BASELINE}"])
    selected_val_risk = _risk_summary(val, selected)
    validation_gate = {
        "selected_is_not_static_fallback": selected != BASELINE,
        "validation_delta_mrr_positive": selected_val_delta > 0.0,
        "validation_top1_noninferior": selected_val_risk.get("top1_delta", -1.0) >= -1e-12,
        "validation_zero_top3_to_below25_catastrophes": selected_val_risk.get("catastrophic_demotions_static_top3_to_below25", 1) == 0,
    }
    validation_gate["pass"] = bool(all(validation_gate.values()))

    summary = {
        "preregistration": {
            "split_salt": SPLIT_SALT,
            "split_rule": "SHA256(salt:query_mol_id) first 64 bits mod 10; buckets 0-6 development, 7-9 internal validation",
            "primary_candidate_coverage": PRIMARY_MIN_COVERAGE,
            "method_menu_fixed_before_ordered_path_ranks": list(METHODS),
            "selection": "maximize development MRR@25 tie-aware; then development Top-5; then fixed simplicity/safety order",
            "selection_tiebreak_order": list(SELECTION_TIEBREAK_ORDER),
            "internal_validation_gate": "non-static selection, positive delta MRR, Top-1 noninferior, zero static-top3 to below-25 catastrophic demotions",
            "casmi2022_lockbox_policy": "do not inspect CASMI 2022 ranking outcomes until a single method passes this internal validation gate and is frozen",
        },
        "theory": {
            "ordered_path": "project query onto union conv(10eV,20eV) U conv(20eV,40eV), excluding arbitrary full-triangle mixtures",
            "bounded_fusion": "z_static + alpha*tanh(z_path); tanh bounds candidate correction to [-alpha,+alpha]",
            "margin_preservation": "for bounded fusion, any candidate pair with static-z gap > 2*alpha cannot reverse order",
            "top1_guard": "exact static top-tie set is forced to remain the top tie; therefore Top-1 credit equals static baseline by construction",
        },
        "coverage": {
            "positive_mh_query_spectra": int(len(spec)),
            "complete_cfm_molecules": int(len(cfm_groups)),
            "scored_ge_90pct": int(len(result_rows)),
            "primary_ge_95pct": int(len(primary)),
            "development_n": int(len(dev)),
            "internal_validation_n": int(len(val)),
            "excluded": excluded,
        },
        "development": {
            "method_ranking": dev_ranking,
            "selected_method": selected,
            "metrics": dev_metrics,
            "risk": {m: _risk_summary(dev, m) for m in METHODS if m != BASELINE},
            "selected_vs_static_inference": _paired_inference(dev, selected, BASELINE, args.seed + 100),
        },
        "internal_validation": {
            "selected_method": selected,
            "metrics": val_metrics,
            "risk": {m: _risk_summary(val, m) for m in METHODS if m != BASELINE},
            "selected_vs_static_inference": _paired_inference(val, selected, BASELINE, args.seed + 200),
            "selected_delta_mrr": selected_val_delta,
            "gate": validation_gate,
        },
        "all_primary_diagnostic_only": {
            "metrics": _summarize(primary, METHODS),
            "risk": {m: _risk_summary(primary, m) for m in METHODS if m != BASELINE},
            "warning": "not used for model selection because it mixes development and held-out internal-validation queries",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if result_rows:
        with (args.out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result_rows[0]))
            w.writeheader(); w.writerows(result_rows)
    if candidate_rows:
        with gzip.open(args.out_dir / "candidate_features.csv.gz", "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(candidate_rows[0]))
            w.writeheader(); w.writerows(candidate_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
