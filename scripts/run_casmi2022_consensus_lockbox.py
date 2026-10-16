from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.consensus_ranker import (
    CONSENSUS_FEATURE_NAMES,
    consensus_feature_matrix,
    selective_consensus_scores,
)
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test
from qfd.spectra import matched_cosine

from run_casmi2016_cfm_transfer import (
    ENERGIES,
    TOLERANCE_DA,
    _clean_raw,
    _find_one,
    _pessimistic_missing_bounds,
    _sqrt_view,
    _truth_metrics,
)
from run_casmi2016_trajectory_dev import _path_score


PRIMARY_COVERAGE = 0.99
N_BOOT = 100_000
N_SIGNFLIP = 500_000
BOOT_SEED = 20260922
SIGNFLIP_SEED = 20260923
CONTROL_SEED = 20260924


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _metric(candidate_ids, scores, truth):
    m = _truth_metrics(list(candidate_ids), list(map(float, scores)), int(truth))
    return {
        "rr": float(m["rr_at_k"]),
        "top1": float(m["top1_credit"]),
        "top5": float(m["top5_credit"]),
        "expected_rank": float(m["expected_rank"]),
        "rank_min": int(m["rank_min"]),
        "rank_max": int(m["rank_max"]),
        "tie_size": int(m["tie_size"]),
    }, m


def _assert_order_invariant(candidate_ids, scores, truth, reference):
    rev, _ = _metric(list(reversed(candidate_ids)), list(reversed(scores)), truth)
    for key in ("rr", "top1", "top5", "expected_rank"):
        if not math.isclose(float(reference[key]), float(rev[key]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"candidate-order invariance failed for {key}")
    for key in ("rank_min", "rank_max", "tie_size"):
        if int(reference[key]) != int(rev[key]):
            raise RuntimeError(f"candidate-order invariance failed for {key}")


def _summarize(rows, method):
    if not rows:
        return {"n": 0}
    return {
        "n": int(len(rows)),
        "n_truth_blocks": int(len(set(int(r["query_mol_id"]) for r in rows))),
        "mrr25": float(np.mean([r[f"rr_{method}"] for r in rows])),
        "top1": float(np.mean([r[f"top1_{method}"] for r in rows])),
        "top5": float(np.mean([r[f"top5_{method}"] for r in rows])),
        "median_expected_rank": float(np.median([r[f"expected_rank_{method}"] for r in rows])),
        "pessimistic_mrr25_missing_above_truth": float(np.mean([r[f"rr_pessimistic_{method}"] for r in rows])),
        "pessimistic_top1_missing_above_truth": float(np.mean([r[f"top1_pessimistic_{method}"] for r in rows])),
        "pessimistic_top5_missing_above_truth": float(np.mean([r[f"top5_pessimistic_{method}"] for r in rows])),
    }


def _risk(rows, method):
    return {
        "rr_wins": int(sum(r[f"rr_{method}"] > r["rr_static"] + 1e-15 for r in rows)),
        "rr_losses": int(sum(r[f"rr_{method}"] < r["rr_static"] - 1e-15 for r in rows)),
        "rr_ties": int(sum(abs(r[f"rr_{method}"] - r["rr_static"]) <= 1e-15 for r in rows)),
        "delta_mrr": float(np.mean([r[f"rr_{method}"] - r["rr_static"] for r in rows])),
        "delta_top1": float(np.mean([r[f"top1_{method}"] - r["top1_static"] for r in rows])),
        "delta_top5": float(np.mean([r[f"top5_{method}"] - r["top5_static"] for r in rows])),
        "catastrophic_static_top3_to_below25": int(sum(
            r["expected_rank_static"] <= 3.0 and r[f"expected_rank_{method}"] > 25.0
            for r in rows
        )),
        "catastrophic_static_top5_to_below25": int(sum(
            r["expected_rank_static"] <= 5.0 and r[f"expected_rank_{method}"] > 25.0
            for r in rows
        )),
    }


def _paired(rows, method, seed_offset):
    a = [r[f"rr_{method}"] for r in rows]
    b = [r["rr_static"] for r in rows]
    blocks = [str(r["query_mol_id"]) for r in rows]
    return {
        "bootstrap": paired_block_bootstrap_delta(
            a, b, blocks, n_boot=N_BOOT, seed=BOOT_SEED + seed_offset
        ),
        "signflip": paired_block_signflip_test(
            a,
            b,
            blocks,
            n_permutations=N_SIGNFLIP,
            seed=SIGNFLIP_SEED + seed_offset,
            exact_max_blocks=20,
        ),
    }


def _load_energy_frames(cfm_path: Path):
    cfm = pd.read_pickle(cfm_path)
    cfm = cfm[
        (cfm["prec_type"] == "[M+H]+") & cfm["ace"].isin(ENERGIES)
    ][["mol_id", "ace", "peaks", "prec_mz"]].copy()
    frames = {}
    for e in ENERGIES:
        f = cfm[cfm["ace"] == e][["mol_id", "peaks", "prec_mz"]].copy()
        f["mol_id"] = f["mol_id"].astype(int)
        if f["mol_id"].duplicated().any():
            dup = int(f.loc[f["mol_id"].duplicated(), "mol_id"].iloc[0])
            raise RuntimeError(f"duplicate CASMI 2022 CFM prediction for mol_id={dup}, energy={e}")
        frames[int(e)] = f.set_index("mol_id", drop=True)
    complete = set(int(x) for x in frames[ENERGIES[0]].index)
    for e in ENERGIES[1:]:
        complete.intersection_update(int(x) for x in frames[e].index)
    return frames, complete


def main():
    p = argparse.ArgumentParser(
        description="One-shot CASMI 2022 evaluation of the frozen CASMI-2016 consensus ranker"
    )
    p.add_argument("casmi_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    model = json.loads(args.frozen_model.read_text())
    if not bool(model.get("freeze_gate_passed", False)):
        raise RuntimeError("refusing lockbox evaluation: frozen CASMI-2016 gate did not pass")
    if tuple(model.get("feature_names", ())) != CONSENSUS_FEATURE_NAMES:
        raise RuntimeError("frozen model feature schema mismatch")
    weights = np.asarray(model.get("weights", ()), dtype=float)
    if weights.shape != (len(CONSENSUS_FEATURE_NAMES),) or not np.all(np.isfinite(weights)):
        raise RuntimeError("invalid frozen weights")
    gate = model.get("gate", {})
    if gate.get("type") != "top-k static/expert set overlap" or gate.get("fallback") != "static_best":
        raise RuntimeError("unexpected frozen gate definition")
    top_k = int(gate["top_k"])
    overlap_threshold = float(gate["overlap_threshold"])

    spec = pd.read_pickle(_find_one(args.casmi_root, "spec_df.pkl"))
    cand = pd.read_pickle(_find_one(args.casmi_root, "cand_df.pkl"))
    cfm_path = _find_one(args.cfm_root, "casmi22_spec_df.pkl")
    spec = spec[spec["prec_type"] == "[M+H]+"].copy()
    by_query = cand.groupby("query_mol_id")["candidate_mol_id"].agg(
        lambda s: tuple(sorted(set(int(x) for x in s)))
    )
    energy_frames, complete_ids = _load_energy_frames(cfm_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_fp = args.out_dir / "candidate_features.csv.gz"
    candidate_fields = [
        "spec_id", "query_mol_id", "candidate_mol_id", "is_truth",
        "cos10", "cos20", "cos40", "static_best", "ordered_path_score",
        "path_segment", "path_t", "expert_score",
    ]

    rows = []
    coverage_rows = []
    excluded = {"no_candidate_list": 0, "truth_not_predicted": 0, "coverage_below_0_99": 0}
    with gzip.open(candidate_fp, "wt", newline="", encoding="utf-8") as cf:
        cw = csv.DictWriter(cf, fieldnames=candidate_fields)
        cw.writeheader()
        eligible_index = 0
        for qrow in spec.itertuples(index=False):
            truth = int(qrow.mol_id)
            original = tuple(int(x) for x in by_query.get(truth, ()))
            if not original:
                excluded["no_candidate_list"] += 1
                continue
            supported = tuple(c for c in original if c in complete_ids)
            coverage = len(supported) / len(original)
            coverage_rows.append({
                "spec_id": int(qrow.spec_id),
                "query_mol_id": truth,
                "candidate_count_original": len(original),
                "candidate_count_supported": len(supported),
                "candidate_coverage_fraction": float(coverage),
                "truth_predicted": truth in complete_ids,
            })
            if truth not in complete_ids:
                excluded["truth_not_predicted"] += 1
                continue
            if coverage < PRIMARY_COVERAGE:
                excluded["coverage_below_0_99"] += 1
                continue

            q_raw = _clean_raw(qrow.peaks, qrow.prec_mz, 0.0)
            q_sqrt = _sqrt_view(q_raw)
            ids = list(supported)
            energy_slices = {
                e: list(energy_frames[e].loc[ids].itertuples(index=False))
                for e in ENERGIES
            }
            static_scores = []
            path_scores = []
            energy_cosines = []
            path_diags = []
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
                path, seg, t = _path_score(q_raw, refs_raw)
                if not math.isfinite(path):
                    raise RuntimeError(f"non-finite ordered-path score for spec={qrow.spec_id}, candidate={cid}")
                static_scores.append(float(max(cos)))
                path_scores.append(float(path))
                energy_cosines.append(cos)
                path_diags.append((seg, t))

            C = np.asarray(energy_cosines, dtype=float)
            X = consensus_feature_matrix(C[:, 0], C[:, 1], C[:, 2], path_scores)
            expert = X @ weights
            gated, overlap, used = selective_consensus_scores(
                static_scores,
                expert,
                overlap_threshold=overlap_threshold,
                k=top_k,
            )
            rng = np.random.default_rng(CONTROL_SEED + int(qrow.spec_id))
            perm_expert = expert[rng.permutation(len(expert))]
            control, control_overlap, control_used = selective_consensus_scores(
                static_scores,
                perm_expert,
                overlap_threshold=overlap_threshold,
                k=top_k,
            )

            score_sets = {
                "static": np.asarray(static_scores, dtype=float),
                "expert": np.asarray(expert, dtype=float),
                "gated": np.asarray(gated, dtype=float),
                "permuted_control": np.asarray(control, dtype=float),
            }
            missing = len(original) - len(ids)
            out = {
                "spec_id": int(qrow.spec_id),
                "query_mol_id": truth,
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
                _assert_order_invariant(ids, scores.tolist(), truth, metric)
                pess = _pessimistic_missing_bounds(raw_metric, missing)
                for k, v in metric.items():
                    out[f"{k}_{name}"] = v
                out[f"rr_pessimistic_{name}"] = float(pess["rr"])
                out[f"top1_pessimistic_{name}"] = float(pess["top1"])
                out[f"top5_pessimistic_{name}"] = float(pess["top5"])
            rows.append(out)

            for i, cid in enumerate(ids):
                seg, t = path_diags[i]
                cw.writerow({
                    "spec_id": int(qrow.spec_id),
                    "query_mol_id": truth,
                    "candidate_mol_id": int(cid),
                    "is_truth": int(cid == truth),
                    "cos10": float(C[i, 0]),
                    "cos20": float(C[i, 1]),
                    "cos40": float(C[i, 2]),
                    "static_best": float(static_scores[i]),
                    "ordered_path_score": float(path_scores[i]),
                    "path_segment": int(seg),
                    "path_t": float(t),
                    "expert_score": float(expert[i]),
                })
            eligible_index += 1
            print(
                f"[{eligible_index}] spec={qrow.spec_id} truth={truth} coverage={coverage:.5f} "
                f"candidates={len(ids)}/{len(original)} overlap={overlap:.2f} expert={used} "
                f"static_rank={out['expected_rank_static']:.1f} gated_rank={out['expected_rank_gated']:.1f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("no CASMI 2022 lockbox queries met the predeclared coverage threshold")

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

    summary = {
        "protocol": {
            "dataset": "CASMI 2022 processed challenge from MassFormer release",
            "dataset_role": "one-shot external lockbox; no outcome-driven model changes permitted",
            "candidate_spectra": "published CFM predictions at 10/20/40 eV only; no measured candidate spectra",
            "ion_scope": "[M+H]+",
            "primary_candidate_prediction_coverage": PRIMARY_COVERAGE,
            "feature_names": list(CONSENSUS_FEATURE_NAMES),
            "model_sha256": _sha256(args.frozen_model),
            "frozen_weights": [float(x) for x in weights],
            "frozen_gate_top_k": top_k,
            "frozen_gate_overlap_threshold": overlap_threshold,
            "candidate_order_invariance": "asserted for every method on every scored query",
            "negative_control": "seeded within-query permutation of frozen expert scores before applying the same reliability gate",
            "inference_block": "query_mol_id (truth molecule)",
            "no_casmi2022_tuning": True,
        },
        "coverage": {
            "positive_mh_spectra": int(len(spec)),
            "primary_scored_spectra": int(len(rows)),
            "primary_truth_blocks": int(len(set(r["query_mol_id"] for r in rows))),
            "complete_cfm_molecules": int(len(complete_ids)),
            "excluded": excluded,
            "coverage_min": float(min(x["candidate_coverage_fraction"] for x in coverage_rows)),
            "coverage_median": float(np.median([x["candidate_coverage_fraction"] for x in coverage_rows])),
            "coverage_mean": float(np.mean([x["candidate_coverage_fraction"] for x in coverage_rows])),
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
        "interpretation_guardrail": (
            "A passing gate is independent evidence that the CASMI-2016-frozen energy-consensus ranker transfers to CASMI 2022 conditional on the published CFM candidate spectra and candidate lists. "
            "It does not prove universal small-molecule identification, perfect candidate generation, negative-ion generalization, or invariance to prediction model and instrument domain."
        ),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.out_dir / "per_query.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.out_dir / "coverage.csv", index=False)
    print(json.dumps({
        "coverage": summary["coverage"],
        "static": metrics["static"],
        "gated": metrics["gated"],
        "gated_risk": gated_risk,
        "gated_inference": inference["gated_vs_static"],
        "control_risk": risks["permuted_control"],
        "expert_usage_fraction": summary["primary_ge_0_99"]["expert_usage_fraction"],
        "replication_gate": replication_gate,
    }, indent=2))


if __name__ == "__main__":
    main()
