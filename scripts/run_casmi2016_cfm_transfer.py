from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.dynamic_variants import candidate_zscores
from qfd.merged import _cluster_grid
from qfd.metrics import tie_aware_truth_metrics
from qfd.small_hull import convex_hull_distance_active_set
from qfd.spectra import matched_cosine, preprocess_spectrum
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test
from qfd.trajectory import SpectrumView


PRIMARY_METHOD = "hybrid_static_convex_100"
BASELINE = "static_best"
TOLERANCE_DA = 0.01
ENERGIES = (10, 20, 40)
PRIMARY_MIN_COVERAGE = 0.95
SCORED_MIN_COVERAGE = 0.90


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name!r} under {root}, found {len(hits)}")
    return hits[0]


def _peak_arrays(peaks):
    mz, it = [], []
    for p in peaks:
        if len(p) < 2:
            continue
        a, b = float(p[0]), float(p[1])
        if np.isfinite(a) and np.isfinite(b) and b >= 0:
            mz.append(a)
            it.append(b)
    return np.asarray(mz, float), np.asarray(it, float)


def _clean_raw(peaks, precursor_mz, energy):
    mz, it = _peak_arrays(peaks)
    mz, it = preprocess_spectrum(
        mz,
        it,
        precursor_mz=float(precursor_mz) if precursor_mz is not None and np.isfinite(float(precursor_mz)) else None,
        intensity_floor=0.001,
        max_peaks=128,
        intensity_transform="none",
    )
    return SpectrumView.make(mz, it, (float(energy),), adduct="[M+H]+")


def _sqrt_view(raw: SpectrumView):
    it = np.sqrt(np.maximum(np.asarray(raw.intensities, float), 0.0))
    norm = float(np.linalg.norm(it))
    if norm > 0:
        it = it / norm
    return SpectrumView.make(raw.mzs, it, raw.collision_energy_ev, adduct=raw.adduct)


def _convex_score(query_raw: SpectrumView, refs_raw: list[SpectrumView]):
    _, M = _cluster_grid([query_raw] + refs_raw, TOLERANCE_DA)
    M = np.maximum(np.asarray(M, float), 0.0)
    if M.shape[1] != 4:
        return float("nan")
    sums = M.sum(axis=0)
    if np.any(sums <= 0):
        return float("nan")
    P = M / sums[None, :]
    d, w = convex_hull_distance_active_set(P[:, 0], P[:, 1:])
    if not np.all(np.isfinite(w)):
        return float("nan")
    return -float(d)


def _truth_metrics(ids, scores, truth):
    vals = [float(x) if math.isfinite(float(x)) else -math.inf for x in scores]
    got = tie_aware_truth_metrics([str(x) for x in ids], vals, str(truth), k=25)
    rev = tie_aware_truth_metrics(
        [str(x) for x in reversed(ids)], list(reversed(vals)), str(truth), k=25
    )
    if got != rev:
        raise RuntimeError("tie-aware metric changed under candidate-order reversal")
    return got


def _pessimistic_missing_bounds(metric, missing_count: int, *, k: int = 25):
    """Worst-case absolute rank credit if every unsupported candidate beats truth."""
    lo = int(metric["rank_min"]) + int(missing_count)
    hi = int(metric["rank_max"]) + int(missing_count)
    rr = float(np.mean([1.0 / r if r <= k else 0.0 for r in range(lo, hi + 1)]))
    top1 = (1.0 / (hi - lo + 1)) if lo == 1 else 0.0
    top5 = float(np.mean([r <= 5 for r in range(lo, hi + 1)]))
    return {"rr": rr, "top1": top1, "top5": top5, "rank_min": lo, "rank_max": hi}


def _summarize(rows, methods):
    out = {"n_queries": len(rows), "n_truth_molecule_blocks": len({r["query_mol_id"] for r in rows})}
    if not rows:
        return out
    out["candidate_count_supported"] = {
        "min": int(min(r["candidate_count"] for r in rows)),
        "median": float(np.median([r["candidate_count"] for r in rows])),
        "max": int(max(r["candidate_count"] for r in rows)),
        "mean": float(np.mean([r["candidate_count"] for r in rows])),
    }
    out["candidate_coverage_fraction"] = {
        "min": float(min(r["candidate_coverage_fraction"] for r in rows)),
        "median": float(np.median([r["candidate_coverage_fraction"] for r in rows])),
        "mean": float(np.mean([r["candidate_coverage_fraction"] for r in rows])),
        "max": float(max(r["candidate_coverage_fraction"] for r in rows)),
    }
    for method in methods:
        out[f"mrr25_tieaware_{method}"] = float(np.mean([r[f"rr_{method}"] for r in rows]))
        out[f"top1_tieaware_{method}"] = float(np.mean([r[f"top1_{method}"] for r in rows]))
        out[f"top5_tieaware_{method}"] = float(np.mean([r[f"top5_{method}"] for r in rows]))
        out[f"median_expected_rank_{method}"] = float(np.median([r[f"expected_rank_{method}"] for r in rows]))
        out[f"pessimistic_mrr25_missing_above_truth_{method}"] = float(np.mean([r[f"rr_pessimistic_{method}"] for r in rows]))
        out[f"pessimistic_top1_missing_above_truth_{method}"] = float(np.mean([r[f"top1_pessimistic_{method}"] for r in rows]))
        out[f"pessimistic_top5_missing_above_truth_{method}"] = float(np.mean([r[f"top5_pessimistic_{method}"] for r in rows]))
    return out


def _paired_inference(rows, method, baseline, seed):
    if not rows:
        return {}
    a = [r[f"rr_{method}"] for r in rows]
    b = [r[f"rr_{baseline}"] for r in rows]
    blocks = [r["query_mol_id"] for r in rows]
    return {
        "bootstrap": paired_block_bootstrap_delta(a, b, blocks, n_boot=10000, seed=seed),
        "signflip": paired_block_signflip_test(
            a, b, blocks, seed=seed + 1, n_permutations=200000, exact_max_blocks=20
        ),
    }


def main():
    p = argparse.ArgumentParser(description="Frozen QFD/convex transfer benchmark on CASMI 2016 using CFM-predicted candidate spectra")
    p.add_argument("casmi_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=314159)
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

    eligible_specs = []
    coverage_rows = []
    excluded = {"no_candidate_list": 0, "truth_not_predicted": 0, "coverage_below_90pct": 0}
    for row in spec.itertuples(index=False):
        truth = int(row.mol_id)
        original_candidates = tuple(int(x) for x in by_query.get(truth, ()))
        if not original_candidates:
            excluded["no_candidate_list"] += 1
            continue
        supported = tuple(c for c in original_candidates if c in cfm_groups)
        frac = len(supported) / len(original_candidates)
        coverage_rows.append({
            "spec_id": int(row.spec_id),
            "truth": truth,
            "candidate_count_original": len(original_candidates),
            "candidate_count_supported": len(supported),
            "candidate_coverage_fraction": frac,
            "truth_predicted": truth in cfm_groups,
        })
        if truth not in cfm_groups:
            excluded["truth_not_predicted"] += 1
            continue
        if frac < SCORED_MIN_COVERAGE:
            excluded["coverage_below_90pct"] += 1
            continue
        eligible_specs.append((row, original_candidates, supported, frac))

    methods = (BASELINE, "convex_hull", PRIMARY_METHOD, "hybrid_permuted_convex_control")
    result_rows = []

    for q_index, (qrow, original_candidates, candidate_ids, coverage_fraction) in enumerate(eligible_specs):
        q_raw = _clean_raw(qrow.peaks, qrow.prec_mz, 0.0)
        q_sqrt = _sqrt_view(q_raw)
        static_scores = []
        convex_scores = []

        for cid in candidate_ids:
            refs_raw = []
            refs_sqrt = []
            for e in ENERGIES:
                r = cfm_groups[cid][e]
                raw = _clean_raw(r.peaks, r.prec_mz, e)
                refs_raw.append(raw)
                refs_sqrt.append(_sqrt_view(raw))
            static = max(
                matched_cosine(q_sqrt.mzs, q_sqrt.intensities, r.mzs, r.intensities, tolerance_da=TOLERANCE_DA)
                for r in refs_sqrt
            )
            convex = _convex_score(q_raw, refs_raw)
            static_scores.append(float(static))
            convex_scores.append(float(convex) if math.isfinite(convex) else -math.inf)

        static_map = {str(k): v for k, v in zip(candidate_ids, static_scores)}
        convex_map = {str(k): v for k, v in zip(candidate_ids, convex_scores)}
        zs = candidate_zscores(static_map)
        zh = candidate_zscores(convex_map)
        hybrid_scores = [zs[str(k)] + zh[str(k)] for k in candidate_ids]

        rng = np.random.default_rng(args.seed + int(qrow.spec_id))
        perm = rng.permutation(len(candidate_ids))
        permuted_convex = np.asarray([zh[str(k)] for k in candidate_ids], float)[perm]
        control_scores = np.asarray([zs[str(k)] for k in candidate_ids], float) + permuted_convex

        score_sets = {
            BASELINE: static_scores,
            "convex_hull": convex_scores,
            PRIMARY_METHOD: hybrid_scores,
            "hybrid_permuted_convex_control": control_scores.tolist(),
        }
        missing_count = len(original_candidates) - len(candidate_ids)
        out = {
            "spec_id": int(qrow.spec_id),
            "query_mol_id": int(qrow.mol_id),
            "candidate_count_original": len(original_candidates),
            "candidate_count": len(candidate_ids),
            "missing_candidate_count": missing_count,
            "candidate_coverage_fraction": float(coverage_fraction),
        }
        for method, values in score_sets.items():
            m = _truth_metrics(list(candidate_ids), list(values), int(qrow.mol_id))
            pessimistic = _pessimistic_missing_bounds(m, missing_count)
            out[f"rank_min_{method}"] = m["rank_min"]
            out[f"rank_max_{method}"] = m["rank_max"]
            out[f"expected_rank_{method}"] = m["expected_rank"]
            out[f"tie_size_{method}"] = m["tie_size"]
            out[f"rr_{method}"] = m["rr_at_k"]
            out[f"top1_{method}"] = m["top1_credit"]
            out[f"top5_{method}"] = m["top5_credit"]
            out[f"rr_pessimistic_{method}"] = pessimistic["rr"]
            out[f"top1_pessimistic_{method}"] = pessimistic["top1"]
            out[f"top5_pessimistic_{method}"] = pessimistic["top5"]
        result_rows.append(out)
        print(
            f"[{q_index + 1}/{len(eligible_specs)}] spec={qrow.spec_id} coverage={coverage_fraction:.3f} "
            f"supported={len(candidate_ids)}/{len(original_candidates)} "
            f"static_rank={out['expected_rank_static_best']:.1f} hybrid_rank={out['expected_rank_hybrid_static_convex_100']:.1f}",
            flush=True,
        )

    strata = {}
    for threshold in (0.90, 0.95, 0.99, 1.0):
        rows = [r for r in result_rows if r["candidate_coverage_fraction"] >= threshold]
        key = f"coverage_ge_{threshold:.2f}"
        strata[key] = {
            "metrics": _summarize(rows, methods),
            "primary_vs_static_inference": _paired_inference(rows, PRIMARY_METHOD, BASELINE, args.seed + int(threshold * 10000)),
        }

    primary_rows = [r for r in result_rows if r["candidate_coverage_fraction"] >= PRIMARY_MIN_COVERAGE]
    summary = {
        "protocol": {
            "dataset": "CASMI 2016 processed candidate lists from MassFormer release",
            "candidate_spectra": "published CFM predictions only; no measured candidate reference spectra",
            "ion_scope": "[M+H]+ only, because the main published CFM CASMI prediction table is positive [M+H]+",
            "cfm_energies_ev": list(ENERGIES),
            "spectrum_preprocessing": "0.1% relative-intensity floor, top 128 peaks, precursor filter; sqrt/L2 for cosine and linear then L1 for convex geometry",
            "mass_tolerance_da": TOLERANCE_DA,
            "baseline": BASELINE,
            "primary_method_frozen_from_msnlib": PRIMARY_METHOD,
            "hybrid_definition": "within-query z(static_best) + 1.0 * z(convex_hull)",
            "primary_min_candidate_prediction_coverage": PRIMARY_MIN_COVERAGE,
            "scored_min_candidate_prediction_coverage": SCORED_MIN_COVERAGE,
            "primary_estimand": "paired reranking improvement on the identical CFM-supported candidate intersection for CASMI queries retaining >=95% of the original candidate pool",
            "absolute_rank_sensitivity": "also report worst-case bounds with every unsupported candidate inserted above the truth",
            "negative_control": "same static z-scores plus a seeded permutation of the convex component across candidate identities",
            "candidate_order_invariance": "asserted per query by reversing candidate order before tie-aware scoring",
            "no_external_weight_tuning": True,
            "coverage_threshold_chosen_before_external_ranking_outcomes": True,
        },
        "coverage": {
            "positive_mh_query_spectra": int(len(spec)),
            "complete_cfm_molecules": int(len(cfm_groups)),
            "scored_queries_ge_90pct": int(len(result_rows)),
            "primary_queries_ge_95pct": int(len(primary_rows)),
            "excluded": excluded,
            "candidate_coverage_fraction": {
                "min": float(min((r["candidate_coverage_fraction"] for r in coverage_rows), default=0.0)),
                "median": float(np.median([r["candidate_coverage_fraction"] for r in coverage_rows])) if coverage_rows else 0.0,
                "mean": float(np.mean([r["candidate_coverage_fraction"] for r in coverage_rows])) if coverage_rows else 0.0,
                "max": float(max((r["candidate_coverage_fraction"] for r in coverage_rows), default=0.0)),
            },
        },
        "primary_ge_95pct": {
            "metrics": _summarize(primary_rows, methods),
            "paired_inference": {
                "primary_vs_static": _paired_inference(primary_rows, PRIMARY_METHOD, BASELINE, args.seed + 1000),
                "convex_vs_static": _paired_inference(primary_rows, "convex_hull", BASELINE, args.seed + 2000),
                "permuted_control_vs_static": _paired_inference(primary_rows, "hybrid_permuted_convex_control", BASELINE, args.seed + 3000),
            },
        },
        "coverage_strata_sensitivity": strata,
        "interpretation_guardrail": (
            "A positive paired result establishes transfer of the frozen reranker from measured-library MSnLib ranking to structure-only CASMI candidates conditional on CFM prediction support. "
            "It is not full CASMI accuracy because some original candidates lack CFM predictions. Pessimistic missing-candidate rank bounds quantify that limitation. "
            "It still does not prove universal metabolite identification: candidate-generation recall, prediction-model error, negative/adduct coverage, and cross-instrument shift remain separate failure modes."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if result_rows:
        with (args.out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result_rows[0]))
            w.writeheader()
            w.writerows(result_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
