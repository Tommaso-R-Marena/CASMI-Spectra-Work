from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .benchmark import static_best_score
from .cone import convex_hull_distance
from .dynamic_variants import (
    candidate_zscores,
    geometric_joint_score,
    minimum_joint_score,
    multi_energy_delta_score,
)
from .merged import _choose_energy_columns, _cluster_grid
from .metrics import rank_by_score, reciprocal_rank_at_k, tie_aware_truth_metrics
from .msnlib import MSnLibRecord
from .msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from .trajectory import joint_energy_score


def _finite_or_neg_inf(v):
    return float(v) if math.isfinite(float(v)) else -math.inf


def _summarize(rows, methods):
    out = {"n_queries": len(rows)}
    if not rows:
        return out
    for method in methods:
        ranks = [r[f"rank_{method}"] for r in rows]
        rrs = [r[f"rr_{method}"] for r in rows]
        out[f"mrr25_{method}"] = float(np.mean(rrs))
        out[f"top1_{method}"] = float(np.mean([x == 1 for x in ranks]))
        out[f"top5_{method}"] = float(np.mean([x <= 5 for x in ranks]))
        out[f"mrr25_tieaware_{method}"] = float(np.mean([r[f"rr_tie_{method}"] for r in rows]))
        out[f"top1_tieaware_{method}"] = float(np.mean([r[f"top1_tie_{method}"] for r in rows]))
        out[f"top5_tieaware_{method}"] = float(np.mean([r[f"top5_tie_{method}"] for r in rows]))
        out[f"truth_tied_queries_{method}"] = int(sum(r[f"tie_size_{method}"] > 1 for r in rows))
        out[f"max_truth_tie_size_{method}"] = int(max(r[f"tie_size_{method}"] for r in rows))
    return out


def _rank_score_map(score_map):
    ids = list(score_map)
    vals = [_finite_or_neg_inf(score_map[k]) for k in ids]
    return rank_by_score(ids, vals), dict(zip(ids, vals))


def _add_tie_metrics(row, method, clean, truth_id):
    ids = list(clean)
    tie = tie_aware_truth_metrics(ids, [clean[k] for k in ids], truth_id, k=25)
    row[f"rank_min_{method}"] = tie["rank_min"]
    row[f"rank_max_{method}"] = tie["rank_max"]
    row[f"expected_rank_{method}"] = tie["expected_rank"]
    row[f"tie_size_{method}"] = tie["tie_size"]
    row[f"rr_tie_{method}"] = tie["rr_at_k"]
    row[f"top1_tie_{method}"] = tie["top1_credit"]
    row[f"top5_tie_{method}"] = tie["top5_credit"]


def evaluate_exact_condition_variants(
    records: list[MSnLibRecord],
    *,
    hard_static_threshold: float = 0.8,
):
    """Same-formula ranking with identical collision-energy support for every candidate."""
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)
    eligible = {}
    for skey, g in groups.items():
        common = sorted(set(g["single_best"]) & set(g["same_energy"]))
        if len(common) < 2:
            continue
        eligible[skey] = {
            "formula": g["formula"],
            "inchikey14": g["inchikey14"],
            "energies": common,
            "qraw": {e: _clean_view(g["single_best"][e], transform="none") for e in common},
            "rraw": {e: _clean_view(g["same_energy"][e], transform="none") for e in common},
            "qsqrt": {e: _clean_view(g["single_best"][e], transform="sqrt") for e in common},
            "rsqrt": {e: _clean_view(g["same_energy"][e], transform="sqrt") for e in common},
        }

    by_formula_adduct = defaultdict(list)
    for skey, g in eligible.items():
        by_formula_adduct[(g["formula"], skey[1])].append(skey)

    base_methods = [
        "static_best",
        "joint_mean",
        "joint_geom",
        "joint_min",
        "delta_hellinger",
        "delta_clr_1e4",
    ]
    hybrid_methods = [
        "hybrid_mean_hell_025",
        "hybrid_mean_hell_050",
        "hybrid_mean_hell_100",
        "hybrid_mean_clr_050",
    ]
    methods = tuple(base_methods + hybrid_methods)
    rows = []

    for truth, g in sorted(eligible.items()):
        required = tuple(g["energies"])
        candidate_keys = sorted([
            cid for cid in by_formula_adduct[(g["formula"], truth[1])]
            if set(required).issubset(set(eligible[cid]["energies"]))
        ])
        if len(candidate_keys) < 2:
            continue

        qraw = [g["qraw"][e] for e in required]
        qsqrt = [g["qsqrt"][e] for e in required]
        score_maps = {m: {} for m in base_methods}
        for cid in candidate_keys:
            cg = eligible[cid]
            rraw = [cg["rraw"][e] for e in required]
            rsqrt = [cg["rsqrt"][e] for e in required]
            candidate_id = cid[0]
            score_maps["static_best"][candidate_id] = static_best_score(qsqrt, rsqrt)
            score_maps["joint_mean"][candidate_id] = joint_energy_score(qsqrt, rsqrt, max_energy_gap_ev=0)
            score_maps["joint_geom"][candidate_id] = geometric_joint_score(qsqrt, rsqrt)
            score_maps["joint_min"][candidate_id] = minimum_joint_score(qsqrt, rsqrt)
            score_maps["delta_hellinger"][candidate_id] = multi_energy_delta_score(qraw, rraw, mode="hellinger")
            score_maps["delta_clr_1e4"][candidate_id] = multi_energy_delta_score(qraw, rraw, mode="clr", eps=1e-4)

        zmean = candidate_zscores(score_maps["joint_mean"])
        zhell = candidate_zscores(score_maps["delta_hellinger"])
        zclr = candidate_zscores(score_maps["delta_clr_1e4"])
        score_maps["hybrid_mean_hell_025"] = {k: zmean[k] + 0.25 * zhell[k] for k in zmean}
        score_maps["hybrid_mean_hell_050"] = {k: zmean[k] + 0.50 * zhell[k] for k in zmean}
        score_maps["hybrid_mean_hell_100"] = {k: zmean[k] + 1.00 * zhell[k] for k in zmean}
        score_maps["hybrid_mean_clr_050"] = {k: zmean[k] + 0.50 * zclr[k] for k in zmean}

        truth_id = truth[0]
        wrong_static = [v for k, v in score_maps["static_best"].items() if k != truth_id and math.isfinite(v)]
        best_decoy = max(wrong_static) if wrong_static else -math.inf
        row = {
            "full_inchikey": truth_id,
            "inchikey14": g["inchikey14"],
            "formula": g["formula"],
            "adduct": truth[1],
            "candidate_count": len(candidate_keys),
            "n_energies": len(required),
            "energies": ";".join(str(x) for x in required),
            "best_decoy_static_score": best_decoy,
            "hard_static": best_decoy >= hard_static_threshold,
        }
        for method in methods:
            ranked, clean = _rank_score_map(score_maps[method])
            row[f"rank_{method}"] = ranked.index(truth_id) + 1
            row[f"rr_{method}"] = reciprocal_rank_at_k(ranked, truth_id, 25)
            row[f"score_{method}"] = clean[truth_id]
            _add_tie_metrics(row, method, clean, truth_id)
        rows.append(row)

    return {
        "all": _summarize(rows, methods),
        "hard": _summarize([r for r in rows if r["hard_static"]], methods),
        "rows": rows,
        "methods": methods,
        "eligible_structure_adduct_groups": len(eligible),
        "indexed_full_identity_groups": len(indexed),
        "excluded_ambiguous_full_identity_groups": len(indexed) - len(groups),
        "identity_definition": "full InChIKey + adduct",
        "ranking_primary": "tie-aware expected reciprocal rank and fractional top-k credit",
        "guardrail": "Every candidate for a query has the exact same required collision energies in both SINGLE_BEST_SCAN and SAME_ENERGY representations.",
    }


def _merged_components(query, reference, *, tolerance_da: float = 0.01):
    chosen = _choose_energy_columns(
        reference, query.collision_energy_ev, adduct=query.adduct, max_gap_ev=0,
    )
    if chosen is None:
        return float("nan"), float("nan")
    _, M = _cluster_grid([query] + chosen, tolerance_da)
    q = np.maximum(M[:, 0], 0.0)
    C = np.maximum(M[:, 1:], 0.0)
    qsum = q.sum()
    csums = C.sum(axis=0)
    if qsum <= 0 or np.any(csums <= 0):
        return float("nan"), float("nan")
    q = q / qsum
    C = C / csums[None, :]
    hull_dist, _ = convex_hull_distance(q, C)
    vertex_dist = float(np.min(np.linalg.norm(C - q[:, None], axis=0)))
    return -float(hull_dist), float(vertex_dist - hull_dist)


def evaluate_merged_variants(
    records: list[MSnLibRecord],
    *,
    hard_static_threshold: float = 0.8,
):
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)
    by_formula_adduct = defaultdict(list)
    for skey, g in groups.items():
        by_formula_adduct[(g["formula"], skey[1])].append(skey)

    methods = (
        "static_best",
        "merged_convex",
        "convex_advantage",
        "hybrid_static_convex_025",
        "hybrid_static_convex_050",
        "hybrid_static_convex_100",
        "hybrid_static_adv_050",
    )
    rows = []

    for truth, g in sorted(groups.items()):
        for qrec in g["all_energies"]:
            required = tuple(dict.fromkeys(qrec.collision_energies))
            if len(required) < 2 or not all(e in g["single_best"] for e in required):
                continue
            candidate_keys = sorted([
                cid for cid in by_formula_adduct[(g["formula"], truth[1])]
                if all(e in groups[cid]["single_best"] for e in required)
            ])
            if len(candidate_keys) < 2:
                continue

            qraw = _clean_view(qrec, transform="none")
            qsqrt = _clean_view(qrec, transform="sqrt")
            score_maps = {m: {} for m in methods}
            for cid in candidate_keys:
                refs_raw = [_clean_view(groups[cid]["single_best"][e], transform="none") for e in required]
                refs_sqrt = [_clean_view(groups[cid]["single_best"][e], transform="sqrt") for e in required]
                candidate_id = cid[0]
                score_maps["static_best"][candidate_id] = static_best_score([qsqrt], refs_sqrt)
                hull, advantage = _merged_components(qraw, refs_raw)
                score_maps["merged_convex"][candidate_id] = hull
                score_maps["convex_advantage"][candidate_id] = advantage

            zs = candidate_zscores(score_maps["static_best"])
            zh = candidate_zscores(score_maps["merged_convex"])
            za = candidate_zscores(score_maps["convex_advantage"])
            score_maps["hybrid_static_convex_025"] = {k: zs[k] + 0.25 * zh[k] for k in zs}
            score_maps["hybrid_static_convex_050"] = {k: zs[k] + 0.50 * zh[k] for k in zs}
            score_maps["hybrid_static_convex_100"] = {k: zs[k] + 1.00 * zh[k] for k in zs}
            score_maps["hybrid_static_adv_050"] = {k: zs[k] + 0.50 * za[k] for k in zs}

            truth_id = truth[0]
            wrong_static = [v for k, v in score_maps["static_best"].items() if k != truth_id and math.isfinite(v)]
            best_decoy = max(wrong_static) if wrong_static else -math.inf
            row = {
                "full_inchikey": truth_id,
                "inchikey14": g["inchikey14"],
                "formula": g["formula"],
                "adduct": truth[1],
                "candidate_count": len(candidate_keys),
                "n_energies": len(required),
                "energies": ";".join(str(x) for x in required),
                "best_decoy_static_score": best_decoy,
                "hard_static": best_decoy >= hard_static_threshold,
            }
            for method in methods:
                ranked, clean = _rank_score_map(score_maps[method])
                row[f"rank_{method}"] = ranked.index(truth_id) + 1
                row[f"rr_{method}"] = reciprocal_rank_at_k(ranked, truth_id, 25)
                row[f"score_{method}"] = clean[truth_id]
                _add_tie_metrics(row, method, clean, truth_id)
            rows.append(row)

    return {
        "all": _summarize(rows, methods),
        "hard": _summarize([r for r in rows if r["hard_static"]], methods),
        "rows": rows,
        "methods": methods,
        "indexed_full_identity_groups": len(indexed),
        "excluded_ambiguous_full_identity_groups": len(indexed) - len(groups),
        "identity_definition": "full InChIKey + adduct",
        "ranking_primary": "tie-aware expected reciprocal rank and fractional top-k credit",
        "guardrail": "Merged-query candidates all contain every collision energy listed by that query; hybrids use within-query z-scores and predeclared weights.",
    }
