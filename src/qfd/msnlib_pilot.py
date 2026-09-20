from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .benchmark import rank_library, static_best_score
from .merged import merged_convex_score
from .metrics import rank_by_score, reciprocal_rank_at_k
from .msnlib import MSnLibRecord
from .qfd_profile import two_energy_qfd_score
from .spectra import preprocess_spectrum
from .trajectory import SpectrumView


def _clean_view(rec: MSnLibRecord, *, transform: str) -> SpectrumView:
    mz, it = preprocess_spectrum(
        rec.view.mzs, rec.view.intensities, precursor_mz=rec.precursor_mz,
        intensity_floor=0.001, max_peaks=128, intensity_transform=transform,
    )
    return SpectrumView.make(
        mz, it, rec.collision_energies, adduct=rec.adduct,
        base_peak_intensity=rec.view.base_peak_intensity,
    )


def index_records(records: list[MSnLibRecord]):
    """Index MSnLib without collapsing stereochemical identity or choosing replicates.

    Full InChIKey + adduct is the benchmark identity. The 14-character InChIKey
    connectivity block is intentionally not used as an identity because distinct
    full InChIKeys in the release can share that block. If a full-identity slot has
    repeated records, the group is marked ambiguous and downstream benchmarks exclude
    the whole group rather than silently selecting one acquisition.
    """
    groups = defaultdict(lambda: {
        "formula": None,
        "inchikey14": None,
        "single_best": {},
        "same_energy": {},
        "all_energies": [],
        "pseudo_ms2": [],
        "ambiguous_slots": [],
    })
    slots = defaultdict(list)

    for r in records:
        key = (r.full_inchikey, r.adduct)
        g = groups[key]
        if g["formula"] is None:
            g["formula"] = r.molecular_formula
            g["inchikey14"] = r.inchikey14
        elif g["formula"] != r.molecular_formula:
            raise ValueError(
                f"inconsistent formulas for full_inchikey={key[0]!r}, adduct={key[1]!r}: "
                f"{g['formula']!r} vs {r.molecular_formula!r}"
            )
        slots[(key, r.merge_type, tuple(r.collision_energies))].append(r)

    for (key, merge_type, energies), recs in slots.items():
        g = groups[key]
        if len(recs) != 1:
            g["ambiguous_slots"].append({
                "merge_type": merge_type,
                "collision_energies": energies,
                "count": len(recs),
                "spectrum_ids": tuple(r.spectrum_id for r in recs),
            })
            continue
        r = recs[0]
        if r.is_single_energy:
            e = r.collision_energies[0]
            if merge_type == "SINGLE_BEST_SCAN":
                g["single_best"][e] = r
            elif merge_type == "SAME_ENERGY":
                g["same_energy"][e] = r
        elif merge_type == "ALL_ENERGIES":
            g["all_energies"].append(r)
        elif merge_type == "ALL_MSN_TO_PSEUDO_MS2":
            g["pseudo_ms2"].append(r)

    return groups


def _unambiguous_groups(groups):
    return {k: g for k, g in groups.items() if not g["ambiguous_slots"]}


def _summarize(rows, modes):
    out = {"n_queries": len(rows)}
    if not rows:
        return out
    for mode in modes:
        ranks = [r[f"rank_{mode}"] for r in rows]
        rrs = [r[f"rr_{mode}"] for r in rows]
        out[f"mrr25_{mode}"] = float(np.mean(rrs))
        out[f"top1_{mode}"] = float(np.mean([x == 1 for x in ranks]))
        out[f"top5_{mode}"] = float(np.mean([x <= 5 for x in ranks]))
    return out


def evaluate_same_energy_trajectory(records: list[MSnLibRecord], *, hard_static_threshold: float = 0.8):
    """Class-1-style exact-condition test: SINGLE_BEST_SCAN query vs SAME_ENERGY reference.

    Candidate sets are restricted so every candidate has references at every
    collision energy used by that query. Thus dynamic methods cannot gain merely
    because a wrong candidate lacks the required experimental condition.
    SAME_ENERGY spectra are derived from replicate acquisitions, so this is a
    consistency/ranking test, not independent or structure-held-out validation.
    """
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
            "query_raw": [_clean_view(g["single_best"][e], transform="none") for e in common],
            "ref_raw": [_clean_view(g["same_energy"][e], transform="none") for e in common],
            "query_sqrt": [_clean_view(g["single_best"][e], transform="sqrt") for e in common],
            "ref_sqrt": [_clean_view(g["same_energy"][e], transform="sqrt") for e in common],
        }
    by_formula_adduct = defaultdict(list)
    for skey, g in eligible.items():
        by_formula_adduct[(g["formula"], skey[1])].append(skey)

    rows = []
    for truth, g in sorted(eligible.items()):
        wanted = set(g["energies"])
        candidates = sorted([
            cid for cid in by_formula_adduct[(g["formula"], truth[1])]
            if wanted.issubset(set(eligible[cid]["energies"]))
        ])
        if len(candidates) < 2:
            continue

        def select(views):
            by_e = {v.collision_energy_ev[0]: v for v in views}
            return [by_e[e] for e in g["energies"]]

        static_lib = {cid[0]: select(eligible[cid]["ref_sqrt"]) for cid in candidates}
        raw_lib = {cid[0]: select(eligible[cid]["ref_raw"]) for cid in candidates}
        ranked_static, scores_static = rank_library(g["query_sqrt"], static_lib, mode="static_best")
        ranked_joint, scores_joint = rank_library(
            g["query_sqrt"], static_lib, mode="joint_energy", max_energy_gap_ev=0,
        )
        ids, qfd_scores = [], []
        for cid in candidates:
            score = two_energy_qfd_score(g["query_raw"], raw_lib[cid[0]], max_energy_gap_ev=0)
            ids.append(cid[0])
            qfd_scores.append(score if math.isfinite(score) else -math.inf)
        ranked_qfd = rank_by_score(ids, qfd_scores)
        qfd_map = dict(zip(ids, qfd_scores))
        truth_id = truth[0]
        decoys = [v for k, v in scores_static.items() if k != truth_id and math.isfinite(v)]
        best_decoy = max(decoys) if decoys else -math.inf
        row = {
            "full_inchikey": truth_id,
            "inchikey14": g["inchikey14"],
            "formula": g["formula"],
            "adduct": truth[1],
            "candidate_count": len(candidates),
            "n_energies": len(g["energies"]),
            "energies": g["energies"],
            "best_decoy_static_score": best_decoy,
            "hard_static": best_decoy >= hard_static_threshold,
        }
        for mode, ranked, scoremap in [
            ("static_best", ranked_static, scores_static),
            ("joint_energy", ranked_joint, scores_joint),
            ("two_energy_qfd", ranked_qfd, qfd_map),
        ]:
            row[f"rank_{mode}"] = ranked.index(truth_id) + 1
            row[f"rr_{mode}"] = reciprocal_rank_at_k(ranked, truth_id, 25)
            row[f"score_{mode}"] = scoremap[truth_id]
        rows.append(row)
    modes = ("static_best", "joint_energy", "two_energy_qfd")
    return {
        "all": _summarize(rows, modes),
        "hard": _summarize([r for r in rows if r["hard_static"]], modes),
        "rows": rows,
        "eligible_structure_adduct_groups": len(eligible),
        "indexed_full_identity_groups": len(indexed),
        "excluded_ambiguous_full_identity_groups": len(indexed) - len(groups),
        "identity_definition": "full InChIKey + adduct",
        "caveat": "SAME_ENERGY references are derived from replicate acquisitions; this is a Class-1 consistency/ranking test, not structure-held-out generalization.",
    }


def evaluate_merged_convex(records: list[MSnLibRecord], *, hard_static_threshold: float = 0.8):
    """Rank same-formula structures for ALL_ENERGIES merged queries.

    Candidate references are SINGLE_BEST_SCAN spectra at every collision energy
    listed by the merged query. Static-best and convex-hull scores use identical
    candidate sets for each query. Ambiguous repeated-acquisition identities are
    excluded rather than resolved by arbitrary record order.
    """
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)
    by_formula_adduct = defaultdict(list)
    for skey, g in groups.items():
        by_formula_adduct[(g["formula"], skey[1])].append(skey)

    rows = []
    for truth, g in sorted(groups.items()):
        for qrec in g["all_energies"]:
            wanted = tuple(dict.fromkeys(qrec.collision_energies))
            if len(wanted) < 2 or not all(e in g["single_best"] for e in wanted):
                continue
            candidate_keys = sorted([
                cid for cid in by_formula_adduct[(g["formula"], truth[1])]
                if all(e in groups[cid]["single_best"] for e in wanted)
            ])
            if len(candidate_keys) < 2:
                continue
            q_raw = _clean_view(qrec, transform="none")
            q_sqrt = _clean_view(qrec, transform="sqrt")
            static_scores, convex_scores = {}, {}
            for cid in candidate_keys:
                refs_raw = [_clean_view(groups[cid]["single_best"][e], transform="none") for e in wanted]
                refs_sqrt = [_clean_view(groups[cid]["single_best"][e], transform="sqrt") for e in wanted]
                static_scores[cid[0]] = static_best_score([q_sqrt], refs_sqrt)
                convex_scores[cid[0]] = merged_convex_score(q_raw, refs_raw, max_energy_gap_ev=0)
            ranked_static = rank_by_score(list(static_scores), list(static_scores.values()))
            ranked_convex = rank_by_score(list(convex_scores), list(convex_scores.values()))
            truth_id = truth[0]
            decoys = [v for k, v in static_scores.items() if k != truth_id and math.isfinite(v)]
            best_decoy = max(decoys) if decoys else -math.inf
            row = {
                "full_inchikey": truth_id,
                "inchikey14": g["inchikey14"],
                "formula": g["formula"],
                "adduct": truth[1],
                "candidate_count": len(candidate_keys),
                "energies": list(wanted),
                "best_decoy_static_score": best_decoy,
                "hard_static": best_decoy >= hard_static_threshold,
                "rank_static_best": ranked_static.index(truth_id) + 1,
                "rr_static_best": reciprocal_rank_at_k(ranked_static, truth_id, 25),
                "score_static_best": static_scores[truth_id],
                "rank_merged_convex": ranked_convex.index(truth_id) + 1,
                "rr_merged_convex": reciprocal_rank_at_k(ranked_convex, truth_id, 25),
                "score_merged_convex": convex_scores[truth_id],
            }
            rows.append(row)
    modes = ("static_best", "merged_convex")
    return {
        "all": _summarize(rows, modes),
        "hard": _summarize([r for r in rows if r["hard_static"]], modes),
        "rows": rows,
        "indexed_full_identity_groups": len(indexed),
        "excluded_ambiguous_full_identity_groups": len(indexed) - len(groups),
        "identity_definition": "full InChIKey + adduct",
        "caveat": "ALL_ENERGIES queries are constructed from the same released acquisition family as the single-energy references; this tests the mixture geometry and Class-1 ranking, not de novo generalization.",
    }
