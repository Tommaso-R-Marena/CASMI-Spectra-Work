from __future__ import annotations

from collections.abc import Mapping
import math

from .metrics import rank_by_score
from .spectra import matched_cosine
from .trajectory import SpectrumView, joint_energy_score


def static_best_score(query: list[SpectrumView], reference: list[SpectrumView], *, tolerance_da: float = 0.01) -> float:
    best = float("-inf")
    for q in query:
        for r in reference:
            if q.adduct is not None and r.adduct is not None and q.adduct != r.adduct:
                continue
            s = matched_cosine(q.mzs, q.intensities, r.mzs, r.intensities, tolerance_da=tolerance_da)
            if s > best:
                best = s
    return float("nan") if best == float("-inf") else best


def rank_library(
    query: list[SpectrumView],
    library: Mapping[str, list[SpectrumView]],
    *,
    mode: str = "static_best",
    tolerance_da: float = 0.01,
    max_energy_gap_ev: float = 5.0,
):
    ids: list[str] = []
    scores: list[float] = []
    for structure_id, refs in library.items():
        if mode == "static_best":
            score = static_best_score(query, refs, tolerance_da=tolerance_da)
        elif mode == "joint_energy":
            score = joint_energy_score(
                query, refs, tolerance_da=tolerance_da,
                max_energy_gap_ev=max_energy_gap_ev, require_same_adduct=True,
            )
        else:
            raise ValueError("mode must be 'static_best' or 'joint_energy'")
        if math.isnan(score):
            score = -math.inf
        ids.append(str(structure_id))
        scores.append(score)
    return rank_by_score(ids, scores, higher_is_better=True), dict(zip(ids, scores))
