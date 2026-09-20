from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .benchmark import rank_library
from .metrics import reciprocal_rank_at_k
from .spectra import preprocess_spectrum
from .trajectory import SpectrumView


@dataclass
class PilotRecord:
    inchikey14: str
    molecular_formula: str
    spectrum_id: str
    adduct: str
    energy_ev: float
    view: SpectrumView


def rows_to_records(rows, *, intensity_floor=0.001, max_peaks=128, transform="sqrt"):
    out: list[PilotRecord] = []
    for row in rows:
        energies = row["collision_energy_ev"]
        if energies is None or len(energies) != 1:
            continue
        mz, it = preprocess_spectrum(
            row["ms2_mzs"], row["ms2_normalized_intensities"],
            precursor_mz=row.get("precursor_mz"), intensity_floor=intensity_floor,
            max_peaks=max_peaks, intensity_transform=transform,
        )
        if len(mz) == 0:
            continue
        e = float(energies[0])
        view = SpectrumView.make(mz, it, [e], adduct=row.get("adduct"), base_peak_intensity=row.get("base_peak_intensity"))
        out.append(PilotRecord(
            inchikey14=str(row["inchikey14"]), molecular_formula=str(row["molecular_formula"]),
            spectrum_id=str(row["spectrum_id"]), adduct=str(row.get("adduct")), energy_ev=e, view=view,
        ))
    return out


def split_query_reference(records: list[PilotRecord], *, min_energies_each: int = 2):
    """Alternate distinct energies into disjoint query/reference sets per structure/adduct."""
    by_group = defaultdict(list)
    for r in records:
        by_group[(r.inchikey14, r.adduct)].append(r)
    result = {}
    for (key, adduct), group in by_group.items():
        by_energy = defaultdict(list)
        for r in group:
            by_energy[r.energy_ev].append(r)
        energies = sorted(by_energy)
        q_energies = energies[::2]
        r_energies = energies[1::2]
        if len(q_energies) < min_energies_each or len(r_energies) < min_energies_each:
            continue
        q = [r.view for e in q_energies for r in sorted(by_energy[e], key=lambda x: x.spectrum_id)]
        ref = [r.view for e in r_energies for r in sorted(by_energy[e], key=lambda x: x.spectrum_id)]
        formula = group[0].molecular_formula
        prev = result.get(key)
        candidate = {"formula": formula, "adduct": adduct, "query": q, "reference": ref, "n_energies": len(energies), "split_mode": "interleaved"}
        if prev is None or (len(q) + len(ref), str(adduct)) > (len(prev["query"]) + len(prev["reference"]), str(prev["adduct"])):
            result[key] = candidate
    return result


def split_query_reference_replicates(records: list[PilotRecord], *, min_energies: int = 2):
    """Hold out distinct spectrum IDs at the same energies for clean Class-1 testing."""
    by_group = defaultdict(list)
    for r in records:
        by_group[(r.inchikey14, r.adduct)].append(r)
    result = {}
    for (key, adduct), group in by_group.items():
        by_energy = defaultdict(list)
        for r in group:
            by_energy[r.energy_ev].append(r)
        replicated = {e: sorted(vals, key=lambda x: x.spectrum_id) for e, vals in by_energy.items() if len(vals) >= 2}
        if len(replicated) < min_energies:
            continue
        q = []
        ref = []
        for e in sorted(replicated):
            vals = replicated[e]
            q.append(vals[0].view)
            ref.append(vals[1].view)
        formula = group[0].molecular_formula
        candidate = {"formula": formula, "adduct": adduct, "query": q, "reference": ref, "n_energies": len(replicated), "split_mode": "replicate"}
        prev = result.get(key)
        if prev is None or (len(q) + len(ref), str(adduct)) > (len(prev["query"]) + len(prev["reference"]), str(prev["adduct"])):
            result[key] = candidate
    return result


def evaluate_formula_candidates(groups: dict, *, modes=("static_best", "joint_energy", "two_energy_qfd"), k=25, max_energy_gap_ev=10.0):
    """Evaluate every query only against structures sharing its molecular formula."""
    by_formula = defaultdict(list)
    for key, g in groups.items():
        by_formula[g["formula"]].append(key)
    rows = []
    for truth, g in sorted(groups.items()):
        candidates = sorted(by_formula[g["formula"]])
        if len(candidates) < 2:
            continue
        library = {cid: groups[cid]["reference"] for cid in candidates}
        result = {"inchikey14": truth, "molecular_formula": g["formula"], "candidate_count": len(candidates)}
        for mode in modes:
            ranked, scores = rank_library(g["query"], library, mode=mode, max_energy_gap_ev=max_energy_gap_ev)
            result[f"rr_{mode}"] = reciprocal_rank_at_k(ranked, truth, k=k)
            result[f"rank_{mode}"] = ranked.index(truth) + 1 if truth in ranked else math.inf
            result[f"score_{mode}"] = scores.get(truth, -math.inf)
            if mode == "static_best":
                decoys = [score for cid, score in scores.items() if cid != truth and math.isfinite(score)]
                best_decoy = max(decoys) if decoys else -math.inf
                result["best_decoy_static_score"] = best_decoy
                result["static_margin"] = result[f"score_{mode}"] - best_decoy
        rows.append(result)
    return rows


def summarize_results(rows, modes=("static_best", "joint_energy", "two_energy_qfd")):
    if not rows:
        return {"n_queries": 0}
    summary = {"n_queries": len(rows)}
    for mode in modes:
        vals = [float(r[f"rr_{mode}"]) for r in rows]
        summary[f"mrr25_{mode}"] = sum(vals) / len(vals)
        summary[f"top1_{mode}"] = sum(r[f"rank_{mode}"] == 1 for r in rows) / len(rows)
        summary[f"top5_{mode}"] = sum(r[f"rank_{mode}"] <= 5 for r in rows) / len(rows)
    return summary
