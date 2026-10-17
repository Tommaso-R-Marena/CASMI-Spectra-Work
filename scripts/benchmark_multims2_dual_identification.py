"""Exploratory real MultiMS2 same-formula CID+EAD identification benchmark.

The original CID/EAD spectrum-prediction outcomes were already known before
this experiment was specified. This is a NEW identification endpoint, but
NOT independent-source confirmation. No parameters are fitted here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

# The previously checked v2 adapter only fixes parsing of literal CID CE headers.
import benchmark_multims2_third_source_geometry_v2  # noqa: F401
from benchmark_multims2_third_source_geometry import load_selected_source_spectra
from benchmark_multims2_ead_lockbox import read_selected_spectra
import audit_multims2_third_source_metadata as source
import audit_multims2_ead_metadata as ead
from audit_multims2_dual_identification_gate import sha, SOURCE_SHA, RANK_FIELDS
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity
)
from qfd.stats import (
    paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie
)

METHODS = ("cid_h", "ead_h", "fused_hh", "cid_q", "fused_qh")
PRIMARY = (("fused_hh_vs_cid_h", "fused_hh", "cid_h"),
           ("fused_hh_vs_ead_h", "fused_hh", "ead_h"))


def rank_scores(scores: dict[str, float], truth: str, *, tol: float = 1e-12) -> dict:
    """Deterministic tie-aware reciprocal rank and fractional top-1."""
    if len(scores) < 2 or truth not in scores:
        raise ValueError("true structure and at least one competitor required")
    if any(not math.isfinite(float(v)) or not -1e-9 <= v <= 1+1e-9
           for v in scores.values()):
        raise ValueError("nonfinite or non-affinity retrieval score")
    s = scores[truth]
    above = sum(v > s+tol for v in scores.values())
    tied = sum(abs(v-s) <= tol for v in scores.values())
    if not tied or above+tied > len(scores):
        raise RuntimeError("nontransitive rank tolerance")
    rank = 1.0 + above + (tied-1)/2
    return {
        "reciprocal_rank": float(1/rank),
        "top1_fractional": float(1/tied if above == 0 else 0),
        "average_rank": float(rank),
        "ties_with_truth": int(tied)
    }


def predictor(low, high, t: float, alpha: float):
    """Prediction grid depends exclusively on candidate reference spectra."""
    grid = reference_only_grid(low, high, tolerance_da=.01)
    predicted = power_mean_distribution(grid.low, grid.high, t, alpha)
    return grid, predicted


def candidate_affinity(query, prediction) -> float:
    grid, distribution = prediction
    aligned, unmatched = query_mass_on_reference_grid(query, grid, tolerance_da=.01)
    if abs(float(aligned.sum()) + unmatched - 1) > 1e-7:
        raise RuntimeError("unmatched target signal mass not conserved")
    result = affinity(aligned, distribution)
    if not math.isfinite(result) or not 0 <= result <= 1+1e-9:
        raise RuntimeError("affinity is nonfinite or outside [0,1]")
    return float(result)


def infer(rows, a: str, b: str, seed: int) -> dict:
    x = np.asarray([r[a]["reciprocal_rank"] for r in rows], float)
    y = np.asarray([r[b]["reciprocal_rank"] for r in rows], float)
    blocks = [r["formula"] for r in rows]
    return {
        "mean_mrr_delta": float(np.mean(x-y)),
        "wins_losses_ties": paired_win_loss_tie(x, y, atol=1e-12),
        "formula_block_bootstrap_95": paired_block_bootstrap_delta(
            x, y, blocks, n_boot=5000, seed=seed),
        "formula_block_signflip_2sided": paired_block_signflip_test(
            x, y, blocks, n_permutations=20000, seed=seed),
    }


def summarize(rows: list[dict]) -> dict:
    result = {
        "n_queries": len(rows),
        "n_formula_blocks": len({r["formula"] for r in rows}),
        "n_competition_groups": len({r["group_id"] for r in rows}),
        "candidate_count_distribution": {
            str(k): sum(r["candidate_count"] == k for r in rows)
            for k in sorted({r["candidate_count"] for r in rows})
        },
        "metrics": {m: {
            "MRR": float(np.mean([r[m]["reciprocal_rank"] for r in rows])),
            "top1_fractional": float(np.mean([r[m]["top1_fractional"] for r in rows])),
        } for m in METHODS},
        "primary_comparisons": {},
    }
    for i, (name, a, b) in enumerate(PRIMARY):
        result["primary_comparisons"][name] = infer(rows, a, b, seed=20261017 + 101*i)
    # Familywise Holm correction across exactly two predeclared primary comparisons.
    ordered = sorted((result["primary_comparisons"][name]["formula_block_signflip_2sided"]["p_value_two_sided"], name)
                     for name, _, _ in PRIMARY)
    previous = 0.0
    for i, (p, name) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered)-i)*p))
        result["primary_comparisons"][name]["holm_p"] = previous
    result["joint_dual_modal_identification_success"] = all(
        result["primary_comparisons"][name]["mean_mrr_delta"] > 0
        and result["primary_comparisons"][name]["formula_block_bootstrap_95"]["ci_low"] > 0
        and result["primary_comparisons"][name]["holm_p"] < .05
        for name, _, _ in PRIMARY)
    result["secondary_fusion_qh_vs_fusion_hh"] = infer(rows, "fused_qh", "fused_hh", 20261217)
    return result


def validate_source(row, modality: str, scans: dict):
    suffix = "cid" if modality == "CID" else "ead"
    exact_ce = (20, 40, 60) if modality == "CID" else (12, 16, 24)
    expected = {
        "inchikey": row["inchikey"], "formula": row["formula"],
        "adduct": row["adduct"], "ion_mode": row["ion_mode"],
        "source_dataset_id": row["source_dataset_id"], "fragmentation": modality
    }
    values = []
    for role, ce in zip(("low", "query", "high"), exact_ce):
        key = row[f"{suffix}_indices"][role]
        if key not in scans:
            raise RuntimeError("previously metadata-selected index missing")
        meta, spec = scans[key]
        if any(str(meta.get(k)) != str(v) for k, v in expected.items()):
            raise RuntimeError("selected scan chemical identity or modality mismatch")
        if meta["energy_eV"] != ce:
            raise RuntimeError("collision energy differs from frozen manifest")
        values.append(spec)
    return values


def evaluate(mgf: Path, gate_dir: Path, out: Path):
    meta_file = gate_dir/"feasibility.json"
    manifest_file = gate_dir/"eligible_isomer_queries.jsonl"
    gate = json.loads(meta_file.read_text())
    rows = [json.loads(t) for t in manifest_file.read_text().splitlines() if t.strip()]
    if (gate.get("status") != "MULTIMS2_DUAL_MODAL_ISOMER_IDENTIFICATION_METADATA_V1"
        or gate.get("gate_pass") is not True
        or gate.get("target_peak_positions_or_intensities_opened") is not False
        or gate.get("candidate_selection_uses_outcome_scores") is not False
        or gate.get("source_sha256") != SOURCE_SHA
        or len(rows) != gate["coverage"]["eligible_isomer_queries"]
        or len({r["group_id"] for r in rows}) != gate["coverage"]["same_formula_isomer_groups"]):
        raise RuntimeError("metadata-only identification gate not passed")
    if (mgf.stat().st_size != source.UPSTREAM_BYTES
        or source.git_blob_sha1(mgf) != source.UPSTREAM_BLOB_SHA1
        or sha(mgf) != SOURCE_SHA):
        raise RuntimeError("original 77MB MultiMS2 raw source changed")
    # Identify and open only the metadata-selected scans; no truth-conditioned
    # candidate deletion, no reuse of a candidate's withheld middle spectrum.
    cid_ids = {r["cid_indices"][role] for r in rows for role in ("low", "query", "high")}
    ead_ids = {r["ead_indices"][role] for r in rows for role in ("low", "query", "high")}
    if cid_ids & ead_ids:
        raise RuntimeError("CID/EAD spectrum index collision")
    cid_scans = load_selected_source_spectra(mgf, cid_ids)
    ead_scans = read_selected_spectra(mgf, ead_ids)
    groups = defaultdict(list)
    refs = {}
    targets = {}
    for r in rows:
        c = validate_source(r, "CID", cid_scans)
        e = validate_source(r, "EAD", ead_scans)
        key = (r["group_id"], r["inchikey"])
        if key in refs:
            raise RuntimeError("duplicate full chemical identity")
        refs[key] = {
            "cid_h": predictor(c[0], c[2], .5, .5),
            "cid_q": predictor(c[0], c[2], .5, .25),
            "ead_h": predictor(e[0], e[2], 1/3, .5),
        }
        targets[key] = {"cid": c[1], "ead": e[1]}
        groups[r["group_id"]].append(r)
    scored = []
    for group_id in sorted(groups):
        members = sorted(groups[group_id], key=lambda r: r["inchikey"])
        if len(members) < 2 or any(m["candidate_count"] != len(members) for m in members):
            raise RuntimeError("metadata candidate count mismatch")
        for row in members:
            t_key = (group_id, row["inchikey"])
            query = targets[t_key]
            scores = {name: {} for name in METHODS}
            for candidate in members:
                structure = candidate["inchikey"]
                ref = refs[(group_id, structure)]
                cid_h = candidate_affinity(query["cid"], ref["cid_h"])
                cid_q = candidate_affinity(query["cid"], ref["cid_q"])
                ead_h = candidate_affinity(query["ead"], ref["ead_h"])
                scores["cid_h"][structure] = cid_h
                scores["cid_q"][structure] = cid_q
                scores["ead_h"][structure] = ead_h
                scores["fused_hh"][structure] = .5*(cid_h+ead_h)
                scores["fused_qh"][structure] = .5*(cid_q+ead_h)
            scored.append({
                "group_id": group_id,
                "formula": row["formula"],
                "inchikey": row["inchikey"],
                "adduct": row["adduct"],
                "candidate_count": len(members),
                **{m: rank_scores(scores[m], row["inchikey"]) for m in METHODS},
            })
    result = {
        "status": "POSTHOC_MULTIMS2_CID_EAD_SAME_FORMULA_IDENTIFICATION",
        "evaluation_class": "EXPLORATORY: previously studied same-lab source; new retrieval endpoint, not prospective external confirmation",
        "data_source_sha256": SOURCE_SHA,
        "gate_feasibility_sha256": sha(meta_file),
        "gate_manifest_sha256": sha(manifest_file),
        "model_training_or_target_based_candidate_selection": False,
        "candidate_pool": "entire metadata-eligible CID+EAD exact full-InChIKey same-formula/adduct/polarity/dataset cohort",
        "prediction_grid": "candidate low/high ONLY, 0.01 Da diameter-bounded; query contributes only to post-prediction scoring",
        "source_energies": {"CID": [20, 40, 60], "EAD": [12, 16, 24]},
        "predeclared": {
            "primary": ["fused_hh_vs_cid_h", "fused_hh_vs_ead_h"],
            "secondary": ["fused_qh_vs_fused_hh"],
            "fusion_weights": [0.5, 0.5],
            "alpha_h": 0.5, "alpha_cid_q": 0.25,
            "inference_unit": "formula cluster",
        },
        "result": summarize(scored),
        "limitations": (
            "Not de novo structure elucidation: every candidate has experimentally measured "
            "reference spectra at two energies. Both modalities are from the same instrument, "
            "and hypothesis was designed after prior MultiMS2 outcomes. Top-1 ties are fractional. "
            "No cross-laboratory evidence or new CASMI identification benchmark."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    with (out/"query_metrics.jsonl").open("w") as f:
        for r in scored:
            f.write(json.dumps(r, sort_keys=True)+"\n")
    lines = [
        "# Same-formula CID + EAD identification (exploratory, one instrument)",
        "",
        f"Eligible queries: {len(scored)}; exact-formula candidate groups: {len(groups)}.",
        "",
        "| Method | MRR | fractional top-1 |",
        "|---|---:|---:|",
    ]
    for name, v in result["result"]["metrics"].items():
        lines.append(f"| {name} | {v['MRR']:.6f} | {v['top1_fractional']:.6f} |")
    for name, value in result["result"]["primary_comparisons"].items():
        ci = value["formula_block_bootstrap_95"]
        lines.append(f"- {name}: MRR delta {value['mean_mrr_delta']:+.6f}, "
                     f"formula-block 95% CI [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}], "
                     f"Holm p={value['holm_p']:.6f}")
    lines += [
        "", f"Predeclared joint two-baseline result: **{result['result']['joint_dual_modal_identification_success']}**.",
        "", result["limitations"], ""
    ]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mgf", type=Path, required=True)
    p.add_argument("--gate-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    evaluate(a.mgf, a.gate_dir, a.out)


if __name__ == "__main__":
    main()
