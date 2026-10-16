"""Auditable cross-cohort verdict for the pre-registered alpha-quarter MSnLib transfer.

Only consumes per-dataset workflow artifacts. It never fits a model or opens
raw MS/MS outcomes. A missing artifact is NOT a failed scientific hypothesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COHORTS = (
    "mcebio_neg", "mcedrug_pos", "mcedrug_neg", "mcescaf_pos",
    "mcescaf_neg", "enammol_pos", "enammol_neg", "enamdisc_neg",
)
PREFIX = "energy-geometry-power-transfer-"
TRANSFER_SHA = "be7813155a5dada31c8c557952a060b664a1e3ac"
PRIMARY = "alpha_quarter_vs_hellinger"


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def holm_adjust(pvals: dict[str, float], family_size: int = len(COHORTS)) -> dict[str, float]:
    """Holm FWER-adjustment; absent/unopened prespecified tests count as p=1."""
    if family_size < len(pvals):
        raise ValueError("family_size smaller than tested hypotheses")
    ordered = sorted(pvals, key=lambda x: (pvals[x], x))
    out = {}
    floor = 0.0
    for i, k in enumerate(ordered):
        floor = max(floor, min(1.0, (family_size - i) * pvals[k]))
        out[k] = floor
    return out


def summarize(root: Path) -> dict:
    entries = []
    pvals = {}
    for ds in COHORTS:
        folder = root / (PREFIX + ds)
        feasibility = _read(folder / "feasibility.json")
        result = _read(folder / "result.json")
        prov = _read(folder / "provenance.json")
        entry = {"dataset": ds, "status": None}
        if feasibility is None:
            entry["status"] = "MISSING_FEASIBILITY_ARTIFACT"
        elif feasibility.get("dataset") != ds:
            entry["status"] = "INVALID_FEASIBILITY_DATASET"
        elif not isinstance(feasibility.get("gate_pass"), bool):
            entry["status"] = "INVALID_GATE"
        elif not feasibility["gate_pass"]:
            entry["coverage"] = feasibility.get("coverage")
            entry["status"] = "PROTOCOL_BREACH_SCORED_FAILED_GATE" if result is not None else "GATE_FAILED_UNOPENED"
        elif result is None:
            entry["status"] = "MISSING_SCORE_ARTIFACT"
        elif prov is None:
            entry["status"] = "MISSING_PROVENANCE"
        elif result.get("dataset") != ds or prov.get("dataset") != ds:
            entry["status"] = "DATASET_MISMATCH"
        elif prov.get("frozen_transfer_sha") != TRANSFER_SHA or result.get("frozen_parameters", {}).get("alpha_quarter") != 0.25:
            entry["status"] = "FROZEN_PARAMETER_MISMATCH"
        elif result.get("coverage", {}).get("query_triplets") != feasibility.get("coverage", {}).get("query_triplets"):
            entry["status"] = "COHORT_SIZE_MISMATCH"
        else:
            comparison = result.get("comparisons", {}).get(PRIMARY, {})
            ci = comparison.get("formula_block_bootstrap_95", {})
            rand = comparison.get("formula_block_signflip_two_sided", {})
            try:
                delta = float(comparison["mean_delta"])
                low = float(ci["ci_low"])
                high = float(ci["ci_high"])
                p = float(rand["p_value_two_sided"])
                assert -1 <= delta <= 1 and -1 <= low <= high <= 1 and 0 <= p <= 1
            except (AssertionError, KeyError, ValueError, TypeError):
                entry["status"] = "INVALID_INFERENCE"
            else:
                entry.update({
                    "status": "SCORED",
                    "queries": comparison["n_queries"],
                    "formula_blocks": comparison["n_formula_blocks"],
                    "delta": delta, "ci_low": low, "ci_high": high, "p_two_sided": p,
                    "positive_predeclared": bool(delta > 0 and low > 0 and p < 0.05),
                    "alpha_quarter_mean": result["methods"]["alpha_quarter"]["mean"],
                    "hellinger_mean": result["methods"]["hellinger_chord"]["mean"],
                    "oracle_mixture_delta": result["comparisons"]["alpha_quarter_vs_oracle_probability_mixture"]["mean_delta"],
                })
                pvals[ds] = p
        entries.append(entry)
    adjusted = holm_adjust(pvals)
    for e in entries:
        if e["dataset"] in adjusted:
            e["holm_p_eight_cohorts"] = adjusted[e["dataset"]]
            e["holm_supported"] = bool(e["delta"] > 0 and e["ci_low"] > 0 and adjusted[e["dataset"]] < 0.05)
    counts = {s: sum(e["status"] == s for e in entries) for s in sorted({e["status"] for e in entries})}
    fully_observed = all(e["status"] in {"SCORED", "GATE_FAILED_UNOPENED"} for e in entries)
    return {
        "status": "COMPLETE" if fully_observed else "INCOMPLETE_OR_INVALID",
        "claim": "Frozen alpha=0.25 transfer superiority to Hellinger alpha=0.5",
        "scope": "MSnLib within-ecosystem; does not prove unique or universal geometry",
        "family_size_for_holm": len(COHORTS),
        "no_pooled_independence_claim": True,
        "counts": counts, "cohorts": entries,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Frozen alpha-quarter MSnLib transfer verdict", "",
        "**Overall:** " + report["status"],
        "Predeclared primary outcome: alpha=0.25 minus Hellinger chord (alpha=0.5), Bhattacharyya affinity.",
        "Holm correction conservatively covers all eight named cohorts, including gate failures.",
        "Missing/failed infrastructure is never converted into a scientific negative. No pooled independence claim.", "",
        "| Cohort | Status | Queries | Delta | 95% formula-block CI | Raw p | Holm p |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for c in report["cohorts"]:
        if c["status"] == "SCORED":
            lines.append(f"| {c['dataset']} | scored {'(supports)' if c['positive_predeclared'] else '(no support)'} | {c['queries']} | {c['delta']:+.6f} | [{c['ci_low']:+.6f}, {c['ci_high']:+.6f}] | {c['p_two_sided']:.3g} | {c['holm_p_eight_cohorts']:.3g} |")
        else:
            lines.append(f"| {c['dataset']} | {c['status']} | — | — | — | — | — |")
    lines += ["", "This report does not tune parameters, compare new exponents, establish an optimum, or validate cross-instrument generalization.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "verdict.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
