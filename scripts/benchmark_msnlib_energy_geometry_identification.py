from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qfd.energy_geometry_controls import geometry_control_scores
from qfd.energy_interpolation import _bhattacharyya, _common_grid_strict, _l1_columns
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.power_geometry import logit_energy_warp, power_mean_score
from qfd.stats import (
    paired_block_bootstrap_delta,
    paired_block_signflip_test,
    paired_win_loss_tie,
    stable_group_is_validation,
)


SPLIT_SALT = "energy-geometry-identification-v1"
VALIDATION_FRACTION = 0.5
ALPHAS = (0.25, 1.0 / 3.0, 0.40, 0.45, 0.50, 0.55, 0.60, 2.0 / 3.0, 0.75, 1.00, 1.25)
MIN_QUERIES_PER_SPLIT = 800
MIN_FORMULAS_PER_SPLIT = 600


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _comparison(rows, a: str, b: str, *, seed: int) -> dict[str, object]:
    av = np.asarray([float(r[a]) for r in rows], dtype=float)
    bv = np.asarray([float(r[b]) for r in rows], dtype=float)
    blocks = [str(r["formula"]) for r in rows]
    return {
        "a": a,
        "b": b,
        "n_queries": int(len(rows)),
        "n_formula_blocks": int(len(set(blocks))),
        "mean_delta": float(np.mean(av - bv)),
        "median_delta": float(np.median(av - bv)),
        "win_loss_tie": paired_win_loss_tie(av, bv, atol=1e-15),
        "formula_block_bootstrap_95": paired_block_bootstrap_delta(
            av, bv, blocks, n_boot=10_000, seed=seed
        ),
        "formula_block_signflip_two_sided": paired_block_signflip_test(
            av, bv, blocks, seed=seed, n_permutations=200_000
        ),
    }


def _summary(rows, key: str) -> dict[str, float]:
    x = np.asarray([float(r[key]) for r in rows], dtype=float)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
    }


def _score_alpha(rows, alpha: float) -> np.ndarray:
    return np.asarray(
        [power_mean_score(r["q"], r["p0"], r["p1"], float(r["t"]), alpha) for r in rows],
        dtype=float,
    )


def _select_alpha(discovery) -> tuple[float, list[dict[str, float]]]:
    profile = []
    for alpha in ALPHAS:
        scores = _score_alpha(discovery, alpha)
        profile.append({"alpha": float(alpha), "mean_affinity": float(np.mean(scores))})
    best = max(x["mean_affinity"] for x in profile)
    eligible = [x for x in profile if best - x["mean_affinity"] <= 1e-12]
    chosen = min(eligible, key=lambda x: (abs(x["alpha"] - 0.5), x["alpha"]))
    return float(chosen["alpha"]), profile


def _fit_hellinger_warp(discovery) -> dict[str, object]:
    def objective(x):
        a, b = float(x[0]), float(x[1])
        vals = [
            power_mean_score(
                r["q"], r["p0"], r["p1"],
                logit_energy_warp(float(r["t"]), a, b),
                0.5,
            )
            for r in discovery
        ]
        return -float(np.mean(vals))

    identity = -objective(np.asarray([1.0, 0.0], dtype=float))
    res = minimize(
        objective,
        x0=np.asarray([1.0, 0.0], dtype=float),
        method="L-BFGS-B",
        bounds=((0.25, 4.0), (-2.0, 2.0)),
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not res.success or not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
        raise RuntimeError(f"energy-warp optimization failed: {res.message}")
    return {
        "a": float(res.x[0]),
        "b": float(res.x[1]),
        "discovery_mean_affinity": float(-res.fun),
        "identity_discovery_mean_affinity": float(identity),
        "optimizer": "L-BFGS-B",
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
    }


def _serializable_row(r):
    return {
        k: v for k, v in r.items()
        if k not in {"q", "p0", "p1"}
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Formula-disjoint identification of collision-energy path geometry.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    rows = []
    for skey, g in sorted(groups.items()):
        refs = sorted(float(e) for e in g["same_energy"])
        if len(refs) < 2:
            continue
        formula = str(g["formula"])
        split = (
            "validation"
            if stable_group_is_validation(
                formula,
                validation_fraction=VALIDATION_FRACTION,
                salt=SPLIT_SALT,
            )
            else "discovery"
        )
        for eq in sorted(float(e) for e in g["single_best"]):
            lower = [e for e in refs if e < eq]
            higher = [e for e in refs if e > eq]
            if not lower or not higher:
                continue
            elo, ehi = max(lower), min(higher)
            query = _clean_view(g["single_best"][eq], transform="none")
            low = _clean_view(g["same_energy"][elo], transform="none")
            high = _clean_view(g["same_energy"][ehi], transform="none")
            X = _l1_columns(_common_grid_strict([query, low, high], 0.01))
            if X is None:
                raise ValueError("strict grid failed to produce normalized triplet")
            q, p0, p1 = X[:, 0], X[:, 1], X[:, 2]
            t = float((eq - elo) / (ehi - elo))
            rows.append({
                "structure_adduct": f"{skey[0]}|{skey[1]}",
                "formula": formula,
                "split": split,
                "query_energy_ev": float(eq),
                "low_energy_ev": float(elo),
                "high_energy_ev": float(ehi),
                "t": t,
                "q": q,
                "p0": p0,
                "p1": p1,
            })

    discovery = [r for r in rows if r["split"] == "discovery"]
    validation = [r for r in rows if r["split"] == "validation"]
    d_formulas = {str(r["formula"]) for r in discovery}
    v_formulas = {str(r["formula"]) for r in validation}
    if d_formulas & v_formulas:
        raise RuntimeError("formula leakage across identification split")

    coverage = {
        "all_queries": int(len(rows)),
        "all_formula_blocks": int(len(d_formulas | v_formulas)),
        "discovery_queries": int(len(discovery)),
        "discovery_formula_blocks": int(len(d_formulas)),
        "validation_queries": int(len(validation)),
        "validation_formula_blocks": int(len(v_formulas)),
    }
    gate = {
        "queries_per_split_min": MIN_QUERIES_PER_SPLIT,
        "formula_blocks_per_split_min": MIN_FORMULAS_PER_SPLIT,
    }
    gate_pass = bool(
        len(discovery) >= MIN_QUERIES_PER_SPLIT
        and len(validation) >= MIN_QUERIES_PER_SPLIT
        and len(d_formulas) >= MIN_FORMULAS_PER_SPLIT
        and len(v_formulas) >= MIN_FORMULAS_PER_SPLIT
    )
    if not gate_pass:
        result = {
            "status": "ENERGY_GEOMETRY_IDENTIFICATION_GATE_FAILED",
            "dataset": args.dataset,
            "coverage": coverage,
            "predeclared_gate": gate,
            "guardrail": "No alpha selection, warp fitting, or validation comparison was performed after gate failure.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    selected_alpha, discovery_alpha_profile = _select_alpha(discovery)
    warp = _fit_hellinger_warp(discovery)

    validation_alpha_profile = []
    for alpha in ALPHAS:
        validation_alpha_profile.append({
            "alpha": float(alpha),
            "mean_affinity": float(np.mean(_score_alpha(validation, alpha))),
        })

    for r in validation:
        q, p0, p1, t = r["q"], r["p0"], r["p1"], float(r["t"])
        low = float(_bhattacharyya(q, p0))
        high = float(_bhattacharyya(q, p1))
        r["best_endpoint"] = max(low, high)
        r["linear"] = power_mean_score(q, p0, p1, t, 1.0)
        r["hellinger_chord"] = power_mean_score(q, p0, p1, t, 0.5)
        r["selected_power"] = power_mean_score(q, p0, p1, t, selected_alpha)
        warped_t = logit_energy_warp(t, float(warp["a"]), float(warp["b"]))
        r["warped_t"] = warped_t
        r["warped_hellinger"] = power_mean_score(q, p0, p1, warped_t, 0.5)
        ctl = geometry_control_scores(q, p0, p1, t)
        r["oracle_probability_mixture"] = float(ctl.oracle_probability_mixture)

    methods = [
        "best_endpoint",
        "linear",
        "hellinger_chord",
        "selected_power",
        "warped_hellinger",
        "oracle_probability_mixture",
    ]
    comparisons = {
        "selected_power_vs_hellinger": _comparison(
            validation, "selected_power", "hellinger_chord", seed=args.seed
        ),
        "hellinger_vs_linear": _comparison(
            validation, "hellinger_chord", "linear", seed=args.seed + 1
        ),
        "warped_hellinger_vs_hellinger": _comparison(
            validation, "warped_hellinger", "hellinger_chord", seed=args.seed + 2
        ),
        "hellinger_vs_oracle_probability_mixture": _comparison(
            validation, "hellinger_chord", "oracle_probability_mixture", seed=args.seed + 3
        ),
        "selected_power_vs_oracle_probability_mixture": _comparison(
            validation, "selected_power", "oracle_probability_mixture", seed=args.seed + 4
        ),
        "hellinger_vs_best_endpoint": _comparison(
            validation, "hellinger_chord", "best_endpoint", seed=args.seed + 5
        ),
    }

    result = {
        "status": "MSNLIB_ENERGY_GEOMETRY_IDENTIFICATION",
        "dataset": args.dataset,
        "input": {"path": str(args.jsonl), "sha256": _sha256(args.jsonl), "records": int(len(records))},
        "protocol": {
            "source_plan": "docs/ENERGY_GEOMETRY_IDENTIFICATION_PROTOCOL.md",
            "split_unit": "molecular formula",
            "split_salt": SPLIT_SALT,
            "validation_fraction": VALIDATION_FRACTION,
            "power_family": "normalize(((1-t)p0^alpha+t p1^alpha)^(1/alpha))",
            "alpha_candidates": [float(x) for x in ALPHAS],
            "alpha_half_identity": "Hellinger chord",
            "alpha_one_identity": "probability-linear interpolation",
            "warp": "sigmoid(a*logit(t)+b), discovery fit only",
            "inference_unit": "molecular formula block",
        },
        "coverage": coverage,
        "predeclared_gate": gate,
        "gate_pass": True,
        "discovery_fit": {
            "selected_alpha": float(selected_alpha),
            "alpha_profile": discovery_alpha_profile,
            "hellinger_energy_warp": warp,
        },
        "validation": {
            "alpha_profile_descriptive": validation_alpha_profile,
            "methods": {m: _summary(validation, m) for m in methods},
            "comparisons": comparisons,
        },
        "query_diagnostics": [_serializable_row(r) for r in validation],
        "guardrail": (
            "Power exponent and energy-warp parameters were selected using discovery formulas only; "
            "validation formulas were formula-disjoint and were not used for tuning."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    printable = {k: v for k, v in result.items() if k != "query_diagnostics"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
