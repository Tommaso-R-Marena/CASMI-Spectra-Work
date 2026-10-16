from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qfd.energy_geometry_controls import geometry_control_scores
from qfd.energy_interpolation import _bhattacharyya, _common_grid_strict, _l1_columns
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.power_geometry import logit_energy_warp, power_mean_score
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie


ALPHA_QUARTER = 0.25
WARP_A = 1.1884079253066788
WARP_B = 0.44288663984932064


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


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen transfer of MCEBIO-selected alpha-quarter geometry.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    groups = _unambiguous_groups(index_records(records))
    rows = []

    for skey, g in sorted(groups.items()):
        refs = sorted(float(e) for e in g["same_energy"])
        if len(refs) < 2:
            continue
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
                raise ValueError("strict grid failed to normalize triplet")
            q, p0, p1 = X[:, 0], X[:, 1], X[:, 2]
            t = float((eq - elo) / (ehi - elo))
            low_aff = float(_bhattacharyya(q, p0))
            high_aff = float(_bhattacharyya(q, p1))
            warped_t = logit_energy_warp(t, WARP_A, WARP_B)
            ctl = geometry_control_scores(q, p0, p1, t)
            rows.append({
                "formula": str(g["formula"]),
                "structure_adduct": f"{skey[0]}|{skey[1]}",
                "query_energy_ev": float(eq),
                "low_energy_ev": float(elo),
                "high_energy_ev": float(ehi),
                "t": t,
                "warped_t": warped_t,
                "best_endpoint": max(low_aff, high_aff),
                "linear": power_mean_score(q, p0, p1, t, 1.0),
                "hellinger_chord": power_mean_score(q, p0, p1, t, 0.5),
                "alpha_quarter": power_mean_score(q, p0, p1, t, ALPHA_QUARTER),
                "warped_hellinger": power_mean_score(q, p0, p1, warped_t, 0.5),
                "oracle_probability_mixture": float(ctl.oracle_probability_mixture),
            })

    if not rows:
        raise RuntimeError("no eligible LOEO triplets")

    methods = [
        "best_endpoint", "linear", "hellinger_chord", "alpha_quarter",
        "warped_hellinger", "oracle_probability_mixture",
    ]
    result = {
        "status": "MSNLIB_ENERGY_GEOMETRY_POWER_TRANSFER",
        "dataset": args.dataset,
        "input": {"path": str(args.jsonl), "sha256": _sha256(args.jsonl), "records": int(len(records))},
        "frozen_parameters": {
            "alpha_quarter": ALPHA_QUARTER,
            "warp_a": WARP_A,
            "warp_b": WARP_B,
            "source": "MCEBIO positive discovery formulas only",
        },
        "coverage": {
            "query_triplets": int(len(rows)),
            "formula_blocks": int(len({str(r['formula']) for r in rows})),
            "structure_adduct_groups": int(len({str(r['structure_adduct']) for r in rows})),
        },
        "methods": {m: _summary(rows, m) for m in methods},
        "comparisons": {
            "alpha_quarter_vs_hellinger": _comparison(rows, "alpha_quarter", "hellinger_chord", seed=args.seed),
            "alpha_quarter_vs_oracle_probability_mixture": _comparison(rows, "alpha_quarter", "oracle_probability_mixture", seed=args.seed + 1),
            "alpha_quarter_vs_best_endpoint": _comparison(rows, "alpha_quarter", "best_endpoint", seed=args.seed + 2),
            "warped_hellinger_vs_hellinger": _comparison(rows, "warped_hellinger", "hellinger_chord", seed=args.seed + 3),
            "hellinger_vs_linear": _comparison(rows, "hellinger_chord", "linear", seed=args.seed + 4),
        },
        "guardrail": "alpha=0.25 and Hellinger warp parameters were frozen from MCEBIO discovery before this transfer dataset was scored.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
