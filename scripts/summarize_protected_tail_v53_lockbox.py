from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie


def main() -> None:
    p = argparse.ArgumentParser(description="Predeclared formula-block inference for frozen protected-tail v5.3 lockboxes.")
    p.add_argument("candidate_features", type=Path)
    p.add_argument("analysis_json", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    df = pd.read_csv(args.candidate_features)
    analysis = json.loads(args.analysis_json.read_text())
    diagnostics = analysis.get("query_diagnostics", [])
    if not diagnostics:
        raise ValueError("analysis JSON contains no query diagnostics")

    meta = (
        df.groupby("spec_id", sort=False)
        .agg(formula=("formula", "first"), candidate_count=("candidate_id", "count"))
        .reset_index()
    )
    by_id = {str(r.spec_id): (str(r.formula), int(r.candidate_count)) for r in meta.itertuples(index=False)}

    rr_static = []
    rr_v53 = []
    blocks = []
    candidate_counts = []
    for row in diagnostics:
        sid = str(row["spec_id"])
        if sid not in by_id:
            raise ValueError(f"missing candidate metadata for {sid}")
        formula, n = by_id[sid]
        rr_static.append(float(row["rr_static"]))
        rr_v53.append(float(row["rr_v53"]))
        blocks.append(formula)
        candidate_counts.append(n)

    a = np.asarray(rr_v53, dtype=float)
    b = np.asarray(rr_static, dtype=float)
    counts = np.asarray(candidate_counts, dtype=int)

    def infer(mask: np.ndarray) -> dict[str, object]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return {"n_queries": 0, "n_formula_blocks": 0}
        aa = a[idx]
        bb = b[idx]
        bl = [blocks[int(i)] for i in idx]
        return {
            "n_queries": int(idx.size),
            "n_formula_blocks": int(len(set(bl))),
            "mean_delta_mrr25": float(np.mean(aa - bb)),
            "win_loss_tie": paired_win_loss_tie(aa, bb, atol=1e-15),
            "formula_block_bootstrap_95": paired_block_bootstrap_delta(
                aa, bb, bl, n_boot=10000, seed=args.seed, alpha=0.05
            ),
            "formula_block_signflip_two_sided": paired_block_signflip_test(
                aa, bb, bl, seed=args.seed, n_permutations=200000, exact_max_blocks=20, atol=1e-15
            ),
        }

    result = {
        "status": "PROTECTED_TAIL_V53_LOCKBOX_INFERENCE",
        "primary_estimand": "query-weighted delta tie-aware MRR@25 (v5.3 - static)",
        "independent_sampling_unit": "molecular formula block",
        "all_eligible_queries": infer(np.ones(len(a), dtype=bool)),
        "candidate_pool_ge_5_sensitivity": infer(counts >= 5),
        "candidate_pool_ge_10_sensitivity": infer(counts >= 10),
        "top1_note": "Top-1 is a structural invariant of the frozen protected_k=1 selector and is not an efficacy endpoint.",
        "guardrail": (
            "The full-cohort formula-block analysis is primary. Pool-size sensitivities were predeclared before "
            "the external lockbox rankings were opened and use only label-free candidate-pool size."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
