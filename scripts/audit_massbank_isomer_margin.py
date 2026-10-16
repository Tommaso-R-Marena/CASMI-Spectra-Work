"""Recompute the secondary correct-vs-best-decoy margin contrast from immutable ranks.

Post-hoc inference; margin was a reported endpoint but CI planned after results.
Does not replace the null primary MRR/accuracy comparison.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qfd.stats import paired_block_bootstrap_delta,paired_block_signflip_test

ROWS_SHA="32e3b1240b48cc82717fdf43ec21abdf0cdd528f31137c9f138c0435911375fa"
RESULT_SHA="20ea49a7d8465ad4f7791fda2ed640382b88b210598e75cfd9b47e8a07f6f0a0"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(rows):
    if len(rows)!=252 or len({r["query_path"] for r in rows})!=252 or len({r["formula"] for r in rows})!=44:
        raise ValueError("original frozen 252-query rank cohort changed")
    dif=np.array([r["alpha_quarter_margin"]-r["hellinger_fixed_margin"] for r in rows])
    if not np.isfinite(dif).all():
        raise ValueError("invalid margin differences")
    groups=[r["formula"] for r in rows]
    a=np.array([r["alpha_quarter_margin"] for r in rows])
    b=np.array([r["hellinger_fixed_margin"] for r in rows])
    bysource=defaultdict(list)
    for r in rows:
        bysource[(r["contributor"],r["instrument_type"])].append(r)
    original_mrr=np.mean([r["alpha_quarter_rr"]-r["hellinger_fixed_rr"] for r in rows])
    return {
        "status":"POSTHOC_SECONDARY_DISCRIMINATION_MARGIN_AUDIT",
        "original_rank_workflow":38963742589,
        "scientific_class":"secondary descriptive / posthoc CI, NOT improved top1 identification",
        "queries":len(rows),"formula_blocks":len(set(groups)),
        "mean_margin_alpha_quarter":float(a.mean()),
        "mean_margin_hellinger":float(b.mean()),
        "mean_margin_uplift":float(dif.mean()),
        "margin_gain_loss_tie":{
            "gains":int((dif>1e-12).sum()),
            "losses":int((dif<-1e-12).sum()),
            "ties":int((abs(dif)<=1e-12).sum()),
        },
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,groups,n_boot=10000,seed=20261016),
        "formula_block_signflip_two_sided":paired_block_signflip_test(
            a,b,groups,n_permutations=50000,seed=20261016),
        "original_primary_mean_mrr_delta":float(original_mrr),
        "source_instrument_descriptive":[
            {"contributor":k[0],"instrument_type":k[1],"queries":len(v),
             "mean_margin_delta":float(np.mean([
                 r["alpha_quarter_margin"]-r["hellinger_fixed_margin"] for r in v])),
             "mean_reciprocal_rank_delta":float(np.mean([
                 r["alpha_quarter_rr"]-r["hellinger_fixed_rr"] for r in v]))}
            for k,v in sorted(bysource.items())],
        "interpretation":"Positive discrimination margins with nearly unchanged isomer rankings can coexist. Do not claim improved identification or external replication.",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-run-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    source=a.original_run_dir/"rows.jsonl"
    result=a.original_run_dir/"result.json"
    if sha(source)!=ROWS_SHA or sha(result)!=RESULT_SHA:
        raise ValueError("original isomer-rank evidence has changed")
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    frozen=json.loads(result.read_text())
    if abs(frozen["all"]["paired_comparisons"]["quarter_minus_hellinger"]["mean_paired_rr_delta"]
           -np.mean([r["alpha_quarter_rr"]-r["hellinger_fixed_rr"] for r in rows]))>1e-12:
        raise ValueError("primary original rank metric mismatch")
    out=audit(rows)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"result.json").write_text(json.dumps(out,indent=2)+"\n")
    ci=out["formula_block_bootstrap_95"]
    lines=["# MassBank exact-formula isomer discriminative-margin sensitivity","",
           "**POSTHOC secondary inference; primary MRR remains negative/non-significant.**","",
           f"Mean correct-vs-best-wrong margin α=.25: {out['mean_margin_alpha_quarter']:.6f}",
           f"Mean correct-vs-best-wrong margin Hellinger: {out['mean_margin_hellinger']:.6f}",
           f"Mean paired margin difference: {out['mean_margin_uplift']:+.6f}",
           f"95% formula-block bootstrap CI: [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}]",
           f"Sign-flip two-sided p: {out['formula_block_signflip_two_sided']['p_value_two_sided']:.6f}",
           f"Original primary MRR difference: {out['original_primary_mean_mrr_delta']:+.6f}",
           "",out["interpretation"],""]
    (a.out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__=="__main__":
    main()
