"""Retrospective three-source falsification: does a nongeometric support control
also beat the query-aware global Hellinger-chord oracle?

NO new experiment, new independent source, prospective test, fitted parameter,
or chemical-identification claim is made here. Only original score artifacts
are re-analyzed. The nondeployable oracle sees the withheld query spectrum.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from qfd.stats import (
    paired_block_bootstrap_delta,
    paired_block_signflip_test,
    paired_win_loss_tie,
)

# Original immutable outcomes, not a rescore of any raw spectra.
SOURCES = {
    "MassBank": {
        "sha256": "2c973a7b83c1a48f95f923107ee72bb44a2a95f7d7da403f9d6f541d662efc8b",
        "n": 2138,
        "n_formula": 613,
        "quarter": "alpha_quarter",
        "hellinger": "hellinger_fixed",
        "support": "support_only",
        "oracle": "hellinger_oracle",
        "mode": "ion_mode",
        "run": 38918658343,
    },
    "ExpoLib": {
        "sha256": "f85b67d61be82f62ee1072944f9d300eb2d1d81462eba6ad221566dfaa276695",
        "n": 1125,
        "n_formula": 114,
        "quarter": "quarter",
        "hellinger": "hellinger",
        "support": "support_only",
        "oracle": "hellinger_oracle",
        "mode": "mode",
        "run": 38964797817,
    },
    "MultiMS2": {
        "sha256": "f97bfaee05e9197cdaa96ad995ab9019c995274820ff57f3b04bd04de12c2be6",
        "n": 1781,
        "n_formula": 1182,
        "quarter": "quarter",
        "hellinger": "hellinger",
        "support": "support_only",
        "oracle": "hellinger_oracle",
        "mode": "ion_mode",
        "run": 38970610885,
    },
}
SEED = 20261016
BC_BINS = (0, .05, .1, .2, .35, .5, .7, 1.0000001)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def canonical_rows(path, name):
    spec = SOURCES[name]
    if digest(path) != spec["sha256"]:
        raise RuntimeError(f"{name}: ORIGINAL ROW SHA256 MISMATCH; fail closed")
    raw = [json.loads(t) for t in path.read_text().splitlines() if t.strip()]
    if len(raw) != spec["n"] or len({r["formula"] for r in raw}) != spec["n_formula"]:
        raise RuntimeError(f"{name}: original cohort size/formula inventory differs")
    rows = []
    for ix, item in enumerate(raw):
        values = {k: float(item[spec[k]]) for k in (
            "quarter", "hellinger", "support", "oracle")}
        if not all(np.isfinite(v) and -1e-9 <= v <= 1+1e-9 for v in values.values()):
            raise ValueError(f"{name} nonfinite/unphysical score at row {ix}")
        if values["oracle"] + 1e-8 < values["hellinger"]:
            raise ValueError(f"{name} Hellinger oracle does not dominate fixed Hellinger")
        formula = item["formula"]
        if not isinstance(formula, str) or not formula:
            raise ValueError("missing molecular formula")
        mode = str(item.get(spec["mode"], "UNKNOWN")).upper()
        bc = item.get("endpoint_bhattacharyya")
        if bc is not None:
            bc = float(bc)
            if not np.isfinite(bc) or not -1e-8 <= bc <= 1+1e-8:
                raise ValueError("invalid endpoint overlap")
        rows.append({
            "formula": formula, "inchikey": str(item["inchikey"]),
            "mode": mode, "bc": bc, **values,
        })
    return rows


def contrast(rows, a, b, seed, infer=True):
    left = np.asarray([r[a] for r in rows])
    right = np.asarray([r[b] for r in rows])
    d = left-right
    result = {
        "n": len(rows),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "win_loss_tie": paired_win_loss_tie(left, right, atol=1e-12),
    }
    if infer:
        blocks = [r["formula"] for r in rows]
        result["formula_block_bootstrap_95"] = paired_block_bootstrap_delta(
            left, right, blocks, n_boot=5000, seed=seed)
        result["formula_block_signflip_2sided"] = paired_block_signflip_test(
            left, right, blocks, n_permutations=20000, seed=seed)
    return result


def subgroups(rows):
    """Fixed, descriptive stratification; no optimal cutoffs or p-value search."""
    parts = {"all": rows}
    modes = sorted({r["mode"] for r in rows})
    for m in modes:
        parts["ion_mode:"+m] = [r for r in rows if r["mode"] == m]
    if all(r["bc"] is not None for r in rows):
        for low, high in zip(BC_BINS[:-1], BC_BINS[1:]):
            sub = [r for r in rows if low <= r["bc"] < high]
            if sub:
                parts[f"endpoint_bc:[{low},{high})"] = sub
    return parts


def evaluate_source(rows, name):
    q, s, o, h = (
        np.asarray([r[key] for r in rows]) for key in
        ("quarter", "support", "oracle", "hellinger")
    )
    tol = 1e-12
    flags_q, flags_s = q > o+tol, s > o+tol
    flags_both = flags_q & flags_s
    contrasts = [
        ("quarter_minus_oracle", "quarter", "oracle"),
        ("support_minus_oracle", "support", "oracle"),
        ("quarter_minus_support", "quarter", "support"),
        ("support_minus_hellinger", "support", "hellinger"),
    ]
    out = {
        "source": name,
        "n": len(rows),
        "n_formulas": len({r["formula"] for r in rows}),
        "n_structures": len({r["inchikey"] for r in rows}),
        "mean_affinity": {k: float(v.mean()) for k,v in (
            ("quarter",q),("support",s),("oracle",o),("hellinger",h))},
        "classification_vs_oracle": {
            "both_strictly_above": int(flags_both.sum()),
            "quarter_only_above": int((flags_q & ~flags_s).sum()),
            "support_only_above": int((~flags_q & flags_s).sum()),
            "neither_above": int((~flags_q & ~flags_s).sum()),
            "oracle_dominates_fixed_hellinger": bool(np.all(o+1e-8>=h)),
            "quarter_and_support_within_0.002_BC": int((np.abs(q-s)<=.002).sum()),
        },
        "paired": {
            label: contrast(rows,a,b,SEED+19*i)
            for i,(label,a,b) in enumerate(contrasts)
        },
        "descriptive_subgroups": {},
    }
    numerator = float((s-o).mean())
    denominator = float((q-o).mean())
    out["ratio_of_mean_support_oracle_to_quarter_oracle"] = (
        numerator/denominator if abs(denominator)>1e-8 else None)
    for label, subs in subgroups(rows).items():
        if label == "all":
            continue
        out["descriptive_subgroups"][label] = {
            "n":len(subs),"formulas":len({r["formula"] for r in subs}),
            "quarter_minus_oracle":contrast(subs,"quarter","oracle",SEED,False)["mean_delta"],
            "support_minus_oracle":contrast(subs,"support","oracle",SEED,False)["mean_delta"],
            "quarter_minus_support":contrast(subs,"quarter","support",SEED,False)["mean_delta"],
        }
    if sum(out["classification_vs_oracle"][x] for x in (
            "both_strictly_above","quarter_only_above","support_only_above",
            "neither_above")) != len(rows):
        raise RuntimeError("oracle comparison partition error")
    return out


def render(results):
    lines = [
        "# Retrospective three-source oracle-versus-support mechanism falsification",
        "",
        "**This is a new analysis of old prediction scores, NOT another prospective experiment.** "
        "No score is recomputed and no α, CE coordinate, thresholds, or alignments are tuned.",
        "",
        "An answer-aware best Hellinger-chord point is not deployable; beating it "
        "excludes Hellinger scalar reparameterization but does not establish "
        "a uniquely physical power-geometry mechanism.",
        "",
        "| Source | n | quarter−Hellinger oracle | support-only−Hellinger oracle | quarter−support-only | Both beat oracle | Quarter-only beat oracle | Support-only beat oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in results.items():
        p = v["paired"]
        c = v["classification_vs_oracle"]
        lines.append(
            f"| {name} | {v['n']} | "
            f"{p['quarter_minus_oracle']['mean_delta']:+.6f} | "
            f"{p['support_minus_oracle']['mean_delta']:+.6f} | "
            f"{p['quarter_minus_support']['mean_delta']:+.6f} | "
            f"{c['both_strictly_above']} | {c['quarter_only_above']} | "
            f"{c['support_only_above']} |"
        )
    for name, v in results.items():
        lines += ["",f"## {name} — formula-block uncertainty"]
        for label, detail in v["paired"].items():
            ci = detail["formula_block_bootstrap_95"]
            test = detail["formula_block_signflip_2sided"]
            lines.append(
                f"- {label}: Δ={detail['mean_delta']:+.6f}, "
                f"95% CI [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}], "
                f"p={test['p_value_two_sided']:.6f}, "
                f"W/L/T={detail['win_loss_tie']}."
            )
        lines.append(
            "- Descriptive ratio of mean (support−oracle)/(quarter−oracle): "
            + str(v["ratio_of_mean_support_oracle_to_quarter_oracle"])
            + ". Not a causal mediated proportion."
        )
        lines += ["", "| Stratum (descriptive) | n | quarter−oracle | support−oracle | quarter−support |",
                  "|---|---:|---:|---:|---:|"]
        for subgroup, detail in v["descriptive_subgroups"].items():
            lines.append(
                f"| {subgroup} | {detail['n']} | "
                f"{detail['quarter_minus_oracle']:+.6f} | "
                f"{detail['support_minus_oracle']:+.6f} | "
                f"{detail['quarter_minus_support']:+.6f} |"
            )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- MassBank and MultiMS2 share no guarantee of fully independent chemical identity despite their separate sources; ExpoLib uses nominal CE whose physical units remain uncertain.",
        "- Formula-block resampling does not turn correlated spectra into new laboratories. There are only three acquisition sources, and there is **no valid cross-laboratory meta-significance claim**.",
        "- The support-only control is a different nongeometric interpolation heuristic; if it also beats the best Hellinger chord, the previously observed oracle gap does **not** uniquely identify a special geometric law.",
        "- These scores concern spectral prediction affinity. No structural identification MRR/top-1 has been improved by this audit.",
        "- Question, bins and comparisons are chosen after previously published outcomes: every inferential contrast here remains retrospective/exploratory.",
        "",
    ]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--massbank",type=Path,required=True)
    p.add_argument("--expolib",type=Path,required=True)
    p.add_argument("--multims2",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    paths={"MassBank":a.massbank,"ExpoLib":a.expolib,"MultiMS2":a.multims2}
    out={}
    for name,path in paths.items():
        rows=canonical_rows(path,name)
        out[name]=evaluate_source(rows,name)
    a.out.mkdir(parents=True,exist_ok=True)
    report=render(out)
    (a.out/"oracle_support_discrimination.json").write_text(
        json.dumps({"evaluation_class":"POSTHOC PREEXISTING SCORES; NOT INDEPENDENT",
                    "original_run_ids":{k:v["run"] for k,v in SOURCES.items()},
                    "results":out},indent=2)+"\n")
    (a.out/"report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
