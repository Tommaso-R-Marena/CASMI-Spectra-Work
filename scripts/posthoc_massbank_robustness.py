"""Post-release, post-hoc audit of unchanged MassBank spectral prediction scores.

Inputs are immutable artifacts from completed runs 38918658343 and 38920155068.
Nothing is fitted, ranked, or re-scored. All subset findings are exploratory.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

METHODS = ("alpha_quarter", "hellinger_fixed", "support_only", "hellinger_oracle", "convex_oracle")
CONTROLS = METHODS[1:]
EXPECTED = {"queries": 2138, "structures": 715, "formula_blocks": 613}
ARCHIVE_SHA = "12c6a922929f9bf2484c46586b6101512a415e44680558883b0630c6e47905fb"
PREVIOUS = ("mcebio_neg","mcescaf_pos","enammol_pos","enamdisc_neg")

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def verify(rows, original_manifest, overlap_manifest, original, audit):
    if not (len(rows) == len(original_manifest) == len(overlap_manifest) == EXPECTED["queries"]):
        raise ValueError("frozen query count mismatch")
    fields = ("query_path","low_path","high_path","inchikey","formula","contributor",
              "instrument","instrument_type","ion_mode","fragmentation","adduct",
              "low_eV","query_eV","high_eV","t")
    def canonical(items):
        result = {}
        for row in items:
            key = row["query_path"]
            if key in result:
                raise ValueError("duplicate target accession")
            result[key] = tuple(row[k] for k in fields)
        return result
    if canonical(rows) != canonical(original_manifest) or canonical(rows) != canonical(overlap_manifest):
        raise ValueError("scored observation differs from original label-free manifests")
    if original.get("status") != "MASSBANK_2026_03_EXTERNAL_EFFICACY_IF_GATE_PASSED":
        raise ValueError("wrong original result")
    if original.get("archive_sha256") != ARCHIVE_SHA or original["all"]["coverage"] != EXPECTED:
        raise ValueError("original archive or coverage differs")
    if audit.get("status") != "POSTHOC_MASSBANK_FULL_INCHIKEY_SOURCE_OVERLAP":
        raise ValueError("wrong prior overlap audit")
    if audit.get("frozen_massbank_source_sha256") != ARCHIVE_SHA:
        raise ValueError("different MassBank source archive")
    if audit.get("n_query_triplets") != len(rows) or audit.get("discovery_excluded_structures_reappearing") != 0:
        raise ValueError("original overlap audit fails")
    if len({r["formula"] for r in rows}) != EXPECTED["formula_blocks"]:
        raise ValueError("formula count differs")
    if len({r["inchikey"] for r in rows}) != EXPECTED["structures"]:
        raise ValueError("structure count differs")
    for method in METHODS:
        arr = np.array([float(r[method]) for r in rows])
        if not np.isfinite(arr).all() or (arr < -1e-12).any() or (arr > 1+1e-12).any():
            raise ValueError("invalid affinity value")
        if abs(float(arr.mean()) - original["all"]["methods"][method]["mean"]) > 1e-12:
            raise ValueError("original frozen score mean changed")
    for row in rows:
        if row["hellinger_oracle"] + 1e-9 < row["hellinger_fixed"]:
            raise ValueError("invalid fixed-path Hellinger oracle")
        if not 0 <= row["unmatched_query_mass"] <= 1:
            raise ValueError("invalid unmatched query mass")

def exact_prior_keys(inv, rows, audit):
    if inv.get("status") != "PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC":
        raise ValueError("wrong prior inventory")
    if set(inv["sources"]) != set(PREVIOUS):
        raise ValueError("must include all four prior evaluated cohorts")
    keys = inv["all_prior_evaluated_full_inchikeys"]
    if keys != sorted(set(keys)) or any(k != k.upper() for k in keys):
        raise ValueError("key inventory must be uppercase sorted unique")
    prior = set(keys)
    common = prior & {r["inchikey"].upper() for r in rows}
    n_queries = sum(r["inchikey"].upper() in common for r in rows)
    if (len(common) != audit["n_overlapping_prior_msnlib_structures"]
            or n_queries != audit["n_overlapping_prior_msnlib_queries"]):
        raise ValueError("computed exact-structure overlap conflicts with original audit")
    return prior

def delta_summary(rows, against, draws=3000, seed=0, infer=True):
    if not rows:
        return None
    diff = np.array([r["alpha_quarter"] - r[against] for r in rows], dtype=float)
    groups = defaultdict(list)
    for row, d in zip(rows, diff):
        groups[row["formula"]].append(float(d))
    result = {
        "mean_query_weighted_delta":float(diff.mean()),
        "mean_equal_formula_weight_delta":float(np.mean([np.mean(x) for x in groups.values()])),
        "median_query_delta":float(np.median(diff)),
        "wins":int(np.sum(diff > 1e-12)),
        "losses":int(np.sum(diff < -1e-12)),
        "ties":int(np.sum(abs(diff) <= 1e-12)),
    }
    if infer and len(groups) >= 3:
        rng = np.random.default_rng(seed)
        sums = np.array([np.sum(v) for v in groups.values()])
        sizes = np.array([len(v) for v in groups.values()])
        ix = rng.integers(len(groups), size=(draws,len(groups)))
        replicate = sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)
        ci = np.quantile(replicate,[.025,.975])
        result["formula_block_bootstrap_95"] = {
            "ci_low":float(ci[0]),"ci_high":float(ci[1]),
            "n_blocks":len(groups),"n_draws":draws,"seed":seed,
            "estimand":"query-weighted paired mean; whole-formula resampling",
        }
    return result

def summarize(rows, draws=3000, seed=0, infer=True):
    if not rows:
        return None
    return {
        "queries":len(rows),
        "structures":len({r["inchikey"] for r in rows}),
        "formula_blocks":len({r["formula"] for r in rows}),
        "mean_unmatched_query_mass":float(np.mean([r["unmatched_query_mass"] for r in rows])),
        "mean_affinity":{m:float(np.mean([r[m] for r in rows])) for m in METHODS},
        "comparisons":{m:delta_summary(rows,m,draws,seed+101*i,infer) for i,m in enumerate(CONTROLS)},
    }

def analyze(rows,audit,prior=None,draws=3000):
    rows = sorted(rows,key=lambda r:r["query_path"])
    out = {
        "status":"POSTHOC_MASSBANK_SENSITIVITY_ONLY",
        "evaluation_class":"EXPLORATORY; NO NEW HELD-OUT VALIDATION",
        "original_efficacy_run":38918658343,
        "prior_metadata_audit_run":38920155068,
        "unchanged_original_scores":True,
        "no_refitting_or_retuning":True,
        "all":summarize(rows,draws,20261016),
    }
    cuts = {
        "non_orbitrap":[r for r in rows if r["non_orbitrap"]],
        "other_instrumentation":[r for r in rows if not r["non_orbitrap"]],
        "offcenter_energy":[r for r in rows if abs(r["t"]-.5)>1e-12],
        "midpoint_energy":[r for r in rows if abs(r["t"]-.5)<=1e-12],
        "positive_ion":[r for r in rows if r["ion_mode"]=="POSITIVE"],
        "negative_ion":[r for r in rows if r["ion_mode"]=="NEGATIVE"],
    }
    if prior is not None:
        cuts["no_prior_msnlib_structure_overlap"]=[r for r in rows if r["inchikey"].upper() not in prior]
        cuts["prior_msnlib_structure_overlap"]=[r for r in rows if r["inchikey"].upper() in prior]
        if len(cuts["no_prior_msnlib_structure_overlap"]) != audit["n_nonoverlapping_prior_msnlib_queries"]:
            raise ValueError("nonoverlap cohort not the original exact 2032 queries")
        if len(cuts["prior_msnlib_structure_overlap"]) != audit["n_overlapping_prior_msnlib_queries"]:
            raise ValueError("overlap cohort not the original exact 106 queries")
        cuts["non_orbitrap_no_prior_msnlib_structure_overlap"]=[
            r for r in cuts["no_prior_msnlib_structure_overlap"] if r["non_orbitrap"]]
    out["subgroups"]={name:summarize(sub,draws,20261100+i*100)
                      for i,(name,sub) in enumerate(cuts.items())}
    cohort,contributors=defaultdict(list),defaultdict(list)
    for r in rows:
        cohort[(r["contributor"],r["instrument_type"])].append(r)
        contributors[r["contributor"]].append(r)
    out["source_instrument_descriptive"]=[
        {"contributor":c,"instrument_type":t,**summarize(v,draws,0,infer=False)}
        for (c,t),v in sorted(cohort.items())]
    out["leave_one_contributor_out_descriptive"]=[
        {"omitted_contributor":s,"remaining_queries":len(rows)-len(v),
         "mean_quarter_minus_fixed_hellinger":float(np.mean([
             r["alpha_quarter"]-r["hellinger_fixed"] for r in rows if r["contributor"]!=s]))}
        for s,v in sorted(contributors.items())]
    out["equal_contributor_weighted_mean_delta"]=float(np.mean([
        np.mean([r["alpha_quarter"]-r["hellinger_fixed"] for r in v])
        for v in contributors.values()]))
    out["n_contributors"]=len(contributors)
    # Seven contributors are not seven interchangeable or randomized laboratories.
    # Enumerate all 2^n group-wise flips rather than treating individual spectra
    # as independent source-level replications. EXPLORATORY, not confirmatory.
    from itertools import product
    by_source=[np.asarray([r["alpha_quarter"]-r["hellinger_fixed"] for r in v])
               for _,v in sorted(contributors.items())]
    means=np.asarray([x.mean() for x in by_source])
    counts=np.asarray([len(x) for x in by_source])
    sign_patterns=np.asarray(list(product((-1.,1.),repeat=len(means))))
    obs_macro=float(means.mean())
    obs_query=float(np.dot(means,counts)/counts.sum())
    flipped_macro=(sign_patterns*means).mean(axis=1)
    flipped_query=(sign_patterns*(means*counts)).sum(axis=1)/counts.sum()
    out["contributor_level_signflip_descriptive"]={
        "n_sources":len(by_source),"assignments":len(sign_patterns),
        "source_equal_weighted_mean":obs_macro,
        "source_equal_weighted_two_sided_exact_p":float(np.mean(
            np.abs(flipped_macro)>=abs(obs_macro)-1e-15)),
        "query_weighted_mean":obs_query,
        "query_weighted_source_flip_two_sided_exact_p":float(np.mean(
            np.abs(flipped_query)>=abs(obs_query)-1e-15)),
        "classification":"POSTHOC / source-level diagnostics, NOT a confirmatory sampling-based trial",
    }
    out["scientific_limitations"]=(
        "All follow-up subgroups are post-hoc, correlated and exploratory. "
        "Exact full-InChIKey nonoverlap cannot establish independent acquisition "
        "or chemical-family independence. Contributor and instrument confounded. "
        "No change to the original all-query frozen primary analysis.")
    return out

def render(out):
    def row(name,part):
        if not part:
            return ""
        a=part["comparisons"]["hellinger_fixed"]
        b=part["comparisons"]["hellinger_oracle"]
        ci=a.get("formula_block_bootstrap_95")
        ci_text=f"[{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}]" if ci else "n/a"
        return f"| {name} | {part['queries']} | {part['formula_blocks']} | {a['mean_query_weighted_delta']:+.6f} | {ci_text} | {b['mean_query_weighted_delta']:+.6f} |"
    lines=[
        "# MassBank 2026.03 post-hoc sensitivity of frozen scored queries","",
        "**EXPLORATORY, not a new independent external validation.**","",
        "| Cohort | n | Formula blocks | α=.25 − fixed Hellinger | 95% formula-block bootstrap | α=.25 − optimized Hellinger oracle |",
        "|---|---:|---:|---:|---|---:|",
        row("Original complete cohort",out["all"]),
    ]
    for name,part in out["subgroups"].items():
        lines.append(row(name.replace("_"," "),part))
    lines += ["","## Contributor/instrument heterogeneity","",
              "| Contributor | Instrument | n | α=.25 − fixed Hellinger | α=.25 − oracle |",
              "|---|---|---:|---:|---:|"]
    for part in out["source_instrument_descriptive"]:
        lines.append(f"| {part['contributor']} | {part['instrument_type']} | {part['queries']} | "
                     f"{part['comparisons']['hellinger_fixed']['mean_query_weighted_delta']:+.6f} | "
                     f"{part['comparisons']['hellinger_oracle']['mean_query_weighted_delta']:+.6f} |")
    lines += ["","## Leave-one-contributor-out","",
              "| Omitted contributor | Remaining n | Mean gain over fixed Hellinger |",
              "|---|---:|---:|"]
    for part in out["leave_one_contributor_out_descriptive"]:
        lines.append(f"| {part['omitted_contributor']} | {part['remaining_queries']} | "
                     f"{part['mean_quarter_minus_fixed_hellinger']:+.6f} |")
    lines += ["",f"Contributor-equal-weighted Δ: {out['equal_contributor_weighted_mean_delta']:+.6f} "
              f"({out['n_contributors']} sources; not a population-generalization estimate).",
              "",f"Exploratory exact source-label flips (2-sided): source-equal-weighted p="
              f"{out['contributor_level_signflip_descriptive']['source_equal_weighted_two_sided_exact_p']:.6f}; "
              f"query-weighted source-flip p="
              f"{out['contributor_level_signflip_descriptive']['query_weighted_source_flip_two_sided_exact_p']:.6f}.",
              "",out["scientific_limitations"],""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--efficacy-dir",type=Path,required=True)
    p.add_argument("--overlap-dir",type=Path,required=True)
    p.add_argument("--prior-inventory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap-draws",type=int,default=3000)
    a=p.parse_args()
    if not 100<=a.bootstrap_draws<=10000:
        raise ValueError("bootstrap-draws outside supported range")
    rows=read_jsonl(a.efficacy_dir/"rows.jsonl")
    original=json.loads((a.efficacy_dir/"result.json").read_text())
    audit=json.loads((a.overlap_dir/"source_overlap.json").read_text())
    verify(rows,read_jsonl(a.efficacy_dir/"label_free_triplets.jsonl"),
           read_jsonl(a.overlap_dir/"label_free_triplets.jsonl"),original,audit)
    prior=exact_prior_keys(json.loads(a.prior_inventory.read_text()),rows,audit)
    out=analyze(rows,audit,prior,draws=a.bootstrap_draws)
    out["input_sha256"]={
        "rows":hashlib.sha256((a.efficacy_dir/"rows.jsonl").read_bytes()).hexdigest(),
        "source_overlap":hashlib.sha256((a.overlap_dir/"source_overlap.json").read_bytes()).hexdigest(),
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"result.json").write_text(json.dumps(out,indent=2)+"\n")
    (a.out/"report.md").write_text(render(out))
    print(render(out))

if __name__=="__main__":
    main()
