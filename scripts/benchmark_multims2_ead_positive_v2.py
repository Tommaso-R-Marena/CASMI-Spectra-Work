"""V2 POSITIVE-only EAD scoring after metadata-only V1 polarity gate failed.

Original predictor/individual scoring methods are unchanged and frozen at
4295a1ae7ead5f56a0d5b57a772cd651032d1362. This changed only population
to the entire positive-ion V1-eligible cohort BEFORE EAD target peaks opened.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import audit_multims2_third_source_metadata as original
import audit_multims2_ead_metadata as ead
from benchmark_multims2_ead_lockbox import read_selected_spectra,score_one,summary
from ead_positive_only_v2_gate import STATUS, SOURCE_SHA

def run(args):
    gatefile=args.gate_dir/"feasibility.json"
    manifestfile=args.gate_dir/"frozen_triplets.jsonl"
    original_gatefile=args.original_v1_dir/"feasibility.json"
    original_manifestfile=args.original_v1_dir/"frozen_triplets.jsonl"
    gate=json.loads(gatefile.read_text())
    v1=json.loads(original_gatefile.read_text())
    rows=[json.loads(s) for s in manifestfile.read_text().splitlines() if s.strip()]
    originals=[json.loads(s) for s in original_manifestfile.read_text().splitlines() if s.strip()]
    if (gate.get("status")!=STATUS or gate.get("gate_pass") is not True or
        gate.get("new_v2_target_intensity_opened") is not False or
        gate.get("new_v2_outcome_scores_opened") is not False or
        gate.get("original_v1_gate_sha256")!=original.digest(original_gatefile) or
        gate.get("original_v1_manifest_sha256")!=original.digest(original_manifestfile) or
        v1.get("gate_pass") is not False or
        v1.get("target_scores_or_predictor_outcomes_opened") is not False or
        gate.get("source_sha256")!=SOURCE_SHA):
        raise RuntimeError("V2 positive-only lockbox data gate unavailable or altered")
    original_positive=[r for r in originals if r["ion_mode"]=="POSITIVE"]
    if (len(originals)!=695 or len(rows)!=694 or len(original_positive)!=694 or
        rows!=original_positive or
        gate["coverage"]["triplets"]!=694 or
        any(r["ion_mode"]!="POSITIVE" for r in rows)):
        raise RuntimeError("V2 is not complete original metadata-selected positive cohort")
    if (args.mgf.stat().st_size!=original.UPSTREAM_BYTES or
        original.git_blob_sha1(args.mgf)!=original.UPSTREAM_BLOB_SHA1 or
        original.digest(args.mgf)!=SOURCE_SHA):
        raise RuntimeError("original raw input bytes differ from pre-score gate")
    if original.digest(args.cid_manifest)!=ead.PINNED_CID_MANIFEST_SHA256:
        raise RuntimeError("original CID manifest differs from prior knowledge")
    prior=[json.loads(s) for s in args.cid_manifest.read_text().splitlines() if s.strip()]
    evaluated_keys={r["inchikey"] for r in prior}
    required={r[role+"_index"] for r in rows for role in ("low","query","high")}
    spectra=read_selected_spectra(args.mgf,required)
    scored=[score_one(r,spectra) for r in rows]
    if len(scored)!=694 or len({r["formula"] for r in scored})!=gate["coverage"]["formula_blocks"]:
        raise RuntimeError("scored population differs from entire selected positive cohort")
    nonoverlap=[r for r in scored if r["inchikey"] not in evaluated_keys]
    result={
        "status":"FROZEN_EAD_POSITIVE_ONLY_V2_MODALITY_TRANSFER",
        "evaluation_class":"V2 metadata-conditioned positive EAD new TARGET intensity transfer; same previously used laboratory and MultiMS2 source",
        "original_v1_gate_run":38985385615,
        "original_v1_gate_failed":True,
        "original_v1_negative_triplets_excluded_by_polarity":1,
        "v2_gate_file_sha256":original.digest(gatefile),
        "v2_manifest_sha256":original.digest(manifestfile),
        "source_sha256":SOURCE_SHA,
        "frozen_predictor_original_sha":"4295a1ae7ead5f56a0d5b57a772cd651032d1362",
        "model_fit_on_EAD":False,
        "predeclared_primary_hybrid_vs_both_criteria":"positive mean, positive formula CI low, Holm adjusted p<0.05 for hybrid−quarter and hybrid−support",
        "all_positive":summary(scored,infer=True),
        "no_prior_CID_same_exact_full_inchikey":summary(nonoverlap,infer=(len(nonoverlap)>=30 and len({r["formula"] for r in nonoverlap})>=15)),
        "source_dataset_ids":dict(Counter(r["source_dataset_id"] for r in scored)),
        "scientific_limits":"the original two-polarity EAD gate failed. V2 is positive-only, same laboratory, after CID data were observed, and not molecular-structure identification.",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.out/"rows.jsonl").open("w") as output:
        for r in scored:
            output.write(json.dumps(r,sort_keys=True)+"\n")
    report=[
        "# EAD cross-fragmentation prospective-in-modality positive-only V2 result","",
        "Previous V1 all-polarity feasibility FAILED: 694 positive and 1 negative, required >=5 negative. V2 is separately declared after HEADER metadata, before EAD 16-eV peaks. No post-outcome filtering.",
        "",
        "| Cohort | Triplets | Formulas | Hybrid | Quarter | Support-only | Hellinger | Hybrid−quarter | Hybrid−support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name,k in (("Complete positive V2","all_positive"),("No exact CID prior identity","no_prior_CID_same_exact_full_inchikey")):
        v=result[k]
        if v is None:continue
        s=v["mean_scores"];d=v["descriptive_deltas"]
        report.append(f"| {name} | {v['n_queries']} | {v['n_formulas']} | "+
                      f"{s['hybrid']:.6f} | {s['quarter']:.6f} | {s['support']:.6f} | "+
                      f"{s['hellinger']:.6f} | {d['hybrid_minus_quarter']:+.6f} | "+
                      f"{d['hybrid_minus_support']:+.6f} |")
    v=result["all_positive"]
    for name,data in v["paired"].items():
        ci=data["formula_block_bootstrap_95"]
        report.append(f"- {name}: Δ={data['mean_delta']:+.6f}; 95% formula CI "+
                      f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}], "+
                      f"block p={data['formula_block_signflip_2sided']['p_value_two_sided']:.6f}, "+
                      f"W/L/T={data['win_loss_tie']}")
    report.extend(["",
        f"Predeclared joint hybrid-beats-quarter-and-support: **{v['breakthrough_gate_hybrid_better_than_both']}**.",
        f"Holm primary p-values: {v['holm_two_primary_p_values']}.",
        "",result["scientific_limits"],""])
    (args.out/"report.md").write_text("\n".join(report))
    print("\n".join(report))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--original-v1-dir",type=Path,required=True)
    p.add_argument("--cid-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    return run(p.parse_args())

if __name__=="__main__":
    main()
