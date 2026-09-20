from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qfd.pilot_eval import evaluate_formula_candidates, rows_to_records, split_query_reference, split_query_reference_replicates, summarize_results
from qfd.pilot_manifest import build_pilot_manifest, build_replicate_pilot_manifest
from qfd.stats import paired_bootstrap_delta, paired_win_loss_tie


def _polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise SystemExit("run_library_pilot requires: pip install -e '.[data]'") from exc
    return pl


def _json_safe(x):
    if isinstance(x, float) and (x == float("inf") or x == float("-inf")):
        return None
    return x


def main():
    p = argparse.ArgumentParser(description="Leakage-aware same-formula multi-energy CASMI library pilot")
    p.add_argument("train_parquet", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results/library_pilot"))
    p.add_argument("--ingest-lib", default="enveda-180")
    p.add_argument("--max-structures", type=int, default=1000)
    p.add_argument("--split-mode", choices=["replicate", "interleaved"], default="replicate")
    p.add_argument("--min-energies-each", type=int, default=2)
    p.add_argument("--intensity-floor", type=float, default=0.001)
    p.add_argument("--max-peaks", type=int, default=128)
    p.add_argument("--transform", choices=["sqrt", "log1p", "none"], default="sqrt")
    p.add_argument("--max-energy-gap-ev", type=float, default=10.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--hard-static-threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.split_mode == "replicate":
        min_distinct_energies = args.min_energies_each
        manifest = build_replicate_pilot_manifest(
            args.train_parquet, ingest_lib=args.ingest_lib,
            min_replicates_per_energy=2, min_energies=min_distinct_energies,
            max_structures=args.max_structures, min_formula_structures=2,
        )
    else:
        min_distinct_energies = 2 * args.min_energies_each
        manifest = build_pilot_manifest(
            args.train_parquet, ingest_lib=args.ingest_lib,
            min_spectra=min_distinct_energies, min_energies=min_distinct_energies,
            max_structures=args.max_structures, min_formula_structures=2,
        )
    ids = manifest["inchikey14"].unique().to_list()
    if len(ids) < 2:
        raise SystemExit("pilot manifest produced fewer than two structures; no meaningful same-formula benchmark is possible")

    pl = _polars()
    columns = [
        "inchikey14", "molecular_formula", "spectrum_id", "adduct",
        "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities",
        "base_peak_intensity", "precursor_mz", "ingest_lib",
    ]
    data = (
        pl.scan_parquet(str(args.train_parquet))
        .select(columns)
        .filter(pl.col("ingest_lib") == args.ingest_lib)
        .filter(pl.col("inchikey14").is_in(ids))
        .filter(pl.col("collision_energy_ev").is_not_null())
        .filter(pl.col("collision_energy_ev").list.len() == 1)
        .collect()
    )
    records = rows_to_records(
        data.to_dicts(), intensity_floor=args.intensity_floor,
        max_peaks=args.max_peaks, transform=args.transform,
    )
    if args.split_mode == "replicate":
        groups = split_query_reference_replicates(records, min_energies=args.min_energies_each)
        score_gap = 0.0
    else:
        groups = split_query_reference(records, min_energies_each=args.min_energies_each)
        score_gap = args.max_energy_gap_ev
    per_query = evaluate_formula_candidates(groups, max_energy_gap_ev=score_gap)
    summary = summarize_results(per_query)

    modes = ("static_best", "joint_energy", "two_energy_qfd")
    if per_query:
        baseline = [r["rr_static_best"] for r in per_query]
        summary["paired_vs_static"] = {}
        for mode in modes[1:]:
            vals = [r[f"rr_{mode}"] for r in per_query]
            summary["paired_vs_static"][mode] = {
                **paired_bootstrap_delta(vals, baseline, n_boot=args.bootstrap, seed=args.seed),
                **paired_win_loss_tie(vals, baseline),
            }

    hard_rows = [r for r in per_query if r.get("best_decoy_static_score", float("-inf")) >= args.hard_static_threshold]
    hard_summary = summarize_results(hard_rows)
    hard_summary["threshold"] = args.hard_static_threshold
    if hard_rows:
        baseline = [r["rr_static_best"] for r in hard_rows]
        hard_summary["paired_vs_static"] = {}
        for mode in modes[1:]:
            vals = [r[f"rr_{mode}"] for r in hard_rows]
            hard_summary["paired_vs_static"][mode] = {
                **paired_bootstrap_delta(vals, baseline, n_boot=args.bootstrap, seed=args.seed),
                **paired_win_loss_tie(vals, baseline),
            }
    summary["hard_static_subset"] = hard_summary

    summary["config"] = {
        "ingest_lib": args.ingest_lib,
        "max_structures": args.max_structures,
        "split_mode": args.split_mode,
        "min_energies_each": args.min_energies_each,
        "required_distinct_energies": min_distinct_energies,
        "intensity_floor": args.intensity_floor,
        "max_peaks": args.max_peaks,
        "transform": args.transform,
        "max_energy_gap_ev": score_gap,
        "hard_static_threshold": args.hard_static_threshold,
        "seed": args.seed,
    }
    summary["cohort"] = {
        "manifest_rows": int(len(manifest)),
        "selected_structures": int(len(ids)),
        "loaded_spectra": int(len(records)),
        "evaluable_structures": int(len(groups)),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(args.out_dir / "manifest.parquet")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if per_query:
        fields = list(per_query[0].keys())
        with (args.out_dir / "per_query.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in per_query:
                writer.writerow({k: _json_safe(v) for k, v in row.items()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
