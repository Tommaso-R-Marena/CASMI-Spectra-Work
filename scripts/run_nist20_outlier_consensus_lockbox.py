from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name!r} under {root}, found {len(hits)}")
    return hits[0]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="One-shot NIST20 Outlier evaluation of the unchanged CASMI-2016-frozen consensus ranker")
    p.add_argument("nist_root", type=Path)
    p.add_argument("cfm_root", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    spec_path = _find_one(args.nist_root, "spec_df.pkl")
    cand_path = _find_one(args.nist_root, "cand_df.pkl")
    cfm_path = _find_one(args.cfm_root, "pcasmi_spec_df.pkl")

    spec = pd.read_pickle(spec_path)
    required = {"spec_id", "mol_id", "prec_type", "peaks", "prec_mz"}
    if not required.issubset(spec.columns):
        raise RuntimeError(f"NIST20 spec_df missing {sorted(required - set(spec.columns))}")
    pos = spec[spec["prec_type"] == "[M+H]+"].copy()
    if pos.empty:
        raise RuntimeError("no positive [M+H]+ NIST20 spectra")

    # Prospectively fixed molecule-balanced primary sample: exactly one query spectrum
    # per truth molecule, chosen without using any candidate-ranking outcome.
    primary = (
        pos.sort_values(["mol_id", "spec_id"], kind="mergesort")
        .groupby("mol_id", sort=True, as_index=False, group_keys=False)
        .head(1)
        .copy()
    )
    if primary["mol_id"].duplicated().any():
        raise RuntimeError("molecule-balanced selection failed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "all_positive_mh_spectra": int(len(pos)),
        "unique_positive_truth_molecules": int(pos["mol_id"].nunique()),
        "primary_selected_spectra": int(len(primary)),
        "rule": "within each mol_id, choose lowest spec_id after stable sort by (mol_id, spec_id)",
        "uses_ranking_outcomes": False,
        "selected_spec_ids": [int(x) for x in primary["spec_id"].tolist()],
    }
    (args.out_dir / "primary_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="nist20_lockbox_") as td:
        td = Path(td)
        data_root = td / "nist"
        cfm_root = td / "cfm"
        data_root.mkdir(parents=True)
        cfm_root.mkdir(parents=True)
        primary.to_pickle(data_root / "spec_df.pkl")
        _link_or_copy(cand_path, data_root / "cand_df.pkl")
        # The frozen evaluator is intentionally reused unchanged. It expects the
        # CASMI-2022 prediction filename, so expose the NIST20 pCASMI table under
        # that local alias only; bytes and scores are otherwise untouched.
        _link_or_copy(cfm_path, cfm_root / "casmi22_spec_df.pkl")

        env = dict(os.environ)
        root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "src"), str(root / "scripts"), env.get("PYTHONPATH", "")]
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_casmi2022_consensus_lockbox.py"),
                str(data_root),
                str(cfm_root),
                str(args.frozen_model),
                "--out-dir",
                str(args.out_dir),
            ],
            check=True,
            env=env,
        )

    summary_path = args.out_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    protocol = summary["protocol"]
    protocol["dataset"] = "NIST20 Outlier / MassFormer pCASMI"
    protocol["dataset_role"] = "one-shot third-domain external lockbox; no NIST20 outcome-driven model changes permitted"
    protocol["candidate_spectra"] = "published pCASMI CFM predictions at 10/20/40 eV only; no measured candidate spectra"
    protocol["primary_unit"] = "one prospectively selected [M+H]+ spectrum per unique truth molecule"
    protocol["primary_selection_rule"] = selection["rule"]
    protocol["source_positive_mh_spectra"] = selection["all_positive_mh_spectra"]
    protocol["source_unique_truth_molecules"] = selection["unique_positive_truth_molecules"]
    protocol.pop("no_casmi2022_tuning", None)
    protocol["no_nist20_tuning"] = True
    summary["coverage"]["source_positive_mh_spectra"] = selection["all_positive_mh_spectra"]
    summary["coverage"]["source_unique_truth_molecules"] = selection["unique_positive_truth_molecules"]
    summary["interpretation_guardrail"] = (
        "A passing gate is independent evidence that the CASMI-2016-frozen energy-consensus ranker transfers to a third data domain (NIST20 Outlier) conditional on the published pCASMI CFM candidate spectra, candidate lists, and the prospectively fixed molecule-balanced query selection. It does not prove universal small-molecule identification, perfect candidate generation, negative-ion generalization, or invariance to prediction model and instrument domain."
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({
        "selection": selection,
        "coverage": summary["coverage"],
        "static": summary["primary_ge_0_99"]["metrics"]["static"],
        "gated": summary["primary_ge_0_99"]["metrics"]["gated"],
        "gated_risk": summary["primary_ge_0_99"]["risk"]["gated"],
        "gated_inference": summary["primary_ge_0_99"]["inference"]["gated_vs_static"],
        "replication_gate": summary["primary_ge_0_99"]["replication_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
