from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def _structure_keys(smiles: str) -> tuple[str, str, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"RDKit could not parse SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey or len(inchikey.split("-")[0]) != 14:
        raise RuntimeError(f"could not derive standard InChIKey for {canonical!r}")
    return canonical, inchikey, inchikey.split("-")[0]


def _load_selected_smiles(path: Path, wanted_ids: set[int], label: str) -> dict[int, str]:
    """Load one large molecular table at a time and immediately discard it.

    CASMI 2022 contains >1M candidate molecules.  The overlap audit needs only
    the frozen 2016 training truths and the 2022 lockbox truths, so retaining
    both complete mol_df tables at once is unnecessary and can exhaust a hosted
    runner.  This helper preserves the exact structural-overlap definition while
    minimizing peak memory.
    """
    print(f"[{label}] loading molecular table: {path}", flush=True)
    frame = pd.read_pickle(path)
    needed = {"mol_id", "smiles"}
    if not needed.issubset(frame.columns):
        raise RuntimeError(f"{label} mol_df missing columns: {sorted(needed - set(frame.columns))}")

    ids = frame["mol_id"].astype(int)
    selected = frame.loc[ids.isin(wanted_ids), ["mol_id", "smiles"]].copy()
    selected["mol_id"] = selected["mol_id"].astype(int)
    if selected["mol_id"].duplicated().any():
        raise RuntimeError(f"{label} mol_df contains duplicate selected mol_id values")

    found = set(int(x) for x in selected["mol_id"])
    missing = sorted(wanted_ids - found)
    if missing:
        raise RuntimeError(f"{label} mol_ids missing from mol_df: {missing[:10]}")
    if len(selected) != len(wanted_ids):
        raise RuntimeError(
            f"{label}: expected {len(wanted_ids)} selected rows, found {len(selected)}"
        )

    out = {
        int(r.mol_id): str(r.smiles)
        for r in selected[["mol_id", "smiles"]].itertuples(index=False)
    }
    del selected, ids, frame
    gc.collect()
    print(f"[{label}] retained {len(out)} requested structures; full table released", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-outcome structural-overlap audit for frozen CASMI-2016 -> CASMI-2022 transfer"
    )
    p.add_argument("casmi2016_root", type=Path)
    p.add_argument("casmi2022_root", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    model = json.loads(args.frozen_model.read_text())
    train_ids = tuple(int(x) for x in model.get("training_queries", ()))
    if not train_ids:
        raise RuntimeError("frozen model has no training_queries")
    train_set = set(train_ids)

    # Determine the lockbox truth IDs from the much smaller spectrum table first.
    print("[CASMI2022] loading spectrum table to identify positive-ion truths", flush=True)
    spec22 = pd.read_pickle(_find_one(args.casmi2022_root, "spec_df.pkl"))
    positive = spec22[spec22["prec_type"] == "[M+H]+"][["mol_id"]].copy()
    truth_ids = tuple(sorted(set(int(x) for x in positive["mol_id"])))
    truth_set = set(truth_ids)
    spec_count = positive.groupby("mol_id").size().to_dict()
    del positive, spec22
    gc.collect()
    print(f"[CASMI2022] identified {len(truth_ids)} unique positive-ion truths", flush=True)

    # Load and release each very large molecule table sequentially.  Never keep
    # the full 2016 and 2022 molecular universes resident at the same time.
    map16 = _load_selected_smiles(
        _find_one(args.casmi2016_root, "mol_df.pkl"), train_set, "CASMI2016"
    )

    exact_to_train: dict[str, list[int]] = defaultdict(list)
    conn_to_train: dict[str, list[int]] = defaultdict(list)
    for i, mid in enumerate(train_ids, 1):
        _, ik, conn = _structure_keys(map16[mid])
        exact_to_train[ik].append(mid)
        conn_to_train[conn].append(mid)
        if i % 10 == 0 or i == len(train_ids):
            print(f"[CASMI2016] keyed {i}/{len(train_ids)} frozen training truths", flush=True)
    del map16
    gc.collect()

    map22 = _load_selected_smiles(
        _find_one(args.casmi2022_root, "mol_df.pkl"), truth_set, "CASMI2022"
    )

    rows = []
    for i, mid in enumerate(truth_ids, 1):
        canonical, ik, conn = _structure_keys(map22[mid])
        exact_matches = sorted(exact_to_train.get(ik, ()))
        conn_matches = sorted(conn_to_train.get(conn, ()))
        rows.append({
            "query_mol_id": mid,
            "positive_mh_spectrum_count": int(spec_count.get(mid, 0)),
            "canonical_isomeric_smiles": canonical,
            "inchikey": ik,
            "connectivity_block": conn,
            "exact_training_overlap": bool(exact_matches),
            "connectivity_training_overlap": bool(conn_matches),
            "exact_matching_training_mol_ids": ";".join(map(str, exact_matches)),
            "connectivity_matching_training_mol_ids": ";".join(map(str, conn_matches)),
        })
        if i % 10 == 0 or i == len(truth_ids):
            print(f"[CASMI2022] keyed {i}/{len(truth_ids)} lockbox truths", flush=True)
    del map22
    gc.collect()

    out = pd.DataFrame(rows)
    n = int(len(out))
    n_exact = int(out["exact_training_overlap"].sum())
    n_conn = int(out["connectivity_training_overlap"].sum())
    summary = {
        "protocol": {
            "timing": "computed before CASMI 2022 ranking outcomes",
            "training_set": "exact 71 CASMI-2016 truth mol_ids embedded in the frozen model",
            "lockbox_truth_scope": "CASMI 2022 [M+H]+ truth molecules",
            "exact_identity_definition": "equality of standard RDKit InChIKey derived from processed canonical SMILES",
            "connectivity_identity_definition": "equality of the 14-character first block of the standard InChIKey",
            "effect_on_primary_analysis": "none; overlap labels are secondary sensitivity strata only",
            "memory_policy": "large mol_df tables loaded sequentially; only predeclared truth/training IDs retained",
        },
        "counts": {
            "training_truth_molecules": int(len(train_ids)),
            "training_unique_exact_inchikeys": int(len(exact_to_train)),
            "training_unique_connectivity_blocks": int(len(conn_to_train)),
            "casmi2022_positive_mh_truth_molecules": n,
            "exact_training_overlap_truths": n_exact,
            "exact_unseen_truths": n - n_exact,
            "connectivity_training_overlap_truths": n_conn,
            "connectivity_unseen_truths": n - n_conn,
        },
        "exact_overlap_query_mol_ids": [
            int(x) for x in out.loc[out["exact_training_overlap"], "query_mol_id"]
        ],
        "connectivity_overlap_query_mol_ids": [
            int(x) for x in out.loc[out["connectivity_training_overlap"], "query_mol_id"]
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "truth_overlap.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
