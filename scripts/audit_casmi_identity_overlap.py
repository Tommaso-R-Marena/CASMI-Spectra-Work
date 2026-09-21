from __future__ import annotations

import argparse
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


def _id_to_smiles(mol_df: pd.DataFrame) -> dict[int, str]:
    needed = {"mol_id", "smiles"}
    if not needed.issubset(mol_df.columns):
        raise RuntimeError(f"mol_df missing columns: {sorted(needed - set(mol_df.columns))}")
    if mol_df["mol_id"].duplicated().any():
        raise RuntimeError("mol_df contains duplicate mol_id values")
    return {int(r.mol_id): str(r.smiles) for r in mol_df[["mol_id", "smiles"]].itertuples(index=False)}


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-outcome structural-overlap audit for frozen CASMI-2016 -> CASMI-2022 transfer")
    p.add_argument("casmi2016_root", type=Path)
    p.add_argument("casmi2022_root", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    model = json.loads(args.frozen_model.read_text())
    train_ids = tuple(int(x) for x in model.get("training_queries", ()))
    if not train_ids:
        raise RuntimeError("frozen model has no training_queries")

    mol16 = pd.read_pickle(_find_one(args.casmi2016_root, "mol_df.pkl"))
    mol22 = pd.read_pickle(_find_one(args.casmi2022_root, "mol_df.pkl"))
    spec22 = pd.read_pickle(_find_one(args.casmi2022_root, "spec_df.pkl"))
    map16 = _id_to_smiles(mol16)
    map22 = _id_to_smiles(mol22)

    missing_train = sorted(set(train_ids) - set(map16))
    if missing_train:
        raise RuntimeError(f"frozen training mol_ids missing from CASMI 2016 mol_df: {missing_train[:10]}")

    exact_to_train: dict[str, list[int]] = defaultdict(list)
    conn_to_train: dict[str, list[int]] = defaultdict(list)
    for mid in train_ids:
        _, ik, conn = _structure_keys(map16[mid])
        exact_to_train[ik].append(mid)
        conn_to_train[conn].append(mid)

    positive = spec22[spec22["prec_type"] == "[M+H]+"].copy()
    truth_ids = tuple(sorted(set(int(x) for x in positive["mol_id"])))
    missing_truth = sorted(set(truth_ids) - set(map22))
    if missing_truth:
        raise RuntimeError(f"CASMI 2022 [M+H]+ truth mol_ids missing from mol_df: {missing_truth[:10]}")

    spec_count = positive.groupby("mol_id").size().to_dict()
    rows = []
    for mid in truth_ids:
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
        "exact_overlap_query_mol_ids": [int(x) for x in out.loc[out["exact_training_overlap"], "query_mol_id"]],
        "connectivity_overlap_query_mol_ids": [int(x) for x in out.loc[out["connectivity_training_overlap"], "query_mol_id"]],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "truth_overlap.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
