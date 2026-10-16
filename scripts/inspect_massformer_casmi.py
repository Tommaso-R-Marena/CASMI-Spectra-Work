from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _short_repr(value, limit: int = 300) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def inspect_pickle(path: Path) -> dict:
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        samples = {}
        for col in obj.columns:
            vals = obj[col].dropna()
            samples[str(col)] = _short_repr(vals.iloc[0]) if len(vals) else None
        return {
            "type": "DataFrame",
            "rows": int(obj.shape[0]),
            "columns": int(obj.shape[1]),
            "column_names": [str(c) for c in obj.columns],
            "dtypes": {str(c): str(obj[c].dtype) for c in obj.columns},
            "nunique": {
                str(c): int(obj[c].nunique(dropna=True))
                for c in obj.columns
                if obj[c].dtype != object or str(c) in {
                    "spec_id", "mol_id", "query_mol_id", "candidate_mol_id",
                    "prec_type", "ion_mode", "formula", "inchikey_s"
                }
            },
            "sample_values": samples,
        }
    return {"type": type(obj).__name__, "repr": _short_repr(obj)}


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect MassFormer processed CASMI archives without changing data")
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    root = args.root.resolve()
    report = {
        "root": str(root),
        "files": [],
        "pickles": {},
    }
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = str(path.relative_to(root))
        report["files"].append({"path": rel, "size_bytes": int(path.stat().st_size)})
        if path.suffix == ".pkl":
            try:
                report["pickles"][rel] = inspect_pickle(path)
            except Exception as exc:
                report["pickles"][rel] = {
                    "type": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
