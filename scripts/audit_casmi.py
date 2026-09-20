from __future__ import annotations

import argparse
import json
from pathlib import Path

from qfd.data import metadata_audit


def main():
    p = argparse.ArgumentParser(description="Memory-safe CASMI train.parquet metadata audit")
    p.add_argument("train_parquet", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/data_audit.json"))
    args = p.parse_args()
    result = metadata_audit(args.train_parquet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
