from __future__ import annotations

import argparse
from pathlib import Path

from qfd.pilot_manifest import write_pilot_manifest


def main():
    p = argparse.ArgumentParser(description="Build a deterministic multi-energy CASMI pilot manifest")
    p.add_argument("train_parquet", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/pilot_manifest.parquet"))
    p.add_argument("--ingest-lib", default="enveda-180")
    p.add_argument("--min-spectra", type=int, default=3)
    p.add_argument("--min-energies", type=int, default=2)
    p.add_argument("--max-structures", type=int, default=5000)
    args = p.parse_args()
    manifest = write_pilot_manifest(
        args.train_parquet,
        args.out,
        ingest_lib=args.ingest_lib,
        min_spectra=args.min_spectra,
        min_energies=args.min_energies,
        max_structures=args.max_structures,
    )
    print(f"wrote {len(manifest)} structure/adduct groups to {args.out}")


if __name__ == "__main__":
    main()
