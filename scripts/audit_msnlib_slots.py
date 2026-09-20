from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl


def _energy_key(rec):
    return tuple(float(e) for e in rec.collision_energies)


def main():
    p = argparse.ArgumentParser(description="Audit repeated MSnLib acquisition slots")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    slots = defaultdict(list)
    for r in records:
        slots[(r.inchikey14, r.adduct, r.merge_type, _energy_key(r))].append(r)

    duplicates = []
    for (key, adduct, merge_type, energies), recs in sorted(slots.items()):
        if len(recs) <= 1:
            continue
        duplicates.append({
            "inchikey14": key,
            "adduct": adduct,
            "merge_type": merge_type,
            "collision_energies": list(energies),
            "count": len(recs),
            "spectrum_ids": [r.spectrum_id for r in recs],
            "compound_names": sorted({r.compound_name for r in recs}),
            "formulas": sorted({r.molecular_formula for r in recs}),
            "peak_counts": [len(r.view.mzs) for r in recs],
        })

    merge_counts = Counter(d["merge_type"] for d in duplicates)
    multiplicities = Counter(d["count"] for d in duplicates)
    affected_structures = sorted({d["inchikey14"] for d in duplicates})
    out = {
        "n_records": len(records),
        "n_slots": len(slots),
        "n_duplicate_slots": len(duplicates),
        "n_affected_structures": len(affected_structures),
        "duplicate_slots_by_merge_type": dict(sorted(merge_counts.items())),
        "duplicate_slot_multiplicity": {str(k): v for k, v in sorted(multiplicities.items())},
        "duplicates": duplicates,
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
