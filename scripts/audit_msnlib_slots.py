from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl


def _energy_key(rec):
    return tuple(float(e) for e in rec.collision_energies)


def _duplicate_report(slots, *, key_label):
    duplicates = []
    for slot, recs in sorted(slots.items(), key=lambda kv: repr(kv[0])):
        if len(recs) <= 1:
            continue
        identity, adduct, merge_type, energies = slot
        full_keys = sorted({r.full_inchikey for r in recs})
        duplicates.append({
            key_label: identity,
            "adduct": adduct,
            "merge_type": merge_type,
            "collision_energies": list(energies),
            "count": len(recs),
            "n_full_inchikeys": len(full_keys),
            "full_inchikeys": full_keys,
            "spectrum_ids": [r.spectrum_id for r in recs],
            "compound_names": sorted({r.compound_name for r in recs}),
            "formulas": sorted({r.molecular_formula for r in recs}),
            "peak_counts": [len(r.view.mzs) for r in recs],
        })
    return duplicates


def _summary(duplicates, *, identity_field):
    merge_counts = Counter(d["merge_type"] for d in duplicates)
    multiplicities = Counter(d["count"] for d in duplicates)
    identities = sorted({d[identity_field] for d in duplicates})
    return {
        "n_duplicate_slots": len(duplicates),
        "n_affected_identities": len(identities),
        "duplicate_slots_by_merge_type": dict(sorted(merge_counts.items())),
        "duplicate_slot_multiplicity": {str(k): v for k, v in sorted(multiplicities.items())},
    }


def main():
    p = argparse.ArgumentParser(description="Audit repeated MSnLib acquisition slots and identity-layer collisions")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    slots14 = defaultdict(list)
    slots_full = defaultdict(list)
    full_keys_by_connectivity = defaultdict(set)
    for r in records:
        energy = _energy_key(r)
        slots14[(r.inchikey14, r.adduct, r.merge_type, energy)].append(r)
        slots_full[(r.full_inchikey, r.adduct, r.merge_type, energy)].append(r)
        full_keys_by_connectivity[r.inchikey14].add(r.full_inchikey)

    dup14 = _duplicate_report(slots14, key_label="inchikey14")
    dup_full = _duplicate_report(slots_full, key_label="full_inchikey")
    mixed_identity_dup14 = [d for d in dup14 if d["n_full_inchikeys"] > 1]
    same_identity_dup14 = [d for d in dup14 if d["n_full_inchikeys"] == 1]
    connectivity_collisions = {
        k: sorted(v) for k, v in sorted(full_keys_by_connectivity.items()) if len(v) > 1
    }

    out = {
        "n_records": len(records),
        "connectivity_level": {
            "n_slots": len(slots14),
            **_summary(dup14, identity_field="inchikey14"),
            "n_duplicate_slots_mixed_full_identity": len(mixed_identity_dup14),
            "n_duplicate_slots_same_full_identity": len(same_identity_dup14),
        },
        "full_identity_level": {
            "n_slots": len(slots_full),
            **_summary(dup_full, identity_field="full_inchikey"),
        },
        "n_connectivity_keys_with_multiple_full_inchikeys": len(connectivity_collisions),
        "connectivity_to_full_identity_collisions": connectivity_collisions,
        "duplicates_connectivity_level": dup14,
        "duplicates_full_identity_level": dup_full,
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
