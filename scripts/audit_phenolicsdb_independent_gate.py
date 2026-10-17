"""Independent source PhenolicsDB exact-identity metadata-only feasibility gate.

Never parse target m/z or intensities. Source is pinned to a public version of
CooperstoneLab's independent Agilent QTOF authentic-standards collection.

All 20/40/60/80 eV records are selected BEFORE target peak outcomes are opened.
40 and 60 eV will be held out, and 20 and 80 eV used as reference endpoints.
This is CID only, not new EAD evidence; all metadata thresholds are fixed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import subprocess

SOURCE_COMMIT = "381693c8435a851710953f6b818f6008e2227377"
SOURCE_BLOB_SHA1 = {
    "Positive": "748ec688c8eeee952260189623275ff892aec1c9",
    "Negative": "19a4cb3a3afac23cc6637f03961aa1bf5ad5851e",
}
SOURCE_BYTES = {"Positive": 137354, "Negative": 169770}
CE_VALUES = (20, 40, 60, 80)
INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
CE_NAME = re.compile(r" (20|40|60|80)eV$", flags=re.IGNORECASE)
MINIM = {"positive_quadruplets": 8, "negative_quadruplets": 8,
         "distinct_structures": 12, "target_queries": 40}


def verify_source(path: Path, mode: str) -> None:
    if mode not in SOURCE_BLOB_SHA1 or not path.is_file() or path.stat().st_size != SOURCE_BYTES[mode]:
        raise RuntimeError("PhenolicsDB source path or size mismatched immutable manifest")
    blob = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    if blob != SOURCE_BLOB_SHA1[mode]:
        raise RuntimeError("PhenolicsDB historical Git blob SHA-1 mismatch")


def header_only(path: Path, mode: str) -> list[dict]:
    """Traverse boundaries and headers only; ignore *contents* of all peak lines."""
    result = []
    active = None
    index = -1
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.upper() == "BEGIN IONS":
                if active is not None:
                    raise RuntimeError("nested MGF scan")
                active = {}
                index += 1
            elif line.upper() == "END IONS":
                if active is None:
                    raise RuntimeError("MGF END without BEGIN")
                required = ("NAME", "INCHI", "IONMODE", "MSLEVEL", "PEPMASS")
                if any(k not in active for k in required):
                    raise RuntimeError("missing PhenolicsDB source metadata")
                match = CE_NAME.search(active["NAME"])
                if not match:
                    raise RuntimeError("unknown energy in original PhenolicsDB spectrum name")
                if active["IONMODE"].lower() != mode.lower() or active["MSLEVEL"] != "2":
                    raise RuntimeError("unexpected ionization or ms level")
                key = active["INCHI"]
                if not INCHIKEY.fullmatch(key):
                    raise RuntimeError("not a full original InChIKey")
                precursor = float(active["PEPMASS"].split()[0])
                if not math.isfinite(precursor) or precursor <= 0:
                    raise RuntimeError("invalid precursor mass")
                result.append({
                    "record_index": index, "inchikey": key,
                    "polarity": mode, "energy_ev": int(match.group(1)),
                    "precursor_mz": precursor, "source_name": active["NAME"],
                })
                active = None
            elif active is not None and "=" in line:
                k, value = line.split("=", 1)
                k = k.strip().upper()
                if k in active:
                    raise RuntimeError("duplicate source header " + k)
                active[k] = value.strip()
            # A non-header peak line is deliberately never parsed or counted.
    if active is not None:
        raise RuntimeError("unterminated source MGF scan")
    return result


def build(positive: list[dict], negative: list[dict]) -> tuple[list[dict], dict]:
    all_records = positive + negative
    groups = defaultdict(dict)
    for row in all_records:
        key = (row["polarity"], row["inchikey"])
        e = row["energy_ev"]
        if e in groups[key]:
            raise RuntimeError("duplicate identity/polarity/energy cannot be silently selected")
        groups[key][e] = row
    quads = []
    for (mode, key), m in sorted(groups.items()):
        if any(e not in m for e in CE_VALUES):
            continue
        precursors = [m[e]["precursor_mz"] for e in CE_VALUES]
        if max(precursors)-min(precursors) > .01:
            raise RuntimeError("precursor identity shifts across energy records")
        quads.append({
            "inchikey": key, "polarity": mode,
            "source_indices": {str(e): m[e]["record_index"] for e in CE_VALUES},
            "source_precursor_mz": precursors,
            "heldout_target_energies_ev": [40, 60],
            "reference_energies_ev": [20, 80],
        })
    coverage = {
        "total_records_positive": len(positive),
        "total_records_negative": len(negative),
        "positive_quadruplets": sum(x["polarity"]=="Positive" for x in quads),
        "negative_quadruplets": sum(x["polarity"]=="Negative" for x in quads),
        "complete_quadruplets": len(quads),
        "distinct_structures": len({x["inchikey"] for x in quads}),
        "target_queries": 2*len(quads),
    }
    return quads, coverage


def evaluate(pos_path: Path, neg_path: Path, out: Path) -> dict:
    for mode, path in (("Positive", pos_path), ("Negative", neg_path)):
        verify_source(path, mode)
    pos, neg = header_only(pos_path, "Positive"), header_only(neg_path, "Negative")
    quads, coverage = build(pos, neg)
    passed = all(coverage[k]>=v for k, v in MINIM.items())
    result = {
        "status": "PHENOLICSDB_INDEPENDENT_CID_METADATA_GATE_V1",
        "source": "CooperstoneLab/PhenolicsDB",
        "source_commit": SOURCE_COMMIT,
        "source_blob_sha1": SOURCE_BLOB_SHA1,
        "source_bytes": SOURCE_BYTES,
        "gate_pass": passed,
        "thresholds": MINIM,
        "coverage": coverage,
        "metadata_only": True,
        "target_peak_positions_or_intensities_opened": False,
        "candidate_selection_based_on_target_scores": False,
        "independent_lab_and_qtof_vendor": True,
        "earlier_experiment_outcomes_already_known": True,
        "claim_limitations": "Only 15 nominal distinct chemistry structures expected; compound-cluster inference is small and heterogeneous; spectra may mix Agilent 6545/6546 batches; nominal eV alone does not prove calibrated collision energies.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/"feasibility.json").write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    with (out/"frozen_quadruplets.jsonl").open("w") as f:
        for row in quads:
            f.write(json.dumps(row, sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--positive", type=Path, required=True)
    p.add_argument("--negative", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(a.positive, a.negative, a.out), indent=2))


if __name__=="__main__":
    main()
