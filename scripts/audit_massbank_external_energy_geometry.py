"""Label-free cross-source MassBank 2026.03 feasibility and leakage audit.

No peak intensities are loaded or scored. This source is not a model-fitting
dataset. All data structures used to construct the gate are metadata only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from qfd.msnlib import load_msnlib_jsonl

# Intentionally refuse bare values, normalized CE %, ramps and voltage conventions.
_EXACT_EV = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*eV\s*$", re.I)
_FULL_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def header_value(lines: list[str], prefix: str):
    matches = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def parse_record_metadata(text: str, path: str):
    header = []
    for line in text.splitlines():
        if line.startswith("PK$PEAK:"):
            break
        header.append(line.strip())
    key = header_value(header, "CH$LINK: INCHIKEY ")
    formula = header_value(header, "CH$FORMULA: ")
    energy_raw = header_value(header, "AC$MASS_SPECTROMETRY: COLLISION_ENERGY ")
    energy_match = _EXACT_EV.fullmatch(energy_raw or "")
    if not energy_match:
        return None
    energy = float(energy_match.group(1))
    if not (0 < energy < 1000):
        return None
    ms_type = header_value(header, "AC$MASS_SPECTROMETRY: MS_TYPE ")
    mode = header_value(header, "AC$MASS_SPECTROMETRY: ION_MODE ")
    fragmentation = header_value(header, "AC$MASS_SPECTROMETRY: FRAGMENTATION_MODE ")
    instrument = header_value(header, "AC$INSTRUMENT: ")
    instrument_type = header_value(header, "AC$INSTRUMENT_TYPE: ")
    adduct = header_value(header, "MS$FOCUSED_ION: PRECURSOR_TYPE ")
    n_peaks = header_value(header, "PK$NUM_PEAK: ")
    accession = header_value(header, "ACCESSION: ")
    if not key or not _FULL_KEY.fullmatch(key.upper()) or not formula:
        return None
    if not ms_type or ms_type.upper() != "MS2":
        return None
    if not mode or mode.upper() not in {"POSITIVE", "NEGATIVE"}:
        return None
    if not fragmentation or fragmentation.upper() not in {"CID", "HCD"}:
        return None
    if not instrument or not instrument_type or not adduct or not accession:
        return None
    try:
        n = int(n_peaks)
    except (ValueError, TypeError):
        return None
    if n < 5:
        return None
    parts = PurePosixPath(path).parts
    if "record" not in parts:
        return None
    idx = parts.index("record")
    if idx + 1 >= len(parts) - 1:
        return None
    contributor = parts[idx + 1]
    return {
        "inchikey": key.upper(), "formula": formula, "eV": energy,
        "instrument": instrument, "instrument_type": instrument_type,
        "ion_mode": mode.upper(), "fragmentation": fragmentation.upper(),
        "adduct": adduct, "contributor": contributor,
        "accession": accession, "path": path, "n_peaks": n,
    }


def eligible_triplets(records, excluded):
    groups=defaultdict(lambda:defaultdict(list))
    rejected_overlap=0
    for r in records:
        if r["inchikey"] in excluded:
            rejected_overlap+=1
            continue
        group = (r["inchikey"], r["adduct"], r["instrument"],
                 r["instrument_type"], r["ion_mode"],
                 r["fragmentation"], r["contributor"])
        groups[group][r["eV"]].append(r)
    rows=[]
    ambiguous_groups=0
    formula_conflicts=0
    for group, by_e in sorted(groups.items()):
        if any(len(slot)!=1 for slot in by_e.values()):
            ambiguous_groups+=1
            continue
        vals=[by_e[e][0] for e in sorted(by_e)]
        if len({v["formula"] for v in vals})!=1:
            formula_conflicts+=1
            continue
        for i in range(1,len(vals)-1):
            low,mid,high=vals[i-1:i+2]
            t=(mid["eV"]-low["eV"])/(high["eV"]-low["eV"])
            if not 0<t<1:
                continue
            rows.append({
                "inchikey":mid["inchikey"], "formula":mid["formula"],
                "contributor":mid["contributor"], "instrument":mid["instrument"],
                "instrument_type":mid["instrument_type"],
                "ion_mode":mid["ion_mode"],"fragmentation":mid["fragmentation"],
                "adduct":mid["adduct"],
                "low_path":low["path"],"query_path":mid["path"],"high_path":high["path"],
                "low_eV":low["eV"],"query_eV":mid["eV"],"high_eV":high["eV"],
                "t":t,
            })
    return rows, {"excluded_discovery_overlap_records":rejected_overlap,
                  "excluded_ambiguous_groups":ambiguous_groups,
                  "excluded_formula_conflict_groups":formula_conflicts}


def _sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("massbank_zip",type=Path)
    ap.add_argument("--discovery-msnlib",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    seen={r.full_inchikey.upper() for r in load_msnlib_jsonl(args.discovery_msnlib)}
    records=[]
    total_mb_files=0
    with ZipFile(args.massbank_zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith((".txt",".mb")):
                continue
            total_mb_files+=1
            with archive.open(name) as fp:
                raw=fp.read().decode("utf-8",errors="replace")
            item=parse_record_metadata(raw,name)
            if item is not None:
                records.append(item)
    rows,exclusion=eligible_triplets(records,seen)
    structures={r["inchikey"] for r in rows}
    formulas={r["formula"] for r in rows}
    contributors={r["contributor"] for r in rows}
    types={r["instrument_type"] for r in rows}
    # Instrument brand/model distinguishes a new measurement source from MSnLib.
    non_orbitrap=[
        r for r in rows if "orbitrap" not in (r["instrument_type"]+" "+r["instrument"]).lower()
        and "q exactive" not in (r["instrument_type"]+" "+r["instrument"]).lower()
        and "exactive" not in (r["instrument_type"]+" "+r["instrument"]).lower()
    ]
    offcenter=[r for r in rows if abs(r["t"]-.5)>1e-12]
    coverage={
        "query_triplets":len(rows), "structures":len(structures),
        "formula_blocks":len(formulas), "offcenter_triplets":len(offcenter),
        "contributor_sources":len(contributors), "instrument_types":len(types),
        "non_orbitrap_triplets":len(non_orbitrap),
        "non_orbitrap_structures":len({r["inchikey"] for r in non_orbitrap}),
        "instrument_type_hist":dict(Counter(r["instrument_type"] for r in rows).most_common()),
        "contributor_hist":dict(Counter(r["contributor"] for r in rows).most_common()),
        "energy_triples_hist":dict(Counter(f"{r['low_eV']:g}-{r['query_eV']:g}-{r['high_eV']:g}"
                                            for r in rows).most_common(30)),
    }
    gate={
        "query_triplets":300, "structures":200, "formula_blocks":150,
        "offcenter_triplets":100, "non_orbitrap_triplets":100,
        "non_orbitrap_structures":50,
    }
    passed=all(coverage[k]>=v for k,v in gate.items())
    args.out.mkdir(parents=True,exist_ok=True)
    result={
        "status":"EXTERNAL_MASSBANK_2026_03_FEASIBILITY_ONLY",
        "massbank_release":"2026.03", "source_doi":"10.5281/zenodo.19590880",
        "massbank_archive_sha256":_sha256(args.massbank_zip),
        "discovery_msnlib_sha256":_sha256(args.discovery_msnlib),
        "discovery_unique_structures":len(seen),
        "massbank_record_files":total_mb_files,
        "parsable_conservative_metadata_records":len(records),
        "selection":"MS2 CID/HCD; explicit single-value eV; full InChIKey+adduct+instrument+mode+contributor; unique energy slots; target interior",
        "coverage":coverage,"predeclared_gate":gate,"gate_pass":passed,
        "exclusions":exclusion,
        "guardrail":"NO spectrum intensities or ranking/similarity scores opened; pass required for subsequent external efficacy",
    }
    (args.out/"feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.out/"label_free_triplets.jsonl").write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in rows)
    )
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
