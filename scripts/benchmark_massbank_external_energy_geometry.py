"""Conditional external MassBank efficacy: no scores unless frozen label-free gate passes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from qfd.power_geometry import power_mean_distribution
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity,
    support_only_suppression, best_hellinger_chord_oracle, best_convex_oracle,
)

METHODS=("alpha_quarter","hellinger_fixed","support_only",
         "hellinger_oracle","convex_oracle")


def parse_peak_spectrum(raw: str, *, record_path: str):
    """MassBank PK$PEAK second column is raw peak intensity."""
    found=False
    mz=[]
    intensity=[]
    for line in raw.splitlines():
        s=line.strip()
        if s.startswith("PK$PEAK:"):
            if found:
                raise ValueError(f"duplicate peak header: {record_path}")
            found=True
            continue
        if not found:
            continue
        if s=="//":
            break
        fields=s.split()
        if len(fields)<2:
            continue
        try:
            m=float(fields[0])
            v=float(fields[1])
        except ValueError:
            continue
        if np.isfinite(m) and np.isfinite(v) and m>0 and v>0:
            mz.append(m)
            intensity.append(v)
    if len(mz)<5:
        raise ValueError(f"fewer than five valid positive peaks: {record_path}")
    return SimpleNamespace(mzs=np.asarray(mz,dtype=float),
                           intensities=np.asarray(intensity,dtype=float))


def compare(rows,left,right,seed):
    x=np.asarray([float(r[left]) for r in rows])
    y=np.asarray([float(r[right]) for r in rows])
    blocks=[str(r["formula"]) for r in rows]
    return {
        "mean_delta":float(np.mean(x-y)),
        "n_queries":len(rows),"n_formula_blocks":len(set(blocks)),
        "win_loss_tie":paired_win_loss_tie(x,y,atol=1e-15),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            x,y,blocks,n_boot=3000,seed=seed),
        "formula_block_signflip_two_sided":paired_block_signflip_test(
            x,y,blocks,n_permutations=50000,seed=seed),
    }


def summarize(rows,seed=20261016):
    if not rows:
        raise ValueError("empty external cohort")
    comparisons=[
        ("alpha_quarter","hellinger_fixed"),
        ("alpha_quarter","support_only"),
        ("alpha_quarter","hellinger_oracle"),
        ("alpha_quarter","convex_oracle"),
    ]
    return {
        "coverage":{"queries":len(rows),
                    "formula_blocks":len(set(r["formula"] for r in rows)),
                    "structures":len(set(r["inchikey"] for r in rows))},
        "methods":{m:{"mean":float(np.mean([r[m] for r in rows])),
                      "median":float(np.median([r[m] for r in rows]))}
                   for m in METHODS},
        "mean_unmatched_query_mass":float(np.mean([r["unmatched_query_mass"] for r in rows])),
        "comparisons":{f"{a}_vs_{b}":compare(rows,a,b,seed+i)
                       for i,(a,b) in enumerate(comparisons)},
    }


def non_orbitrap(row):
    s=(str(row["instrument_type"])+" "+str(row["instrument"])).lower()
    return not any(x in s for x in ("orbitrap","q exactive","exactive"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("massbank_zip",type=Path)
    p.add_argument("--feasibility-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()

    gate=json.loads((args.feasibility_dir/"feasibility.json").read_text())
    if gate.get("status")!="EXTERNAL_MASSBANK_2026_03_FEASIBILITY_ONLY":
        raise RuntimeError("wrong/unknown feasibility protocol")
    if not isinstance(gate.get("gate_pass"),bool) or gate["gate_pass"] is not True:
        raise RuntimeError("FROZEN_EXTERNAL_GATE_FAILED: no spectra read or scored")
    import hashlib
    h=hashlib.sha256()
    with args.massbank_zip.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    if h.hexdigest()!=gate.get("massbank_archive_sha256"):
        raise RuntimeError("massbank archive mismatch to label-free feasibility")
    manifest=[json.loads(line) for line in
              (args.feasibility_dir/"label_free_triplets.jsonl").read_text().splitlines()
              if line.strip()]
    if len(manifest)!=gate["coverage"]["query_triplets"]:
        raise RuntimeError("manifest/feasibility query count mismatch")
    cache={}
    rows=[]
    with ZipFile(args.massbank_zip) as archive:
        names=set(archive.namelist())
        for entry in manifest:
            paths=(entry["query_path"],entry["low_path"],entry["high_path"])
            if any(p not in names for p in paths):
                raise RuntimeError("missing required accession in pinned archive")
            def view(p):
                if p not in cache:
                    cache[p]=parse_peak_spectrum(
                        archive.read(p).decode("utf-8",errors="replace"),record_path=p)
                return cache[p]
            query,lo,hi=(view(x) for x in paths)
            grid=reference_only_grid(lo,hi,tolerance_da=0.01)
            q,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=0.01)
            t=float(entry["t"])
            p0,p1=grid.low,grid.high
            score=lambda predicted:affinity(q,predicted)
            quarter=score(power_mean_distribution(p0,p1,t,0.25))
            hellinger=score(power_mean_distribution(p0,p1,t,0.5))
            support=score(support_only_suppression(p0,p1,t,power=4))
            oracle=best_hellinger_chord_oracle(q,p0,p1)
            convex=best_convex_oracle(q,p0,p1)
            if oracle["affinity"]+1e-9<hellinger:
                raise ValueError("oracle must dominate fixed Hellinger")
            row={
                **entry,
                "alpha_quarter":quarter,
                "hellinger_fixed":hellinger,
                "support_only":support,
                "hellinger_oracle":oracle["affinity"],
                "convex_oracle":convex,
                "unmatched_query_mass":unmatched,
                "oracle_hellinger_weight":oracle["oracle_weight"],
                "non_orbitrap":non_orbitrap(entry),
            }
            if any(not np.isfinite(row[k]) for k in METHODS):
                raise RuntimeError("nonfinite spectral score")
            rows.append(row)
    if len(rows)!=len(manifest):
        raise RuntimeError("incomplete scored cohort")
    non_orb=[r for r in rows if r["non_orbitrap"]]
    if len(non_orb)<gate["predeclared_gate"]["non_orbitrap_triplets"] or \
       len(set(r["inchikey"] for r in non_orb))<gate["predeclared_gate"]["non_orbitrap_structures"]:
        raise RuntimeError("non-Orbitrap subset changed since feasibility")
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r["contributor"],r["instrument_type"])].append(r)
    cohorts=[
        {"contributor":k[0],"instrument_type":k[1],"n":len(v),
         "n_structures":len(set(r["inchikey"] for r in v)),
         "delta_quarter_vs_hellinger":float(np.mean([r["alpha_quarter"]-r["hellinger_fixed"] for r in v]))}
        for k,v in sorted(grouped.items())]
    result={
        "status":"MASSBANK_2026_03_EXTERNAL_EFFICACY_IF_GATE_PASSED",
        "source":"official MassBank 2026.03", "posthoc":False,
        "discovery_exclusion":"all exact full InChIKeys from MCEBIO+ model discovery",
        "archive_sha256":gate["massbank_archive_sha256"],
        "baseline_model_exponent":0.25,
        "all":summarize(rows),
        "non_orbitrap":summarize(non_orb,seed=20261022),
        "source_cohorts_descriptive_only":cohorts,
        "caveat":"separate archive not enough to prove independent acquisitions, structures, instruments or provenance",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.out/"rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
