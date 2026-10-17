"""Frozen 4th-source PhenolicsDB real CID interpolation; no fitted parameters.

Runs ONLY after independent metadata gate. Target 40/60 eV spectra never
contribute to prediction grids for their 20/80 eV reference pair. Source is
different lab / QTOF vendor from prior datasets. Candidate signatures were
predeclared before this SOURCE'S target spectral scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity,
    support_only_suppression,
)
from qfd.stats import (
    paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie,
)
from audit_phenolicsdb_independent_gate import (
    verify_source, SOURCE_BLOB_SHA1, CE_VALUES, INCHIKEY, CE_NAME
)

METHODS=("hellinger", "quarter", "support_only", "linear", "best_endpoint")


def load_selected(path: Path, mode: str, need: set[int]) -> dict[int, tuple[dict, object]]:
    """Open peaks only for previously metadata-selected scan positions."""
    selected={}
    active=None
    index=-1
    mz=[]
    intens=[]
    with path.open("r",encoding="utf-8") as f:
        for raw in f:
            line=raw.strip()
            if line.upper()=="BEGIN IONS":
                if active is not None:
                    raise RuntimeError("nested MGF scan")
                active={}
                mz=[];intens=[]
                index+=1
            elif line.upper()=="END IONS":
                if active is None:
                    raise RuntimeError("MGF END without BEGIN")
                if index in need:
                    if index in selected or not mz or len(mz)!=len(intens):
                        raise RuntimeError("selected scan has duplicate or missing positive data")
                    match=CE_NAME.search(active.get("NAME",""))
                    if (match is None or active.get("IONMODE","").lower()!=mode.lower()
                        or not INCHIKEY.fullmatch(active.get("INCHI",""))):
                        raise RuntimeError("selected source chemical identity/CE metadata corrupted")
                    selected[index]=(
                        {"inchikey":active["INCHI"],"mode":mode,"ce":int(match.group(1)),
                         "precursor":float(active["PEPMASS"].split()[0])},
                        SimpleNamespace(mzs=np.asarray(mz,float), intensities=np.asarray(intens,float))
                    )
                active=None
            elif active is not None and index in need and "=" in line:
                k,v=line.split("=",1)
                k=k.strip().upper()
                if k in active:
                    raise RuntimeError("duplicate selected scan metadata")
                active[k]=v.strip()
            elif active is not None and index in need and line:
                fields=line.split()
                if len(fields)<2:
                    raise RuntimeError("malformed selected target peak row")
                try:
                    m,i=float(fields[0]),float(fields[1])
                except ValueError as ex:
                    raise RuntimeError("non-numeric selected peak") from ex
                if not (math.isfinite(m) and math.isfinite(i) and m>0 and i>=0):
                    raise RuntimeError("selected peak nonfinite/negative or malformed")
                if i>0:
                    mz.append(m);intens.append(i)
    if active is not None or set(selected)!=need:
        raise RuntimeError("selected scan inventory incomplete")
    return selected


def cmp(rows:list[dict], left:str, right:str,seed:int)->dict:
    a=np.asarray([r[left] for r in rows],float)
    b=np.asarray([r[right] for r in rows],float)
    # CRUCIAL: every repeated energy and polarity of the SAME structure
    # stays in one chemical-identity block; 50 rows are NOT 50 independent molecules.
    keys=[r["inchikey"] for r in rows]
    return {
        "mean_delta":float(np.mean(a-b)),
        "wins_losses_ties":paired_win_loss_tie(a,b,atol=1e-12),
        "structure_block_bootstrap_95":paired_block_bootstrap_delta(
            a,b,keys,n_boot=10000,seed=seed),
        "structure_block_signflip_2sided":paired_block_signflip_test(
            a,b,keys,n_permutations=100000,seed=seed),
    }


def summary(rows:list[dict], infer:bool)->dict:
    result={
        "n_queries":len(rows),
        "n_unique_structure_blocks":len({r["inchikey"] for r in rows}),
        "n_mode_structure_groups":len({(r["polarity"],r["inchikey"]) for r in rows}),
        "mean_affinity":{m:float(np.mean([r[m] for r in rows])) for m in METHODS},
        "mean_endpoint_bc":float(np.mean([r["endpoint_bhattacharyya"] for r in rows])),
        "mean_target_unmatched_mass":float(np.mean([r["unmatched_target_mass"] for r in rows])),
        "quarter_minus_hellinger_descriptive":float(np.mean([r["quarter"]-r["hellinger"] for r in rows])),
        "support_minus_hellinger_descriptive":float(np.mean([r["support_only"]-r["hellinger"] for r in rows])),
    }
    if infer:
        primary=cmp(rows,"quarter","hellinger",20261017)
        ci=primary["structure_block_bootstrap_95"]
        p=primary["structure_block_signflip_2sided"]["p_value_two_sided"]
        # Single frozen primary hypothesis; support is a mechanistic control.
        result["primary_quarter_vs_hellinger"]=primary
        result["primary_transfer_confirmed"]=bool(
            primary["mean_delta"]>0 and ci["ci_low"]>0 and p<.05)
        result["secondary_quarter_vs_support"]=cmp(rows,"quarter","support_only",20261018)
    return result


def evaluate(pos:Path, neg:Path, gatedir:Path,out:Path)->dict:
    gate_file=gatedir/"feasibility.json"
    manifest_file=gatedir/"frozen_quadruplets.jsonl"
    gate=json.loads(gate_file.read_text())
    quads=[json.loads(x) for x in manifest_file.read_text().splitlines() if x.strip()]
    if (gate.get("status")!="PHENOLICSDB_INDEPENDENT_CID_METADATA_GATE_V1"
        or gate.get("gate_pass") is not True
        or gate.get("target_peak_positions_or_intensities_opened") is not False
        or gate.get("candidate_selection_based_on_target_scores") is not False
        or gate.get("source_blob_sha1")!=SOURCE_BLOB_SHA1
        or gate.get("coverage",{}).get("complete_quadruplets")!=len(quads)):
        raise RuntimeError("frozen fourth-source metadata-only gate has not passed")
    for mode,path in (("Positive",pos),("Negative",neg)):
        verify_source(path,mode)
    if len({(r["inchikey"],r["polarity"]) for r in quads})!=len(quads):
        raise RuntimeError("duplicate chemistry/polarity in locked population")
    needed=defaultdict(set)
    for row in quads:
        if set(row["source_indices"])!=set(map(str,CE_VALUES)):
            raise RuntimeError("missing or extra CE records")
        needed[row["polarity"]].update(row["source_indices"].values())
    if len(needed["Positive"] & needed["Negative"]):
        # Numeric indices are relative to different source MGF files; overlap
        # is normal and NOT a cross-file duplicate, so no global de-duplication.
        pass
    selected={
        "Positive":load_selected(pos,"Positive",needed["Positive"]),
        "Negative":load_selected(neg,"Negative",needed["Negative"]),
    }
    scored=[]
    for row in quads:
        scans=selected[row["polarity"]]
        specs={}
        for e in CE_VALUES:
            idx=row["source_indices"][str(e)]
            metadata, spectrum=scans[idx]
            if (metadata["inchikey"]!=row["inchikey"]
                or metadata["mode"]!=row["polarity"] or metadata["ce"]!=e
                or abs(metadata["precursor"]-row["source_precursor_mz"][CE_VALUES.index(e)])>.0000001):
                raise RuntimeError("raw heldout chemical identity/CE mismatch")
            specs[e]=spectrum
        grid=reference_only_grid(specs[20],specs[80],tolerance_da=.01)
        p0,p1=grid.low,grid.high
        endpoint_bc=float(np.sum(np.sqrt(p0*p1)))
        for e in (40,60):
            t=(e-20)/60
            q,unmatched=query_mass_on_reference_grid(specs[e],grid,tolerance_da=.01)
            if abs(float(q.sum())+unmatched-1)>1e-7:
                raise RuntimeError("heldout query mass not conserved")
            preds={
                "hellinger":power_mean_distribution(p0,p1,t,.5),
                "quarter":power_mean_distribution(p0,p1,t,.25),
                "support_only":support_only_suppression(p0,p1,t,power=4.),
                "linear":power_mean_distribution(p0,p1,t,1.),
                "best_endpoint":p0 if affinity(q,p0)>affinity(q,p1) else p1,
            }
            vals={name:affinity(q,p) for name,p in preds.items()}
            if any(not math.isfinite(v) or not 0<=v<=1+1e-9 for v in vals.values()):
                raise RuntimeError("invalid or out-of-bounds affinity")
            scored.append({
                "inchikey":row["inchikey"],"polarity":row["polarity"],
                "target_energy_ev":e,"low_energy_ev":20,"high_energy_ev":80,
                "interpolation_t":t,"endpoint_bhattacharyya":endpoint_bc,
                "unmatched_target_mass":unmatched,
                "reference_bin_count":len(grid.mz),
                **vals,
            })
    if (len(scored)!=gate["coverage"]["target_queries"]
        or len({r["inchikey"] for r in scored})!=gate["coverage"]["distinct_structures"]):
        raise RuntimeError("scored results do not cover all metadata-only selected queries")
    result={
        "status":"INDEPENDENT_PHENOLICSDB_FIXED_CID_POWER_TRANSFER",
        "source":"CooperstoneLab/PhenolicsDB: independently collected Agilent QTOF CID",
        "source_blob_sha1":SOURCE_BLOB_SHA1,
        "frozen_manifest_source":"original external source MGF metadata, before target peaks scored",
        "fitted_parameters_on_new_source":False,
        "source_only_small_chemical_cohort":True,
        "predictors":"fixed quarter α=.25, fixed Hellinger α=.5, support-only power=4, linear, best endpoint",
        "all":summary(scored,True),
        "by_polarity":{m:summary([r for r in scored if r["polarity"]==m],False)
                       for m in ("Positive","Negative")},
        "by_target_energy":{str(e):summary([r for r in scored if r["target_energy_ev"]==e],False)
                            for e in (40,60)},
        "scientific_limitations":(
            "Only ~15 unique structure blocks, correlated duplicate target energies and polarity; "
            "no formula is given as a directly verified per-spectrum MGF field. "
            "Acquisition/instrument batch may differ within PhenolicsDB; 20/80 eV endpoints "
            "and 40/60 eV targets differ from all prior protocols. "
            "Only a cross-laboratory CID energy-interpolation observation; "
            "not same-formula identification, EAD confirmation, mechanistic kinetics, "
            "or a universal physical law."
        ),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (out/"query_scores.jsonl").open("w") as f:
        for row in scored:f.write(json.dumps(row,sort_keys=True)+"\n")
    lines=[
        "# Fourth-source independent PhenolicsDB CID transfer result",
        "",
        "Full source SHA anchored, metadata-only selection, 20/80 reference endpoints, 40/60 eV heldouts. Predictions target-blind.",
        "",
        f"**{len(scored)}** scored spectra, **{result['all']['n_unique_structure_blocks']}** unique structures.",
        "",
        "| Polarity | n | Quarter affinity | Hellinger | Support-only | Quarter−Hellinger |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in ("Positive","Negative"):
        s=result["by_polarity"][k]
        lines.append(f"| {k} | {s['n_queries']} | {s['mean_affinity']['quarter']:.6f} | "
                     f"{s['mean_affinity']['hellinger']:.6f} | "
                     f"{s['mean_affinity']['support_only']:.6f} | "
                     f"{s['quarter_minus_hellinger_descriptive']:+.6f} |")
    s=result["all"];p=s["primary_quarter_vs_hellinger"]
    ci=p["structure_block_bootstrap_95"]
    lines += [
        "",f"All: quarter−Hellinger {p['mean_delta']:+.6f}; "
        f"95% identity-cluster CI [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}]; "
        f"two-sided p={p['structure_block_signflip_2sided']['p_value_two_sided']:.6f}.",
        f"Predeclared fixed-quarter superiority: **{s['primary_transfer_confirmed']}**.",
        "",result["scientific_limitations"],"",
    ]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--positive",type=Path,required=True)
    p.add_argument("--negative",type=Path,required=True)
    p.add_argument("--gate-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.positive,a.negative,a.gate_dir,a.out)


if __name__=="__main__":
    main()
