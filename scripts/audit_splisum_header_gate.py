"""Fail-closed SPLISUM external source header audit. ZERO fragment peaks parsed.

This phase is deliberately separate from query-scoring. A failed header gate
does not authorize opening any query peak m/z or intensity.
"""
from __future__ import annotations
import argparse
import bisect
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

MD5={
 "qpos":"3142dab3cd679611d0f0d50b95f248ca",
 "qneg":"dce2f8f93746fb859c19ae798cf9306f",
 "tpos":"1675977e3d76d5f1b0ada7e5968ee95d",
 "tneg":"ab73102f4a2e008cc2db606f6c144ed5",
}
MINIMA={"evaluable_queries":100,"distinct_evaluable_structures":50,
        "isomer_or_isobar_queries":60}
IK=re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
MASS_KEYS=("PEPMASS","PRECURSORMZ","PRECURSOR_MZ","PARENTMASS","PRECURSOR_MASS")
IK_KEYS=("INCHIKEY","INCHIKEY_FULL","INCHI_KEY","INCHIAUX")
SMILES_KEYS=("SMILES","CANONICAL_SMILES","STRUCTURE_SMILES","SMILE")
MISSING={"","NA","N/A","NONE","NULL","UNKNOWN","*"}
ION_KEYS=("IONMODE","ION_MODE","IONIZATION_MODE","POLARITY")


def file_md5(path):
    h=hashlib.md5()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):h.update(buf)
    return h.hexdigest()


def explicit_identity(header):
    for k in IK_KEYS:
        s=header.get(k,"").strip().upper()
        if IK.fullmatch(s):
            return "inchikey:"+s
    for k in SMILES_KEYS:
        s=header.get(k,"").strip()
        if s and s.upper() not in MISSING:
            return "literal_smiles:"+s
    return None


def precursor(header):
    for k in MASS_KEYS:
        value=header.get(k,"").split()
        if not value:continue
        try: n=float(value[0])
        except ValueError: continue
        if math.isfinite(n) and n>0:return n
    return None


def headers_only(path:Path,mode:str):
    """Peaks are not counted, tokenized as floats, matched, or stored."""
    rows=[];fields=Counter(); has_name=Counter(); active=None;index=-1
    with path.open("r",encoding="utf-8",errors="replace") as f:
        for raw in f:
            line=raw.strip()
            if line.upper()=="BEGIN IONS":
                if active is not None:raise RuntimeError("nested MGF IONS")
                active={};index+=1
            elif line.upper()=="END IONS":
                if active is None:raise RuntimeError("END IONS without header")
                for key in active:fields[key]+=1
                label=explicit_identity(active);mass=precursor(active)
                native_mode=next((active.get(k) for k in ION_KEYS if active.get(k)),None)
                if native_mode is not None:
                    sign=native_mode.upper()
                    if (mode=="positive" and sign not in ("POSITIVE","POS","P","+","1+")) or (
                        mode=="negative" and sign not in ("NEGATIVE","NEG","N","-","1-")):
                        raise RuntimeError("native ion polarity contradicts original source file")
                rows.append({"index":index,"label":label,"precursor_mz":mass,
                             "mode":mode})
                if label:has_name[label.split(":")[0]]+=1
                active=None
            elif active is not None and "=" in line and not line.upper().startswith("BEGIN"):
                key,val=line.split("=",1)
                k=key.strip().upper()
                # Only official-looking one-token MGF header keys.
                if re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}",k):
                    if k in active:raise RuntimeError("duplicate native MGF header key "+k)
                    active[k]=val.strip()
            # Otherwise ignore the entire peak line without inspecting it.
    if active is not None:raise RuntimeError("unterminated MGF")
    return rows,{"scans":len(rows),"header_keys_count":dict(fields),
                 "explicit_identity_type_count":dict(has_name),
                 "with_precursor":sum(r["precursor_mz"] is not None for r in rows),
                 "with_molecule_identity":sum(r["label"] is not None for r in rows)}


def candidate_pool(targets,query):
    qmass=query["precursor_mz"]
    if qmass is None:return []
    tol=max(.02,20e-6*qmass)
    mzs=[t["precursor_mz"] for t in targets]
    start=bisect.bisect_left(mzs,qmass-tol)
    end=bisect.bisect_right(mzs,qmass+tol)
    return targets[start:end]


def check(files:dict[str,Path],out:Path):
    rec={};summary={}
    for key in MD5:
        p=files[key]
        if not p.is_file() or file_md5(p)!=MD5[key]:
            raise RuntimeError(f"upstream source {key} MD5 mismatch; fail closed")
        mode="positive" if key.endswith("pos") else "negative"
        rec[key],summary[key]=headers_only(p,mode)
    qualifying=[]
    modes={}
    for mode in ("positive","negative"):
        suffix="pos" if mode=="positive" else "neg"
        source=sorted([r for r in rec["t"+suffix] if r["precursor_mz"] is not None],
                      key=lambda r:(r["precursor_mz"],r["index"]))
        used=0;ident=0;inpool=0;hard=0;groups=Counter()
        for q in rec["q"+suffix]:
            if not q["label"] or q["precursor_mz"] is None:continue
            ident+=1
            pool=candidate_pool(source,q)
            competitors={p["label"] for p in pool if p["label"]}
            if q["label"] in competitors:
                inpool+=1
                if len(competitors)>=2:
                    hard+=1
                qualifying.append({
                    "query_source":"q"+suffix,"query_index":q["index"],
                    "identity":q["label"],"polarity":mode,
                    "precursor_mz":q["precursor_mz"],
                    "target_candidate_structures":len(competitors),
                    "target_candidate_scans":len(pool),
                    "verified_truth_in_mass_pool":True,
                })
                groups[q["label"]]+=1
                used+=1
        modes[mode]={"query_total":len(rec["q"+suffix]),
                     "target_total":len(rec["t"+suffix]),
                     "query_with_identity_and_mass":ident,
                     "query_true_label_in_mass_pool":inpool,
                     "query_with_at_least_2_structures":hard,
                     "distinct_molecule_identity_labels":len(groups)}
    coverage={
        "evaluable_queries":len(qualifying),
        "distinct_evaluable_structures":len({q["identity"] for q in qualifying}),
        "isomer_or_isobar_queries":sum(q["target_candidate_structures"]>=2 for q in qualifying),
        "positive_evaluable":modes["positive"]["query_true_label_in_mass_pool"],
        "negative_evaluable":modes["negative"]["query_true_label_in_mass_pool"],
    }
    passed=all(coverage[k]>=v for k,v in MINIMA.items())
    result={
        "status":"SPLISUM_2026_FIFTH_ORIGIN_HEADER_ONLY_GATE_V1",
        "source_doi":"10.5281/zenodo.19834896",
        "source_checksums_md5":MD5,
        "method":"strict full InChIKey or literal raw SMILES; NO title/name inference; max(.02 Da,20ppm) same-mode candidate pool",
        "gate_pass":passed,"thresholds":MINIMA,
        "coverage":coverage,"polarity":modes,"source_metadata_audits":summary,
        "query_fragment_mass_values_or_intensities_opened":False,
        "source_target_fragments_read":False,
        "score_model_selected_using_this_source":False,
        "scientific_interpretation":(
            "PASS permits later frozen scoring on an independent single-energy identification corpus; "
            "FAIL requires no source peak scoring and no efficacy claims."
        ),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"feasibility.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    with (out/"eligible_queries.jsonl").open("w") as f:
        for q in qualifying:f.write(json.dumps(q,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for n in MD5:
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    files={k:getattr(a,k) for k in MD5}
    print(json.dumps(check(files,a.out),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
