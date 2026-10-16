"""Post-release exact-structure overlap audit on immutable ExpoLib transfer scores.

Preserve original 1,125-query result. Nonoverlap is retrospective exploratory,
not a fresh acquisition. Never fit or recompute spectral predictors.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import numpy as np
from qfd.stats import paired_block_bootstrap_delta,paired_block_signflip_test

DIGESTS={
  "expolib_rows":"f85b67d61be82f62ee1072944f9d300eb2d1d81462eba6ad221566dfaa276695",
  "expolib_result":"62d8716688b3eb6d2e086a5783c1b141aaa0433666e1fe26754c08a6c04e4d27",
  "expolib_manifest":"095340d967b177077ce517da7aa332795ce691899cc1320563cce07a76a68560",
  "prior_keys":"5520ecb3f134290953fd5ad63b97cc1561dd39c2ea658aeaf4a90b2be348d07a",
  "massbank_manifest":"265dfb75e33d2eff3f19945255bddb5db0c0f7f9161593ada80cd2b50ecac9c4",
}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def paired(rows, against="hellinger",seed=20261016):
    if not rows:
        return None
    x=np.array([r["quarter"] for r in rows])
    y=np.array([r[against] for r in rows])
    formula=[r["formula"] for r in rows]
    return {
      "n_queries":len(rows),
      "n_structures":len({r["inchikey"] for r in rows}),
      "n_formulas":len(set(formula)),
      "mean_difference":float(np.mean(x-y)),
      "mean_quarter_affinity":float(np.mean(x)),
      "mean_comparator_affinity":float(np.mean(y)),
      "wins":int(np.sum(x-y>1e-12)),
      "losses":int(np.sum(x-y< -1e-12)),
      "formula_bootstrap":paired_block_bootstrap_delta(x,y,formula,n_boot=10000,seed=seed),
      "formula_signflip_two_sided":paired_block_signflip_test(
            x,y,formula,n_permutations=50000,seed=seed),
    }

def analyze(rows,manifest,previous_keys,massbank_keys,original):
    if len(rows)!=1125 or len(manifest)!=1125:
        raise ValueError("original full ExpoLib cohort changed")
    def identity(x):
        return (x["source"],x["query_index"],x["inchikey"],x["formula"],
                x["adduct"],x["mode"],x["fragmentation"],x["t"])
    if set(map(identity,rows))!=set(map(identity,manifest)):
        raise ValueError("scored query does not match original label-free ExpoLib manifest")
    original_mean=original["all"]["mean_quarter_minus_hellinger"]
    if abs(original_mean-np.mean([r["quarter"]-r["hellinger"] for r in rows]))>1e-12:
        raise ValueError("frozen ExpoLib result mean changed")
    structures={r["inchikey"] for r in rows}
    msn_overlap=structures&previous_keys
    mass_overlap=structures&massbank_keys
    union_overlap=msn_overlap|mass_overlap
    controls={
      "no_prior_four_MSnLib_structure_overlap":[r for r in rows if r["inchikey"] not in previous_keys],
      "no_prior_MassBank_eligible_structure_overlap":[r for r in rows if r["inchikey"] not in massbank_keys],
      "no_prior_MSnLib_or_MassBank_structure_overlap":[r for r in rows if r["inchikey"] not in previous_keys|massbank_keys],
      "original_midpoint_nominal_CE":[r for r in rows if abs(r["t"]-.5)<1e-12],
      "original_offcenter_nominal_CE":[r for r in rows if abs(r["t"]-.5)>1e-12],
    }
    if (len(msn_overlap)!=8 or len(mass_overlap)!=25 or len(union_overlap)!=32
        or len(controls["no_prior_MSnLib_or_MassBank_structure_overlap"])!=823):
        raise ValueError("original independent source metadata overlap changed")
    return {
       "status":"POSTHOC_EXPOLIB_SOURCE_OVERLAP_EXPLORATORY",
       "evidence":"FROZEN ORIGIN RESULTS UNMODIFIED; NONOVERLAP RETROSPECTIVE",
       "original_source_scoring_run":38964797817,
       "original_source_metadata_run":38964190641,
       "original_source_queries":len(rows),
       "exact_structure_overlaps":{
            "prior_MSnLib_evaluated":len(msn_overlap),
            "prior_MassBank_eligible":len(mass_overlap),
            "union_previous_MSnLib_or_MassBank":len(union_overlap),
            "no_overlap_structures":len(structures-union_overlap),
            "no_overlap_queries":len(controls["no_prior_MSnLib_or_MassBank_structure_overlap"]),
        },
       "original_all":paired(rows),
       "subgroups":{key:paired(cohort,seed=20261017+i*7)
                    for i,(key,cohort) in enumerate(controls.items())},
       "no_training_or_retuning":True,
       "limits":"Exact full-InChIKey nonoverlap does not prove separate acquisition or distinct chemical scaffold. Units of collision energy unknown. Exploratory source audit, no new source replication.",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expolib-result-dir",required=True,type=Path)
    p.add_argument("--expolib-metadata-dir",required=True,type=Path)
    p.add_argument("--prior-inventory",required=True,type=Path)
    p.add_argument("--massbank-manifest",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    items={
       "expolib_rows":args.expolib_result_dir/"rows.jsonl",
       "expolib_result":args.expolib_result_dir/"result.json",
       "expolib_manifest":args.expolib_metadata_dir/"nominal_energy_triplets.jsonl",
       "prior_keys":args.prior_inventory,
       "massbank_manifest":args.massbank_manifest,
    }
    for name,path in items.items():
        if sha(path)!=DIGESTS[name]:
            raise RuntimeError(f"source artifact {name} checksum changed")
    original=json.loads(items["expolib_result"].read_text())
    inventory=json.loads(items["prior_keys"].read_text())
    if inventory.get("status")!="PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC":
        raise ValueError("wrong prior-library inventory")
    previous=set(inventory["all_prior_evaluated_full_inchikeys"])
    massbank={r["inchikey"] for r in lines(items["massbank_manifest"])}
    result=analyze(lines(items["expolib_rows"]),lines(items["expolib_manifest"]),previous,massbank,original)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    text=["# ExpoLib exact-structure nonoverlap and nominal-energy falsification",
          "","**POSTHOC sensitivity only. Original ExpoLib failure of universal α=.25 is preserved.**","",
          "| Set | n queries | n structures | Formula blocks | Δ quarter minus Hellinger | 95% formula-block CI | p |",
          "|---|---:|---:|---:|---:|---|---:|"]
    for name,row in [("All frozen source",result["original_all"]),*result["subgroups"].items()]:
        ci=row["formula_bootstrap"]
        text.append(f"| {name} | {row['n_queries']} | {row['n_structures']} | {row['n_formulas']} | "
                    f"{row['mean_difference']:+.6f} | [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}] | "
                    f"{row['formula_signflip_two_sided']['p_value_two_sided']:.6f} |")
    text += ["",result["limits"],""]
    (args.out/"report.md").write_text("\n".join(text))
    print("\n".join(text))

if __name__=="__main__":
    main()
