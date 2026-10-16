"""Post-hoc r=.25 versus answer-aware optimal Hellinger chord position."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.power_geometry import power_mean_distribution
from qfd.target_blind_geometry import (
    reference_only_grid, query_mass_on_reference_grid, affinity,
    best_hellinger_chord_oracle,
)
from qfd.stats import (
    paired_block_bootstrap_delta, paired_block_signflip_test,
    paired_win_loss_tie,
)


def _sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _compare(rows,a,b,seed):
    x=np.asarray([r[a] for r in rows],dtype=float)
    y=np.asarray([r[b] for r in rows],dtype=float)
    labels=[r["formula"] for r in rows]
    return {
        "delta":float(np.mean(x-y)),
        "n_queries":len(rows), "n_formula_blocks":len(set(labels)),
        "win_loss_tie":paired_win_loss_tie(x,y,atol=1e-15),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            x,y,labels,n_boot=3000,seed=seed
        ),
        "formula_block_signflip_two_sided":paired_block_signflip_test(
            x,y,labels,n_permutations=50000,seed=seed
        ),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("jsonl",type=Path)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=20261016)
    args=ap.parse_args()
    groups=_unambiguous_groups(index_records(load_msnlib_jsonl(args.jsonl)))
    rows=[]
    for skey,g in sorted(groups.items()):
        refs=sorted(float(e) for e in g["same_energy"])
        if len(refs)<2:
            continue
        for eq in sorted(float(e) for e in g["single_best"]):
            before=[e for e in refs if e<eq]
            after=[e for e in refs if e>eq]
            if not before or not after:
                continue
            e0,e1=max(before),min(after)
            t=(eq-e0)/(e1-e0)
            query=_clean_view(g["single_best"][eq],transform="none")
            low=_clean_view(g["same_energy"][e0],transform="none")
            high=_clean_view(g["same_energy"][e1],transform="none")
            grid=reference_only_grid(low,high,tolerance_da=.01)
            q,unmatched=query_mass_on_reference_grid(query,grid,tolerance_da=.01)
            quarter=affinity(q,power_mean_distribution(grid.low,grid.high,t,.25))
            chord=affinity(q,power_mean_distribution(grid.low,grid.high,t,.5))
            oracle=best_hellinger_chord_oracle(q,grid.low,grid.high)
            if oracle["affinity"]+1e-9<chord:
                raise AssertionError("Hellinger oracle failed to dominate fixed chord")
            row={
                "formula":str(g["formula"]),
                "structure_adduct":f"{skey[0]}|{skey[1]}",
                "query_eV":eq,"low_eV":e0,"high_eV":e1,"t":t,
                "alpha_quarter":quarter,
                "hellinger_fixed":chord,
                "hellinger_oracle":oracle["affinity"],
                "oracle_t":oracle["oracle_weight"],
                "query_unmatched_mass":unmatched,
            }
            if not np.isfinite(np.asarray(list(row.values())[6:],dtype=float)).all():
                raise ValueError("non-finite spectral score")
            rows.append(row)
    if not rows:
        raise RuntimeError("no eligible query triplets")
    x={
        "status":"POSTHOC_HELLINGER_SCALAR_ORACLE_FALSIFICATION",
        "dataset":args.dataset,"posthoc":True,
        "input_sha256":_sha(args.jsonl),
        "coverage":{"queries":len(rows),"formula_blocks":len(set(r["formula"] for r in rows))},
        "methods":{
            k:{"mean":float(np.mean([r[k] for r in rows])),
               "median":float(np.median([r[k] for r in rows]))}
            for k in ("alpha_quarter","hellinger_fixed","hellinger_oracle","oracle_t","query_unmatched_mass")
        },
        "comparisons":{
            "quarter_minus_answer_aware_hellinger_oracle":_compare(rows,"alpha_quarter","hellinger_oracle",args.seed),
            "quarter_minus_fixed_hellinger":_compare(rows,"alpha_quarter","hellinger_fixed",args.seed+1),
            "oracle_minus_fixed_hellinger":_compare(rows,"hellinger_oracle","hellinger_fixed",args.seed+2),
        },
        "guardrail":"oracle uses withheld answer to optimize scalar chord position; nondeployable diagnostic, post-hoc",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(x,indent=2)+"\n")
    with (args.out/"rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r,sort_keys=True)+"\n")
    print(json.dumps(x,indent=2))


if __name__=="__main__":
    main()
