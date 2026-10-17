"""Retrospective exact-identity paired CID-vs-EAD geometry reversal.

Replay frozen original CID scores and independently frozen EAD V2 scores;
no rescoring, outcome-conditioned matching, new model fitting, or rounding
of chemical identities. Both measurements originate in SAME MultiMS2 lab.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from qfd.stats import paired_block_bootstrap_delta,paired_block_signflip_test,paired_win_loss_tie

CID_ORIGINAL_ROWS_SHA="f97bfaee05e9197cdaa96ad995ab9019c995274820ff57f3b04bd04de12c2be6"
CID_RUN=38970610885
EAD_RUN=38985827065
ID=("inchikey","formula","adduct","ion_mode","source_dataset_id")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(t) for t in path.read_text().splitlines() if t.strip()]

def identity(r):
    return tuple(str(r[x]) for x in ID)

def unique_rows(rows,method):
    out={}
    for r in rows:
        k=identity(r)
        if k in out:
            raise RuntimeError("multiple distinct triplets for same full chemical/modal identity in "+method)
        out[k]=r
    return out

def report_paired(rows):
    if len(rows)<30 or len({r["formula"] for r in rows})<20:
        raise RuntimeError("insufficient exact-identity matched cross-fragmentation transfer")
    cid=np.asarray([r["cid_delta_quarter_hellinger"] for r in rows],float)
    ead=np.asarray([r["ead_delta_quarter_hellinger"] for r in rows],float)
    diff=cid-ead
    f=[r["formula"] for r in rows]
    bc_c=np.asarray([r["cid_endpoint_bc"] for r in rows],float)
    bc_e=np.asarray([r["ead_endpoint_bc"] for r in rows],float)
    result={
        "n_exact_identity_adduct_mode_dataset_pairs":len(rows),
        "n_full_structure_keys":len({r["inchikey"] for r in rows}),
        "n_formula_blocks":len(set(f)),
        "cid_quarter_minus_hellinger_mean":float(cid.mean()),
        "ead_quarter_minus_hellinger_mean":float(ead.mean()),
        "matched_within_compound_cid_minus_ead_delta":float(diff.mean()),
        "paired_wins_losses_ties_cid_over_ead":paired_win_loss_tie(cid,ead,atol=1e-12),
        "formula_block_bootstrap_95":paired_block_bootstrap_delta(
            cid,ead,f,n_boot=10000,seed=20261016),
        "formula_block_signflip_2sided":paired_block_signflip_test(
            cid,ead,f,n_permutations=50000,seed=20261016),
        "cid_endpoint_overlap_mean":float(bc_c.mean()),
        "ead_endpoint_overlap_mean":float(bc_e.mean()),
        "mean_within_compound_endpoint_overlap_difference_cid_minus_ead":float((bc_c-bc_e).mean()),
        "both_modalities_quarter_beats_hellinger":int(((cid>0)&(ead>0)).sum()),
        "cid_only_quarter_beats_hellinger":int(((cid>0)&(ead<=0)).sum()),
        "ead_only_quarter_beats_hellinger":int(((cid<=0)&(ead>0)).sum()),
        "neither_quarter_beats_hellinger":int(((cid<=0)&(ead<=0)).sum()),
    }
    for bound in (.05,.1,.2):
        sub=np.abs(bc_c-bc_e)<=bound
        result[f"endpoint_bc_matched_abs_diff_le_{bound}_DESCRIPTIVE_ONLY"]={
            "n":int(sub.sum()),
            "mean_cid_minus_ead_quarter_vs_hellinger":float(diff[sub].mean()) if sub.any() else None,
            "n_formula_blocks":len({f[i] for i,z in enumerate(sub) if z}),
        }
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cid-rows",type=Path,required=True)
    p.add_argument("--ead-rows",type=Path,required=True)
    p.add_argument("--cid-result",type=Path,required=True)
    p.add_argument("--ead-result",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if sha(args.cid_rows)!=CID_ORIGINAL_ROWS_SHA:
        raise RuntimeError("original frozen CID scores changed")
    cid_result=json.loads(args.cid_result.read_text())
    ead_result=json.loads(args.ead_result.read_text())
    if (cid_result.get("status")!="MULTIMS2_THIRD_SOURCE_FROZEN_20_40_60EV_CID_EFFICACY" or
        ead_result.get("status")!="FROZEN_EAD_POSITIVE_ONLY_V2_MODALITY_TRANSFER" or
        ead_result.get("original_v1_gate_failed") is not True or
        ead_result.get("all_positive",{}).get("n_queries")!=694 or
        cid_result.get("all",{}).get("n_queries")!=1781):
        raise RuntimeError("unrecognized score cohort or unreported V1 gate failure")
    cid_rows=read_rows(args.cid_rows)
    ead_rows=read_rows(args.ead_rows)
    if len(cid_rows)!=1781 or len(ead_rows)!=694:
        raise RuntimeError("incomplete original cohort")
    a=unique_rows(cid_rows,"CID")
    b=unique_rows(ead_rows,"EAD")
    common=sorted(set(a)&set(b))
    paired=[]
    for k in common:
        cr,er=a[k],b[k]
        if er["ion_mode"]!="POSITIVE" or cr["ion_mode"]!="POSITIVE":
            raise RuntimeError("incompatible source polarity in matched set")
        x=float(cr["quarter"])-float(cr["hellinger"])
        y=float(er["quarter"])-float(er["hellinger"])
        cb=float(cr["endpoint_bhattacharyya"])
        eb=float(er["endpoint_bc"])
        if not all(np.isfinite(z) for z in (x,y,cb,eb)):
            raise ValueError("non-finite data in pinned prediction score")
        paired.append({
            **dict(zip(ID,k)),
            "cid_query_index":cr["query_index"],
            "ead_query_index":er["query_index"],
            "cid_endpoint_bc":cb,
            "ead_endpoint_bc":eb,
            "cid_delta_quarter_hellinger":x,
            "ead_delta_quarter_hellinger":y,
            "difference_cid_minus_ead":x-y,
        })
    stats=report_paired(paired)
    result={
        "status":"POSTHOC_IDENTITY_MATCHED_CID_VS_EAD_GEOMETRY_REVERSAL",
        "original_CID_run":CID_RUN,"original_EAD_V2_run":EAD_RUN,
        "cid_rows_sha256":sha(args.cid_rows),
        "ead_rows_sha256":sha(args.ead_rows),
        "n_total_CID":len(cid_rows),
        "n_total_EAD_positive_only":len(ead_rows),
        "exact_match_key":ID,
        "no_new_spectrum_peaks_or_outcome_scoring":True,
        "no_fit_or_tuning":True,
        "matched_results":stats,
        "identifiability_limit":"CID 20/40/60 eV at t=0.5 vs EAD 12/16/24 eV at t=1/3; same source/instrument, shared compounds. Geometry differences CANNOT be attributed causally to modality alone because energy positions, activation channels, endpoints and protocol differ.",
        "inferential_status":"RETROSPECTIVE (post-outcome identity match), formula-block inference within one laboratory, not independent lab replication.",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.out/"pairs.jsonl").open("w") as out:
        for row in paired:out.write(json.dumps(row,sort_keys=True)+"\n")
    ci=stats["formula_block_bootstrap_95"]
    msg=["# Retrospective exact-structure and adduct matched CID vs EAD geometry",
         "",f"Exact full-key/formula/adduct/polarity/dataset pairs: **{stats['n_exact_identity_adduct_mode_dataset_pairs']}**, "
         f"molecular formulas **{stats['n_formula_blocks']}**.",
         "",f"CID (quarter−Hellinger): **{stats['cid_quarter_minus_hellinger_mean']:+.6f}**.",
         f"EAD (quarter−Hellinger): **{stats['ead_quarter_minus_hellinger_mean']:+.6f}**.",
         f"Within-exact-identity CID−EAD geometry advantage: **{stats['matched_within_compound_cid_minus_ead_delta']:+.6f}**.",
         f"Formula-cluster 95% CI [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]; "
         f"two-sided signflip p={stats['formula_block_signflip_2sided']['p_value_two_sided']:.6f}.",
         f"CID/EAD mean endpoint BC: {stats['cid_endpoint_overlap_mean']:.6f} / {stats['ead_endpoint_overlap_mean']:.6f}.",
         "",str(stats["paired_wins_losses_ties_cid_over_ead"]),
         "",result["identifiability_limit"],
         "",result["inferential_status"],""]
    (args.out/"report.md").write_text("\n".join(msg))
    print("\n".join(msg))
    return result

if __name__=="__main__":
    main()
