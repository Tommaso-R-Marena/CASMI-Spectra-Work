"""Reference-only certificates for support geometry and retrieval stability.

Mathematical statements are for finite discrete spectral distributions. They
are mathematical consequences of normalization, Cauchy-Schwarz, and total
variation, NOT a proof of physical fragmentation dynamics or CASMI efficacy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np

from .support_factorization import unnormalized_components


def disjoint_support_energy_reparameterization(t:float,from_alpha:float,to_alpha:float)->float:
    """Exact disjoint-support conjugacy: P_from(t)=P_to(s).

    s = sigmoid((to_alpha/from_alpha)*logit(t)); s(0)=0 and s(1)=1.
    Thus no distinct positive power exponent is identifiable purely
    from a disjoint-support spectral path with unconstrained energy warp.
    """
    v=float(t)
    if (not math.isfinite(v) or not 0<=v<=1
        or not math.isfinite(from_alpha) or from_alpha<=0
        or not math.isfinite(to_alpha) or to_alpha<=0):
        raise ValueError("finite t in [0,1] and positive finite exponents required")
    if v in (0.,1.):
        return v
    logit=(to_alpha/from_alpha)*math.log(v/(1-v))
    if logit>=0:
        z=math.exp(-logit)
        return 1/(1+z)
    z=math.exp(logit)
    return z/(1+z)


def score_perturbation_certificate(a,b,t,*,from_shared_alpha:float,to_shared_alpha:float,exclusive_power:float=4.)->dict:
    """Bound ANY normalized/subnormalized query's BC affinity change.

    With U=S1+E and V=S2+E, d=||S1-S2||_1, masses A=sum U,
    B=sum V, normalization implies TV(U/A,V/B) <= d/max(A,B).
    Cauchy-Schwarz then implies |BC(q,U/A)-BC(q,V/B)| <=
    min(1, sqrt(2*d/max(A,B))) for any q with sum(q)<=1.

    A tighter reference-only bound is ||sqrt(U/A)-sqrt(V/B)||_2,
    also via Cauchy-Schwarz. Never treat these as causal attributions.
    """
    s1,l1,r1=unnormalized_components(
        a,b,t,shared_alpha=from_shared_alpha,exclusive_power=exclusive_power)
    s2,l2,r2=unnormalized_components(
        a,b,t,shared_alpha=to_shared_alpha,exclusive_power=exclusive_power)
    if not np.array_equal(l1,l2) or not np.array_equal(r1,r2):
        raise RuntimeError("certification requires literally identical exclusive terms")
    raw1=s1+l1+r1
    raw2=s2+l2+r2
    z1=float(raw1.sum());z2=float(raw2.sum())
    if z1<=0 or z2<=0:
        raise ValueError("nonpositive path normalization")
    p=raw1/z1;q=raw2/z2
    d=float(np.abs(s1-s2).sum())
    tv=float(np.abs(p-q).sum()/2)
    tv_bound=d/max(z1,z2)
    tight=float(np.linalg.norm(np.sqrt(p)-np.sqrt(q)))
    loose=math.sqrt(2*tv_bound)
    if tv>tv_bound+1e-10 or tight>loose+1e-10:
        raise RuntimeError("algebraic TV or Hellinger certificate violated")
    return {
        "shared_unnormalized_l1_change":d,
        "tv_actual":tv,
        "tv_proved_upper":min(1.,tv_bound),
        "affinity_uniform_upper":min(1.,tight,loose),
        "affinity_tv_upper":min(1.,loose),
        "shared_reference_bin_count":int(np.sum((np.asarray(a)>0)&(np.asarray(b)>0))),
    }


def certify_top1_unchanged(baseline_scores:Mapping[str,float],
                           prediction_change_bounds:Mapping[str,float]) ->dict:
    """Sufficient (not necessary) condition for SAME unique top-1 candidate.

    For each candidate, its score can change by at most its bound for EVERY
    possible held-out query. A unique baseline winner w remains winner if
    baseline[w]-baseline[j] > eps[w]+eps[j] for all other j.
    A false result is only 'not certified'; it is NOT evidence of change.
    """
    if set(baseline_scores)!=set(prediction_change_bounds) or len(baseline_scores)<2:
        raise ValueError("same candidate inventory and >=2 candidates required")
    keys=sorted(baseline_scores)
    for k in keys:
        v=float(baseline_scores[k]);e=float(prediction_change_bounds[k])
        if not (math.isfinite(v) and math.isfinite(e) and 0<=v<=1+1e-9
                and 0<=e<=1+1e-9):
            raise ValueError("finite bounded reference scores and certificate required")
    winner=max(keys,key=lambda k:(baseline_scores[k],k))
    margin=min(float(baseline_scores[winner]-baseline_scores[j])
               for j in keys if j!=winner)
    max_required=max(float(prediction_change_bounds[winner]+prediction_change_bounds[j])
                     for j in keys if j!=winner)
    safe=all(float(baseline_scores[winner])-float(baseline_scores[j])>
             float(prediction_change_bounds[winner]+prediction_change_bounds[j])+1e-12
             for j in keys if j!=winner)
    return {"baseline_winner":winner, "certified_unchanged":bool(safe),
            "smallest_score_margin":margin,
            "largest_pairwise_required_margin":max_required,
            "status":"PROVED_STABLE" if safe else "UNKNOWN_NOT_PROVED"}



def distribution_affinity_difference_upper(p,q)->float:
    """Uniformly bound |BC(target,p)-BC(target,q)| for ANY target.

    For normalized distributions p and q and any query subdistribution
    (including unmatched measured intensity), Cauchy-Schwarz gives
    |<sqrt(target),sqrt(p)-sqrt(q)>| <= ||sqrt(p)-sqrt(q)||_2.
    No target is required to compute the bound.
    """
    x=np.asarray(p,float);y=np.asarray(q,float)
    if (x.ndim!=1 or not len(x) or x.shape!=y.shape
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or np.any(x<0) or np.any(y<0)
        or not np.isclose(x.sum(),1,atol=1e-9)
        or not np.isclose(y.sum(),1,atol=1e-9)):
        raise ValueError("two finite normalized prediction distributions required")
    return float(min(1.,np.linalg.norm(np.sqrt(x)-np.sqrt(y))))
