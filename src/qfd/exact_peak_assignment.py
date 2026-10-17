"""Exact maximum-weight bipartite assignment for sparse MS/MS peak matching.

Given fixed m/z tolerance, each admissible observed/query peak may match at
most one reference peak; unmatched peaks remain in the FULL original L1/L2
normalization denominators. Optimize a GLOBAL assignment independently for
each predeclared similarity functional, never against the identity label.

Greedy nearest-m/z matching can fail to maximize weighted similarity when
multiple peaks lie within a tolerance region. The Hungarian algorithm is
applied only to the connected components of this sparse bipartite graph.
No target labels or reference-identity truth enter peak alignment.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

METHODS=("cosine","hellinger","fragment_overlap","support_composite")


def _peaks(spec):
    mz,intensity=spec
    mz=np.asarray(mz,dtype=float)
    intensity=np.asarray(intensity,dtype=float)
    if (mz.ndim!=1 or intensity.ndim!=1 or not len(mz) or mz.shape!=intensity.shape
        or not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensity))
        or np.any(mz<=0) or np.any(intensity<=0)
        or np.any(mz[1:]<mz[:-1])):
        raise ValueError("nonempty sorted finite positive measured spectral peaks required")
    return mz,intensity


def admissible_edges(query_mz,reference_mz,tolerance_da:float):
    if not math.isfinite(tolerance_da) or tolerance_da<=0 or tolerance_da>=.5:
        raise ValueError("fixed positive tolerance below 0.5 Da required")
    ed=[]
    for i,m in enumerate(query_mz):
        lo=int(np.searchsorted(reference_mz,m-tolerance_da,side="left"))
        hi=int(np.searchsorted(reference_mz,m+tolerance_da,side="right"))
        for j in range(lo,hi):
            if abs(float(m-reference_mz[j]))<=tolerance_da:
                ed.append((i,j))
    return ed


def _components(nq,nr,edges):
    parent=list(range(nq+nr))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    for i,j in edges:
        a=find(i);b=find(nq+j)
        if a!=b:parent[b]=a
    groups=defaultdict(list)
    for i,j in edges:groups[find(i)].append((i,j))
    out=[]
    for value in groups.values():
        qs=sorted({i for i,j in value})
        rs=sorted({j for i,j in value})
        out.append((qs,rs,value))
    return out


def _one_method(qi,ri,comps,method):
    qsum=float(qi.sum());rsum=float(ri.sum())
    q2=float(np.linalg.norm(qi));r2=float(np.linalg.norm(ri))
    if not all(math.isfinite(z) and z>0 for z in (qsum,rsum,q2,r2)):
        raise ValueError("positive finite full-spectrum normalizers required")
    value=0.
    ambiguous_components=0
    for qs,rs,edges in comps:
        qidx={v:i for i,v in enumerate(qs)}
        ridx={v:i for i,v in enumerate(rs)}
        # A query row has dummy unmatched columns; forbidden real edges
        # have a negative score, dummy edges 0. All allowable weights >=0.
        n,m=len(qs),len(rs)
        matrix=np.full((n,m+n),-1.,dtype=float)
        matrix[:,m:]=0.
        for i,j in edges:
            q=float(qi[i]);s=float(ri[j])
            cosine=(q/q2)*(s/r2)
            hellinger=math.sqrt((q/qsum)*(s/rsum))
            overlap=min(q/qsum,s/rsum)
            weight={
                "cosine":cosine,
                "hellinger":hellinger,
                "fragment_overlap":overlap,
                "support_composite":.5*cosine+.5*overlap
            }[method]
            if not math.isfinite(weight) or weight<0:
                raise RuntimeError("nonfinite matching edge weight")
            matrix[qidx[i],ridx[j]]=weight
        assignment=linear_sum_assignment(-matrix)
        for a,b in zip(*assignment):
            if b<m:
                if matrix[a,b]<0:raise RuntimeError("forbidden tolerance edge selected")
                value+=float(matrix[a,b])
        if len(edges)>min(len(qs),len(rs)):ambiguous_components+=1
    if not math.isfinite(value) or value<0 or value>1+1e-8:
        raise RuntimeError("global assignment similarity invalid")
    return float(min(1.,max(0.,value))),ambiguous_components


def optimal_ion_spectra(query,reference,tolerance_da:float=.01):
    q,qi=_peaks(query);r,ri=_peaks(reference)
    ed=admissible_edges(q,r,tolerance_da)
    comps=_components(len(q),len(r),ed)
    values={}
    nambiguous=0
    for method in METHODS:
        values[method],count=_one_method(qi,ri,comps,method)
        nambiguous=max(nambiguous,count)
    return values,{
        "n_admissible_edges":len(ed),
        "n_connected_matching_components":len(comps),
        "n_ambiguous_components":nambiguous,
        "n_query_peaks":len(q),"n_reference_peaks":len(r),
    }
