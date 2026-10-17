"""Fixed RDKit graph cut connected-fragment masses for chemical isomer ranking.

Deliberately simple falsifiable chemical hypothesis, NOT physically complete
bond kinetics. Native GNPS candidate SMILES ONLY; never input query SMILES
or true chemical identity to generate fragment signatures.
"""
from __future__ import annotations
import bisect
import math
from functools import lru_cache

import numpy as np

H_ATOM=1.00782503223
PROTON=1.007276466621
MAX_HEAVY=160
MAX_BREAKS=24
MIN_COMPONENT_HEAVY=2
TOLERANCE=.01


def _fragment_hypotheses(smiles,formula):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    if not smiles or not formula:
        return [],"MISSING_NATIVE_STRUCTURE_OR_FORMULA",0
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        return [],"INVALID_NATIVE_SMILES",0
    if CalcMolFormula(mol)!=formula:
        return [],"NATIVE_SMILES_FORMULA_MISMATCH",0
    n=mol.GetNumAtoms()
    if not 4<=n<=MAX_HEAVY:
        return [],"UNSUPPORTED_HEAVY_ATOM_COUNT",0
    if len(Chem.GetMolFrags(mol))!=1:
        return [],"DISCONNECTED_NATIVE_SMILES",0
    # All atoms and fixed native hydrogen counts are from the candidate
    # structure itself, never matched to query ions or outcomes.
    table=Chem.GetPeriodicTable()
    mass=[float(table.GetMostCommonIsotopeMass(a.GetAtomicNum()))+
          float(a.GetTotalNumHs())*H_ATOM for a in mol.GetAtoms()]
    full=sum(mass)
    adjacent=[[] for _ in range(n)]
    for bond in mol.GetBonds():
        a=bond.GetBeginAtomIdx();z=bond.GetEndAtomIdx()
        adjacent[a].append((z,bond.GetIdx()))
        adjacent[z].append((a,bond.GetIdx()))
    bridges=[]
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType()!=Chem.BondType.SINGLE:
            continue
        skip=bond.GetIdx()
        visited=1  # Rooted at atom 0, so cut components are nested/disjoint.
        stack=[0]
        while stack:
            node=stack.pop()
            for nbr,edge in adjacent[node]:
                if edge==skip or (visited>>nbr)&1:continue
                visited|=1<<nbr
                stack.append(nbr)
        other=((1<<n)-1)^visited
        count=other.bit_count()
        if not MIN_COMPONENT_HEAVY<=count<=n-MIN_COMPONENT_HEAVY:
            continue
        a=bond.GetBeginAtom()
        z=bond.GetEndAtom()
        hetero_first=(0 if a.GetAtomicNum() not in (1,6) or
                      z.GetAtomicNum() not in (1,6) else 1)
        bridges.append((hetero_first,bond.GetIdx(),other))
    bridges.sort()
    bridges=bridges[:MAX_BREAKS]
    if not bridges:
        return [],"NO_ELIGIBLE_NONRING_BREAKS",0
    def maskmass(bits):
        acc=0.
        while bits:
            bit=bits & -bits
            acc+=mass[bit.bit_length()-1]
            bits-=bit
        return acc
    masses=[(mask,maskmass(mask)) for _,_,mask in bridges]
    neutral=[]
    for mask,v in masses:
        neutral.append(v)
        neutral.append(full-v)
    for i in range(len(masses)):
        am,ma=masses[i]
        for j in range(i+1,len(masses)):
            bm,mb=masses[j]
            if am&bm==0:
                options=(ma,mb,full-ma-mb)
            elif am&bm==am:
                options=(ma,mb-ma,full-mb)
            elif am&bm==bm:
                options=(mb,ma-mb,full-ma)
            else:
                # Two sides of bridge cuts in a connected graph form a
                # rooted tree. Overlap with neither nesting nor
                # disjointness would violate the graph invariant.
                raise RuntimeError("invalid nested graph bridge relation")
            neutral.extend(v for v in options if v>0)
    shifts=(-H_ATOM,0.,H_ATOM,PROTON)
    values=set()
    # No isotope/ion calibration fitted; small fixed ion mass hypotheses.
    for v in neutral:
        if v<=0 or v>=full:continue
        for delta in shifts:
            mz=v+delta
            if mz>0:
                values.add(round(mz,6))
    return sorted(values),"OK",len(bridges)


@lru_cache(maxsize=60000)
def signatures(smiles:str,formula:str):
    """Return immutable sorted m/z signatures, reason and bounded cut count."""
    masses,status,bonds=_fragment_hypotheses(smiles,formula)
    return tuple(masses),status,bonds


def candidate_match_vector(query_mz, candidate_masses, tolerance_da=TOLERANCE):
    """Label-free Boolean coverage of all query fragments by graph-cut m/z."""
    masses=np.asarray(candidate_masses,float)
    q=np.asarray(query_mz,float)
    if (q.ndim!=1 or not np.all(np.isfinite(q)) or np.any(q<=0)
        or not math.isfinite(tolerance_da) or tolerance_da<=0):
        raise ValueError("finite positive m/z and fixed tolerance needed")
    if not len(masses):
        return np.zeros(len(q),dtype=bool)
    if not np.all(np.isfinite(masses)) or np.any(masses<=0) or np.any(masses[1:]<masses[:-1]):
        raise ValueError("sorted finite candidate theoretical fragments required")
    at=np.searchsorted(masses,q)
    after=masses[np.minimum(at,len(masses)-1)]
    before=masses[np.maximum(at-1,0)]
    return (np.abs(after-q)<=tolerance_da)|(np.abs(before-q)<=tolerance_da)


def discriminative_graph_scores(query, candidates:dict[str,tuple],tol=TOLERANCE):
    """Candidate rarity 1-df/K prevents universal formula peaks affecting rank.

    No test label is read; the full query spectrum is used for matching, as
    in standard library search. The full candidate pool is defined upstream.
    """
    q,qi=query
    qi=np.asarray(qi,float)
    if len(candidates)<2 or not len(q) or len(q)!=len(qi) or qi.sum()<=0:
        raise ValueError("nontrivial fixed candidate pool and measured query needed")
    p=qi/float(qi.sum())
    features={k:candidate_match_vector(q,values,tol) for k,values in candidates.items()}
    df=np.sum(np.stack(list(features.values()),axis=0),axis=0)
    k=len(candidates)
    rarity=1.-df/k
    scores={key:float(np.sum(p*rarity*matched))
            for key,matched in features.items()}
    if any(not math.isfinite(v) or v<0 or v>1+1e-10 for v in scores.values()):
        raise RuntimeError("candidate graph-score invalid")
    return scores,{"n_competing_structures":k,"n_query_peaks":len(p),
                   "n_query_peaks_with_any_structure_explanation":int(np.count_nonzero(df)),
                   "fraction_of_query_intensity_explained_by_any":float(np.sum(p*(df>0)))}
