"""Parameter-free reference-only persistent-fragment / transient-support hybrid.

Designed and fixed before scoring EAD 16-eV queries. The shared-support
shape uses the normalized Hellinger chord; endpoint-exclusive fragments
use fourth-power survival coefficients. No query spectrum is accessed.
"""
from __future__ import annotations
import math
import numpy as np

def persistent_hellinger_survival(low, high, t: float):
    """Normalized hybrid with exact endpoints, reversal and disjoint-support laws.

    a,b must be normalized nonnegative endpoint distributions on an identical,
    endpoint-constructed reference grid. All values are *unnormalized* until
    final normalization. No arbitrary intensity floor is introduced.
    """
    a=np.asarray(low,dtype=float)
    b=np.asarray(high,dtype=float)
    if (a.ndim!=1 or a.shape!=b.shape or len(a)==0 or
        not np.isfinite(a).all() or not np.isfinite(b).all() or
        np.any(a<0) or np.any(b<0) or
        not math.isfinite(t) or not 0<=t<=1):
        raise ValueError("invalid reference distribution or energy coordinate")
    if a.sum()<=0 or b.sum()<=0:
        raise ValueError("both endpoint distributions must contain mass")
    a=a/float(a.sum()); b=b/float(b.sum())
    shared=(a>0)&(b>0)
    h=((1-t)*np.sqrt(a)+t*np.sqrt(b))**2
    z=np.where(shared, h, ((1-t)**4)*a+(t**4)*b)
    mass=float(z.sum())
    if mass<=0 or not math.isfinite(mass):
        raise ValueError("hybrid has degenerate predicted mass")
    return z/mass
