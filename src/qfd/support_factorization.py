"""Exact shared/exclusive support factorization of power-mean interpolation.

A falsification instrument, not a new empirical fit. For each peak, the
normalized power path separates its unnormalized shared-bin curvature from
exact endpoint-exclusive survival coefficients. With exclusive exponent
gamma=1/alpha and shared exponent alpha, this is IDENTICALLY P_alpha.

Do not interpret score differences after normalization as causal effects.
"""
from __future__ import annotations

import math
import numpy as np


def _validated(a, b, t, shared_alpha, exclusive_power):
    x=np.asarray(a,dtype=float)
    y=np.asarray(b,dtype=float)
    if (x.ndim!=1 or not len(x) or x.shape!=y.shape
        or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
        or np.any(x<0) or np.any(y<0) or x.sum()<=0 or y.sum()<=0
        or not math.isfinite(float(t)) or not 0<=float(t)<=1
        or not math.isfinite(float(shared_alpha)) or shared_alpha<=0
        or not math.isfinite(float(exclusive_power)) or exclusive_power<=0):
        raise ValueError("finite normalized nonnegative reference distributions and positive powers required")
    return x/float(x.sum()),y/float(y.sum()),float(t)


def unnormalized_components(a, b, t, *, shared_alpha:float, exclusive_power:float):
    """Return exact nonnegative disjoint components (shared, left, right).

    Shared support has power-mean curvature alpha; exclusive support has
    survival gamma. Normalization occurs ONLY after their combination.
    In particular gamma=1/alpha gives the unnormalized power interpolant,
    while (alpha,gamma)=(1,4) is the established support-only control.
    """
    x,y,t=_validated(a,b,t,shared_alpha,exclusive_power)
    shared=(x>0)&(y>0)
    left=(x>0)&(y==0)
    right=(x==0)&(y>0)
    zshared=np.zeros_like(x)
    zshared[shared]=((1-t)*np.power(x[shared],shared_alpha)
                     +t*np.power(y[shared],shared_alpha))**(1/shared_alpha)
    zleft=np.where(left,(1-t)**exclusive_power*x,0.)
    zright=np.where(right,t**exclusive_power*y,0.)
    return zshared,zleft,zright


def factorized_distribution(a,b,t,*,shared_alpha:float,exclusive_power:float):
    s,l,r=unnormalized_components(a,b,t,shared_alpha=shared_alpha,
                                  exclusive_power=exclusive_power)
    z=s+l+r
    if not np.all(np.isfinite(z)) or float(z.sum())<=0:
        raise ValueError("degenerate or overflowing factored path")
    return z/float(z.sum())


def fixed_mechanism_predictions(a,b,t):
    """No tuning: three shared-bin laws, SAME fourth-power exclusive bins.

    quarter: alpha=1/4 on shared; Hellinger-hybrid: alpha=1/2 on
    shared; support-only: alpha=1 (ordinary linear shared).
    Thus all three predictions have EXACT identical *unnormalized*
    exclusive terms, unlike comparisons to fixed Hellinger alone.
    """
    return {
        "quarter_factored":factorized_distribution(
            a,b,t,shared_alpha=.25,exclusive_power=4.),
        "shared_hellinger_exclusive_fourth":factorized_distribution(
            a,b,t,shared_alpha=.5,exclusive_power=4.),
        "shared_linear_exclusive_fourth":factorized_distribution(
            a,b,t,shared_alpha=1.,exclusive_power=4.),
    }


def reference_support_diagnostics(a,b):
    x,y,_=_validated(a,b,.5,1.,4.)
    shared=(x>0)&(y>0)
    return {
        "shared_bin_count":int(shared.sum()),
        "left_only_bin_count":int(((x>0)&(y==0)).sum()),
        "right_only_bin_count":int(((x==0)&(y>0)).sum()),
        "shared_endpoint_mass_low":float(x[shared].sum()),
        "shared_endpoint_mass_high":float(y[shared].sum()),
        "reference_endpoint_bc":float(np.sqrt(x*y).sum()),
    }
