"""Closed-form global oracle for a normalized square-root (Hellinger) chord.

Independent diagnostic: the previously frozen numerical oracle is unchanged.
"""
from __future__ import annotations
import math
import numpy as np


def analytic_hellinger_chord_oracle(q, p0, p1) -> dict[str, float]:
    """Compute the global max of <sqrt(q), ((1-w)a+w*b)/||...||>, 0<=w<=1.

    a=sqrt(p0/sum(p0)), b=sqrt(p1/sum(p1)), u=sqrt(q).
    Let A=<u,a>, B=<u,b>, c=<a,b>. In odds k=w/(1-w):
    F(k)=(A+k*B)/sqrt(1+2*c*k+k*k), with derivative sign
    (B-A*c)+k*(B*c-A). The only interior maximum, if any,
    has k=(B-A*c)/(A-B*c). Otherwise the global max is an endpoint.

    q is allowed to have mass <1: unmatched query peaks remain excluded
    from matching, but not from original intensity normalization.
    """
    x, y, z = (np.asarray(v, dtype=float) for v in (q, p0, p1))
    if x.ndim != 1 or x.size == 0 or x.shape != y.shape or y.shape != z.shape:
        raise ValueError("q and both endpoints must be nonempty equally shaped vectors")
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in (x, y, z)):
        raise ValueError("nonfinite or negative intensity")
    sy, sz = float(y.sum()), float(z.sum())
    if sy <= 0 or sz <= 0 or not math.isfinite(sy + sz):
        raise ValueError("nonpositive endpoint mass")
    a, b, u = np.sqrt(y / sy), np.sqrt(z / sz), np.sqrt(x)
    A, B = float(np.dot(u, a)), float(np.dot(u, b))
    c = float(np.clip(np.dot(a, b), 0.0, 1.0))
    candidates = [(A, 0.0), (B, 1.0)]
    n0, n1 = A-B*c, B-A*c
    if n0 > 0.0 and n1 > 0.0:
        w = n1 / (n0+n1)
        v = (1.0-w)*a+w*b
        candidates.append((float(np.dot(u, v) / np.linalg.norm(v)), float(w)))
    best = max(candidates, key=lambda item: item[0])
    return {"affinity": best[0], "oracle_weight": best[1]}
