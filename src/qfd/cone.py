from __future__ import annotations
import numpy as np
from scipy.optimize import nnls, minimize

def cone_distance(y,columns):
    w,_=nnls(np.asarray(columns,float),np.asarray(y,float)); r=y-columns@w
    return float(np.linalg.norm(r)),w

def convex_hull_distance(y,columns):
    y=np.asarray(y,float); C=np.asarray(columns,float); m=C.shape[1]
    fun=lambda w: 0.5*float((y-C@w)@(y-C@w))
    res=minimize(fun,np.full(m,1/m),bounds=[(0,None)]*m,constraints=({'type':'eq','fun':lambda w:np.sum(w)-1},),method='SLSQP')
    return float(np.linalg.norm(y-C@res.x)),res.x
