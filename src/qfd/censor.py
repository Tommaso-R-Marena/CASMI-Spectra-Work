from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

def censored_profile_nll(observed_log,observed_mask,candidate_log,threshold_log,sigma=0.2,maxiter=100):
    obs=np.asarray(observed_mask,bool); mu=np.asarray(candidate_log,float); Y=np.asarray(observed_log,float)
    nf,ne=mu.shape; T=np.broadcast_to(np.asarray(threshold_log,float),mu.shape)
    p=nf+max(ne-1,0)
    def nuisance(th):
        u=th[:nf,None]; v=np.r_[th[nf:],0.0][None,:] if ne>1 else np.zeros((1,1)); return u+v
    def obj(th):
        eta=mu+nuisance(th); z=(Y[obs]-eta[obs])/sigma
        val=.5*np.sum(z*z)+np.count_nonzero(obs)*np.log(sigma)
        cen=~obs
        if np.any(cen): val-=np.sum(log_ndtr((T[cen]-eta[cen])/sigma))
        return float(val)
    x0=np.zeros(p)
    if np.any(obs):
        rows=[]; rhs=[]
        for f,e in np.argwhere(obs):
            row=np.zeros(p); row[f]=1
            if e<ne-1: row[nf+e]=1
            rows.append(row); rhs.append(Y[f,e]-mu[f,e])
        x0,*_=np.linalg.lstsq(np.vstack(rows),np.asarray(rhs),rcond=None)
    return float(minimize(obj,x0,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-9}).fun)
