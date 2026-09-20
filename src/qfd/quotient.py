from __future__ import annotations
import numpy as np

def feim(X):
    X=np.asarray(X,float)
    if np.any(X<=0): raise ValueError('FEIM requires strictly positive entries')
    L=np.log(X)
    return L-L.mean(1,keepdims=True)-L.mean(0,keepdims=True)+L.mean()

def observation_design(mask):
    mask=np.asarray(mask,bool); nf,ne=mask.shape; edges=np.argwhere(mask)
    A=np.zeros((len(edges), nf+max(ne-1,0)))
    for r,(f,e) in enumerate(edges):
        A[r,f]=1
        if e<ne-1: A[r,nf+e]=1
    return A,edges

def cycle_rank(mask):
    mask=np.asarray(mask,bool); nf,ne=mask.shape; edges=np.argwhere(mask)
    if len(edges)==0: return 0
    adj={i:[] for i in range(nf+ne)}; active=set()
    for f,e in edges:
        u,v=int(f),nf+int(e); adj[u].append(v); adj[v].append(u); active|={u,v}
    seen=set(); c=0
    for u in active:
        if u in seen: continue
        c+=1; seen.add(u); stack=[u]
        while stack:
            w=stack.pop()
            for z in adj[w]:
                if z not in seen: seen.add(z); stack.append(z)
    return int(len(edges)-len(active)+c)

def quotient_residual(observed_log,candidate_log,mask,sigma=1.0):
    A,edges=observation_design(mask)
    if len(edges)==0: return 0.0,np.empty(0)
    d=np.array([observed_log[f,e]-candidate_log[f,e] for f,e in edges],float)
    s=np.full_like(d,float(sigma)) if np.isscalar(sigma) else np.array([np.asarray(sigma)[f,e] for f,e in edges],float)
    WA=A/s[:,None]; Wd=d/s
    th,*_=np.linalg.lstsq(WA,Wd,rcond=None); r=Wd-WA@th
    return float(r@r),r

def two_energy_signature(X):
    X=np.asarray(X,float)
    if X.shape[1]!=2 or np.any(X<=0): raise ValueError('positive n x 2 required')
    r=np.log(X[:,1])-np.log(X[:,0]); return r-r.mean()
