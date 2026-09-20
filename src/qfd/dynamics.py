from __future__ import annotations
import numpy as np
from scipy.linalg import expm, logm

def trajectory(Q, energies, x0):
    return np.stack([expm(float(E)*Q) @ x0 for E in energies], axis=1)

def l1_normalize_columns(X, eps=1e-15):
    d=np.sum(X,axis=0,keepdims=True)
    return X/np.maximum(d,eps)

def reconstruct_propagator(states):
    n=states.shape[0]
    if states.shape[1] < n+1: raise ValueError('need n+1 states')
    K=states[:,:n]; Y=states[:,1:n+1]
    return Y @ np.linalg.inv(K)

def reconstruct_generator(states, delta_E):
    return np.real_if_close(logm(reconstruct_propagator(states))/delta_E)
