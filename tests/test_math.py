import numpy as np
from qfd.dynamics import trajectory,l1_normalize_columns,reconstruct_propagator
from qfd.quotient import feim,cycle_rank,quotient_residual,two_energy_signature
from qfd.cone import cone_distance,convex_hull_distance

def test_projective_gauge():
    Q=np.array([[-1.,0,0],[.7,-.6,0],[.3,.4,-.2]]); x=np.array([1.,0,0]); E=np.array([.2,1.,2.])
    z1=l1_normalize_columns(trajectory(Q,E,x)); z2=l1_normalize_columns(trajectory(Q+.37*np.eye(3),E,x))
    assert np.max(np.abs(z1-z2))<1e-12

def test_krylov_recovery():
    A=np.array([[.8,.1,0],[0,.7,.2],[.1,0,.6]]); x=np.array([1.,.3,.2]); ss=[x]
    for _ in range(3): ss.append(A@ss[-1])
    assert np.max(np.abs(A-reconstruct_propagator(np.stack(ss,1))))<1e-10

def test_feim_invariant():
    r=np.random.default_rng(0); X=np.exp(r.normal(size=(7,4))); a=np.exp(r.normal(size=7)); b=np.exp(r.normal(size=4))
    assert np.max(np.abs(feim(X)-feim(a[:,None]*X*b[None,:])))<1e-12

def test_cycle_rank_complete(): assert cycle_rank(np.ones((5,3),bool))==8

def test_cycle_rank_tree_zero(): assert cycle_rank(np.array([[1,1,0],[0,1,1],[0,0,1]],bool))==0

def test_quotient_nuisance_zero():
    r=np.random.default_rng(2); mu=r.normal(size=(6,3)); y=mu+r.normal(size=(6,1))+r.normal(size=(1,3)); d,_=quotient_residual(y,mu,np.ones_like(mu,bool)); assert d<1e-20

def test_two_energy_invariant():
    r=np.random.default_rng(3); X=np.exp(r.normal(size=(8,2))); Y=np.exp(r.normal(size=8))[:,None]*X*np.exp(r.normal(size=2))[None,:]
    assert np.max(np.abs(two_energy_signature(X)-two_energy_signature(Y)))<1e-12

def test_cone_exact_mix():
    r=np.random.default_rng(4); C=np.abs(r.normal(size=(20,3))); w=np.array([.2,1.1,.4]); y=C@w; d,_=cone_distance(y,C); assert d<1e-9

def test_convex_exact_mix():
    r=np.random.default_rng(5); C=np.abs(r.normal(size=(10,3))); w=np.array([.2,.3,.5]); y=C@w; d,wh=convex_hull_distance(y,C); assert d<1e-5 and abs(wh.sum()-1)<1e-8

def test_qfd_detects_nonseparable():
    mu=np.zeros((4,3)); y=mu.copy(); y[0,0]=1; y[1,1]=-1; d,_=quotient_residual(y,mu,np.ones_like(mu,bool)); assert d>.1
