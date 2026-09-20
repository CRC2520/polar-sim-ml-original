"""Numerical helpers shared by the five R10 experiments."""
from __future__ import annotations
import hashlib, json, math
import numpy as np

def sigmoid(x):
    x=np.asarray(x,float)
    return 1.0/(1.0+np.exp(-np.clip(x,-40,40)))

def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    return np.log(p/(1-p))

def ridge_fit(x,y,lam=1e-3,intercept=True):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if y.ndim==1: y=y[:,None]
    if intercept:
        a=np.c_[np.ones(len(x)),x]
        reg=np.eye(a.shape[1])*lam; reg[0,0]=0
    else:
        a=x; reg=np.eye(a.shape[1])*lam
    w=np.linalg.solve(a.T@a+reg,a.T@y)
    return w

def ridge_predict(x,w,intercept=True):
    x=np.asarray(x,float)
    a=np.c_[np.ones(len(x)),x] if intercept else x
    y=a@w
    return y[:,0] if y.shape[1]==1 else y

def pca_fit(x,d):
    x=np.asarray(x,float); mean=x.mean(0)
    _,_,vt=np.linalg.svd(x-mean,full_matrices=False)
    basis=vt[:d].T
    return mean,basis

def pca_project(x,mean,basis):
    x=np.asarray(x,float)
    return mean+(x-mean)@basis@basis.T

def balanced_accuracy(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    vals=np.unique(y_true)
    scores=[]
    for v in vals:
        m=y_true==v
        if m.any(): scores.append(float(np.mean(y_pred[m]==v)))
    return float(np.mean(scores)) if scores else float("nan")

def brier(y,p):
    return float(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))

def bootstrap_median(values,seed=12345,n=4000):
    v=np.asarray(values,float); rng=np.random.default_rng(seed)
    if len(v)==0: return [float("nan"),float("nan")]
    idx=rng.integers(0,len(v),(n,len(v)))
    med=np.median(v[idx],axis=1)
    return [float(np.quantile(med,.025)),float(np.quantile(med,.975))]

def bootstrap_mean(values,seed=12345,n=4000):
    v=np.asarray(values,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),(n,len(v)))
    m=v[idx].mean(1)
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def orthogonal(dim,seed):
    rng=np.random.default_rng(seed)
    q,r=np.linalg.qr(rng.normal(size=(dim,dim)))
    signs=np.sign(np.diag(r)); signs[signs==0]=1
    return q*signs

def fit_logistic(x,y,steps=500,lr=.08,l2=1e-3):
    x=np.asarray(x,float); y=np.asarray(y,float)
    a=np.c_[np.ones(len(x)),x]; w=np.zeros(a.shape[1])
    for _ in range(steps):
        p=sigmoid(a@w)
        g=a.T@(p-y)/len(y)
        g[1:]+=l2*w[1:]
        w-=lr*g
    return w

def logistic_predict(x,w):
    return sigmoid(np.c_[np.ones(len(x)),np.asarray(x,float)]@w)

def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def digest_record(obj):
    return hashlib.sha256(json.dumps(jsonable(obj),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
