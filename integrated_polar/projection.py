"""Weighted capped-simplex projection via exact piecewise-linear breakpoints."""
import numpy as np


def project(x,costs,budget,allowed=None,weights=None,lower=None):
    x=np.asarray(x,float);c=np.asarray(costs,float);n=x.size
    w=np.ones(n) if weights is None else np.asarray(weights,float)
    lo=np.zeros(n) if lower is None else np.asarray(lower,float)
    mask=np.ones(n,bool) if allowed is None else np.asarray(allowed)
    if x.ndim!=1 or any(a.shape!=(n,) or not np.isfinite(a).all() for a in (x,c,w,lo)):
        raise ValueError('projection arrays must be finite equal vectors')
    if np.any(c<=0) or np.any(w<=0) or np.any((lo<0)|(lo>1)) or not np.isfinite(budget) or budget<0:
        raise ValueError('invalid projection parameters')
    if mask.dtype!=np.bool_ or mask.shape!=(n,):raise ValueError('invalid permission mask')
    if np.any(lo[~mask]>0) or c@lo>budget+1e-12:raise ValueError('infeasible mandatory action')
    hi=mask.astype(float);u=np.clip(x,lo,hi)
    if c@u<=budget:return u
    slope=c/w
    points=np.unique(np.maximum(0.,np.concatenate(([0.],(x-hi)/slope,(x-lo)/slope))))
    costs_at=np.clip(x[None,:]-points[:,None]*slope[None,:],lo,hi)@c
    candidates=np.flatnonzero(costs_at<=budget+1e-14)
    if not len(candidates):return lo.copy()
    k=int(candidates[0])
    if k==0:return u
    l,r=points[k-1:k+1];fl,fr=costs_at[k-1:k+1]
    lam=l+(r-l)*(fl-budget)/max(fl-fr,1e-300)
    return np.clip(x-np.nextafter(lam,np.inf)*slope,lo,hi)
