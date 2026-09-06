"""Delayed linear prediction and constrained finite-horizon decisions."""
from __future__ import annotations
import numpy as np
from integrated_polar.numerics import array

ITERATIONS=40
REGULARIZATION=.0002
CONTINUITY=.0002


def project_rows(x,costs,budgets,allowed,lower=None):
    """Euclidean capped-simplex projection, independently for each horizon row.

    Piecewise-linear breakpoints avoid unconstrained inverse commands. Priorities
    belong in the planning objective, not in hidden evaluation-only weights.
    """
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite plan matrix required')
    H,m=x.shape;c=array(costs,(m,),'costs',1e-12);bud=array(budgets,(H,),'budgets',0)
    mask=np.asarray(allowed)
    if mask.dtype!=np.bool_ or mask.shape not in ((m,),(H,m)):raise ValueError('permission shape/type')
    hi=np.broadcast_to(mask,(H,m)).astype(float)
    lo=np.zeros((H,m)) if lower is None else np.broadcast_to(np.asarray(lower,float),(H,m)).copy()
    if not np.isfinite(lo).all() or np.any(lo<0) or np.any(lo>hi) or np.any(lo@c>bud+1e-12):raise ValueError('infeasible obligation')
    initial=np.clip(x,lo,hi)
    points=np.sort(np.maximum(0.,np.concatenate((np.zeros((H,1)),(x-hi)/c,(x-lo)/c),axis=1)),axis=1)
    levels=np.clip(x[:,None,:]-points[:,:,None]*c,lo[:,None,:],hi[:,None,:])@c
    crosses=levels<=bud[:,None]+1e-14;k=np.argmax(crosses,axis=1)
    if not crosses.any(axis=1).all():raise ValueError('projection has no feasible breakpoint')
    rows=np.arange(H);left=np.maximum(k-1,0)
    l=points[rows,left];r=points[rows,k];fl=levels[rows,left];fr=levels[rows,k]
    fraction=np.divide(fl-bud,fl-fr,out=np.zeros(H),where=(fl-fr)>1e-15)
    lam=l+(r-l)*np.clip(fraction,0,1)
    projected=np.clip(x-np.nextafter(lam,np.inf)[:,None]*c,lo,hi)
    return np.where((initial@c<=bud)[:,None],initial,projected)


def operators(F,B,actuator,drift,state,held):
    """x'=F*x+d+B*h'; h'=a*h+(1-a)*u. Return stacked x Jacobian."""
    B=np.asarray(B,float);n,m=B.shape;F=array(F,(n,n),'F');drift=np.asarray(drift,float)
    H=drift.shape[0]
    if drift.shape!=(H,n) or not np.isfinite(drift).all() or not 0<=actuator<1:raise ValueError('dynamics contract')
    x=array(state,(n,),'belief');h=array(held,(m,),'actuator state')
    sx=np.zeros((n,H*m));sh=np.zeros((m,H*m));A=[];offset=[]
    for t in range(H):
        h=actuator*h;sh=actuator*sh
        sh[:,t*m:(t+1)*m]+=(1-actuator)*np.eye(m)
        x=F@x+drift[t]+B@h;sx=F@sx+B@sh
        A.append(sx.copy());offset.append(x.copy())
    return np.vstack(A),np.asarray(offset).ravel()


def plan(F,B,actuator,drift,state,held,targets,priorities,internal,costs,budgets,allowed,
         *,lower=None,generic=False,iterations=ITERATIONS):
    B=np.asarray(B,float);n,m=B.shape;H=len(targets)
    r=array(targets,(H,n),'targets');w=array(priorities,(H,n),'priorities',1e-12)
    p=array(internal,(m,),'polar continuity',0,1)
    if iterations<1:raise ValueError('iterations must be positive')
    A,b=operators(F,B,actuator,drift,state,held);q=w.ravel()/w.sum()
    if generic:
        Q=np.diag(q);R=A.T@Q@A;linear=A.T@Q@(b-r.ravel())
    else:
        R=(A.T*q)@A;linear=A.T@(q*(b-r.ravel()))
    R+=(REGULARIZATION+CONTINUITY)*np.eye(H*m);linear-=CONTINUITY*np.tile(p,H)
    L=max(float(np.linalg.eigvalsh(R)[-1]),1e-12)
    u=project_rows(np.tile(p,(H,1)),costs,budgets,allowed,lower)
    accelerated=u.copy();momentum=1.
    for _ in range(iterations):
        proposal=(accelerated.ravel()-(R@accelerated.ravel()+linear)/L).reshape(H,m)
        nxt=project_rows(proposal,costs,budgets,allowed,lower)
        mt=(1+np.sqrt(1+4*momentum*momentum))/2
        accelerated=nxt+(momentum-1)/mt*(nxt-u);momentum=mt;u=nxt
    pred=(A@u.ravel()+b).reshape(H,n);residual=r-pred
    pg=project_rows((u.ravel()-(R@u.ravel()+linear)/L).reshape(H,m),costs,budgets,allowed,lower)
    trace={'predicted':pred.tolist(),'signed_residual':residual.tolist(),
           'positive_deficit':np.maximum(residual,0).tolist(),'negative_deficit':np.maximum(-residual,0).tolist(),
           'coactivation':(u[:,0::2]*u[:,1::2]).tolist(),'plan':u.tolist(),
           'tracking_objective':float(np.sum(q*residual.ravel()**2)),
           'objective':float(np.sum(q*residual.ravel()**2)+REGULARIZATION*np.sum(u*u)+CONTINUITY*np.sum((u-p)**2)),
           'projected_step_residual':float(np.max(abs(pg-u))),'iterations':iterations,
           'receiver_sensitivities_first':(A.T*q)[:m].tolist()}
    return u,trace


def diagnostic_step(B,state,target,weights,previous,chi,*,signed=True,conflict=True,directed=True,coupled=True):
    """One synchronous intervention: each of four flags changes ONLY its named term.

    Estimator and state are fixed across arms; the coactivation derivative is
    retained when sign is removed. No retuning or hidden environmental truth.
    """
    B=np.asarray(B,float).copy();n,m=B.shape
    if m!=2*n:raise ValueError('two channels per operational polarity required')
    if not coupled:
        mask=np.zeros_like(B,bool)
        for i in range(n):mask[i,2*i:2*i+2]=True
        B=np.where(mask,B,0.)
    receiver=B.T.copy()
    if not directed:
        receiver=np.repeat(receiver.reshape(n,2,n).mean(axis=1),2,axis=0)
    e=np.asarray(target)-np.asarray(state);message=e if signed else np.abs(e)
    co=np.asarray(previous).reshape(n,2);chi=np.asarray(chi)
    conflict_gradient=(chi[:,None]*co[:,::-1]).ravel() if conflict else np.zeros(m)
    delta=receiver@(np.asarray(weights)*message)-.02*conflict_gradient
    return np.clip(np.asarray(previous)+2*delta,0,1),{
        'signed_error':e.tolist(),'transmitted_error':message.tolist(),
        'conflict_gradient':conflict_gradient.tolist(),'receiver':receiver.tolist(),'delta':delta.tolist()}
