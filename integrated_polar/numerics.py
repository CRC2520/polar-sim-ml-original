"""Identification and finite-horizon control with conventional RLS/quadratic MPC."""
from __future__ import annotations
import numpy as np
from .projection import project


def array(value, shape, name, lower=None, upper=None):
    a=np.asarray(value,dtype=float)
    if a.shape!=tuple(shape) or not np.isfinite(a).all():raise ValueError(f'{name}: expected finite shape {shape}')
    if lower is not None and np.any(a<lower):raise ValueError(f'{name}: below range')
    if upper is not None and np.any(a>upper):raise ValueError(f'{name}: above range')
    return a.copy()


class CoupledRLS:
    """Learn directed action->output relations from own uncensored transitions."""
    def __init__(self,outputs,actions,*,initial=None,forgetting=.995,ridge=.001):
        if outputs<1 or actions<1 or not 0<forgetting<=1 or ridge<=0:raise ValueError('invalid RLS configuration')
        self.n,self.m=outputs,actions;self.forgetting=forgetting
        self.B=np.zeros((outputs,actions)) if initial is None else array(initial,(outputs,actions),'initial B')
        self.P=np.repeat((np.eye(actions)/ridge)[None],outputs,axis=0)
        self.variance=np.full(outputs,1e-4);self.count=np.zeros(outputs,int)
    def predict(self,action):return self.B@array(action,(self.m,),'action')
    def predictive_variance(self,action):
        u=array(action,(self.m,),'action')
        return self.variance*(1+np.einsum('i,nij,j->n',u,self.P,u))
    def update(self,action,response,valid=None):
        u=array(action,(self.m,),'action');z=array(response,(self.n,),'response')
        valid=np.ones(self.n,bool) if valid is None else np.asarray(valid)
        if valid.shape!=(self.n,) or valid.dtype!=np.bool_:raise ValueError('invalid transition mask')
        residual=z-self.B@u
        for i in np.flatnonzero(valid):
            Pu=self.P[i]@u;k=Pu/(self.forgetting+u@Pu)
            self.B[i]+=k*residual[i]
            P=(self.P[i]-np.outer(k,u@self.P[i]))/self.forgetting;self.P[i]=(P+P.T)*.5
            self.variance[i]=max(1e-10,.95*self.variance[i]+.05*residual[i]**2);self.count[i]+=1
        if not np.isfinite(self.B).all() or not np.isfinite(self.P).all():raise FloatingPointError('RLS overflow')
        return {'residual':residual.tolist(),'valid':valid.tolist(),'count':self.count.tolist()}
    def snapshot(self):return {'B':self.B.tolist(),'P':self.P.tolist(),'variance':self.variance.tolist(),'count':self.count.tolist()}


def own_pair_mask(n):
    mask=np.zeros((n,2*n),bool)
    for i in range(n):mask[i,2*i:2*i+2]=True
    return mask


def horizon_operator(B,rho,drift,state,horizon):
    n,m=B.shape
    if not 0<=rho<1 or horizon<1:raise ValueError('invalid dynamics/horizon')
    A=np.zeros((horizon*n,horizon*m));base=[];y=np.asarray(state,float).copy()
    for h in range(horizon):
        y=rho*y+drift;base.extend(y.tolist())
        for k in range(h+1):A[h*n:(h+1)*n,k*m:(k+1)*m]=rho**(h-k)*B
    return A,np.asarray(base)


def solve_plan(B,state,targets,weights,rho,drift,internal,costs,budgets,allowed,*,iterations=64,regularization=.0005,persistence_penalty=.0005,lower=None,generic=False):
    """Projected quadratic MPC; coordinate-equivalent algebra must agree.

    No physical-state safety guarantee follows from this affine approximation.
    """
    B=np.asarray(B,float);n,m=B.shape;H=len(targets)
    r=array(targets,(H,n),'targets');w=array(weights,(H,n),'priorities',1e-12)
    y=array(state,(n,),'state');d=array(drift,(n,),'drift');p=array(internal,(m,),'internal state',0,1)
    c=array(costs,(m,),'costs',1e-12);bud=array(budgets,(H,),'budgets',0)
    if iterations<1:raise ValueError('iterations must be positive')
    A,b=horizon_operator(B,rho,d,y,H);wf=w.ravel()/w.sum();rflat=r.ravel()
    if generic:
        Q=np.diag(wf);R=A.T@Q@A;linear=A.T@Q@(b-rflat)
    else:R=(A.T*wf)@A;linear=A.T@(wf*(b-rflat))
    R+=(regularization+persistence_penalty)*np.eye(H*m);linear-=persistence_penalty*np.tile(p,H)
    step=1./max(float(np.linalg.eigvalsh(R)[-1]),1e-12);u=np.tile(p,(H,1))
    for _ in range(iterations):
        proposal=(u.ravel()-step*(R@u.ravel()+linear)).reshape(H,m)
        u=np.array([project(proposal[h],c,bud[h],allowed,lower=lower) for h in range(H)])
    predicted=(A@u.ravel()+b).reshape(H,n);residual=r-predicted
    messages=(A.T*(wf*residual.ravel())).reshape(H*m,H,n);W_eff=-(A.T*wf)@A
    trace={'predicted_states':predicted.tolist(),'signed_residual':residual.tolist(),
           'positive_deficit':np.maximum(residual,0).tolist(),'negative_deficit':np.maximum(-residual,0).tolist(),
           'coactivation':(u[:,0::2]*u[:,1::2]).tolist(),'jacobian':A.tolist(),'priority':w.tolist(),
           'directed_messages':messages.tolist(),'K_effective':(A.T*wf).tolist(),'W_effective':W_eff.tolist(),
           'planning_B':B.tolist(),'plan':u.tolist(),
           'objective':float(np.sum(wf*(predicted.ravel()-rflat)**2)+regularization*np.sum(u*u)+persistence_penalty*np.sum((u-p)**2)),
           'iterations':iterations}
    return u,trace
