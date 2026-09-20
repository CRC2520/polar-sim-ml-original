"""Compact causal-control agent for POLAR R11 development.

The agent uses only ERR + GOAL + MEM as persistent control-state families.
All model updates use the agent's own factual transitions.  Evaluator-only
source labels and counterfactuals never enter training.
"""
from __future__ import annotations
import copy, math
import numpy as np
from .config_dev import *

ACT=np.asarray(ACTIONS,float)

def _phi(obs):
    r,e,d=np.asarray(obs,float)
    return np.array([1.,r,e,d,r*e,r*d,e*d,r*r,e*e,d*d],float)

def _key(obs,action):
    r,e,d=np.clip(np.asarray(obs,float),0,1)
    q=lambda x:min(MEM_BINS-1,int(x*MEM_BINS))
    return (q(r),q(e),q(d),int(action))

class RLS:
    def __init__(self,nin=10,nout=4):
        self.theta=np.zeros((nin,nout),float)
        # persistence prior for next observation
        self.theta[1,0]=1.; self.theta[2,1]=1.; self.theta[3,2]=1.
        self.P=np.eye(nin)*RLS_RIDGE
        self.n=0
    def predict(self,x): return np.asarray(x,float)@self.theta
    def update(self,x,y):
        x=np.asarray(x,float); y=np.asarray(y,float)
        Px=self.P@x
        den=RLS_FORGET+x@Px
        k=Px/max(den,1e-9)
        err=y-x@self.theta
        self.theta += np.outer(k,err)
        self.P=(self.P-np.outer(k,x)@self.P)/RLS_FORGET
        self.P=(self.P+self.P.T)/2
        self.n+=1
        return err

class CompactWorkspace:
    def __init__(self):
        self.memory={}
        self.goal_weights=np.array([.5,.3,.2],float)
        self.goal_name="balanced"
        self.goal_age=0
        self.world_age=99
        self.last_source=2
        self.error_hist=np.zeros((ERR_WINDOW,4),float)
        self.error_count=0
    def goal_update(self,obs):
        r,e,d=np.asarray(obs,float)
        pressures=np.array([max(0,.58-e)/.58,max(0,.62-r)/.62,d])
        idx=int(np.argmax(pressures))
        names=("survival","reserve","task")
        weights=(np.array([.72,.18,.10]),np.array([.25,.65,.10]),np.array([.25,.15,.60]))
        candidate=names[idx] if pressures[idx]>.12 else "balanced"
        new=np.array([.5,.3,.2]) if candidate=="balanced" else weights[idx]
        if candidate!=self.goal_name and (self.goal_age>=2 or pressures[idx]>.40):
            self.goal_name=candidate; self.goal_weights=new; self.goal_age=0
        else: self.goal_age+=1
        return pressures
    def uncertainty(self):
        if not self.error_count: return .20
        n=min(self.error_count,ERR_WINDOW)
        return float(np.sqrt(np.mean(self.error_hist[-n:,:3]**2)))
    def remember(self,obs,action,target):
        k=_key(obs,action); entry=self.memory.get(k)
        delta=np.asarray(target[:3])-np.asarray(obs)
        reward=float(target[3])
        if entry is None:
            self.memory[k]=dict(delta=delta.copy(),reward=reward,n=1)
        else:
            n=entry["n"]; rate=min(.35,1/(n+1))
            entry["delta"]=(1-rate)*entry["delta"]+rate*delta
            entry["reward"]=(1-rate)*entry["reward"]+rate*reward
            entry["n"]=n+1
    def recall(self,obs,action):
        e=self.memory.get(_key(obs,action))
        if e is None: return None
        return np.r_[np.clip(np.asarray(obs)+e["delta"],0,1),e["reward"]]

class CompactCausalAgent:
    def __init__(self,seed,relation_mode="polar"):
        self.seed=int(seed); self.rng=np.random.default_rng(seed)
        self.local=[RLS() for _ in range(4)]
        self.cross=[RLS() for _ in range(4)]
        self.workspace=CompactWorkspace()
        # Utility gate regression: context -> observed utility advantage full-local
        self.gate_P=np.eye(8)*GATE_RIDGE
        self.gate_coef=np.zeros(8)
        self.gate_n=0
        self.relation_mode=relation_mode
        self.last=None
        self.reset_episode()
    def reset_episode(self):
        self.last=None
    def _context(self,obs):
        r,e,d=np.asarray(obs,float)
        u=self.workspace.uncertainty()
        w=self.workspace.goal_weights
        return np.array([1.,r,e,d,u,w[0],w[1],w[2]],float)
    def _interaction_phi(self,obs):
        x=_phi(obs).copy()
        if self.relation_mode=="polar":
            return x
        if self.relation_mode=="generic":
            # fixed invertible renaming of the observable coordinates before
            # constructing products; information and model size are unchanged.
            r,e,d=np.asarray(obs,float)[[2,0,1]]
            return np.array([1.,r,e,d,r*e,r*d,e*d,r*r,e*e,d*d],float)
        raise ValueError(self.relation_mode)
    def gate_prob(self,obs):
        z=self._context(obs)
        v=float(z@self.gate_coef)
        return 1/(1+math.exp(-max(-30,min(30,5*v))))
    def predictions(self,obs,force_gate=None,permuted_content=False,no_memory=False):
        lp=[]; fp=[]; memories=[]
        xlocal=_phi(obs); xcross=self._interaction_phi(obs)
        for a in range(4):
            l=self.local[a].predict(xlocal)
            f=self.cross[a].predict(xcross)
            lp.append(l); fp.append(f)
            memories.append(None if no_memory else self.workspace.recall(obs,a))
        lp=np.asarray(lp); fp=np.asarray(fp)
        g=self.gate_prob(obs) if force_gate is None else float(force_gate)
        pred=(1-g)*lp+g*fp
        pred[:,:3]=np.clip(pred[:,:3],0,1)
        pred[:,3]=np.clip(pred[:,3],0,1)
        # Typed memory directly modifies the same candidate values.
        for a,m in enumerate(memories):
            if m is not None:
                mm=m.copy()
                if permuted_content:
                    mm[:3]=mm[[2,0,1]]
                pred[a]=.8*pred[a]+.2*mm
        return pred,g,lp,fp
    def prepare(self,obs,force_gate=None,permuted_content=False,no_memory=False,no_cross=False):
        self.workspace.goal_update(obs)
        fg=0. if no_cross else force_gate
        pred,g,lp,fp=self.predictions(obs,fg,permuted_content,no_memory)
        w=self.workspace.goal_weights
        # [reserve, health, task/reward] utility with explicit uncertainty penalty
        u=self.workspace.uncertainty()
        utility=w[1]*pred[:,0]+w[0]*pred[:,1]+w[2]*pred[:,3]-.12*u
        floor=.18
        feasible=pred[:,1]-.75*u>=floor
        if not feasible.any(): feasible[int(np.argmax(pred[:,1]-.75*u))]=True
        action=int(np.argmax(np.where(feasible,utility,-np.inf)))
        return dict(action=action,pred=pred,local=lp,full=fp,gate=float(g),
                    utility=utility,feasible=feasible,uncertainty=float(u),
                    goals=w.copy(),context=self._context(obs))
    def complete(self,obs,action,nxt,reward,policy,learn=True,learn_gate=True):
        target=np.r_[np.asarray(nxt,float),float(reward)]
        xl=_phi(obs); xc=self._interaction_phi(obs)
        pred_l=self.local[action].predict(xl); pred_f=self.cross[action].predict(xc)
        if learn:
            el=self.local[action].update(xl,target)
            ef=self.cross[action].update(xc,target)
            self.workspace.remember(obs,action,target)
            # factual prediction residual enters ERR
            err=target-policy["pred"][action]
            self.workspace.error_hist=np.roll(self.workspace.error_hist,-1,axis=0)
            self.workspace.error_hist[-1]=err
            self.workspace.error_count+=1
            # native source estimate: own expected action effect vs residual
            base=self.local[0].predict(xl)[:3]
            self_effect=pred_l[:3]-base
            residual=np.asarray(nxt)-np.clip(pred_l[:3],0,1)
            sm=float(np.linalg.norm(self_effect)); wm=float(np.linalg.norm(residual))
            if wm>1.35*max(sm,1e-8): src=1
            elif sm>1.35*max(wm,1e-8): src=0
            else: src=2
            self.workspace.last_source=src
            self.workspace.world_age=0 if src==1 else self.workspace.world_age+1
            if learn_gate:
                # utility target is factual squared-prediction improvement of cross over local
                gain=float(np.mean((target-pred_l)**2)-np.mean((target-pred_f)**2))
                z=self._context(obs)
                Pz=self.gate_P@z; den=1+z@Pz; k=Pz/den
                self.gate_coef += k*(gain-z@self.gate_coef)
                self.gate_P=self.gate_P-np.outer(k,z)@self.gate_P
                self.gate_n+=1
        self.last=dict(obs=np.asarray(obs).copy(),action=int(action),target=target.copy())
    def copy(self): return copy.deepcopy(self)
