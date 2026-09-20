"""R11 causal-invariance adapter layered over the unchanged R9 NativeAgent."""
from __future__ import annotations
import copy, hashlib
import numpy as np
from r9_completion.agent import NativeAgent

def soft(x,t):
    return np.sign(x)*np.maximum(np.abs(x)-t,0.)

class OnlineAdapter:
    """Action-specific RLS with light L1 shrinkage and factual residual tracking."""
    def __init__(self, dim, forgetting=.985, l1=.0001, min_samples=5):
        self.dim=int(dim); self.forgetting=float(forgetting); self.l1=float(l1); self.min_samples=int(min_samples)
        self.W=np.zeros((4,self.dim+1,4),float)
        self.P=np.tile(np.eye(self.dim+1)[None]*8.,(4,1,1))
        self.count=np.zeros(4,int)
        self.err=np.ones(4,float)*.35
        self.base_err=np.ones(4,float)*.35
    def predict_raw(self,x):
        z=np.r_[1.,np.asarray(x,float)]
        return np.einsum('d,ado->ao',z,self.W)
    def update(self,x,action,target,base_prediction):
        a=int(action); z=np.r_[1.,np.asarray(x,float)]; y=np.asarray(target,float)
        pred=z@self.W[a]
        P=self.P[a]
        denom=self.forgetting+z@P@z
        k=(P@z)/max(denom,1e-9)
        self.W[a]+=np.outer(k,y-pred)
        self.W[a,1:]=soft(self.W[a,1:],self.l1)
        self.P[a]=(P-np.outer(k,z)@P)/self.forgetting
        self.P[a]=(self.P[a]+self.P[a].T)/2
        self.count[a]+=1
        ae=float(np.mean(np.abs(y-pred)))
        be=float(np.mean(np.abs(y-np.asarray(base_prediction,float))))
        rate=.08
        self.err[a]=(1-rate)*self.err[a]+rate*ae
        self.base_err[a]=(1-rate)*self.base_err[a]+rate*be
    def usable(self,margin):
        return (self.count>=self.min_samples)&(self.err+float(margin)<self.base_err)
    def state(self):
        return dict(W=self.W.tolist(),count=self.count.tolist(),err=self.err.tolist(),base_err=self.base_err.tolist())

class R11Agent:
    """R9 controller + causal-core adapter. R9 learned parameters are not rewritten."""
    def __init__(self, base:NativeAgent, *, generic=False, forgetting=.985,l1=.0001,min_samples=5,
                 reliability_margin=.002,adapter_weight=.60,risk_floor=.25):
        self.base=copy.deepcopy(base); self.generic=bool(generic)
        self.reliability_margin=float(reliability_margin); self.adapter_weight=float(adapter_weight)
        self.risk_floor=float(risk_floor)
        self.dim=39
        self.adapter=OnlineAdapter(self.dim,forgetting,l1,min_samples)
        rng=np.random.default_rng(int(base.seed)^0x11CA)
        q,_=np.linalg.qr(rng.normal(size=(self.dim,self.dim)))
        self.rotation=q
        self.step_index=0; self.last_policy=None; self.last_obs=None
        self.source_class=2; self.world_age=99; self.confidence=.5
        self.last_world_strength=0.; self.last_self_strength=0.
        self.memory_err=.30; self.persistence_err=.30; self.memory_count=0
    def reset_episode(self,obs,variant='full',reset_adapter=False):
        self.base.reset_episode(np.asarray(obs,float),variant)
        self.step_index=0; self.last_policy=None; self.last_obs=np.asarray(obs,float).copy()
        self.source_class=2; self.world_age=99; self.confidence=.5
        if reset_adapter:
            old=self.adapter
            self.adapter=OnlineAdapter(old.dim,old.forgetting,old.l1,old.min_samples)
    def _feature(self,obs,p,variant='full'):
        poles=np.asarray(p['poles'],float)
        orient=(poles[:,0]-poles[:,1]); intensity=(poles[:,0]+poles[:,1])/2
        polar=np.r_[orient,intensity,np.asarray(p['tension'],float)[:,0]]
        if variant in ('noCross','noRelation'):
            polar=np.zeros_like(polar)
        err=np.asarray(obs,float)-np.asarray(p['previous_prediction'],float)
        core=np.r_[err,float(p['gate_context'][-1]),np.asarray(p['goals'],float),
                   np.asarray(p['memory_energy'],float),np.asarray(p['memory_resource'],float)]
        x=np.r_[polar,core]
        if len(x)!=self.dim: raise RuntimeError((len(x),self.dim))
        return self.rotation@x if self.generic else x
    def prepare(self,obs,variant='full',adaptation_probe=False):
        obs=np.asarray(obs,float)
        p=self.base.prepare(obs,variant)
        x=self._feature(obs,p,variant)
        raw=self.adapter.predict_raw(x)
        adap=np.clip(raw,0,1)
        usable=self.adapter.usable(self.reliability_margin)
        pred=np.asarray(p['predictions'],float).copy()
        pred[usable]=adap[usable]
        w=np.asarray(p['goals'],float); w=w/max(w.sum(),1e-9)
        immediate=w[0]*pred[:,1]+w[1]*pred[:,0]+w[2]*pred[:,3]
        memory_usable=bool(self.memory_count>=12 and self.memory_err+.005<self.persistence_err)
        if memory_usable and variant!='noMemory':
            delayed=w[0]*np.asarray(p['memory_energy'])+w[1]*np.asarray(p['memory_resource'])+w[2]*pred[:,3]
        else:
            delayed=w[0]*float(obs[1])+w[1]*float(obs[0])+w[2]*pred[:,3]
        risk=np.maximum(0.,self.risk_floor-pred[:,1])
        native=(self.adapter_weight*immediate+(1-self.adapter_weight)*np.asarray(p['scores'])+
                .12*(delayed-immediate)-1.5*risk)
        base_feasible=np.asarray(p['feasible'],bool)
        if np.any(usable):
            feasible=pred[:,1]>=self.risk_floor
            if not np.any(feasible):
                feasible=np.zeros(4,bool); feasible[int(np.argmax(pred[:,1]))]=True
        else:
            feasible=base_feasible.copy()
        if adaptation_probe and self.step_index<64:
            # Declared uniform causal probe. It bypasses the inherited R9 mask only
            # during the feasibility-audited adaptation window.
            action=int(self.step_index%4)
        else:
            action=int(np.argmax(np.where(feasible,native,-np.inf)))
        self_effect=pred-pred[0:1]
        counterfactual_reward=pred[:,3].copy()
        out=dict(base=p,features=x,adapter_predictions=adap,adapter_usable=usable,
                 predictions=pred,native_scores=native,action=action,feasible=feasible,
                 inherited_feasible=base_feasible,
                 counterfactual_reward=counterfactual_reward,
                 confidence=float(self.confidence),source_class=int(self.source_class),
                 world_age=int(self.world_age),memory_usable=memory_usable,
                 memory_error=float(self.memory_err),persistence_error=float(self.persistence_err))
        self.last_policy=out; self.last_obs=obs.copy()
        return out
    def complete(self,obs,action,nxt,reward,policy,learn_base=False,terminal=False,update_adapter=True):
        obs=np.asarray(obs,float); nxt=np.asarray(nxt,float); a=int(action)
        bp=np.asarray(policy['base']['predictions'][a],float)
        target=np.r_[nxt,float(reward)]
        if update_adapter:
            self.adapter.update(policy['features'],a,target,bp)
        # source estimate uses action-dependent predicted difference vs action-0.
        pred=np.asarray(policy['predictions'],float)
        self_delta=pred[a,:3]-pred[0,:3]
        observed=nxt-obs
        residual=observed-self_delta
        sm=float(np.linalg.norm(self_delta)); wm=float(np.linalg.norm(residual))
        self.last_self_strength=sm; self.last_world_strength=wm
        if wm>1.35*max(sm,1e-8): self.source_class=1
        elif sm>1.35*max(wm,1e-8): self.source_class=0
        else: self.source_class=2
        if self.source_class in (1,2) and wm>.018:
            self.world_age=0
        else:
            self.world_age=min(99,self.world_age+1)
        mem_pred=np.array([policy['base']['memory_energy'][a],policy['base']['memory_resource'][a]],float)
        persist_pred=np.array([obs[1],obs[0]],float)
        mem_target=np.array([nxt[1],nxt[0]],float)
        rate=.08
        self.memory_err=(1-rate)*self.memory_err+rate*float(np.mean(np.abs(mem_target-mem_pred)))
        self.persistence_err=(1-rate)*self.persistence_err+rate*float(np.mean(np.abs(mem_target-persist_pred)))
        self.memory_count+=1
        e=float(np.mean(np.abs(target-policy['predictions'][a])))
        self.confidence=float(1./(1.+5.*e))
        self.base.complete_transition(obs,a,nxt,reward,policy['base'],learn=learn_base,terminal=terminal)
        self.step_index+=1; self.last_obs=nxt.copy()
    def parameter_count(self):
        return int(self.adapter.W.size+self.adapter.P.size)
