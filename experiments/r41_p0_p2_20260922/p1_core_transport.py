#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REF=HERE.parent/"r36_r1_metacog_stabilization_20260922"
SRC=REF/"r36_r1_metacog.py"
b=SRC.read_bytes()
assert hashlib.sha1(b"blob "+str(len(b)).encode()+bytes([0])+b).hexdigest()=="99c6ac712e26abbbe7a1202709701e5af94c6ab4"
sp=importlib.util.spec_from_file_location("r36r1",SRC); base=importlib.util.module_from_spec(sp); sp.loader.exec_module(base)
U=base.ACTIONS.copy(); N=3; T=1760; CENTERS=base.CUE_CENTERS.copy()

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def rng(seed,label):
    h=hashlib.sha256(f"R41P1|{seed}|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8],"little"))

def costs(A,B,x,target):
    x1=x@A.T+U@B.T
    x2=x1[:,None,:]@A.T+(U@B.T)[None,:,:]
    c1=np.mean((x1-target)**2,axis=1)+base.ACTION_PENALTY*np.mean(U*U,axis=1)
    c2=np.mean((x2-target)**2,axis=2)+base.ACTION_PENALTY*np.mean(U*U,axis=1)[None,:]
    return c1[:,None]+base.PLAN_DISCOUNT*c2

def fast_plan(self,x,target,key,A_override=None,B_override=None):
    m=self.model(key); A=m.A if A_override is None else A_override; B=m.B if B_override is None else B_override
    return int(np.argmin(costs(A,B,x,target))//len(U))
def fast_cost(self,x,target,key,first_action):
    m=self.model(key); return float(costs(m.A,m.B,x,target)[int(first_action)].min())
base.PersistentAgent.plan_action=fast_plan; base.PersistentAgent.model_two_step_cost=fast_cost

class World:
    def __init__(self,seed,stress=False):
        self.seed=seed; self.t=0; self.stress=stress
        rm=rng(seed,"matrices"); rs=rng(seed,"schedule"); rg=rng(seed,"goals")
        mats=[]
        for _ in range(4):
            A=rm.normal(0,.12,(3,3)); np.fill_diagonal(A,rm.uniform(.43,.61,3))
            rad=max(abs(np.linalg.eigvals(A)))
            if rad>.80: A*=.80/rad
            mats.append(A)
        self.As=np.asarray(mats); self.schedule=np.r_[rs.permutation(4),rs.permutation(4)]
        self.cue_map=rs.permutation(4); self.goals=rg.uniform(-.55,.55,(8,3))
        self.B=np.diag(rng(seed,"body").uniform(.78,1.22,3))
        self.x=rng(seed,"initial").normal(0,.10,3)
        self.proc=rng(seed,"process").normal(0,.014,(T,3))
        self.sensor=rng(seed,"sensor").normal(0,.030 if stress else .007,(T+1,3))
        self.cnoise=rng(seed,"cue").normal(0,1.05 if stress else .62,(T+1,2))
        if stress:
            rr=rng(seed,"outlier"); bad=rr.random(T+1)<.08
            self.cnoise[bad]+=rr.normal(0,2.5,(int(bad.sum()),2))
        rr=rng(seed,"shock"); bad=rr.random(T)<.015
        self.shock=np.zeros((T,3)); ix=np.flatnonzero(bad)
        if len(ix): self.shock[ix,rr.integers(3,size=len(ix))]=rr.choice([-1.,1.],len(ix))*rr.uniform(.22,.34,len(ix))
    def block(self): return min(7,self.t//220)
    def reg(self): return int(self.schedule[self.block()])
    def target(self): return self.goals[self.block()]
    def obs(self): return np.r_[self.x+self.sensor[self.t],CENTERS[self.cue_map[self.reg()]]+self.cnoise[self.t]]
    def step(self,a):
        A=self.As[self.reg()]; target=self.target()
        self.x=np.clip(A@self.x+self.B@U[int(a)]+self.proc[self.t]+self.shock[self.t],-2.5,2.5)
        c=float(np.mean((self.x-target)**2)+base.ACTION_PENALTY*np.mean(U[int(a)]**2))
        self.t+=1
        return self.obs(),c

class NoC(base.PersistentAgent):
    def update_cue(self,cue):
        self.cue_state=np.asarray(cue,float).copy()
        return base.cue_key(cue)

def dz(v):
    v=np.asarray(v,float)
    if len(v)<20: return 0.0
    return float(v.mean()/(v.std(ddof=1)+1e-12))

def run_agent(seed,stress,agent_cls=base.PersistentAgent,collect=False):
    env=World(seed,stress); ag=agent_cls(); cost=[]; stable=[]; dd=[]; dc=[]; dr=[]
    for _ in range(T):
        obs=env.obs(); x=obs[:3].copy(); cue=obs[3:].copy(); target=env.target()
        action,key,conf,pred,use,adv=ag.act(obs,target); m=ag.model(key); fitted=m.fitted
        cur=ag.model(base.cue_key(cue)).predict(x,action)
        pd=np.repeat(m.A.mean(axis=1,keepdims=True),3,axis=1)@x+m.B@U[action]
        pr=np.diag(np.diag(m.A))@x+m.B@U[action]
        nxt,c=env.step(action); y=nxt[:3].copy(); err=base.mse(pred,y)
        if collect and fitted:
            dd.append(base.mse(pd,y)-err); dc.append(base.mse(cur,y)-err); dr.append(base.mse(pr,y)-err)
        ag.update(x,cue,action,y,False,key,conf,-1,-1)
        cost.append(c); stable.append(float(np.linalg.norm(env.x)<3.))
    return {"cost":float(np.mean(cost)),"stable":float(np.mean(stable)),
            "D_dz":dz(dd),"C_dz":dz(dc),"R_dz":dz(dr)}

def run_control(seed,stress,kind):
    env=World(seed,stress); rr=rng(seed,f"{kind}-{int(stress)}"); cs=[]
    for _ in range(T):
        if kind=="random": a=int(rr.integers(len(U)))
        else:
            tc=costs(env.As[env.reg()],env.B,env.x,env.target())
            a=int(np.argmin(tc)//len(U))
        _,c=env.step(a); cs.append(c)
    return float(np.mean(cs))

def evaluate_seed(seed,stress):
    q=run_agent(seed,stress,base.PersistentAgent,True); noc=run_agent(seed,stress,NoC,False)
    random_cost=run_control(seed,stress,"random"); oracle_cost=run_control(seed,stress,"oracle")
    denom=random_cost-oracle_cost
    score=float((random_cost-q["cost"])/denom) if denom>1e-8 else -999.
    q.update({"seed":seed,"stress":stress,"random_cost":random_cost,"oracle_cost":oracle_cost,
              "noC_cost":noc["cost"],"normalized_score":score,
              "core_pass":bool(score>=.50 and q["D_dz"]>=.20 and q["C_dz"]>=.20 and q["R_dz"]>=.20 and q["stable"]>=.99),
              "instrument_valid":bool(denom>1e-8 and np.isfinite(score))})
    return q

def main(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    dev=[evaluate_seed(s,z) for s in range(2116001,2116005) for z in [False,True]]
    instrument=all(x["instrument_valid"] for x in dev)
    conf=[]
    if instrument:
        for s in range(2117001,2117033):
            for z in [False,True]:
                conf.append(evaluate_seed(s,z))
            print("P1",s,flush=True)
    groups={}
    for z in [False,True]:
        r=[x for x in conf if x["stress"]==z]
        groups["stress" if z else "standard"]={
            "n":len(r),"core_pass_count":sum(x["core_pass"] for x in r),
            "component_counts":{k:sum(x[k]>=v for x in r) for k,v in [("normalized_score",.50),("D_dz",.20),("C_dz",.20),("R_dz",.20),("stable",.99)]},
            "medians":{k:float(np.median([x[k] for x in r])) for k in ["normalized_score","D_dz","C_dz","R_dz","cost","noC_cost"]}
        }
    support=instrument and all(groups[x]["core_pass_count"]>=24 for x in groups)
    result={"resolution":"P1_SCALE_NORMALIZED_CORE_TRANSPORT_PASS" if support else "P1_SCALE_NORMALIZED_CORE_TRANSPORT_FAIL",
            "instrument_pass":instrument,"required":24,"denominator":32,"development":dev,"groups":groups,"records":conf,
            "scope":"fresh internal generator, scale-normalized causal transport; not E6b, not consciousness"}
    dump(out/"P1_RESULTS.json",result); print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",default="r41_results/p1"); a=p.parse_args(); main(a.out)
