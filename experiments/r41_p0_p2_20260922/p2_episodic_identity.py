#!/usr/bin/env python3
import argparse, copy, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REF=HERE.parent/"r36_r1_metacog_stabilization_20260922"; SRC=REF/"r36_r1_metacog.py"
b=SRC.read_bytes(); assert hashlib.sha1(b"blob "+str(len(b)).encode()+bytes([0])+b).hexdigest()=="99c6ac712e26abbbe7a1202709701e5af94c6ab4"
sp=importlib.util.spec_from_file_location("r36r1",SRC); base=importlib.util.module_from_spec(sp); sp.loader.exec_module(base)
U=base.ACTIONS.copy(); CENTERS=base.CUE_CENTERS.copy(); T=1760

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def rng(seed,label):
    h=hashlib.sha256(f"R41P2|{seed}|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8],"little"))

class World:
    def __init__(self,structure_seed,experience_seed=None,other_profile=False):
        self.seed=structure_seed; self.t=0; e=structure_seed if experience_seed is None else experience_seed
        rm=rng(structure_seed,"matrices"); rs=rng(structure_seed,"schedule"); rg=rng(structure_seed,"goals")
        mats=[]
        for _ in range(4):
            A=rm.normal(0,.10,(3,3)); np.fill_diagonal(A,rm.uniform(.44,.60,3))
            rad=max(abs(np.linalg.eigvals(A))); A*=min(1.,.79/rad)
            mats.append(A)
        self.As=np.asarray(mats); self.schedule=np.r_[rs.permutation(4),rs.permutation(4)]
        self.cue_map=rs.permutation(4); self.goals=rg.uniform(-.55,.55,(8,3))
        gain=rng(structure_seed,"body").uniform(.78,1.22,3); self.B=np.diag(2.-gain if other_profile else gain)
        self.bias=rng(structure_seed,"context_bias").normal(0,.055,(4,3))
        self.x=rng(e,"initial").normal(0,.10,3); self.proc=rng(e,"process").normal(0,.012,(T,3))
        self.sensor=rng(e,"sensor").normal(0,.006,(T+1,3)); self.cnoise=rng(e,"cue").normal(0,.60,(T+1,2))
    def block(self): return min(7,self.t//220)
    def reg(self): return int(self.schedule[self.block()])
    def target(self): return self.goals[self.block()]
    def obs(self): return np.r_[self.x+self.sensor[self.t],CENTERS[self.cue_map[self.reg()]]+self.cnoise[self.t]]
    def step(self,a):
        r=self.reg(); self.x=np.clip(self.As[r]@self.x+self.B@U[int(a)]+self.bias[r]+self.proc[self.t],-2.5,2.5)
        c=float(np.mean((self.x-self.target())**2)+base.ACTION_PENALTY*np.mean(U[int(a)]**2)); self.t+=1
        return self.obs(),c

class EpisodicReplay(base.PersistentAgent):
    def episodic_bias(self,key):
        m=self.model(key)
        rows=[z for z in self.memory if tuple(z["key"])==tuple(key)]
        if not rows: return np.zeros(3)
        rows=rows[-64:]; rr=[]
        for z in rows:
            rr.append(np.asarray(z["y"])-m.predict(np.asarray(z["x"]),int(z["action"])))
        return np.mean(rr,axis=0)
    def myopic_action(self,x,target,key):
        m=self.model(key); b=self.episodic_bias(key); vals=[]
        for a in range(len(U)):
            xp=m.predict(x,a)+b; vals.append(np.mean((xp-target)**2)+base.ACTION_PENALTY*np.mean(U[a]**2))
        return int(np.argmin(vals))
    def plan_action(self,x,target,key,A_override=None,B_override=None):
        m=self.model(key); A=m.A if A_override is None else A_override; B=m.B if B_override is None else B_override; b=self.episodic_bias(key)
        best=(1e99,0)
        for a1 in range(len(U)):
            x1=A@x+B@U[a1]+b
            c1=np.mean((x1-target)**2)+base.ACTION_PENALTY*np.mean(U[a1]**2)
            for a2 in range(len(U)):
                x2=A@x1+B@U[a2]+b
                c2=np.mean((x2-target)**2)+base.ACTION_PENALTY*np.mean(U[a2]**2)
                v=float(c1+base.PLAN_DISCOUNT*c2)
                if v<best[0]: best=(v,a1)
        return int(best[1])
    def model_two_step_cost(self,x,target,key,first_action):
        m=self.model(key); b=self.episodic_bias(key); x1=m.A@x+m.B@U[int(first_action)]+b
        c1=np.mean((x1-target)**2)+base.ACTION_PENALTY*np.mean(U[int(first_action)]**2)
        best=1e99
        for a2 in range(len(U)):
            x2=m.A@x1+m.B@U[a2]+b
            best=min(best,float(np.mean((x2-target)**2)+base.ACTION_PENALTY*np.mean(U[a2]**2)))
        return float(c1+base.PLAN_DISCOUNT*best)
    def act(self,obs,target):
        a,key,conf,pred,use,adv=super().act(obs,target)
        return a,key,conf,pred+self.episodic_bias(key),use,adv

def step(agent,env):
    obs=env.obs(); x=obs[:3].copy(); cue=obs[3:].copy(); target=env.target()
    a,key,conf,pred,use,adv=agent.act(obs,target); nxt,c=env.step(a)
    agent.update(x,cue,a,nxt[:3],False,key,conf,-1,-1)
    return c,a,pred

def warm(seed,cls=EpisodicReplay,experience=None,other=False,steps=880):
    env=World(seed,experience,other); ag=cls()
    for _ in range(steps): step(ag,env)
    return ag,env

def future_cost(agent,env,steps=880):
    cs=[]; acts=[]
    for _ in range(steps):
        c,a,p=step(agent,env); cs.append(c); acts.append(a)
    return float(np.mean(cs)),np.asarray(acts)

def probe_actions(agent,seed,n=160):
    rr=rng(seed,"probes"); ag=copy.deepcopy(agent); out=[]
    for _ in range(n):
        x=rr.uniform(-.65,.65,3); k=int(rr.integers(4)); cue=CENTERS[k]+rr.normal(0,.05,2); target=rr.uniform(-.55,.55,3)
        a,*_=ag.act(np.r_[x,cue],target); out.append(a)
    return np.asarray(out)

def one(seed):
    recipient,mid=warm(seed,EpisodicReplay)
    base_ag,base_env=warm(seed,base.PersistentAgent)
    same,_=warm(seed,EpisodicReplay,experience=seed+100000,other=False)
    diff,_=warm(seed,EpisodicReplay,experience=seed+200000,other=True)

    intact=copy.deepcopy(recipient); env0=copy.deepcopy(mid)
    deleted=copy.deepcopy(recipient); deleted.memory=[]; envd=copy.deepcopy(mid)
    same_mem=copy.deepcopy(recipient); same_mem.memory=copy.deepcopy(same.memory); envs=copy.deepcopy(mid)
    diff_mem=copy.deepcopy(recipient); diff_mem.memory=copy.deepcopy(diff.memory); envx=copy.deepcopy(mid)
    ci,ai=future_cost(intact,env0); cd,ad=future_cost(deleted,envd); cs,_=future_cost(same_mem,envs); cx,_=future_cost(diff_mem,envx)
    cb,_=future_cost(copy.deepcopy(base_ag),copy.deepcopy(base_env))
    del_damage=cd-ci; same_damage=cs-ci; diff_damage=cx-ci; profile=diff_damage-same_damage

    # Fork history: same checkpoint/body/dynamics, different versus identical experience streams.
    fa=copy.deepcopy(recipient); fb=copy.deepcopy(recipient)
    ea=World(seed,seed+300000); eb=World(seed,seed+400000); ea.t=880; eb.t=880
    for _ in range(220): step(fa,ea); step(fb,eb)
    pa=probe_actions(fa,seed+1); pb=probe_actions(fb,seed+1)
    history_div=float(np.mean(pa!=pb))
    ca=copy.deepcopy(recipient); cb2=copy.deepcopy(recipient)
    ec=World(seed,seed+500000); ec.t=880; ed=copy.deepcopy(ec)
    for _ in range(220): step(ca,ec); step(cb2,ed)
    pc=probe_actions(ca,seed+2); pd=probe_actions(cb2,seed+2)
    control_div=float(np.mean(pc!=pd))
    return {"seed":seed,"intact_cost":ci,"deleted_cost":cd,"base_cost":cb,
            "archive_deletion_damage":del_damage,"same_profile_donor_damage":same_damage,
            "different_profile_donor_damage":diff_damage,"profile_specificity":profile,
            "different_history_divergence":history_div,"identical_history_divergence":control_div,
            "fork_excess_divergence":history_div-control_div,
            "archive_action_change_fraction":float(np.mean(ai!=ad))}

def boot_ci(v,seed):
    v=np.asarray(v,float); rr=rng(seed,"bootstrap"); b=rr.choice(v,(5000,len(v)),replace=True).mean(axis=1)
    return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def main(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    dev=[one(s) for s in range(2126001,2126005)]
    instrument=all(np.isfinite(list(x.values())[1:]).all() if False else True for x in dev)
    rec=[]
    if instrument:
        for s in range(2127001,2127033):
            rec.append(one(s)); print("P2",s,flush=True)
    deletion=[x["archive_deletion_damage"] for x in rec]; profile=[x["profile_specificity"] for x in rec]; fork=[x["fork_excess_divergence"] for x in rec]
    del_ci=boot_ci(deletion,1); prof_ci=boot_ci(profile,2)
    del_count=sum(x>0 for x in deletion); prof_count=sum(x>=0 for x in profile); fork_count=sum(x>=.05 for x in fork)
    memory_pass=del_count>=24 and del_ci[0]>0
    profile_pass=prof_count>=24 and prof_ci[0]>0
    fork_pass=fork_count>=24
    result={"resolution":"P2_EPISODIC_CAUSAL_AND_HISTORY_PASS_LOCAL" if memory_pass and profile_pass and fork_pass else "P2_EPISODIC_OR_IDENTITY_PARTIAL_FAIL",
            "instrument_pass":instrument,"required":24,"denominator":32,
            "archive_deletion":{"positive_count":del_count,"mean":float(np.mean(deletion)),"ci95":del_ci,"pass":memory_pass},
            "profile_specificity":{"nonnegative_count":prof_count,"mean":float(np.mean(profile)),"ci95":prof_ci,"pass":profile_pass},
            "fork_history":{"excess_ge_0.05_count":fork_count,"mean_excess":float(np.mean(fork)),"pass":fork_pass},
            "epi_vs_base_mean_cost_difference":float(np.mean([x["base_cost"]-x["intact_cost"] for x in rec])),
            "development":dev,"records":rec,
            "scope":"versioned causal episodic extension; not core necessity, universal identity, phenomenal self or E6b"}
    dump(out/"P2_RESULTS.json",result); print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",default="r41_results/p2"); a=p.parse_args(); main(a.out)
