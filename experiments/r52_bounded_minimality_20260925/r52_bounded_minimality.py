#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

POPGYM_COMMIT="e397e5eac9965f9963d18c9f455cd1983bca14fb"
TASKS=("ConcentrationHard","CountRecallHard","AutoencodeHard","RepeatPreviousHard")
CONDITIONS=("FULL_DCR","NO_D_MATCHED","NO_C_MATCHED","NO_R_MATCHED","SHAM_CAPACITY","GENERIC_ISO")
DEV_SEEDS=list(range(2226001,2226017))
CONF_SEEDS=list(range(2227001,2227033))
SMOKE_SEED=2225000
EPISODES_PER_SEED=3
ISO_TOL=1e-12
N_SYMBOLS_13=13
N_POSITIONS=52
N_PAIRS=26

def stable_seed(seed:int,*labels:str)->int:
    raw="|".join(["R52",str(int(seed)),*labels]).encode()
    h=hashlib.sha256(raw).digest()
    return int.from_bytes(h[:8],"little") & 0xffffffff

def phi13(x:int)->int: return 12-int(x)
def phi4(x:int)->int: return 3-int(x)

def cyclic_pick(cands:Sequence[int],last:int,n:int)->int:
    if not cands: raise RuntimeError("no candidates")
    start=(int(last)+1)%n if last>=0 else 0
    return min((int(i) for i in cands),key=lambda i:((i-start)%n,i))

@dataclass
class EpisodeMetric:
    metric: float
    native_return: float
    actions: List[int]

class ConcentrationController:
    def __init__(self,condition:str):
        self.condition=condition
        self.memory:Dict[int,int]={}
        self.failed:Set[Tuple[int,int]]=set()
        self.solved:Set[int]=set()
        self.pair:List[int]=[]
        self.last=-1
        self.dummy=np.zeros(N_POSITIONS,dtype=np.int64)

    def enc(self,s:int)->int:
        if self.condition=="NO_D_MATCHED": return 0
        if self.condition=="GENERIC_ISO": return phi13(s)
        return int(s)

    def target(self,s:int)->int:
        return (int(s)+1)%13 if self.condition=="NO_R_MATCHED" else int(s)

    @staticmethod
    def pk(i,j): return tuple(sorted((int(i),int(j))))

    def unsolved(self,exclude=None):
        exclude=exclude or set()
        return [i for i in range(N_POSITIONS) if i not in self.solved and i not in exclude]

    def known(self,target,exclude,first=None):
        out=[]
        for i in self.unsolved(exclude):
            if i not in self.memory or self.memory[i]!=target: continue
            if first is not None and self.pk(first,i) in self.failed: continue
            out.append(i)
        return out

    def pairable(self):
        known=[i for i in self.unsolved() if i in self.memory]
        out=[]
        for i in known:
            t=self.target(self.memory[i])
            if any(j!=i and self.memory[j]==t and self.pk(i,j) not in self.failed for j in known):
                out.append(i)
        return out

    def act(self):
        if not self.pair:
            p=self.pairable()
            if p: return cyclic_pick(p,self.last,N_POSITIONS)
            u=[i for i in self.unsolved() if i not in self.memory]
            if u: return cyclic_pick(u,self.last,N_POSITIONS)
            return cyclic_pick(self.unsolved(),self.last,N_POSITIONS)
        first=self.pair[0]
        if first not in self.memory:
            u=[i for i in self.unsolved({first}) if i not in self.memory]
            return cyclic_pick(u or self.unsolved({first}),self.last,N_POSITIONS)
        t=self.target(self.memory[first])
        k=self.known(t,{first},first)
        if k: return cyclic_pick(k,self.last,N_POSITIONS)
        u=[i for i in self.unsolved({first}) if i not in self.memory]
        if u: return cyclic_pick(u,self.last,N_POSITIONS)
        rem=[i for i in self.unsolved({first}) if self.pk(first,i) not in self.failed]
        return cyclic_pick(rem or self.unsolved({first}),self.last,N_POSITIONS)

    def observe(self,action:int,symbol:int,reward:float):
        a=int(action)
        z=self.enc(symbol)
        self.memory[a]=z
        self.dummy[a]=(self.dummy[a]+1)%17
        self.pair.append(a); self.last=a
        if len(self.pair)==2:
            i,j=self.pair
            if reward>0:
                self.solved.update((i,j))
                self.memory.pop(i,None); self.memory.pop(j,None)
                self.failed={p for p in self.failed if i not in p and j not in p}
            elif reward<0:
                self.failed.add(self.pk(i,j))
            self.pair=[]
            if self.condition=="NO_C_MATCHED":
                self.memory={k:0 for k in []}
                self.failed.clear()

class CountRecallController:
    def __init__(self,condition:str,n_actions:int):
        self.condition=condition
        self.counts=np.zeros(13,dtype=np.int64)
        self.dummy=np.zeros(13,dtype=np.int64)
        self.n_actions=int(n_actions)
    def act(self,obs):
        v,q=map(int,np.asarray(obs))
        if self.condition=="NO_C_MATCHED":
            self.counts[:]=0
        if self.condition=="NO_D_MATCHED":
            v=q=0
        elif self.condition=="GENERIC_ISO":
            v,q=phi13(v),phi13(q)
        self.counts[v]+=1
        self.dummy[(v+q)%13]=(self.dummy[(v+q)%13]+1)%19
        if self.condition=="NO_R_MATCHED":
            q=(q+1)%13
        return int(min(self.counts[q],self.n_actions-1))

class AutoencodeController:
    def __init__(self,condition:str):
        self.condition=condition
        self.memory:List[Optional[int]]=[]
        self.play_started=False
        self.dummy:List[int]=[]
    def enc(self,s):
        if s is None: return None
        if self.condition=="NO_D_MATCHED": return 0
        if self.condition=="GENERIC_ISO": return phi4(s)
        return int(s)
    def dec(self,s):
        if s is None: return 0
        return phi4(s) if self.condition=="GENERIC_ISO" else int(s)
    def consume(self,s):
        z=self.enc(s)
        if self.condition=="NO_C_MATCHED": self.memory=[z]
        else: self.memory.append(z)
        self.dummy.append((len(self.dummy)*3+1)%7)
    def act(self,obs):
        mode,symbol=int(obs[0]),int(obs[1])
        if mode==1:
            self.consume(symbol); return 0
        if not self.play_started:
            self.consume(symbol); self.play_started=True
        if not self.memory: return 0
        z=self.memory.pop(0) if self.condition=="NO_R_MATCHED" else self.memory.pop(-1)
        return self.dec(z)

class RepeatPreviousController:
    def __init__(self,condition:str,k:int):
        self.condition=condition; self.k=int(k)
        self.memory:List[Optional[int]]=[]
        self.dummy:List[int]=[]
    def enc(self,s):
        if self.condition=="NO_D_MATCHED": return 0
        if self.condition=="GENERIC_ISO": return phi4(s)
        return int(s)
    def dec(self,s):
        if s is None: return 0
        return phi4(s) if self.condition=="GENERIC_ISO" else int(s)
    def act(self,symbol:int):
        z=self.enc(symbol)
        if self.condition=="NO_C_MATCHED": self.memory=[z]
        else: self.memory.append(z)
        self.dummy.append((len(self.dummy)*5+2)%11)
        lag=max(1,self.k-1 if self.condition=="NO_R_MATCHED" else self.k)
        if len(self.memory)<lag: return 0
        return self.dec(self.memory[-lag])

def make_env(task):
    import popgym
    return getattr(popgym,task)()

def concentration_episode(condition,seed):
    env=make_env("ConcentrationHard")
    obs=np.asarray(env.reset(seed=seed))
    agent=ConcentrationController(condition)
    matches=0; ret=0.0; acts=[]; done=False; steps=0
    while not done and steps<int(env.episode_length):
        a=agent.act(); nxt,r,done,_=env.step(a)
        sym=int(np.asarray(nxt)[a])
        if not 0<=sym<13: raise RuntimeError("Concentration reveal contract changed")
        agent.observe(a,sym,float(r))
        matches+=int(r>0); ret+=float(r); acts.append(int(a)); obs=np.asarray(nxt); steps+=1
    env.close()
    return EpisodeMetric(matches/N_PAIRS,ret,acts)

def count_episode(condition,seed):
    env=make_env("CountRecallHard")
    obs=np.asarray(env.reset(seed=seed),dtype=np.int64)
    agent=CountRecallController(condition,int(env.action_space.n))
    correct=steps=0; ret=0.0; acts=[]; done=False
    while not done:
        a=agent.act(obs); nxt,r,done,_=env.step(a)
        correct+=int(r>0); steps+=1; ret+=float(r); acts.append(int(a)); obs=np.asarray(nxt,dtype=np.int64)
    env.close()
    return EpisodeMetric(correct/max(steps,1),ret,acts)

def auto_episode(condition,seed):
    env=make_env("AutoencodeHard")
    obs=env.reset(seed=seed)
    agent=AutoencodeController(condition)
    correct=scored=0; ret=0.0; acts=[]; done=False
    while not done:
        a=agent.act(obs); nxt,r,done,_=env.step(a)
        if abs(float(r))>1e-15: scored+=1; correct+=int(r>0)
        ret+=float(r); acts.append(int(a)); obs=nxt
    env.close()
    return EpisodeMetric(correct/max(scored,1),ret,acts)

def repeat_episode(condition,seed):
    env=make_env("RepeatPreviousHard")
    obs=env.reset(seed=seed)
    agent=RepeatPreviousController(condition,int(env.k))
    correct=scored=0; ret=0.0; acts=[]; done=False
    while not done:
        a=agent.act(int(obs)); nxt,r,done,_=env.step(a)
        if abs(float(r))>1e-15: scored+=1; correct+=int(r>0)
        ret+=float(r); acts.append(int(a)); obs=nxt
    env.close()
    return EpisodeMetric(correct/max(scored,1),ret,acts)

def run_episode(task,condition,seed):
    if task=="ConcentrationHard": return concentration_episode(condition,seed)
    if task=="CountRecallHard": return count_episode(condition,seed)
    if task=="AutoencodeHard": return auto_episode(condition,seed)
    if task=="RepeatPreviousHard": return repeat_episode(condition,seed)
    raise ValueError(task)

def mean_metric(rows):
    return {
        "metric":float(np.mean([r.metric for r in rows])),
        "native_return":float(np.mean([r.native_return for r in rows]))
    }

def agreement(a,b):
    same=total=0
    if len(a)!=len(b): return 0.0
    for x,y in zip(a,b):
        if len(x.actions)!=len(y.actions): return 0.0
        same+=sum(int(i==j) for i,j in zip(x.actions,y.actions)); total+=len(x.actions)
    return same/max(total,1)

def evaluate_seed(seed):
    tasks={}
    for ti,task in enumerate(TASKS):
        raw={}; sm={}
        for cond in CONDITIONS:
            eps=[run_episode(task,cond,stable_seed(seed,task,"env",str(ep))) for ep in range(EPISODES_PER_SEED)]
            raw[cond]=eps; sm[cond]=mean_metric(eps)
        full=sm["FULL_DCR"]["metric"]
        sham_gap=abs(full-sm["SHAM_CAPACITY"]["metric"])
        row={
            "full_metric":full,
            "d_effect":full-sm["NO_D_MATCHED"]["metric"],
            "c_effect":full-sm["NO_C_MATCHED"]["metric"],
            "r_effect":full-sm["NO_R_MATCHED"]["metric"],
            "sham_gap":sham_gap,
            "d_specificity":full-sm["NO_D_MATCHED"]["metric"]-sham_gap,
            "c_specificity":full-sm["NO_C_MATCHED"]["metric"]-sham_gap,
            "r_specificity":full-sm["NO_R_MATCHED"]["metric"]-sham_gap,
            "iso_gap":abs(full-sm["GENERIC_ISO"]["metric"]),
            "iso_action_agreement":agreement(raw["FULL_DCR"],raw["GENERIC_ISO"]),
            "condition_metrics":sm,
        }
        row["task_pass"]=bool(
            row["full_metric"]>=0.90 and
            row["d_effect"]>=0.20 and row["c_effect"]>=0.20 and row["r_effect"]>=0.20 and
            row["sham_gap"]<=0.02 and
            row["d_specificity"]>=0.18 and row["c_specificity"]>=0.18 and row["r_specificity"]>=0.18 and
            row["iso_gap"]<=ISO_TOL and row["iso_action_agreement"]>=1-ISO_TOL
        )
        tasks[task]=row
    return {"seed":int(seed),"tasks":tasks,"seed_guard":all(x["task_pass"] for x in tasks.values())}

def summarize(records,required):
    summaries={}
    for task in TASKS:
        rows=[r["tasks"][task] for r in records]
        med=lambda k:float(np.median([x[k] for x in rows]))
        s={f"median_{k}":med(k) for k in ["full_metric","d_effect","c_effect","r_effect","sham_gap","d_specificity","c_specificity","r_specificity","iso_gap","iso_action_agreement"]}
        s["pass"]=bool(
            s["median_full_metric"]>=0.90 and
            s["median_d_effect"]>=0.20 and s["median_c_effect"]>=0.20 and s["median_r_effect"]>=0.20 and
            s["median_sham_gap"]<=0.02 and
            s["median_d_specificity"]>=0.18 and s["median_c_specificity"]>=0.18 and s["median_r_specificity"]>=0.18 and
            s["median_iso_gap"]<=ISO_TOL and s["median_iso_action_agreement"]>=1-ISO_TOL
        )
        summaries[task]=s
    guards=sum(r["seed_guard"] for r in records)
    passed=all(x["pass"] for x in summaries.values()) and guards>=required
    return {"task_summaries":summaries,"seed_guard_count":int(guards),"seed_guard_required":int(required),"pass":bool(passed)}

def metadata():
    try: ver=importlib.metadata.version("popgym")
    except importlib.metadata.PackageNotFoundError: ver="UNKNOWN"
    return {"repository":"proroklab/popgym","commit":POPGYM_COMMIT,"installed_version":ver,"tasks":list(TASKS)}

def run(mode,output):
    if mode=="smoke": seeds=[SMOKE_SEED]; required=0
    elif mode=="development": seeds=DEV_SEEDS; required=13
    else: seeds=CONF_SEEDS; required=28
    rec=[evaluate_seed(s) for s in seeds]
    summ=summarize(rec,required)
    if mode=="development":
        resolution="R52_DEVELOPMENT_AUTHORIZE_CONFIRM" if summ["pass"] else "R52_DEVELOPMENT_FAIL_NO_CONFIRM"
    elif mode=="confirmatory":
        resolution="R52_BOUNDED_DCR_NECESSITY_PASS" if summ["pass"] else "R52_BOUNDED_DCR_NECESSITY_FAIL"
    else: resolution="R52_ENGINEERING_SMOKE"
    out={
        "campaign":"R52 bounded minimality challenge",
        "mode":mode,"resolution":resolution,
        "authorize_confirm":bool(summ["pass"]) if mode=="development" else None,
        "source":metadata(),"seeds":seeds,"episodes_per_seed":EPISODES_PER_SEED,
        "summary":summ,"records":rec,
        "boundaries":{
            "claim_scope":"bounded necessity across four tested external task families",
            "global_minimality":"NOT_ESTABLISHED",
            "universal_sufficiency":"NOT_ESTABLISHED",
            "unique_decomposition":"NOT_ESTABLISHED",
            "neural_learnability":"NOT_ESTABLISHED",
            "POLAR_superiority":"NOT_ESTABLISHED",
            "E6b":"OPEN","E7":"OPEN","consciousness":"NOT_ESTABLISHED",
            "core_version":"POLAR Core v1.1 unchanged"
        }
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"resolution":resolution,"summary":summ},indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke","development","confirmatory"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(); run(a.mode,a.output)

if __name__=="__main__":
    main()
