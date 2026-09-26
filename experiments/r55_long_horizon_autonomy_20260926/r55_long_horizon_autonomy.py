#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HORIZON=12000
REGIME_LEN=600
N_ZONES=4
N_GOALS=3
PASSIVE_DRAIN=np.array([0.0065,0.0030,0.0018],float)
SATIATION=np.array([0.82,0.85,0.75],float)
EMERGENCY=np.array([0.25,0.30,0.0],float)
GOAL_MIN_STEPS=10
SCRIPT_STEPS=18
TRAVEL_COST=0.012
WORK_SWITCH_COST=0.006
EMA_ALPHA=0.28
PRIOR_YIELD=0.025
UCB_BETA=0.012

DEV_SEEDS=list(range(2246001,2246017))
CONF_SEEDS=list(range(2247001,2247033))

# regime x zone x work-mode(resource)
YIELDS=np.array([
    [[0.078,0.014,0.012],[0.016,0.061,0.015],[0.020,0.018,0.069],[0.037,0.033,0.030]],
    [[0.018,0.019,0.066],[0.074,0.017,0.013],[0.030,0.039,0.032],[0.016,0.063,0.017]],
    [[0.031,0.035,0.029],[0.018,0.016,0.071],[0.077,0.015,0.014],[0.015,0.060,0.018]],
    [[0.016,0.062,0.016],[0.034,0.032,0.031],[0.019,0.020,0.068],[0.076,0.015,0.014]],
],float)

CONDITIONS=("FULL_AUTONOMY","NO_GOAL_PERSISTENCE","NO_LEARNED_MODEL","EXTERNAL_SCRIPT","SHAM_STATE")


def stable_seed(seed:int,label:str)->int:
    raw=hashlib.sha256(f"R55|{int(seed)}|{label}".encode()).digest()
    return int.from_bytes(raw[:8],"little") & 0xffffffff


class World:
    def __init__(self,seed:int):
        self.seed=int(seed)
        rng=np.random.default_rng(stable_seed(seed,"world"))
        self.yield_noise=rng.normal(0,0.0035,(HORIZON,N_ZONES,N_GOALS))
        self.hazard_u=rng.random(HORIZON)
        self.hazard_mag=rng.uniform(0.055,0.11,HORIZON)

    def regime(self,t:int)->int:
        return (t//REGIME_LEN)%4

    def yield_for(self,t:int,zone:int,mode:int)->float:
        y=YIELDS[self.regime(t),zone,mode]+self.yield_noise[t,zone,mode]
        return float(max(0.0,y))


@dataclass
class AgentState:
    resources: np.ndarray
    zone: int
    goal: Optional[int]
    goal_age: int
    last_action: int
    last_work_mode: Optional[int]
    last_yield: float
    estimates: np.ndarray
    counts: np.ndarray
    sham: np.ndarray


class Agent:
    def __init__(self,condition:str):
        self.condition=condition
        self.s=AgentState(
            resources=np.array([0.72,0.78,0.62],float),
            zone=0,
            goal=None,
            goal_age=0,
            last_action=5,
            last_work_mode=None,
            last_yield=0.0,
            estimates=np.full((N_ZONES,N_GOALS),PRIOR_YIELD,float),
            counts=np.zeros((N_ZONES,N_GOALS),int),
            sham=np.zeros(1+N_ZONES*N_GOALS,float),
        )
        self.goal_switches=0
        self.work_switches=0
        self.travel_steps=0
        self.pred_errors=[]
        self.goal_trace=[]

    def urgency(self)->np.ndarray:
        r=self.s.resources
        u=np.maximum(0.0,(SATIATION-r)/SATIATION)
        # knowledge exploration receives a mild epistemic bonus when low
        u[2]+=0.15*max(0.0,0.55-r[2])
        return u

    def desired_goal(self,t:int)->int:
        if self.condition=="EXTERNAL_SCRIPT":
            return int((t//SCRIPT_STEPS)%N_GOALS)
        return int(np.argmax(self.urgency()))

    def choose_goal(self,t:int)->int:
        desired=self.desired_goal(t)
        old=self.s.goal

        if self.condition=="NO_GOAL_PERSISTENCE":
            new=desired
        elif self.condition=="EXTERNAL_SCRIPT":
            new=desired
        else:
            emergency=None
            if self.s.resources[0]<EMERGENCY[0]:
                emergency=0
            elif self.s.resources[1]<EMERGENCY[1]:
                emergency=1

            if emergency is not None:
                new=emergency
            elif old is None:
                new=desired
            elif self.s.resources[old]>=SATIATION[old] and self.s.goal_age>=GOAL_MIN_STEPS:
                new=desired
            elif self.s.goal_age<GOAL_MIN_STEPS:
                new=old
            else:
                # hysteresis: switch only when candidate urgency is materially larger
                u=self.urgency()
                new=desired if u[desired] > u[old]+0.06 else old

        if old is not None and new!=old:
            self.goal_switches+=1
        if new==old:
            self.s.goal_age+=1
        else:
            self.s.goal_age=0
        self.s.goal=new
        self.goal_trace.append(int(new))
        return int(new)

    def target_zone(self,goal:int,t:int)->int:
        if self.condition=="NO_LEARNED_MODEL":
            # uniform-prior tie break is intentionally deterministic.
            return 0

        est=self.s.estimates[:,goal]
        cnt=self.s.counts[:,goal]
        bonus=UCB_BETA*np.sqrt(np.log(t+2.0)/(cnt+1.0))
        score=est+bonus
        return int(np.argmax(score))

    def act(self,t:int)->int:
        g=self.choose_goal(t)
        ztar=self.target_zone(g,t)
        z=self.s.zone
        if z!=ztar:
            cw=(ztar-z)%N_ZONES
            ccw=(z-ztar)%N_ZONES
            self.travel_steps+=1
            return 3 if cw<=ccw else 4
        return int(g)  # work-mode action matches goal

    def update_model(self,zone:int,mode:int,observed_yield:float):
        if self.condition=="NO_LEARNED_MODEL":
            self.s.estimates.fill(PRIOR_YIELD)
            self.s.counts.fill(0)
            return
        pred=float(self.s.estimates[zone,mode])
        self.pred_errors.append(abs(pred-observed_yield))
        self.s.estimates[zone,mode]=(1-EMA_ALPHA)*pred+EMA_ALPHA*observed_yield
        self.s.counts[zone,mode]+=1

    def step(self,world:World,t:int,action:int):
        r=self.s.resources
        r-=PASSIVE_DRAIN

        realized=0.0
        if action in (0,1,2):
            mode=int(action)
            if self.s.last_work_mode is not None and mode!=self.s.last_work_mode:
                r[0]-=WORK_SWITCH_COST
                self.work_switches+=1
            realized=world.yield_for(t,self.s.zone,mode)
            r[mode]+=realized
            self.update_model(self.s.zone,mode,realized)
            self.s.last_work_mode=mode
        elif action==3:
            self.s.zone=(self.s.zone+1)%N_ZONES
            r[0]-=TRAVEL_COST
            self.s.last_work_mode=None
        elif action==4:
            self.s.zone=(self.s.zone-1)%N_ZONES
            r[0]-=TRAVEL_COST
            self.s.last_work_mode=None
        elif action==5:
            r[0]+=0.002
            self.s.last_work_mode=None
        else:
            raise ValueError(action)

        # same exogenous hazard stream; knowledge changes exposure probability
        p=0.014*(1.0-0.72*np.clip(r[2],0,1))
        if world.hazard_u[t] < p:
            r[1]-=world.hazard_mag[t]

        np.clip(r,0.0,1.0,out=r)
        self.s.last_action=int(action)
        self.s.last_yield=float(realized)

        if self.condition=="SHAM_STATE":
            # unused state evolves deterministically but never enters policy.
            self.s.sham[0]=0.99*self.s.sham[0]+0.01*float(np.mean(r))
            flat=self.s.estimates.reshape(-1)
            self.s.sham[1:]=0.995*self.s.sham[1:]+0.005*flat


def run_condition(seed:int,condition:str)->dict:
    world=World(seed)
    agent=Agent(condition)
    safety=[]
    viability=[]
    post_shift=[]
    catastrophic=0
    completed=0
    work_actions=[]

    shift_windows=set()
    for s in range(REGIME_LEN,HORIZON,REGIME_LEN):
        shift_windows.update(range(s,min(s+120,HORIZON)))

    for t in range(HORIZON):
        action=agent.act(t)
        agent.step(world,t,action)
        e,i,k=agent.s.resources
        safe=bool(e>=0.15 and i>=0.15)
        safety.append(float(safe))
        viability.append(float(min(e,i)))
        if t in shift_windows:
            post_shift.append(float(safe))
        if action in (0,1,2):
            work_actions.append(action)
        completed=t+1
        if e<=0.05 or i<=0.05:
            catastrophic+=1
            break

    return {
        "condition":condition,
        "steps_completed":int(completed),
        "survival_fraction":float(completed/HORIZON),
        "safety_fraction":float(np.mean(safety)) if safety else 0.0,
        "mean_viability_margin":float(np.mean(viability)) if viability else 0.0,
        "catastrophic_events":int(catastrophic),
        "post_shift_safety":float(np.mean(post_shift)) if post_shift else 0.0,
        "goal_switch_rate":float(agent.goal_switches/max(completed,1)),
        "travel_fraction":float(agent.travel_steps/max(completed,1)),
        "work_mode_switch_rate":float(agent.work_switches/max(len(work_actions),1)),
        "model_prediction_mae":float(np.mean(agent.pred_errors)) if agent.pred_errors else None,
        "final_resources":[float(x) for x in agent.s.resources],
    }


def evaluate_seed(seed:int)->dict:
    cond={c:run_condition(seed,c) for c in CONDITIONS}
    full=cond["FULL_AUTONOMY"]
    out={
        "seed":int(seed),
        "conditions":cond,
        "full_survival_fraction":full["survival_fraction"],
        "full_safety_fraction":full["safety_fraction"],
        "full_mean_viability_margin":full["mean_viability_margin"],
        "full_catastrophic_events":full["catastrophic_events"],
        "full_post_shift_safety":full["post_shift_safety"],
        "full_goal_switch_rate":full["goal_switch_rate"],
        "goal_effect":full["safety_fraction"]-cond["NO_GOAL_PERSISTENCE"]["safety_fraction"],
        "model_effect":full["safety_fraction"]-cond["NO_LEARNED_MODEL"]["safety_fraction"],
        "self_selection_effect":full["safety_fraction"]-cond["EXTERNAL_SCRIPT"]["safety_fraction"],
        "sham_gap":abs(full["safety_fraction"]-cond["SHAM_STATE"]["safety_fraction"]),
    }
    out["seed_guard"]=bool(
        out["full_survival_fraction"]==1.0 and
        out["full_safety_fraction"]>=0.93 and
        out["full_mean_viability_margin"]>=0.32 and
        out["full_catastrophic_events"]==0 and
        out["full_post_shift_safety"]>=0.85 and
        out["full_goal_switch_rate"]<=0.12 and
        out["goal_effect"]>=0.04 and
        out["model_effect"]>=0.10 and
        out["self_selection_effect"]>=0.06 and
        out["sham_gap"]<=1e-12
    )
    return out


def summarize(rows:List[dict],required:int)->dict:
    med=lambda k:float(np.median([r[k] for r in rows]))
    out={
        "n":len(rows),
        "seed_guard_count":sum(bool(r["seed_guard"]) for r in rows),
        "seed_guard_required":required,
        "median_full_safety":med("full_safety_fraction"),
        "median_full_viability":med("full_mean_viability_margin"),
        "median_goal_effect":med("goal_effect"),
        "median_model_effect":med("model_effect"),
        "median_self_selection_effect":med("self_selection_effect"),
        "median_sham_gap":med("sham_gap"),
        "median_post_shift_safety":med("full_post_shift_safety"),
        "median_goal_switch_rate":med("full_goal_switch_rate"),
    }
    out["pass"]=bool(
        out["seed_guard_count"]>=required and
        out["median_full_safety"]>=0.95 and
        out["median_goal_effect"]>=0.06 and
        out["median_model_effect"]>=0.12 and
        out["median_self_selection_effect"]>=0.08 and
        out["median_sham_gap"]<=1e-12
    )
    return out


def run_panel(lo:int,hi:int,phase:str,output:str):
    seeds=list(range(lo,hi+1))
    rows=[evaluate_seed(s) for s in seeds]
    required=13 if phase=="development" else 28
    s=summarize(rows,required)
    if phase=="development":
        res="R55_DEVELOPMENT_AUTHORIZE_CONFIRM" if s["pass"] else "R55_DEVELOPMENT_FAIL_NO_CONFIRM"
    else:
        res="R55_BOUNDED_LONG_HORIZON_AUTONOMY_PASS" if s["pass"] else "R55_BOUNDED_LONG_HORIZON_AUTONOMY_FAIL"
    out={
        "campaign":"R55 bounded long-horizon functional autonomy",
        "phase":phase,
        "seeds":seeds,
        "resolution":res,
        "authorize_confirm":bool(s["pass"]) if phase=="development" else None,
        "summary":s,
        "records":rows,
        "boundaries":{
            "functional_autonomy":"BOUNDED_OPERATIONAL_ONLY",
            "consciousness":"NOT_ESTABLISHED",
            "free_will":"NOT_ESTABLISHED",
            "intrinsic_value_formation":"NOT_TESTED",
            "E6b":"OPEN",
            "E7":"NO_FULL_PASS"
        }
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2,sort_keys=True))


def smoke(output:str):
    r=evaluate_seed(2245000)
    out={"campaign":"R55 bounded long-horizon functional autonomy","mode":"engineering_smoke","scientific_seed":False,"record":r}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke","development","confirmatory"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.mode=="smoke": smoke(a.output)
    elif a.mode=="development": run_panel(2246001,2246016,"development",a.output)
    else: run_panel(2247001,2247032,"confirmatory",a.output)

if __name__=="__main__":
    main()
