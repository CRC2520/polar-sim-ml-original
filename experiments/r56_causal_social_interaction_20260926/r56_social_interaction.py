#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

HORIZON=6000
BLOCK_LEN=150
BASE_SCHEDULE=[0,1,2,3,0,2,1,3]
BLOCKS=HORIZON//BLOCK_LEN
SCHEDULE=(BASE_SCHEDULE*((BLOCKS+len(BASE_SCHEDULE)-1)//len(BASE_SCHEDULE)))[:BLOCKS]

COOPERATE=0
DEFECT=1
FUTURE_WEIGHT=1.0
EXPLORE_N=12

# responsive partner parameters (P3 is non-responsive)
THRESHOLD=np.array([0.45,0.35,0.62,0.50],float)
COOP_GAIN=np.array([0.18,0.28,0.12,0.0],float)
DEFECT_LOSS=np.array([0.35,0.18,0.48,0.0],float)
TEMP=np.array([0.075,0.085,0.070,1.0],float)
INITIAL_TRUST=np.array([0.62,0.70,0.58,0.50],float)
P3_COOP=0.15

CONDITIONS=("FULL_SOCIAL","NO_PARTNER_ID","NO_CROSS_VISIT_HISTORY","WRONG_SOCIAL_RELATION","MYOPIC","SHAM_STATE")

DEV_SEEDS=list(range(2256001,2256017))
CONF_SEEDS=list(range(2257001,2257033))


def stable_seed(seed:int,label:str)->int:
    raw=hashlib.sha256(f"R56|{int(seed)}|{label}".encode()).digest()
    return int.from_bytes(raw[:8],"little") & 0xffffffff


class SocialWorld:
    def __init__(self,seed:int):
        rng=np.random.default_rng(stable_seed(seed,"world"))
        self.u=rng.random((HORIZON,4))
        self.initial_noise=rng.normal(0,0.025,4)

    def partner(self,t:int)->int:
        return int(SCHEDULE[t//BLOCK_LEN])


def coop_probability(partner:int,trust:float)->float:
    if partner==3:
        return P3_COOP
    z=(trust-THRESHOLD[partner])/TEMP[partner]
    p=1.0/(1.0+np.exp(-z))
    return float(np.clip(p,0.03,0.97))


def immediate_reward(partner_coop:bool,action:int)->float:
    if partner_coop:
        return 1.0 if action==COOPERATE else 1.30
    return -0.20 if action==COOPERATE else 0.05


class SocialAgent:
    def __init__(self,condition:str):
        self.condition=condition
        n_models=1 if condition=="NO_PARTNER_ID" else 4
        # Beta-style prior: cooperation probability 0.6 after either action
        self.success=np.full((n_models,2),3.0,float)
        self.total=np.full((n_models,2),5.0,float)
        self.last_action_by_partner=[None]*4
        self.prev_reward=0.0
        self.action_counts=np.zeros((4,2),int)
        self.sham=np.zeros(12,float)

    def model_index(self,partner:int)->int:
        return 0 if self.condition=="NO_PARTNER_ID" else int(partner)

    def on_block_start(self,partner:int):
        if self.condition=="NO_CROSS_VISIT_HISTORY":
            idx=self.model_index(partner)
            self.success[idx,:]=3.0
            self.total[idx,:]=5.0
            self.last_action_by_partner[partner]=None

    def observe_partner(self,partner:int,partner_coop:bool):
        last=self.last_action_by_partner[partner]
        if last is None:
            return
        idx=self.model_index(partner)
        self.total[idx,last]+=1.0
        self.success[idx,last]+=float(partner_coop)

    def q(self,partner:int,action:int)->float:
        if self.condition=="WRONG_SOCIAL_RELATION":
            idx=self.model_index((partner+1)%4)
        else:
            idx=self.model_index(partner)
        return float(self.success[idx,action]/self.total[idx,action])

    def model_observations(self,partner:int)->float:
        idx=self.model_index(partner)
        return float(np.sum(self.total[idx]-5.0))

    def choose(self,partner:int,partner_coop:bool)->int:
        if self.condition=="MYOPIC":
            r0=immediate_reward(partner_coop,COOPERATE)
            r1=immediate_reward(partner_coop,DEFECT)
            return COOPERATE if r0>r1 else DEFECT

        # Exploration is tied to the true partner's model/history, not a wrong
        # queried relation, so WRONG_RELATION isolates lookup rather than data.
        if self.model_observations(partner)<EXPLORE_N:
            n=int(self.model_observations(partner))
            return COOPERATE if n%2==0 else DEFECT

        values=[]
        for a in (COOPERATE,DEFECT):
            imm=immediate_reward(partner_coop,a)
            q=self.q(partner,a)
            continuation=q*1.0+(1.0-q)*0.05
            values.append(imm+FUTURE_WEIGHT*continuation)
        return int(np.argmax(values))

    def record_action(self,partner:int,action:int,reward:float):
        self.last_action_by_partner[partner]=int(action)
        self.prev_reward=float(reward)
        self.action_counts[partner,action]+=1
        if self.condition=="SHAM_STATE":
            self.sham*=0.99
            self.sham[partner]+=0.01
            self.sham[4+action]+=0.01*reward


def run_condition(seed:int,condition:str)->dict:
    world=SocialWorld(seed)
    agent=SocialAgent(condition)
    trust=np.clip(INITIAL_TRUST+world.initial_noise,0.1,0.9)

    rewards=[]
    partner_coops=[]
    first30_revisit=[]
    last60=[]
    block_rewards=[]
    block_partner_seen={p:0 for p in range(4)}
    responsive_agent_actions=[]
    p3_agent_actions=[]

    prev_block=-1
    cur_block_rewards=[]

    for t in range(HORIZON):
        block=t//BLOCK_LEN
        p=world.partner(t)
        pos=t%BLOCK_LEN

        if block!=prev_block:
            if prev_block>=0:
                block_rewards.append(cur_block_rewards)
            cur_block_rewards=[]
            agent.on_block_start(p)
            block_partner_seen[p]+=1
            prev_block=block

        prob=coop_probability(p,trust[p])
        partner_coop=bool(world.u[t,p]<prob)
        agent.observe_partner(p,partner_coop)
        action=agent.choose(p,partner_coop)
        reward=immediate_reward(partner_coop,action)

        # action causally changes future social state only for responsive partners
        if p<3:
            if action==COOPERATE:
                trust[p]+=COOP_GAIN[p]*(1.0-trust[p])
            else:
                trust[p]-=DEFECT_LOSS[p]*trust[p]
            trust[p]=float(np.clip(trust[p],0.0,1.0))

        agent.record_action(p,action,reward)
        rewards.append(reward)
        partner_coops.append(float(partner_coop))
        cur_block_rewards.append(reward)

        if block_partner_seen[p]>1 and pos<30:
            first30_revisit.append(reward)
        if pos>=BLOCK_LEN-60:
            last60.append(reward)

        if p<3:
            responsive_agent_actions.append(float(action==COOPERATE))
        else:
            p3_agent_actions.append(float(action==COOPERATE))

    block_rewards.append(cur_block_rewards)

    contrasts=[]
    if condition=="NO_PARTNER_ID":
        contrasts=[agent.q(0,COOPERATE)-agent.q(0,DEFECT)]
    else:
        for p in range(3):
            # For FULL/SHAM/NO_HISTORY this is the true model; WRONG_RELATION's
            # endpoint is diagnostic and not used in FULL guard.
            idx=agent.model_index(p)
            contrasts.append(float(agent.success[idx,COOPERATE]/agent.total[idx,COOPERATE]-agent.success[idx,DEFECT]/agent.total[idx,DEFECT]))

    return {
        "condition":condition,
        "mean_reward":float(np.mean(rewards)),
        "total_reward":float(np.sum(rewards)),
        "agent_cooperation_fraction":float(np.mean([a==COOPERATE for p in range(4) for a in []])) if False else float(np.sum(agent.action_counts[:,COOPERATE])/HORIZON),
        "partner_cooperation_fraction":float(np.mean(partner_coops)),
        "revisit_first30_reward":float(np.mean(first30_revisit)) if first30_revisit else 0.0,
        "block_last60_reward":float(np.mean(last60)) if last60 else 0.0,
        "responsive_agent_cooperation":float(np.mean(responsive_agent_actions)),
        "p3_agent_cooperation":float(np.mean(p3_agent_actions)),
        "responsive_action_response_contrast":float(np.mean(contrasts)),
        "final_trust":[float(x) for x in trust],
    }


def evaluate_seed(seed:int)->dict:
    c={name:run_condition(seed,name) for name in CONDITIONS}
    full=c["FULL_SOCIAL"]
    out={
        "seed":int(seed),
        "conditions":c,
        "full_mean_reward":full["mean_reward"],
        "full_revisit_reward":full["revisit_first30_reward"],
        "full_responsive_contrast":full["responsive_action_response_contrast"],
        "full_responsive_cooperation":full["responsive_agent_cooperation"],
        "full_p3_cooperation":full["p3_agent_cooperation"],
        "identity_effect":full["mean_reward"]-c["NO_PARTNER_ID"]["mean_reward"],
        "history_effect":full["mean_reward"]-c["NO_CROSS_VISIT_HISTORY"]["mean_reward"],
        "relation_effect":full["mean_reward"]-c["WRONG_SOCIAL_RELATION"]["mean_reward"],
        "long_horizon_effect":full["mean_reward"]-c["MYOPIC"]["mean_reward"],
        "sham_gap":abs(full["mean_reward"]-c["SHAM_STATE"]["mean_reward"]),
    }
    out["seed_guard"]=bool(
        out["full_mean_reward"]>=0.76 and
        out["full_revisit_reward"]>=0.70 and
        out["full_responsive_contrast"]>=0.30 and
        out["full_responsive_cooperation"]>=0.65 and
        out["full_p3_cooperation"]<=0.35 and
        out["identity_effect"]>=0.05 and
        out["history_effect"]>=0.04 and
        out["relation_effect"]>=0.07 and
        out["long_horizon_effect"]>=0.18 and
        out["sham_gap"]<=1e-12
    )
    return out


def summarize(rows:List[dict],required:int)->dict:
    med=lambda k:float(np.median([r[k] for r in rows]))
    out={
        "n":len(rows),
        "seed_guard_count":sum(bool(r["seed_guard"]) for r in rows),
        "seed_guard_required":required,
        "median_full_reward":med("full_mean_reward"),
        "median_revisit_reward":med("full_revisit_reward"),
        "median_responsive_contrast":med("full_responsive_contrast"),
        "median_responsive_cooperation":med("full_responsive_cooperation"),
        "median_p3_cooperation":med("full_p3_cooperation"),
        "median_identity_effect":med("identity_effect"),
        "median_history_effect":med("history_effect"),
        "median_relation_effect":med("relation_effect"),
        "median_long_horizon_effect":med("long_horizon_effect"),
        "median_sham_gap":med("sham_gap"),
    }
    out["pass"]=bool(
        out["seed_guard_count"]>=required and
        out["median_full_reward"]>=0.78 and
        out["median_identity_effect"]>=0.06 and
        out["median_history_effect"]>=0.05 and
        out["median_relation_effect"]>=0.08 and
        out["median_long_horizon_effect"]>=0.20 and
        out["median_sham_gap"]<=1e-12
    )
    return out


def run_panel(lo:int,hi:int,phase:str,output:str):
    seeds=list(range(lo,hi+1))
    rows=[evaluate_seed(s) for s in seeds]
    required=13 if phase=="development" else 28
    s=summarize(rows,required)
    if phase=="development":
        res="R56_DEVELOPMENT_AUTHORIZE_CONFIRM" if s["pass"] else "R56_DEVELOPMENT_FAIL_NO_CONFIRM"
    else:
        res="R56_BOUNDED_CAUSAL_SOCIAL_INTERACTION_PASS" if s["pass"] else "R56_BOUNDED_CAUSAL_SOCIAL_INTERACTION_FAIL"
    out={
        "campaign":"R56 bounded causal social interaction",
        "phase":phase,
        "seeds":seeds,
        "resolution":res,
        "authorize_confirm":bool(s["pass"]) if phase=="development" else None,
        "summary":s,
        "records":rows,
        "boundaries":{
            "social_interaction":"BOUNDED_OPERATIONAL_ONLY",
            "empathy":"NOT_ESTABLISHED",
            "theory_of_mind":"NOT_ESTABLISHED_HUMAN_SENSE",
            "moral_agency":"NOT_ESTABLISHED",
            "intrinsic_values":"NOT_TESTED",
            "consciousness":"NOT_ESTABLISHED",
            "E6b":"OPEN",
            "E7":"NO_FULL_PASS"
        }
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2,sort_keys=True))


def smoke(output:str):
    r=evaluate_seed(2255000)
    out={"campaign":"R56 bounded causal social interaction","mode":"engineering_smoke","scientific_seed":False,"record":r}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke","development","confirmatory"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.mode=="smoke": smoke(a.output)
    elif a.mode=="development": run_panel(2256001,2256016,"development",a.output)
    else: run_panel(2257001,2257032,"confirmatory",a.output)

if __name__=="__main__":
    main()
