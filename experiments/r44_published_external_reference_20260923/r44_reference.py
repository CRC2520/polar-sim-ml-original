#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

DEV_SEEDS=list(range(2156001,2156009))
CONF_SEEDS=list(range(2157001,2157021))
EPISODES=4
PAPER_REFERENCE={
    "Easy":{"LSTM_mean":1.0,"LSTM_sd":0.0,"MLP_mean":0.722,"MLP_sd":0.001},
    "Medium":{"LSTM_mean":1.0,"LSTM_sd":0.0,"MLP_mean":0.398,"MLP_sd":0.006},
    "Hard":{"LSTM_mean":1.0,"LSTM_sd":0.0,"MLP_mean":0.265,"MLP_sd":0.002},
}

def cartpole_step(s,u,p,dt=0.02):
    x,xd,th,thd=map(float,s); mc,mp,l,g=p["mc"],p["mp"],p["l"],9.8
    force=float(np.clip(u,-10,10)); total=mc+mp; poleml=mp*l
    st,ct=np.sin(th),np.cos(th)
    temp=(force+poleml*thd*thd*st)/total
    thacc=(g*st-ct*temp)/(l*(4.0/3.0-mp*ct*ct/total))
    xacc=temp-poleml*thacc*ct/total
    return np.array([x+dt*xd,xd+dt*xacc,th+dt*thd,thd+dt*thacc],float)

def numeric_linearization():
    p={"mc":1.0,"mp":0.10,"l":0.50}; n=4; eps=1e-5
    s0=np.zeros(n); f0=cartpole_step(s0,0.0,p)
    A=np.zeros((n,n))
    for i in range(n):
        ss=s0.copy(); ss[i]+=eps
        A[:,i]=(cartpole_step(ss,0.0,p)-f0)/eps
    B=((cartpole_step(s0,eps,p)-f0)/eps)[:,None]
    return A,B

def lqr_gain():
    A,B=numeric_linearization()
    Q=np.diag([0.25,0.05,8.0,0.5]); R=np.array([[0.02]])
    P=solve_discrete_are(A,B,Q,R)
    return np.linalg.solve(B.T@P@B+R,B.T@P@A)

K_NOM=lqr_gain()
P_ISO=np.diag([-1.0,1.0,1.0,-1.0])
K_ISO=K_NOM@P_ISO.T
CONTROLLERS=["CORE_C","GENERIC_ISO","NO_C_CURRENT","ORACLE_STATE"]

def make_env(level):
    import popgym
    cls=getattr(popgym,f"StatelessCartPole{level}")
    env=cls()
    shape=tuple(env.observation_space.shape)
    if shape!=(2,):
        raise RuntimeError(f"historical observation contract changed: {shape}")
    if getattr(env.action_space,"n",None)!=2:
        raise RuntimeError("historical action contract changed")
    return env

def reset_env(env,seed):
    q=env.reset(seed=int(seed))
    if isinstance(q,tuple):
        return np.asarray(q[0],float)
    return np.asarray(q,float)

def estimate(obs,prev_obs,prev_v,use_history=True):
    x,theta=float(obs[0]),float(obs[1])
    if (not use_history) or prev_obs is None:
        v=np.zeros(2)
    else:
        raw=np.array([(x-float(prev_obs[0]))/0.02,(theta-float(prev_obs[1]))/0.02])
        raw=np.clip(raw,[-6.0,-12.0],[6.0,12.0])
        v=.65*np.asarray(prev_v,float)+.35*raw
    return np.array([x,v[0],theta,v[1]],float),v

def action(env,obs,prev_obs,prev_v,controller):
    if controller=="ORACLE_STATE":
        state=np.asarray(env.get_state(),float).reshape(-1)
        u=float(-(K_NOM@state)[0])
        return (1 if u>=0 else 0),prev_v
    state,v=estimate(obs,prev_obs,prev_v,controller!="NO_C_CURRENT")
    if controller=="GENERIC_ISO":
        u=float(-(K_ISO@(P_ISO@state))[0])
    else:
        u=float(-(K_NOM@state)[0])
    return (1 if u>=0 else 0),v

def run_episode(level,controller,seed):
    env=make_env(level)
    obs=reset_env(env,seed)
    prev_obs=None; prev_v=np.zeros(2); steps=0; rew=0.0
    max_steps=int(env.max_episode_length)
    while True:
        a,new_v=action(env,obs,prev_obs,prev_v,controller)
        q=env.step(a)
        if len(q)!=4:
            raise RuntimeError("expected historical gym 4-tuple step API")
        nxt,r,done,info=q
        steps+=1; rew+=float(r)
        prev_obs=obs.copy(); prev_v=np.asarray(new_v,float).copy(); obs=np.asarray(nxt,float)
        if done or steps>=max_steps:
            break
    env.close()
    return {"steps":steps,"max_steps":max_steps,"score":steps/max_steps,"reward":rew}

def eval_seed(seed,level):
    rows={}
    for controller in CONTROLLERS:
        vals=[]
        for ep in range(EPISODES):
            vals.append(run_episode(level,controller,seed*1000+ep))
        rows[controller]={
            "score":float(np.mean([x["score"] for x in vals])),
            "reward":float(np.mean([x["reward"] for x in vals])),
            "full_fraction":float(np.mean([x["steps"]==x["max_steps"] for x in vals])),
        }
    return {
        "seed":int(seed),
        "level":level,
        "controllers":rows,
        "core_score":rows["CORE_C"]["score"],
        "noC_score":rows["NO_C_CURRENT"]["score"],
        "generic_score":rows["GENERIC_ISO"]["score"],
        "oracle_score":rows["ORACLE_STATE"]["score"],
        "core_minus_noC":rows["CORE_C"]["score"]-rows["NO_C_CURRENT"]["score"],
        "iso_gap":abs(rows["CORE_C"]["score"]-rows["GENERIC_ISO"]["score"]),
        "paper_lstm_reference":PAPER_REFERENCE[level]["LSTM_mean"],
        "paper_mlp_reference":PAPER_REFERENCE[level]["MLP_mean"],
        "paper_lstm_gap":abs(PAPER_REFERENCE[level]["LSTM_mean"]-rows["CORE_C"]["score"]),
    }

def summarize(records,level,dev=False):
    med=lambda k:float(np.median([r[k] for r in records]))
    out={
        "level":level,
        "n":len(records),
        "median_core_score":med("core_score"),
        "median_core_minus_noC":med("core_minus_noC"),
        "median_iso_gap":med("iso_gap"),
        "median_oracle_score":med("oracle_score"),
        "median_paper_lstm_gap":med("paper_lstm_gap"),
        "mean_full_fraction":float(np.mean([r["controllers"]["CORE_C"]["full_fraction"] for r in records])),
        "paper_reference":PAPER_REFERENCE[level],
    }
    if dev:
        count=sum(
            r["core_score"]>=.90 and
            r["core_minus_noC"]>=.15 and
            r["iso_gap"]<=1e-12
            for r in records
        )
        out["seed_guard_count"]=count; out["seed_guard_required"]=7
        out["pass"]=bool(
            out["median_core_score"]>=.95 and
            out["median_core_minus_noC"]>=.20 and
            out["median_iso_gap"]<=1e-12 and
            out["median_oracle_score"]>=.99 and
            count>=7
        )
    else:
        count=sum(
            r["core_score"]>=.90 and
            r["core_minus_noC"]>=.15 and
            r["iso_gap"]<=1e-12
            for r in records
        )
        out["seed_guard_count"]=count; out["seed_guard_required"]=18
        out["pass"]=bool(
            out["median_core_score"]>=.95 and
            out["median_core_minus_noC"]>=.20 and
            out["median_iso_gap"]<=1e-12 and
            out["median_oracle_score"]>=.99 and
            out["median_paper_lstm_gap"]<=.05 and
            count>=18
        )
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["dev","confirm"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()

    if a.mode=="dev":
        records=[eval_seed(s,"Easy") for s in DEV_SEEDS]
        summary={"Easy":summarize(records,"Easy",True)}
        resolution="R44_DEVELOPMENT_AUTHORIZE_CONFIRM" if summary["Easy"]["pass"] else "R44_DEVELOPMENT_FAIL_NO_CONFIRM"
    else:
        records=[]
        for level in ["Medium","Hard"]:
            records.extend(eval_seed(s,level) for s in CONF_SEEDS)
        summary={
            level:summarize([r for r in records if r["level"]==level],level,False)
            for level in ["Medium","Hard"]
        }
        resolution="R44_PUBLISHED_REFERENCE_COMPATIBILITY_PASS" if all(x["pass"] for x in summary.values()) else "R44_PUBLISHED_REFERENCE_COMPATIBILITY_FAIL"

    out={
        "campaign":"R44 published external-reference compatibility",
        "mode":a.mode,
        "resolution":resolution,
        "external_environment_commit":"proroklab/popgym@e397e5e",
        "paper_reference":{
            "source":"Morad et al. ICLR 2023 Appendix Table 3",
            "metric":"published MMER/max-training reward, not paired held-out evaluation",
            "values":PAPER_REFERENCE
        },
        "development_seeds":DEV_SEEDS,
        "confirmatory_seeds":CONF_SEEDS,
        "episodes_per_seed":EPISODES,
        "summary":summary,
        "records":records,
        "boundaries":{
            "comparison_type":"benchmark-ceiling compatibility, not paired checkpoint noninferiority",
            "POLAR_superiority":"NOT_ESTABLISHED",
            "E6b":"OPEN",
            "E7":"OPEN",
            "D_or_R":"NOT_TESTED",
            "universal_recurrence_benefit":"NOT_ESTABLISHED",
            "consciousness":"NOT_ESTABLISHED",
            "core_version":"POLAR Core v1.1 unchanged"
        }
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2,sort_keys=True))

if __name__=="__main__":
    main()
