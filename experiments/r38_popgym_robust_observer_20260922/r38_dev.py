#!/usr/bin/env python3
import argparse, json, importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r27_path=ROOT/"r27_strong_mpc_20260921"/"r27_strong_mpc.py"
spec=importlib.util.spec_from_file_location("r27base",r27_path)
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

ENV_NAMES=[
    "PositionOnlyCartPoleMedium",
    "PositionOnlyCartPoleHard",
    "NoisyPositionOnlyCartPoleMedium",
    "NoisyPositionOnlyCartPoleHard",
]
VARIANTS=["KF_R01","KF_R04","KF_R09","KF_ADAPT"]
P_NOM={"mc":1.0,"mp":0.10,"l":0.50}
A_NOM,B_NOM=base.numeric_linearization(base.cartpole_step,4,P_NOM)
Q_LQR=np.diag([0.25,0.05,8.0,0.5])
R_LQR=np.array([[0.02]])
K_NOM=base.infinite_lqr_gain(A_NOM,B_NOM,Q_LQR,R_LQR)
if K_NOM is None:
    raise RuntimeError("nominal CartPole LQR failed")
H=np.array([[1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0]])
P_ISO=np.diag([-1.0,1.0,1.0,-1.0])
K_ISO=K_NOM@P_ISO.T

def make_env(name):
    import popgym
    cls=getattr(popgym.envs,name)
    env=cls()
    if tuple(env.observation_space.shape)!=(2,):
        raise RuntimeError(f"{name} observation contract changed")
    if getattr(env.action_space,"n",None)!=2:
        raise RuntimeError(f"{name} action contract changed")
    if not hasattr(env,"get_state"):
        raise RuntimeError(f"{name} missing evaluator state API")
    return env

def force_from_action(a):
    return 10.0 if int(a)==1 else -10.0

class RobustObserver:
    def __init__(self,variant,cfg):
        self.variant=variant
        ocfg=cfg["observer"]
        self.Q=np.diag(np.asarray(ocfg["process_diag"],float))
        self.P0=np.diag(np.asarray(ocfg["initial_cov_diag"],float))
        self.P=self.P0.copy()
        if variant=="KF_ADAPT":
            v=ocfg[variant]
            self.rvar=float(v["initial_measurement_variance"])
            self.rmin=float(v["min_variance"])
            self.rmax=float(v["max_variance"])
            self.rema=float(v["innovation_ema"])
        else:
            self.rvar=float(ocfg[variant]["measurement_variance"])
            self.rmin=self.rmax=self.rvar
            self.rema=0.0
        self.x=None
        self.prev_action=None

    def update(self,obs):
        y=np.asarray(obs,float).reshape(2)
        if self.x is None:
            self.x=np.array([y[0],0.0,y[1],0.0],float)
            self.P=self.P0.copy()
            return self.x.copy()

        xpred=base.cartpole_step(self.x,force_from_action(self.prev_action),P_NOM)
        Ppred=A_NOM@self.P@A_NOM.T+self.Q
        innov=y-H@xpred
        if self.variant=="KF_ADAPT":
            empirical=float(np.mean(innov*innov))
            self.rvar=float(np.clip((1-self.rema)*self.rvar+self.rema*empirical,self.rmin,self.rmax))
        Rm=np.eye(2)*self.rvar
        S=H@Ppred@H.T+Rm
        Kg=Ppred@H.T@np.linalg.inv(S)
        self.x=xpred+Kg@innov
        self.P=(np.eye(4)-Kg@H)@Ppred
        return self.x.copy()

    def set_action(self,a):
        self.prev_action=int(a)

def no_c_state(obs):
    o=np.asarray(obs,float).reshape(2)
    return np.array([o[0],0.0,o[1],0.0],float)

def action_from_state(state,iso=False):
    if iso:
        z=P_ISO@state
        u=float(-(K_ISO@z)[0])
    else:
        u=float(-(K_NOM@state)[0])
    return 1 if u>=0 else 0

def run_episode(env_name,controller,episode_seed,variant,cfg):
    env=make_env(env_name)
    obs,_=env.reset(seed=int(episode_seed))
    max_steps=int(env.max_episode_length)
    observer=None if controller in ("NO_C_CURRENT","ORACLE_STATE") else RobustObserver(variant,cfg)
    steps=0
    reward_sum=0.0
    while True:
        if controller=="ORACLE_STATE":
            state=np.asarray(env.get_state(),float).reshape(4)
            action=action_from_state(state,False)
        elif controller=="NO_C_CURRENT":
            action=action_from_state(no_c_state(obs),False)
        else:
            state=observer.update(obs)
            action=action_from_state(state,controller=="GENERIC_ISO")
            observer.set_action(action)
        obs,reward,terminated,truncated,_=env.step(action)
        reward_sum+=float(reward)
        steps+=1
        if terminated or truncated or steps>=max_steps:
            break
    env.close()
    return float(steps/max_steps),float(reward_sum),max_steps

def baseline_scores(seed,episodes):
    rows={}
    for ei,env_name in enumerate(ENV_NAMES):
        rows[env_name]={}
        for controller in ("NO_C_CURRENT","ORACLE_STATE"):
            vals=[]
            for ep in range(episodes):
                epseed=seed*100000+ei*1000+ep
                d,r,m=run_episode(env_name,controller,epseed,"KF_R04",GLOBAL_CFG)
                vals.append((d,r))
            rows[env_name][controller]=float(np.mean([v[0] for v in vals]))
    return rows

def variant_scores(seed,variant,episodes,baseline):
    rows={}
    for ei,env_name in enumerate(ENV_NAMES):
        rows[env_name]=dict(baseline[env_name])
        for controller in ("CORE_C","GENERIC_ISO"):
            vals=[]
            for ep in range(episodes):
                epseed=seed*100000+ei*1000+ep
                d,r,m=run_episode(env_name,controller,epseed,variant,GLOBAL_CFG)
                vals.append((d,r))
            rows[env_name][controller]=float(np.mean([v[0] for v in vals]))
    core=float(np.mean([rows[e]["CORE_C"] for e in ENV_NAMES]))
    generic=float(np.mean([rows[e]["GENERIC_ISO"] for e in ENV_NAMES]))
    noc=float(np.mean([rows[e]["NO_C_CURRENT"] for e in ENV_NAMES]))
    oracle=float(np.mean([rows[e]["ORACLE_STATE"] for e in ENV_NAMES]))
    return {
        "seed":int(seed),
        "variant":variant,
        "environments":rows,
        "core_score":core,
        "core_minus_noC":float(core-noc),
        "iso_gap":float(abs(core-generic)),
        "oracle_score":oracle,
        "min_env_core":float(min(rows[e]["CORE_C"] for e in ENV_NAMES))
    }

def adjudicate(records,c,required):
    med=lambda k: float(np.median([r[k] for r in records]))
    m={k:med(k) for k in ["core_score","core_minus_noC","iso_gap","oracle_score","min_env_core"]}
    mc=c["medians"]
    checks={
        "core_score":m["core_score"]>=mc["core_score_min"],
        "core_minus_noC":m["core_minus_noC"]>=mc["core_minus_noC_min"],
        "iso_gap":m["iso_gap"]<=mc["iso_gap_max"],
        "oracle_score":m["oracle_score"]>=mc["oracle_score_min"],
    }
    sg=c["seed_guard"]
    count=sum(
        r["core_score"]>=sg["core_score_min"] and
        r["core_minus_noC"]>=sg["core_minus_noC_min"] and
        r["iso_gap"]<=sg["iso_gap_max"]
        for r in records
    )
    return {
        "medians":m,
        "median_checks":checks,
        "seed_guard_pass_count":int(count),
        "seed_guard_required":int(required),
        "eligible":bool(all(checks.values()) and count>=required)
    }

def select_variant(results,c):
    order={v:i for i,v in enumerate(c["selection_order"])}
    eligible=[]
    for v in c["variants"]:
        a=results[v]["adjudication"]
        if a["eligible"]:
            eligible.append(v)
    if not eligible:
        return None
    eligible.sort(key=lambda v:(
        -results[v]["adjudication"]["medians"]["min_env_core"],
        -results[v]["adjudication"]["medians"]["core_score"],
        -results[v]["adjudication"]["medians"]["core_minus_noC"],
        order[v]
    ))
    return eligible[0]

def main():
    global GLOBAL_CFG
    ap=argparse.ArgumentParser()
    ap.add_argument("--criteria",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    c=json.loads(Path(args.criteria).read_text())
    GLOBAL_CFG=c
    baselines={s:baseline_scores(s,c["episodes_per_environment"]) for s in c["development_seeds"]}
    variants={}
    for v in c["variants"]:
        rec=[variant_scores(s,v,c["episodes_per_environment"],baselines[s]) for s in c["development_seeds"]]
        variants[v]={"records":rec,"adjudication":adjudicate(rec,c,c["development_guard_required"])}
    selected=select_variant(variants,c)
    out={
        "campaign":c["campaign"],
        "mode":"development",
        "seeds":c["development_seeds"],
        "variants":variants,
        "selected_variant":selected,
        "confirmatory_authorized":selected is not None,
        "confirmatory_seeds_status":"UNOPENED_2047001_2047012",
        "boundaries":c["boundaries"]
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"selected_variant":selected,"variants":{v:variants[v]["adjudication"] for v in variants}},indent=2,sort_keys=True))

if __name__=="__main__":
    main()
