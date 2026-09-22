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
CONTROLLERS=["CORE_C","GENERIC_ISO","NO_C_CURRENT","ORACLE_STATE"]

def nominal_gain():
    p={"mc":1.0,"mp":0.10,"l":0.50}
    A,B=base.numeric_linearization(base.cartpole_step,4,p)
    Q=np.diag([0.25,0.05,8.0,0.5])
    R=np.array([[0.02]])
    K=base.infinite_lqr_gain(A,B,Q,R)
    if K is None:
        raise RuntimeError("nominal CartPole LQR failed")
    return K

K_NOM=nominal_gain()
P_ISO=np.diag([-1.0,1.0,1.0,-1.0])
K_ISO=K_NOM@P_ISO.T

def estimate_state(obs,prev_obs,prev_v,controller):
    obs=np.asarray(obs,dtype=float).reshape(-1)
    if obs.shape!=(2,):
        raise RuntimeError(f"unexpected POPGym observation shape {obs.shape}")
    x,theta=float(obs[0]),float(obs[1])
    if controller=="NO_C_CURRENT" or prev_obs is None:
        v=np.zeros(2,dtype=float)
    else:
        raw=np.array([(x-float(prev_obs[0]))/0.02,(theta-float(prev_obs[1]))/0.02])
        raw=np.clip(raw,[-6.0,-12.0],[6.0,12.0])
        v=.65*np.asarray(prev_v,dtype=float)+.35*raw
    return np.array([x,v[0],theta,v[1]],dtype=float),v

def policy_action(env,obs,prev_obs,prev_v,controller):
    if controller=="ORACLE_STATE":
        state=np.asarray(env.get_state(),dtype=float).reshape(-1)
        if state.shape!=(4,):
            raise RuntimeError(f"unexpected hidden state shape {state.shape}")
        u=float(-(K_NOM@state)[0])
        return (1 if u>=0 else 0),prev_v
    state,v=estimate_state(obs,prev_obs,prev_v,controller)
    if controller=="GENERIC_ISO":
        z=P_ISO@state
        u=float(-(K_ISO@z)[0])
    else:
        u=float(-(K_NOM@state)[0])
    return (1 if u>=0 else 0),v

def make_env(name):
    import popgym
    cls=getattr(popgym.envs,name)
    env=cls()
    # Frozen contract checks: third-party task must remain exactly 2D partial observation.
    if tuple(env.observation_space.shape)!=(2,):
        raise RuntimeError(f"{name} observation contract changed: {env.observation_space}")
    if getattr(env.action_space,"n",None)!=2:
        raise RuntimeError(f"{name} action contract changed: {env.action_space}")
    if not hasattr(env,"get_state"):
        raise RuntimeError(f"{name} missing evaluator state API")
    return env

def run_episode(env_name,controller,episode_seed):
    env=make_env(env_name)
    obs,_=env.reset(seed=int(episode_seed))
    max_steps=int(env.max_episode_length)
    prev_obs=None
    prev_v=np.zeros(2,dtype=float)
    steps=0
    total_reward=0.0
    while True:
        action,new_v=policy_action(env,obs,prev_obs,prev_v,controller)
        next_obs,reward,terminated,truncated,_=env.step(action)
        total_reward+=float(reward)
        steps+=1
        prev_obs=np.asarray(obs,dtype=float).copy()
        prev_v=np.asarray(new_v,dtype=float).copy()
        obs=next_obs
        if terminated or truncated or steps>=max_steps:
            break
    env.close()
    return {
        "steps":int(steps),
        "max_steps":int(max_steps),
        "duration_score":float(steps/max_steps),
        "environment_reward":float(total_reward),
    }

def eval_seed(seed,episodes):
    rows={}
    for ei,env_name in enumerate(ENV_NAMES):
        rows[env_name]={}
        for controller in CONTROLLERS:
            vals=[]
            for ep in range(episodes):
                episode_seed=seed*100000+ei*1000+ep
                vals.append(run_episode(env_name,controller,episode_seed))
            rows[env_name][controller]={
                "duration_score":float(np.mean([v["duration_score"] for v in vals])),
                "reward":float(np.mean([v["environment_reward"] for v in vals])),
                "episodes":episodes,
                "max_steps":int(vals[0]["max_steps"]),
            }
    def avg(controller,key="duration_score"):
        return float(np.mean([rows[e][controller][key] for e in ENV_NAMES]))
    core=avg("CORE_C"); generic=avg("GENERIC_ISO"); noc=avg("NO_C_CURRENT"); oracle=avg("ORACLE_STATE")
    return {
        "seed":int(seed),
        "environments":rows,
        "core_score":core,
        "generic_score":generic,
        "noC_score":noc,
        "oracle_score":oracle,
        "core_minus_noC":float(core-noc),
        "iso_gap":float(abs(core-generic)),
        "min_env_core":float(min(rows[e]["CORE_C"]["duration_score"] for e in ENV_NAMES)),
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
        "pass":bool(all(checks.values()) and count>=required),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--criteria",required=True)
    ap.add_argument("--mode",choices=["dev","confirm"],required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    c=json.loads(Path(args.criteria).read_text())
    seeds=c["development_seeds"] if args.mode=="dev" else c["confirmatory_seeds"]
    records=[eval_seed(s,c["episodes_per_environment"]) for s in seeds]
    required=c["seed_guard_required_dev"] if args.mode=="dev" else c["seed_guard_required_confirm"]
    out={
        "campaign":c["campaign"],
        "mode":args.mode,
        "popgym_version":c["dependency"]["version"],
        "environment_classes":ENV_NAMES,
        "controllers":CONTROLLERS,
        "records":records,
        "adjudication":adjudicate(records,c,required),
        "boundaries":c["fixed_boundaries"],
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["adjudication"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
