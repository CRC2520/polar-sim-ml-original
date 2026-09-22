#!/usr/bin/env python3
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R38_PATH=ROOT/"r38_popgym_robust_observer_20260922"/"r38_dev.py"
R38_CFG_PATH=ROOT/"r38_popgym_robust_observer_20260922"/"R38_DEV_CRITERIA.json"

spec=importlib.util.spec_from_file_location("r38frozen",R38_PATH)
r38=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r38)
R38_CFG=json.loads(R38_CFG_PATH.read_text())

def make_env(name):
    import popgym
    cls=getattr(popgym.envs,name)
    env=cls()
    if tuple(env.observation_space.shape)!=(2,):
        raise RuntimeError(f"{name} observation contract changed: {env.observation_space}")
    if getattr(env.action_space,"n",None)!=2:
        raise RuntimeError(f"{name} action contract changed: {env.action_space}")
    return env

def train_rppo(env_name,seed,cfg,model_dir):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    env=Monitor(make_env(env_name))
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    h=cfg["training"]["recurrent_ppo"]
    model=RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=float(h["learning_rate"]),
        n_steps=int(h["n_steps"]),
        batch_size=int(h["batch_size"]),
        n_epochs=int(h["n_epochs"]),
        gamma=float(h["gamma"]),
        gae_lambda=float(h["gae_lambda"]),
        ent_coef=float(h["ent_coef"]),
        vf_coef=float(h["vf_coef"]),
        policy_kwargs={"lstm_hidden_size":int(h["lstm_hidden_size"])},
        seed=int(seed),
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=int(cfg["training"]["timesteps_per_environment"]),progress_bar=False)
    model_path=Path(model_dir)/f"{env_name}_recurrent_ppo"
    model.save(str(model_path))
    env.close()
    return model

def eval_rppo_episode(model,env_name,episode_seed):
    env=make_env(env_name)
    obs,_=env.reset(seed=int(episode_seed))
    max_steps=int(env.max_episode_length)
    state=None
    episode_start=np.ones((1,),dtype=bool)
    steps=0
    reward_sum=0.0
    while True:
        action,state=model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start[:]=False
        a=int(np.asarray(action).reshape(-1)[0])
        obs,reward,terminated,truncated,_=env.step(a)
        reward_sum+=float(reward)
        steps+=1
        if terminated or truncated or steps>=max_steps:
            break
    env.close()
    return float(steps/max_steps),float(reward_sum),max_steps

def eval_frozen_core_episode(env_name,controller,episode_seed):
    d,r,m=r38.run_episode(
        env_name,
        controller,
        int(episode_seed),
        "KF_R09",
        R38_CFG,
    )
    return float(d),float(r),int(m)

def eval_seed(seed,models,cfg):
    envs=cfg["environments"]
    episodes=int(cfg["episodes_per_environment"])
    rows={}
    for ei,env_name in enumerate(envs):
        vals={"RPPO":[],"CORE_C":[],"GENERIC_ISO":[],"NO_C_CURRENT":[],"ORACLE_STATE":[]}
        for ep in range(episodes):
            epseed=int(seed)*100000+ei*1000+ep
            d,r,m=eval_rppo_episode(models[env_name],env_name,epseed)
            vals["RPPO"].append((d,r))
            for c in ("CORE_C","GENERIC_ISO","NO_C_CURRENT","ORACLE_STATE"):
                d2,r2,m2=eval_frozen_core_episode(env_name,c,epseed)
                vals[c].append((d2,r2))
        rows[env_name]={}
        for c,v in vals.items():
            rows[env_name][c]={
                "duration_score":float(np.mean([x[0] for x in v])),
                "reward":float(np.mean([x[1] for x in v])),
                "episodes":episodes,
            }

    def avg(controller):
        return float(np.mean([rows[e][controller]["duration_score"] for e in envs]))
    rppo=avg("RPPO")
    core=avg("CORE_C")
    generic=avg("GENERIC_ISO")
    noc=avg("NO_C_CURRENT")
    oracle=avg("ORACLE_STATE")
    return {
        "seed":int(seed),
        "environments":rows,
        "rppo_score":rppo,
        "core_score":core,
        "generic_score":generic,
        "noC_score":noc,
        "oracle_score":oracle,
        "core_minus_rppo":float(core-rppo),
        "rppo_minus_core":float(rppo-core),
        "core_minus_noC":float(core-noc),
        "iso_gap":float(abs(core-generic)),
        "rppo_min_env":float(min(rows[e]["RPPO"]["duration_score"] for e in envs)),
        "core_min_env":float(min(rows[e]["CORE_C"]["duration_score"] for e in envs)),
    }

def med(records,key):
    return float(np.median([r[key] for r in records]))

def adjudicate(records,cfg,mode):
    base=cfg["baseline_validity"]
    bm=base["medians"]
    bsg=base["seed_guard"]
    required=base["development_seed_guard_required"] if mode=="dev" else base["confirm_seed_guard_required"]
    m={
        k:med(records,k)
        for k in [
            "rppo_score","core_score","core_minus_rppo","rppo_minus_core",
            "core_minus_noC","iso_gap","rppo_min_env","core_min_env","oracle_score"
        ]
    }
    baseline_checks={
        "rppo_score":m["rppo_score"]>=bm["rppo_score_min"],
        "rppo_min_env":m["rppo_min_env"]>=bm["rppo_min_env_min"],
    }
    baseline_guard=sum(
        r["rppo_score"]>=bsg["rppo_score_min"] and
        r["rppo_min_env"]>=bsg["rppo_min_env_min"]
        for r in records
    )
    baseline_valid=all(baseline_checks.values()) and baseline_guard>=required

    comp=cfg["frozen_core_competitiveness"]
    creq=comp["development_seed_guard_required"] if mode=="dev" else comp["confirm_seed_guard_required"]
    core_noninferior_check=m["core_minus_rppo"]>=comp["core_minus_rppo_min"]
    core_guard=sum(
        r["core_minus_rppo"]>=comp["seed_guard_core_minus_rppo_min"]
        for r in records
    )
    core_competitive=baseline_valid and core_noninferior_check and core_guard>=creq

    adv=cfg["external_baseline_advantage"]
    adv_guard=sum(
        r["rppo_minus_core"]>=adv["seed_guard_rppo_minus_core_min"]
        for r in records
    )
    rppo_advantage=(
        baseline_valid and
        m["rppo_minus_core"]>=adv["rppo_minus_core_min"] and
        adv_guard>=adv["confirm_seed_guard_required"]
    ) if mode=="confirm" else False

    return {
        "medians":m,
        "baseline_validity_checks":baseline_checks,
        "baseline_seed_guard_pass_count":int(baseline_guard),
        "baseline_seed_guard_required":int(required),
        "baseline_valid":bool(baseline_valid),
        "core_noninferiority_median_check":bool(core_noninferior_check),
        "core_noninferiority_seed_guard_pass_count":int(core_guard),
        "core_noninferiority_seed_guard_required":int(creq),
        "frozen_core_competitive":bool(core_competitive),
        "rppo_advantage_seed_guard_count":int(adv_guard),
        "external_rppo_advantage_over_core":bool(rppo_advantage),
    }

def package_versions():
    import importlib.metadata as md
    return {
        "popgym":md.version("popgym"),
        "stable-baselines3":md.version("stable-baselines3"),
        "sb3-contrib":md.version("sb3-contrib"),
        "torch":md.version("torch"),
        "scipy":md.version("scipy"),
    }

def load_models(cfg,model_dir):
    from sb3_contrib import RecurrentPPO
    models={}
    for env_name in cfg["environments"]:
        path=Path(model_dir)/f"{env_name}_recurrent_ppo.zip"
        if not path.is_file():
            raise FileNotFoundError(path)
        models[env_name]=RecurrentPPO.load(str(path),device="cpu")
    return models

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--criteria",required=True)
    ap.add_argument("--mode",choices=["train","evaluate"],required=True)
    ap.add_argument("--env")
    ap.add_argument("--output",required=True)
    ap.add_argument("--model-dir",required=True)
    args=ap.parse_args()
    cfg=json.loads(Path(args.criteria).read_text())
    model_dir=Path(args.model_dir)
    model_dir.mkdir(parents=True,exist_ok=True)

    if args.mode=="train":
        if args.env not in cfg["environments"]:
            raise ValueError(f"invalid --env {args.env}")
        env_name=args.env
        seed=int(cfg["training"]["training_seeds"][env_name])
        train_rppo(env_name,seed,cfg,model_dir)
        out={
            "campaign":cfg["campaign"],
            "mode":"train",
            "environment":env_name,
            "seed":seed,
            "timesteps":int(cfg["training"]["timesteps_per_environment"]),
            "model_path":str(model_dir/f"{env_name}_recurrent_ppo.zip"),
            "package_versions":package_versions(),
        }
        Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
        print(json.dumps(out,indent=2,sort_keys=True))
        return

    models=load_models(cfg,model_dir)
    training_records={
        env_name:{
            "seed":int(cfg["training"]["training_seeds"][env_name]),
            "timesteps":int(cfg["training"]["timesteps_per_environment"]),
            "model_path":str(model_dir/f"{env_name}_recurrent_ppo.zip"),
        }
        for env_name in cfg["environments"]
    }

    dev=[eval_seed(seed,models,cfg) for seed in cfg["development_eval_seeds"]]
    dev_adj=adjudicate(dev,cfg,"dev")
    out={
        "campaign":cfg["campaign"],
        "criteria_status":cfg["status"],
        "execution_mode":"parallel_training_then_single_evaluation",
        "package_versions":package_versions(),
        "training":training_records,
        "development":{
            "records":dev,
            "adjudication":dev_adj,
        },
        "confirmatory_authorized":bool(dev_adj["baseline_valid"]),
        "confirmatory_eval_seeds_status":"UNOPENED" if not dev_adj["baseline_valid"] else "OPENED_BY_FROZEN_GATE",
        "boundaries":cfg["fixed_boundaries"],
    }

    if dev_adj["baseline_valid"]:
        conf=[eval_seed(seed,models,cfg) for seed in cfg["confirmatory_eval_seeds"]]
        out["confirmatory"]={
            "records":conf,
            "adjudication":adjudicate(conf,cfg,"confirm"),
        }
    else:
        out["confirmatory"]=None

    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "development":out["development"]["adjudication"],
        "confirmatory":None if out["confirmatory"] is None else out["confirmatory"]["adjudication"],
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
