#!/usr/bin/env python3
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

TRAIN_SEEDS=[2146001,2146002,2146003]
DEV_SEEDS=list(range(2146101,2146121))
CONF_SEEDS=list(range(2147001,2147041))

def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def make_env(level):
    from popgym.envs.velocity_only_cartpole import VelocityOnlyCartPoleEasy,VelocityOnlyCartPoleMedium,VelocityOnlyCartPoleHard
    return {"easy":VelocityOnlyCartPoleEasy,"medium":VelocityOnlyCartPoleMedium,"hard":VelocityOnlyCartPoleHard}[level]()

def train(seed,out):
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)

    def factory():
        return make_env("easy")

    env=DummyVecEnv([factory for _ in range(8)])
    common=dict(n_steps=256,batch_size=2048,n_epochs=5,learning_rate=3e-4,gamma=.99,gae_lambda=.95,ent_coef=0.0,clip_range=.2,verbose=0,device="cpu",seed=seed)
    r=RecurrentPPO("MlpLstmPolicy",env,policy_kwargs={"lstm_hidden_size":64,"net_arch":dict(pi=[64],vf=[64])},**common)
    t=time.time(); r.learn(total_timesteps=1048576)
    r.save(out/f"recurrent_{seed}")
    rt=time.time()-t

    f=PPO("MlpPolicy",env,policy_kwargs={"net_arch":dict(pi=[64],vf=[64])},**common)
    t=time.time(); f.learn(total_timesteps=1048576)
    f.save(out/f"feedforward_{seed}")
    ft=time.time()-t
    env.close()

    rec=out/f"recurrent_{seed}.zip"; ff=out/f"feedforward_{seed}.zip"
    dump(out/f"train_{seed}.json",{
        "seed":seed,
        "transitions_per_model":1048576,
        "recurrent_seconds":rt,
        "feedforward_seconds":ft,
        "recurrent_sha256":hashlib.sha256(rec.read_bytes()).hexdigest(),
        "feedforward_sha256":hashlib.sha256(ff.read_bytes()).hexdigest()
    })
    print("TRAIN_DONE",seed,flush=True)

def episode(model,kind,seed,level,reset_each=False):
    env=make_env(level); obs,_=env.reset(seed=seed)
    state=None; start=np.ones(1,dtype=bool); steps=0
    maxlen=int(env.max_episode_length)
    while True:
        if kind=="recurrent":
            if reset_each:
                state=None; start[:]=True
            action,state=model.predict(obs,state=state,episode_start=start,deterministic=True)
            a=int(np.asarray(action).reshape(-1)[0])
        else:
            action,_=model.predict(obs,deterministic=True)
            a=int(np.asarray(action).reshape(-1)[0])
        obs,r,term,trunc,_=env.step(a)
        steps+=1
        start[:]=bool(term or trunc)
        if term or trunc: break
        if steps>maxlen: raise RuntimeError("episode exceeded environment maximum")
    env.close()
    return {"seed":seed,"level":level,"steps":steps,"max_steps":maxlen,"score":steps/maxlen,"reset_each_step":bool(reset_each)}

def eval_model(rec,ff,train_seed,seeds,level):
    ri=[episode(rec,"recurrent",s,level,False) for s in seeds]
    rr=[episode(rec,"recurrent",s,level,True) for s in seeds]
    fi=[episode(ff,"feedforward",s,level,False) for s in seeds]
    intact=float(np.mean([x["score"] for x in ri]))
    reset=float(np.mean([x["score"] for x in rr]))
    feed=float(np.mean([x["score"] for x in fi]))
    return {
        "training_seed":train_seed,
        "level":level,
        "recurrent_score":intact,
        "reset_score":reset,
        "history_benefit":intact-reset,
        "feedforward_score":feed,
        "episodes":{"recurrent":ri,"reset":rr,"feedforward":fi}
    }

def evaluate(out):
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    models={}
    for s in TRAIN_SEEDS:
        models[s]=(RecurrentPPO.load(out/f"recurrent_{s}.zip",device="cpu"),PPO.load(out/f"feedforward_{s}.zip",device="cpu"))

    dev=[]
    for s,(r,f) in models.items():
        q=eval_model(r,f,s,DEV_SEEDS,"medium")
        q["eligible"]=bool(q["recurrent_score"]>=.80 and q["history_benefit"]>=.15)
        dev.append(q)
    eligible=sum(x["eligible"] for x in dev)>=2 and all(np.isfinite([x["recurrent_score"],x["reset_score"],x["feedforward_score"],x["history_benefit"]]).all() for x in dev)

    conf=[]
    if eligible:
        for s,(r,f) in models.items():
            q=eval_model(r,f,s,CONF_SEEDS,"hard")
            q["pass"]=bool(q["recurrent_score"]>=.70 and q["history_benefit"]>=.15)
            conf.append(q)
        resolution="R43_EXTERNAL_RECURRENT_BASELINE_QUALIFIED" if sum(x["pass"] for x in conf)>=2 else "R43_EXTERNAL_RECURRENT_BASELINE_NOT_QUALIFIED"
    else:
        resolution="R43_DEVELOPMENT_FAIL_NO_CONFIRM"

    result={
        "campaign":"R43 external recurrent baseline qualification",
        "resolution":resolution,
        "development":dev,
        "development_eligible":eligible,
        "confirmation_opened":bool(eligible),
        "confirmation":conf,
        "criteria":{
            "dev_recurrent_score_min":.80,
            "dev_history_benefit_min":.15,
            "dev_models_required":2,
            "confirm_recurrent_score_min":.70,
            "confirm_history_benefit_min":.15,
            "confirm_models_required":2
        },
        "training_seeds":TRAIN_SEEDS,
        "development_episode_seeds":DEV_SEEDS,
        "confirmatory_episode_seeds":CONF_SEEDS,
        "boundaries":{
            "POLAR_superiority":"NOT_TESTED",
            "E6b":"OPEN",
            "E7":"OPEN",
            "D_or_R":"NOT_TESTED",
            "consciousness":"NOT_ESTABLISHED"
        }
    }
    dump(out/"R43_RESULTS.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("development","confirmation")},indent=2))
    for x in dev:
        print("DEV",x["training_seed"],x["recurrent_score"],x["history_benefit"],x["feedforward_score"],x["eligible"])
    for x in conf:
        print("CONF",x["training_seed"],x["recurrent_score"],x["history_benefit"],x["feedforward_score"],x["pass"])

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=["train","evaluate"],required=True)
    p.add_argument("--seed",type=int); p.add_argument("--out",default="r43_results")
    a=p.parse_args(); out=Path(a.out)
    if a.stage=="train":
        if a.seed not in TRAIN_SEEDS: raise ValueError(a.seed)
        train(a.seed,out)
    else: evaluate(out)
