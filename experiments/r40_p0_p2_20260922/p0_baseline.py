#!/usr/bin/env python3
"""P0: externally implemented learner, internally executed; NOT independent replication."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

TRAIN_SEEDS = [2066001, 2066002, 2066003]
DEV_SEEDS = list(range(2066101,2066113))
TEST_SEEDS = list(range(2067001,2067033))

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def make_env():
    from popgym.envs.repeat_previous import RepeatPreviousEasy
    return RepeatPreviousEasy()

def train(seed, out):
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env = DummyVecEnv([make_env for _ in range(16)])
    model = RecurrentPPO('MlpLstmPolicy', env, seed=seed, device='cpu',
        n_steps=128, batch_size=256, n_epochs=4, learning_rate=0.0003,
        gamma=0.95, gae_lambda=0.95, ent_coef=0.01,
        policy_kwargs={'lstm_hidden_size':64,'net_arch':dict(pi=[64],vf=[64])}, verbose=0)
    start = time.time()
    model.learn(total_timesteps=262144)
    out.mkdir(parents=True,exist_ok=True)
    model.save(out/f'model_{seed}')
    env.close()
    p=out/f'model_{seed}.zip'
    dump(out/f'train_{seed}.json',{'seed':seed,'steps':int(model.num_timesteps),
        'seconds':time.time()-start,'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'environment':'popgym.envs.repeat_previous.RepeatPreviousEasy',
        'external_algorithm':'sb3_contrib.RecurrentPPO','selection':'final fixed checkpoint only'})
    print('TRAIN_DONE', seed, flush=True)

def episode(model, seed, corruption=0.0, reset_each_step=False, oracle=False):
    env=make_env(); obs,_=env.reset(seed=seed)
    rng=np.random.default_rng(seed+9000000)
    state=None; starts=np.ones(1,dtype=bool); history=[]
    total=0.0; correct=0; scored=0; steps=0
    while True:
        history.append(int(obs))
        seen=int(rng.integers(4)) if rng.random()<corruption else int(obs)
        if oracle:
            # A queue ceiling uses only the observed clean sequence, never env.get_state().
            action=history[-env.k] if len(history)>=env.k else 0
        else:
            if reset_each_step: state=None; starts[:]=True
            action,state=model.predict(seen,state=state,episode_start=starts,deterministic=True)
            action=int(np.asarray(action).reshape(-1)[0])
        obs,reward,terminated,truncated,_=env.step(action)
        total+=float(reward); steps+=1
        if reward!=0:
            scored+=1; correct+=int(reward>0)
        starts[:]=bool(terminated or truncated)
        if terminated or truncated: break
    env.close()
    return {'seed':seed,'corruption':corruption,'reset_each_step':reset_each_step,
        'accuracy':correct/max(scored,1),'return':total,'steps':steps,'scored':scored}

def evaluate(out):
    import torch
    from sb3_contrib import RecurrentPPO
    torch.set_num_threads(1)
    models={s:RecurrentPPO.load(out/f'model_{s}.zip',device='cpu') for s in TRAIN_SEEDS}
    raw=[]; dev=[]
    oracle_scores=[episode(None,s,oracle=True)['accuracy'] for s in DEV_SEEDS]
    for s,m in models.items():
        intact=[episode(m,k) for k in DEV_SEEDS]
        reset=[episode(m,k,reset_each_step=True) for k in DEV_SEEDS]
        for r in intact+reset: r['training_seed']=s; r['split']='development'; raw.append(r)
        acc=float(np.mean([r['accuracy'] for r in intact]))
        benefit=float(np.mean([a['accuracy']-b['accuracy'] for a,b in zip(intact,reset)]))
        dev.append({'training_seed':s,'accuracy':acc,'history_benefit':benefit,
                    'eligible':bool(acc>=.80 and benefit>=.10)})
    instrument=bool(min(oracle_scores)>=1.-1e-12)
    eligible=instrument and sum(d['eligible'] for d in dev)>=2
    confirm=[]
    if eligible:
        for s,m in models.items():
            metrics={}
            for noise in [0.,.10,.20]:
                intact=[episode(m,k,noise) for k in TEST_SEEDS]
                reset=[episode(m,k,noise,True) for k in TEST_SEEDS]
                for r in intact+reset: r['training_seed']=s; r['split']='confirmation'; raw.append(r)
                metrics[str(noise)]={'accuracy':float(np.mean([r['accuracy'] for r in intact])),
                    'history_benefit':float(np.mean([a['accuracy']-b['accuracy'] for a,b in zip(intact,reset)]))}
            ok=metrics['0.0']['accuracy']>=.8 and metrics['0.1']['accuracy']>=.65 and metrics['0.0']['history_benefit']>=.1
            confirm.append({'training_seed':s,'metrics':metrics,'pass':bool(ok)})
        resolution='P0_NEW_MEMORY_BASELINE_PASS' if sum(d['pass'] for d in confirm)>=2 else 'P0_NEW_MEMORY_BASELINE_CONFIRM_FAIL'
    else:
        resolution='P0_BASELINE_DEV_FAIL_NO_CONFIRM'
    report={'resolution':resolution,'oracle_instrument_pass':instrument,'development':dev,
        'confirmation_opened':eligible,'confirmation':confirm,'raw_records':raw,
        'training_seeds':TRAIN_SEEDS,'development_episode_seeds':DEV_SEEDS,
        'reserved_confirmatory_seeds':TEST_SEEDS,
        'scope':'new external delayed-recall task only; NOT old noisy-control validity',
        'E6b':'OPEN_SEPARATE_TEAM_REQUIRED','E7':'OPEN','POLAR_superiority':'NOT_TESTED'}
    dump(out/'P0_RESULTS.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='raw_records'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['train','evaluate'],required=True)
    p.add_argument('--seed',type=int); p.add_argument('--out',default='r40_results/p0')
    a=p.parse_args(); out=Path(a.out)
    if a.stage=='train':
        if a.seed not in TRAIN_SEEDS: raise ValueError('Unregistered training seed')
        train(a.seed,out)
    else: evaluate(out)
