#!/usr/bin/env python3
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np

TRAIN_SEEDS=[2106001,2106002,2106003]
DEV_SEEDS=list(range(2106101,2106117))
CONF_SEEDS=list(range(2107001,2107065))

def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def make_env():
    from popgym.envs.repeat_first import RepeatFirstEasy
    return RepeatFirstEasy()

def train_rl(seed,out):
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    env=DummyVecEnv([make_env for _ in range(16)])
    model=RecurrentPPO("MlpLstmPolicy",env,seed=seed,device="cpu",
        n_steps=256,batch_size=4096,n_epochs=10,learning_rate=5e-5,
        gamma=.99,gae_lambda=1.0,ent_coef=0.0,clip_range=.3,vf_coef=1.0,
        policy_kwargs={"lstm_hidden_size":64,"net_arch":dict(pi=[64],vf=[64])},verbose=0)
    t=time.time(); model.learn(total_timesteps=1048576)
    out.mkdir(parents=True,exist_ok=True); model.save(out/f"ppo_{seed}")
    fp=out/f"ppo_{seed}.zip"
    dump(out/f"train_{seed}.json",{"seed":seed,"steps":int(model.num_timesteps),"seconds":time.time()-t,
        "checkpoint_sha256":hashlib.sha256(fp.read_bytes()).hexdigest()})
    env.close()

def episode(model,seed,reset_each=False):
    env=make_env(); obs,_=env.reset(seed=seed); state=None; start=np.ones(1,bool)
    correct=0; n=0
    while True:
        if reset_each: state=None; start[:]=True
        action,state=model.predict(obs,state=state,episode_start=start,deterministic=True)
        obs,r,term,trunc,_=env.step(int(np.asarray(action).reshape(-1)[0]))
        correct+=int(r>0); n+=1; start[:]=bool(term or trunc)
        if term or trunc: break
    env.close(); return correct/max(n,1)

def supervised_instrument(seed=2105999):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    L=51; ntr=12000; nte=3000
    def batch(n):
        first=rng.integers(0,4,n)
        x=rng.integers(0,4,(n,L)); x[:,0]=first
        X=np.eye(4,dtype=np.float32)[x]
        y=np.repeat(first[:,None],L,axis=1)
        return torch.tensor(X),torch.tensor(y,dtype=torch.long)
    X,Y=batch(ntr); Xt,Yt=batch(nte)
    class Net(nn.Module):
        def __init__(self): super().__init__(); self.r=nn.LSTM(4,64,batch_first=True); self.h=nn.Linear(64,4)
        def forward(self,x): z,_=self.r(x); return self.h(z)
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=2e-3)
    for _ in range(8):
        for i in range(0,ntr,256):
            pred=m(X[i:i+256]); loss=nn.functional.cross_entropy(pred.reshape(-1,4),Y[i:i+256].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p=m(Xt).argmax(-1); acc=(p[:,1:]==Yt[:,1:]).float().mean().item()
        # current-only frequency ceiling on i.i.d. later cards is chance
    return {"accuracy":acc,"chance":.25,"pass":bool(acc>=.98)}

def evaluate(out):
    import torch
    from sb3_contrib import RecurrentPPO
    torch.set_num_threads(1)
    sup=supervised_instrument()
    rows=[]
    models={s:RecurrentPPO.load(out/f"ppo_{s}.zip",device="cpu") for s in TRAIN_SEEDS}
    for s,m in models.items():
        intact=np.array([episode(m,k,False) for k in DEV_SEEDS])
        reset=np.array([episode(m,k,True) for k in DEV_SEEDS])
        rows.append({"training_seed":s,"accuracy":float(intact.mean()),"history_benefit":float((intact-reset).mean()),
                     "eligible":bool(intact.mean()>=.85 and (intact-reset).mean()>=.25)})
    eligible=sup["pass"] and sum(x["eligible"] for x in rows)>=2
    conf=[]
    if eligible:
        for s,m in models.items():
            intact=np.array([episode(m,k,False) for k in CONF_SEEDS])
            reset=np.array([episode(m,k,True) for k in CONF_SEEDS])
            a=float(intact.mean()); h=float((intact-reset).mean())
            conf.append({"training_seed":s,"accuracy":a,"history_benefit":h,"pass":bool(a>=.85 and h>=.25)})
        res="P0_STRONG_MEMORY_BASELINE_PASS" if sum(x["pass"] for x in conf)>=2 else "P0_CONFIRM_FAIL"
    else:
        res="P0_DEV_FAIL_NO_CONFIRM"
    outj={"resolution":res,"task":"POPGym RepeatFirstEasy","supervised_task_instrument":sup,
          "development":rows,"confirmation_opened":eligible,"confirmation":conf,
          "training_seeds":TRAIN_SEEDS,"reserved_confirmatory_seeds":CONF_SEEDS,
          "scope":"strong recurrent memory baseline only; not external control baseline, not E6b"}
    dump(out/"P0_RESULTS.json",outj); print(json.dumps(outj,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=["train","evaluate"],required=True)
    p.add_argument("--seed",type=int); p.add_argument("--out",default="r41_results/p0"); a=p.parse_args(); out=Path(a.out)
    if a.stage=="train":
        if a.seed not in TRAIN_SEEDS: raise ValueError(a.seed)
        train_rl(a.seed,out)
    else: evaluate(out)
