#!/usr/bin/env python3
import argparse, json, importlib.util, itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r30_path=ROOT/"r30_adaptive_lqr_20260921"/"r30_adaptive_lqr.py"
spec=importlib.util.spec_from_file_location("r30base",r30_path)
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

TRAIN_STEPS=3200
EVAL_STEPS=3200
TRAIN_SCHEDULE=[0,1,0,2,1,2,0,1]
EVAL_SCHEDULE=[3,2,0,1,3,0,2,1]

def adaptive_train(seed,task,kind,steps=TRAIN_STEPS):
    rng=np.random.default_rng(seed)
    step,n,params,Q,R,uclip,state,obs,estimate,cost,stable=base.task_spec(task,rng,steps)
    Anom,Bnom=base.base.numeric_linearization(step,n,params[0])
    K_nom=base.base.infinite_lqr_gain(Anom,Bnom,Q,R)
    Ahat=Anom.copy(); Bhat=Bnom.copy()
    K_adapt=K_nom.copy() if K_nom is not None else np.zeros((1,n))
    states=[]; actions=[]
    prev_o=None; prev_v=None; est_prev=None
    seg=steps//len(TRAIN_SCHEDULE)
    for t in range(steps):
        rid=TRAIN_SCHEDULE[min(len(TRAIN_SCHEDULE)-1,t//seg)]
        p=params[rid]
        o=obs(state,t)
        if prev_v is None:
            prev_v=np.zeros(2 if task=="cartpole" else 1)
        est,vel=estimate(o,prev_o,prev_v)
        if est_prev is not None:
            states.append(est_prev.copy()); actions.append(float(u_prev))
        if len(actions)>=80 and t%20==0:
            w=min(360,len(actions))
            seq=np.asarray(states[-w:]+[est.copy()],float)
            act=np.asarray(actions[-w:],float)
            cut=max(55,int(.75*w))
            try:
                Anew,Bnew=base.base.fit_ab(seq[:cut+1],act[:cut],lam=.08)
                if kind=="CORE":
                    Anew=Anew.copy()
                    Anew[np.abs(Anew)<0.02]=0.0
                Xv=seq[cut:-1]; Uv=act[cut:]; Yv=seq[cut+1:]
                oldp=(Xv@Ahat.T)+(Uv[:,None]@Bhat.T)
                newp=(Xv@Anew.T)+(Uv[:,None]@Bhat.T*0 + Uv[:,None]@Bnew.T)
                # same prediction expression, written explicitly for auditability
                newp=(Xv@Anew.T)+(Uv[:,None]@Bnew.T)
                oldm=float(np.mean((oldp-Yv)**2))
                newm=float(np.mean((newp-Yv)**2))
                if newm<=oldm*.95:
                    alpha=.20
                    Ac=(1-alpha)*Ahat+alpha*Anew
                    Bc=(1-alpha)*Bhat+alpha*Bnew
                    Knew=base.base.infinite_lqr_gain(Ac,Bc,Q,R)
                    if Knew is not None and np.max(np.abs(Knew))<120:
                        rho=max(abs(np.linalg.eigvals(Ac-Bc@Knew)))
                        if rho<1.08:
                            Ahat,Bhat,K_adapt=Ac,Bc,Knew
            except Exception:
                pass
        ua=float(-(K_adapt@est)[0]) if K_adapt is not None else 0.0
        un=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
        u=float(np.clip(.70*un+.30*ua,-uclip,uclip))
        next_state=step(state,u,p)
        prev_o=o; prev_v=vel; est_prev=est.copy(); u_prev=u; state=next_state
    return {"K_nom":K_nom,"K_adapt":K_adapt}

def policy_rollout_cost(task,seed,gain_scale,regime,steps=200):
    rng=np.random.default_rng(seed)
    step,n,params,Q,R,uclip,state,obs,estimate,cost,stable=base.task_spec(task,rng,steps)
    Anom,Bnom=base.base.numeric_linearization(step,n,params[0])
    K_nom=base.base.infinite_lqr_gain(Anom,Bnom,Q,R)
    prev_o=None; prev_v=None; costs=[]
    p=params[regime]
    for t in range(steps):
        o=obs(state,t)
        if prev_v is None:
            prev_v=np.zeros(2 if task=="cartpole" else 1)
        est,vel=estimate(o,prev_o,prev_v)
        u=float(np.clip(-(gain_scale*K_nom@est)[0],-uclip,uclip))
        costs.append(cost(state,u))
        state=step(state,u,p)
        prev_o=o; prev_v=vel
    return float(np.mean(costs))

def model_free_bandit_train(seed,task):
    # Seven fixed feedback policies; selection uses realized scalar cost only.
    candidates=[0.40,0.50,0.60,0.70,0.85,1.00,1.15]
    scores={g:[] for g in candidates}
    # Two paired rounds across all policies: 2*7*200 = 2800 interactions.
    for rnd in range(2):
        regime=(seed+rnd)%3
        rollout_seed=seed+rnd*1000
        for g in candidates:
            scores[g].append(policy_rollout_cost(task,rollout_seed,g,regime,200))
    means={g:float(np.mean(v)) for g,v in scores.items()}
    top=sorted(candidates,key=lambda g:means[g])[:2]
    # Final paired refinement of the top two: 2*200 = 400 interactions.
    regime=(seed+2)%3
    rollout_seed=seed+2000
    for g in top:
        scores[g].append(policy_rollout_cost(task,rollout_seed,g,regime,200))
    means={g:float(np.mean(v)) for g,v in scores.items()}
    best=min(candidates,key=lambda g:means[g])
    return {"K_nom":None,"gain_scale":float(best),"candidate_costs":means,"training_interactions":3200}

def eval_controller(seed,task,controller,trained=None,steps=EVAL_STEPS):
    rng=np.random.default_rng(seed)
    step,n,params,Q,R,uclip,state,obs,estimate,cost,stable=base.task_spec(task,rng,steps)
    Anom,Bnom=base.base.numeric_linearization(step,n,params[0])
    K_nom=base.base.infinite_lqr_gain(Anom,Bnom,Q,R)
    prev_o=None; prev_v=None
    costs=[]; stables=[]; shift_costs=[]; rl_nonzero=[]
    seg=steps//len(EVAL_SCHEDULE)
    prev_rid=EVAL_SCHEDULE[0]
    for t in range(steps):
        rid=EVAL_SCHEDULE[min(len(EVAL_SCHEDULE)-1,t//seg)]
        p=params[rid]; within=t%seg
        o=obs(state,t)
        if prev_v is None:
            prev_v=np.zeros(2 if task=="cartpole" else 1)
        est,vel=estimate(o,prev_o,prev_v)
        if controller=="FROZEN":
            u=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
        elif controller=="ORACLE_LQR":
            At,Bt=base.base.numeric_linearization(step,n,p)
            Ko=base.base.infinite_lqr_gain(At,Bt,Q,R)
            u=float(-(Ko@state)[0]) if Ko is not None else 0.0
        elif controller in ("CORE","ADAPTIVE_LQR"):
            K_adapt=trained["K_adapt"]
            ua=float(-(K_adapt@est)[0]) if K_adapt is not None else 0.0
            un=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
            u=.70*un+.30*ua
        elif controller=="MODEL_FREE_RL":
            g=float(trained["gain_scale"])
            u=float(-(g*K_nom@est)[0]) if K_nom is not None else 0.0
            rl_nonzero.append(float(abs(g-1.0)>1e-12))
        else:
            raise ValueError(controller)
        u=float(np.clip(u,-uclip,uclip))
        c=cost(state,u); costs.append(c); stables.append(stable(state))
        if t>0 and rid!=prev_rid and within<80:
            shift_costs.append(c)
        elif rid!=0 and within<80:
            shift_costs.append(c)
        state=step(state,u,p)
        prev_o=o; prev_v=vel; prev_rid=rid
    return {
      "mean_cost":float(np.mean(costs)),
      "postshift_cost":float(np.mean(shift_costs)),
      "stable_fraction":float(np.mean(stables)),
      "rl_nonzero_fraction":float(np.mean(rl_nonzero)) if rl_nonzero else 0.0
    }

def eval_seed(seed):
    tasks={}; rl_scales=[]
    for ti,task in enumerate(("cartpole","pendulum")):
        tr_seed=seed+10000*ti
        ev_seed=seed+500000+10000*ti
        core=adaptive_train(tr_seed,task,"CORE")
        adp=adaptive_train(tr_seed+100,task,"ADAPTIVE_LQR")
        rl=model_free_bandit_train(tr_seed+200,task)
        rl_scales.append(float(rl["gain_scale"]))
        tasks[task]={
          "CORE":eval_controller(ev_seed,task,"CORE",core),
          "ADAPTIVE_LQR":eval_controller(ev_seed,task,"ADAPTIVE_LQR",adp),
          "MODEL_FREE_RL":eval_controller(ev_seed,task,"MODEL_FREE_RL",rl),
          "FROZEN":eval_controller(ev_seed,task,"FROZEN"),
          "ORACLE_LQR":eval_controller(ev_seed,task,"ORACLE_LQR")
        }
    def ratio(a,b,key):
        return float(np.mean([tasks[t][a][key]/(tasks[t][b][key]+1e-12) for t in tasks]))
    return {
      "seed":seed,
      "tasks":tasks,
      "rl_frozen_ratio":ratio("MODEL_FREE_RL","FROZEN","mean_cost"),
      "rl_adaptive_ratio":ratio("MODEL_FREE_RL","ADAPTIVE_LQR","mean_cost"),
      "core_rl_ratio":ratio("CORE","MODEL_FREE_RL","mean_cost"),
      "core_rl_postshift_ratio":ratio("CORE","MODEL_FREE_RL","postshift_cost"),
      "core_adaptive_ratio":ratio("CORE","ADAPTIVE_LQR","mean_cost"),
      "core_stable":float(np.mean([tasks[t]["CORE"]["stable_fraction"] for t in tasks])),
      "rl_stable":float(np.mean([tasks[t]["MODEL_FREE_RL"]["stable_fraction"] for t in tasks])),
      "adaptive_stable":float(np.mean([tasks[t]["ADAPTIVE_LQR"]["stable_fraction"] for t in tasks])),
      "rl_nonzero_fraction":float(np.mean([tasks[t]["MODEL_FREE_RL"]["rl_nonzero_fraction"] for t in tasks])),
      "rl_gain_scale_mean":float(np.mean(rl_scales))
    }

def summarize(records):
    keys=["rl_frozen_ratio","rl_adaptive_ratio","core_rl_ratio","core_rl_postshift_ratio",
          "core_adaptive_ratio","core_stable","rl_stable","adaptive_stable","rl_nonzero_fraction","rl_gain_scale_mean"]
    return {k:float(np.median([r[k] for r in records])) for k in keys}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    a,b=map(int,args.seeds.split(":"))
    seeds=list(range(a,b+1))
    rec=[eval_seed(s) for s in seeds]
    out={"campaign":"R31 matched residual model-free RL","seeds":seeds,"records":rec,"medians":summarize(rec)}
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["medians"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
