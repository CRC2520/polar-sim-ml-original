#!/usr/bin/env python3
import argparse, json, importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r27_path=ROOT/"r27_strong_mpc_20260921"/"r27_strong_mpc.py"
spec=importlib.util.spec_from_file_location("r27base",r27_path)
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def task_spec(task,rng,steps):
    if task=="cartpole":
        step=base.cartpole_step; n=4; dt=0.02; uclip=10.0
        params=[
            {"mc":1.0,"mp":0.10,"l":0.50},
            {"mc":0.72,"mp":0.25,"l":0.80},
            {"mc":1.35,"mp":0.06,"l":0.34},
            {"mc":0.90,"mp":0.18,"l":0.63},
        ]
        Q=np.diag([0.25,0.05,8.0,0.5]); R=np.array([[0.02]])
        state=np.array([rng.normal(0,.03),0,rng.normal(0,.035),0.0])
        noise=rng.normal(0,0.0006,size=(steps+1,2))
        def obs(s,t): return np.array([s[0],s[2]])+noise[t]
        def estimate(o,prev_o,prev_v):
            if prev_o is None: return np.array([o[0],0,o[1],0.0]),np.zeros(2)
            dv=(o-prev_o)/dt; vv=.65*prev_v+.35*dv
            return np.array([o[0],vv[0],o[1],vv[1]]),vv
        def cost(s,u): return float(s@Q@s+R[0,0]*u*u)
        def stable(s): return float(abs(s[0])<2.4 and abs(s[2])<0.35)
    else:
        step=base.pendulum_step; n=2; dt=0.05; uclip=2.0
        params=[
            {"m":1.0,"l":1.0,"b":0.05},
            {"m":1.55,"l":0.62,"b":0.30},
            {"m":0.60,"l":1.38,"b":0.01},
            {"m":1.20,"l":0.82,"b":0.16},
        ]
        Q=np.diag([6.0,0.4]); R=np.array([[0.04]])
        state=np.array([rng.normal(0,.06),0.0])
        noise=rng.normal(0,0.001,size=(steps+1,1))
        def obs(s,t): return np.array([s[0]])+noise[t]
        def estimate(o,prev_o,prev_v):
            if prev_o is None: return np.array([o[0],0.0]),np.zeros(1)
            dv=(o-prev_o)/dt; vv=.65*prev_v+.35*dv
            return np.array([o[0],vv[0]]),vv
        def cost(s,u): return float(s@Q@s+R[0,0]*u*u)
        def stable(s): return float(abs(s[0])<0.5)
    return step,n,params,Q,R,uclip,state,obs,estimate,cost,stable

def run_task(seed,task,controller,steps):
    rng=np.random.default_rng(seed)
    step,n,params,Q,R,uclip,state,obs,estimate,cost,stable=task_spec(task,rng,steps)
    Anom,Bnom=base.numeric_linearization(step,n,params[0])
    K_nom=base.infinite_lqr_gain(Anom,Bnom,Q,R)
    Ahat=Anom.copy(); Bhat=Bnom.copy()
    K_adapt=K_nom.copy() if K_nom is not None else np.zeros((1,n))
    states=[]; actions=[]; costs=[]; stables=[]; shift_costs=[]
    prev_o=None; prev_v=None; est_prev=None
    schedule=[0,1,0,2,1,3,0,2,3,1,0,3,2,0,1,2]
    seg=steps//len(schedule)
    prev_rid=schedule[0]
    for t in range(steps):
        rid=schedule[min(len(schedule)-1,t//seg)]; p=params[rid]
        within=t%seg
        o=obs(state,t)
        if prev_v is None: prev_v=np.zeros(2 if task=="cartpole" else 1)
        est,vel=estimate(o,prev_o,prev_v)

        if est_prev is not None:
            states.append(est_prev.copy()); actions.append(float(u_prev))

        if len(actions)>=80 and t%20==0 and controller in ("CORE","ADAPTIVE_LQR"):
            w=min(360,len(actions))
            seq=np.asarray(states[-w:]+[est.copy()],float)
            act=np.asarray(actions[-w:],float)
            cut=max(55,int(.75*w))
            try:
                Anew,Bnew=base.fit_ab(seq[:cut+1],act[:cut],lam=.08)
                if controller=="CORE":
                    Anew=Anew.copy(); Anew[np.abs(Anew)<0.02]=0.0
                Xv=seq[cut:-1]; Uv=act[cut:]; Yv=seq[cut+1:]
                oldp=(Xv@Ahat.T)+(Uv[:,None]@Bhat.T)
                newp=(Xv@Anew.T)+(Uv[:,None]@Bnew.T)
                oldm=float(np.mean((oldp-Yv)**2)); newm=float(np.mean((newp-Yv)**2))
                if newm<=oldm*.95:
                    alpha=.20
                    Ac=(1-alpha)*Ahat+alpha*Anew
                    Bc=(1-alpha)*Bhat+alpha*Bnew
                    Knew=base.infinite_lqr_gain(Ac,Bc,Q,R)
                    if Knew is not None and np.max(np.abs(Knew))<120:
                        rho=max(abs(np.linalg.eigvals(Ac-Bc@Knew)))
                        if rho<1.08:
                            Ahat,Bhat,K_adapt=Ac,Bc,Knew
            except Exception:
                pass

        if controller=="ORACLE_LQR":
            At,Bt=base.numeric_linearization(step,n,p)
            Ko=base.infinite_lqr_gain(At,Bt,Q,R)
            u=float(-(Ko@state)[0]) if Ko is not None else 0.0
        elif controller=="FROZEN":
            u=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
        else:
            ua=float(-(K_adapt@est)[0]) if K_adapt is not None else 0.0
            un=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
            u=.70*un+.30*ua

        u=float(np.clip(u,-uclip,uclip))
        c=cost(state,u); costs.append(c); stables.append(stable(state))
        if t>0 and rid!=prev_rid and within<80:
            shift_costs.append(c)
        elif rid!=0 and within<80:
            shift_costs.append(c)

        next_state=step(state,u,p)
        prev_o=o; prev_v=vel; est_prev=est.copy(); u_prev=u; state=next_state
        prev_rid=rid

    return {
        "mean_cost":float(np.mean(costs)),
        "postshift_cost":float(np.mean(shift_costs)),
        "stable_fraction":float(np.mean(stables))
    }

def eval_seed(seed,steps):
    ctrls=["CORE","ADAPTIVE_LQR","FROZEN","ORACLE_LQR"]
    tasks={}
    for i,task in enumerate(("cartpole","pendulum")):
        tasks[task]={c:run_task(seed+1000*i,task,c,steps) for c in ctrls}
    def ratio(a,b,key):
        return float(np.mean([tasks[t][a][key]/(tasks[t][b][key]+1e-12) for t in tasks]))
    return {
        "seed":seed,
        "tasks":tasks,
        "core_adaptive_ratio":ratio("CORE","ADAPTIVE_LQR","mean_cost"),
        "core_adaptive_postshift_ratio":ratio("CORE","ADAPTIVE_LQR","postshift_cost"),
        "adaptive_frozen_ratio":ratio("ADAPTIVE_LQR","FROZEN","mean_cost"),
        "core_oracle_ratio":ratio("CORE","ORACLE_LQR","mean_cost"),
        "core_stable":float(np.mean([tasks[t]["CORE"]["stable_fraction"] for t in tasks])),
        "adaptive_stable":float(np.mean([tasks[t]["ADAPTIVE_LQR"]["stable_fraction"] for t in tasks]))
    }

def med(records,key):
    return float(np.median([r[key] for r in records]))

def adjudicate(records,c):
    m={k:med(records,k) for k in [
        "core_adaptive_ratio","core_adaptive_postshift_ratio","adaptive_frozen_ratio",
        "core_oracle_ratio","core_stable","adaptive_stable"
    ]}
    b=c["baseline_validity"]["medians"]
    bchecks={
        "adaptive_stable":m["adaptive_stable"]>=b["adaptive_stable_min"],
        "adaptive_frozen":m["adaptive_frozen_ratio"]<=b["adaptive_frozen_ratio_max"]
    }
    bg=c["baseline_validity"]["seed_guard"]
    bcount=sum(
        r["adaptive_stable"]>=bg["adaptive_stable_min"] and
        r["adaptive_frozen_ratio"]<=bg["adaptive_frozen_ratio_max"]
        for r in records
    )
    valid=all(bchecks.values()) and bcount>=c["baseline_validity"]["seed_guard_required"]

    p=c["core_noninferiority"]["medians"]
    checks={
        "core_stable":m["core_stable"]>=p["core_stable_min"],
        "core_adaptive_low":m["core_adaptive_ratio"]>=p["core_adaptive_ratio_min"],
        "core_adaptive_high":m["core_adaptive_ratio"]<=p["core_adaptive_ratio_max"],
        "core_adaptive_postshift":m["core_adaptive_postshift_ratio"]<=p["core_adaptive_postshift_ratio_max"]
    }
    pg=c["core_noninferiority"]["seed_guard"]
    pcount=sum(
        r["core_stable"]>=pg["core_stable_min"] and
        r["core_adaptive_ratio"]>=pg["core_adaptive_ratio_min"] and
        r["core_adaptive_ratio"]<=pg["core_adaptive_ratio_max"] and
        r["core_adaptive_postshift_ratio"]<=pg["core_adaptive_postshift_ratio_max"]
        for r in records
    )
    noninf=valid and all(checks.values()) and pcount>=c["core_noninferiority"]["seed_guard_required"]

    a=c["exclusive_core_advantage"]
    acount=sum(r["core_adaptive_ratio"]<=a["seed_guard_ratio_max"] for r in records)
    advantage=valid and m["core_adaptive_ratio"]<=a["core_adaptive_ratio_max"] and acount>=a["seed_guard_required"]

    return {
        "medians":m,
        "baseline_validity_checks":bchecks,
        "baseline_validity_seed_guard_count":int(bcount),
        "baseline_valid":bool(valid),
        "core_noninferiority_checks":checks,
        "core_noninferiority_seed_guard_count":int(pcount),
        "core_noninferiority_pass":bool(noninf),
        "exclusive_core_advantage_seed_guard_count":int(acount),
        "exclusive_core_advantage_pass":bool(advantage)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--criteria",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    c=json.loads(Path(args.criteria).read_text())
    dev=[eval_seed(s,c["trajectory_steps"]) for s in c["development_seeds"]]
    conf=[eval_seed(s,c["trajectory_steps"]) for s in c["confirmatory_seeds"]]
    out={
        "campaign":c["campaign"],
        "criteria_status":c["status"],
        "development":{"records":dev,"adjudication":adjudicate(dev,c)},
        "confirmatory":{"records":conf,"adjudication":adjudicate(conf,c)},
        "boundaries":c["fixed_boundaries"]
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["confirmatory"]["adjudication"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
