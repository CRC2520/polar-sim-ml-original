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

def trust_mpc_action(A,B,Q,R,x,K_nom,K_adapt,uclip,horizon=10):
    un=float(np.clip(-(K_nom@x)[0],-uclip,uclip)) if K_nom is not None else 0.0
    ua=float(np.clip(-(K_adapt@x)[0],-uclip,uclip)) if K_adapt is not None else un
    d=0.08*uclip
    cand=np.unique(np.clip(np.array([
        un,
        0.5*(un+ua),
        ua,
        un-d,un+d,
        un-2*d,un+2*d,
        0.75*un+0.25*ua,
        0.25*un+0.75*ua
    ]),-uclip,uclip))
    best=un; bestv=float("inf")
    for u0 in cand:
        xx=A@x+B[:,0]*u0
        val=float(x@Q@x+R[0,0]*u0*u0)
        for _ in range(horizon-1):
            # Safe nominal tail; the adaptive model is used to evaluate candidate consequences.
            uu=float(np.clip(-(K_nom@xx)[0],-uclip,uclip)) if K_nom is not None else 0.0
            val+=float(xx@Q@xx+R[0,0]*uu*uu)
            xx=A@xx+B[:,0]*uu
            if not np.all(np.isfinite(xx)) or np.linalg.norm(xx)>1e4:
                val=float("inf"); break
        # small trust penalty discourages unnecessary deviation from nominal safe action
        val+=0.03*((u0-un)/(uclip+1e-12))**2
        if val<bestv:
            bestv=val; best=float(u0)
    return best

def run_task(seed,task,controller,steps):
    rng=np.random.default_rng(seed)
    step,n,params,Q,R,uclip,state,obs,estimate,cost,stable=task_spec(task,rng,steps)
    Anom,Bnom=base.numeric_linearization(step,n,params[0])
    K_nom=base.infinite_lqr_gain(Anom,Bnom,Q,R)
    Ahat=Anom.copy(); Bhat=Bnom.copy()
    K_adapt=K_nom.copy() if K_nom is not None else np.zeros((1,n))
    states=[]; actions=[]; costs=[]; stables=[]; shift_costs=[]
    prev_o=None; prev_v=None; est_prev=None
    schedule=[0,1,0,2,1,3,0,2]
    seg=steps//len(schedule)
    for t in range(steps):
        rid=schedule[min(len(schedule)-1,t//seg)]; p=params[rid]
        o=obs(state,t)
        if prev_v is None: prev_v=np.zeros(2 if task=="cartpole" else 1)
        est,vel=estimate(o,prev_o,prev_v)
        if est_prev is not None:
            states.append(est_prev.copy()); actions.append(float(u_prev))
        if len(actions)>=80 and t%20==0 and controller in ("CORE","GENERIC","TRUST_MPC"):
            w=min(320,len(actions))
            seq=np.asarray(states[-w:]+[est.copy()],float)
            act=np.asarray(actions[-w:],float)
            cut=max(50,int(.75*w))
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

        if controller=="ORACLE_MPC":
            At,Bt=base.numeric_linearization(step,n,p)
            Ko=base.infinite_lqr_gain(At,Bt,Q,R)
            u=trust_mpc_action(At,Bt,Q,R,state,Ko,Ko,uclip,10)
        elif controller=="FROZEN":
            u=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
        elif controller=="TRUST_MPC":
            u=trust_mpc_action(Ahat,Bhat,Q,R,est,K_nom,K_adapt,uclip,10)
        else:
            ua=float(-(K_adapt@est)[0]) if K_adapt is not None else 0.0
            un=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
            u=.70*un+.30*ua

        u=float(np.clip(u,-uclip,uclip))
        c=cost(state,u)
        costs.append(c); stables.append(stable(state))
        if rid!=0 and (t%seg)<80:
            shift_costs.append(c)
        next_state=step(state,u,p)
        prev_o=o; prev_v=vel; est_prev=est.copy(); u_prev=u; state=next_state

    return {
        "mean_cost":float(np.mean(costs)),
        "postshift_cost":float(np.mean(shift_costs)),
        "stable_fraction":float(np.mean(stables))
    }

def eval_seed(seed,steps):
    ctrls=["CORE","GENERIC","TRUST_MPC","FROZEN","ORACLE_MPC"]
    tasks={}
    for i,task in enumerate(("cartpole","pendulum")):
        tasks[task]={c:run_task(seed+1000*i,task,c,steps) for c in ctrls}
    def ratio(a,b,key):
        return float(np.mean([tasks[t][a][key]/(tasks[t][b][key]+1e-12) for t in tasks]))
    return {
        "seed":seed,
        "tasks":tasks,
        "core_mpc_cost_ratio":ratio("CORE","TRUST_MPC","mean_cost"),
        "core_mpc_postshift_ratio":ratio("CORE","TRUST_MPC","postshift_cost"),
        "core_generic_ratio":ratio("CORE","GENERIC","mean_cost"),
        "mpc_frozen_ratio":ratio("TRUST_MPC","FROZEN","mean_cost"),
        "core_oracle_ratio":ratio("CORE","ORACLE_MPC","mean_cost"),
        "core_stable":float(np.mean([tasks[t]["CORE"]["stable_fraction"] for t in tasks])),
        "mpc_stable":float(np.mean([tasks[t]["TRUST_MPC"]["stable_fraction"] for t in tasks]))
    }

def med(records,key):
    return float(np.median([r[key] for r in records]))

def adjudicate(records,c):
    m={k:med(records,k) for k in [
        "core_mpc_cost_ratio","core_mpc_postshift_ratio","core_generic_ratio",
        "mpc_frozen_ratio","core_oracle_ratio","core_stable","mpc_stable"
    ]}
    cv=c["comparator_validity"]["medians"]
    cv_checks={
        "mpc_stable":m["mpc_stable"]>=cv["mpc_stable_min"],
        "mpc_frozen":m["mpc_frozen_ratio"]<=cv["mpc_frozen_ratio_max"]
    }
    sg=c["comparator_validity"]["seed_guard"]
    cv_sg=sum(
        r["mpc_stable"]>=sg["mpc_stable_min"] and
        r["mpc_frozen_ratio"]<=sg["mpc_frozen_ratio_max"]
        for r in records
    )
    valid=all(cv_checks.values()) and cv_sg>=c["comparator_validity"]["seed_guard_required"]

    p=c["noninferiority"]["medians"]
    ni_checks={
        "core_stable":m["core_stable"]>=p["core_stable_min"],
        "core_mpc_cost":m["core_mpc_cost_ratio"]<=p["core_mpc_cost_ratio_max"],
        "core_mpc_postshift":m["core_mpc_postshift_ratio"]<=p["core_mpc_postshift_ratio_max"],
        "core_generic_low":m["core_generic_ratio"]>=p["core_generic_ratio_min"],
        "core_generic_high":m["core_generic_ratio"]<=p["core_generic_ratio_max"]
    }
    nsg=c["noninferiority"]["seed_guard"]
    ni_sg=sum(
        r["core_stable"]>=nsg["core_stable_min"] and
        r["core_mpc_cost_ratio"]<=nsg["core_mpc_cost_ratio_max"] and
        r["core_mpc_postshift_ratio"]<=nsg["core_mpc_postshift_ratio_max"] and
        r["core_generic_ratio"]>=nsg["core_generic_ratio_min"] and
        r["core_generic_ratio"]<=nsg["core_generic_ratio_max"]
        for r in records
    )
    noninf=valid and all(ni_checks.values()) and ni_sg>=c["noninferiority"]["seed_guard_required"]

    a=c["exclusive_core_advantage"]
    adv_sg=sum(r["core_mpc_cost_ratio"]<=a["seed_guard_ratio_max"] for r in records)
    adv=valid and m["core_mpc_cost_ratio"]<=a["core_mpc_cost_ratio_max"] and adv_sg>=a["seed_guard_required"]

    return {
        "medians":m,
        "comparator_validity_checks":cv_checks,
        "comparator_validity_seed_guard_count":int(cv_sg),
        "comparator_valid":bool(valid),
        "noninferiority_checks":ni_checks,
        "noninferiority_seed_guard_count":int(ni_sg),
        "classical_control_noninferiority_pass":bool(noninf),
        "exclusive_core_advantage_seed_guard_count":int(adv_sg),
        "exclusive_core_advantage_pass":bool(adv)
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
