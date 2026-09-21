#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_are

def ridge_fit(X,Y,lam=1e-3):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if X.ndim==1: X=X[:,None]
    if Y.ndim==1: Y=Y[:,None]
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@Y)

def cartpole_step(s,u,p,dt=0.02):
    x,xd,th,thd=map(float,s); mc,mp,l,g=p["mc"],p["mp"],p["l"],9.8
    force=float(np.clip(u,-10,10)); total=mc+mp; poleml=mp*l
    st,ct=np.sin(th),np.cos(th); temp=(force+poleml*thd*thd*st)/total
    thacc=(g*st-ct*temp)/(l*(4.0/3.0-mp*ct*ct/total))
    xacc=temp-poleml*thacc*ct/total
    return np.array([x+dt*xd,xd+dt*xacc,th+dt*thd,thd+dt*thacc],float)

def pendulum_step(s,u,p,dt=0.05):
    th,om=map(float,s); m,l,b,g=p["m"],p["l"],p["b"],10.0
    torque=float(np.clip(u,-2,2))
    acc=3*g/(2*l)*np.sin(th)+3.0/(m*l*l)*torque-b*om
    om2=np.clip(om+dt*acc,-8,8); th2=((th+dt*om2+np.pi)%(2*np.pi))-np.pi
    return np.array([th2,om2],float)

def numeric_linearization(step_fn,n,params):
    s0=np.zeros(n); eps=1e-5; f0=step_fn(s0,0.0,params)
    A=np.zeros((n,n))
    for i in range(n):
        ss=s0.copy(); ss[i]+=eps
        A[:,i]=(step_fn(ss,0.0,params)-f0)/eps
    B=((step_fn(s0,eps,params)-f0)/eps)[:,None]
    return A,B

def infinite_lqr_gain(A,B,Q,R):
    try:
        P=solve_discrete_are(A,B,Q,R)
        K=np.linalg.solve(B.T@P@B+R,B.T@P@A)
        return K if np.all(np.isfinite(K)) else None
    except Exception:
        return None

def finite_horizon_gain(A,B,Q,R,H=24):
    P=Q.copy(); K=None
    try:
        for _ in range(H):
            G=B.T@P@B+R
            K=np.linalg.solve(G,B.T@P@A)
            P=Q+A.T@P@A-A.T@P@B@K
        if np.all(np.isfinite(K)): return K
    except Exception:
        pass
    return None

def fit_ab(states,actions,lam=0.08):
    S=np.asarray(states,float); U=np.asarray(actions,float).reshape(-1,1)
    X=np.hstack([S[:-1],U]); Y=S[1:]
    W=ridge_fit(X,Y,lam); n=S.shape[1]
    return W[:n,:].T,W[n:,:].T

def task_spec(task,rng,steps):
    if task=="cartpole":
        step=cartpole_step; n=4; dt=0.02; uclip=10.0
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
        step=pendulum_step; n=2; dt=0.05; uclip=2.0
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

def run_task(seed,task,controller,steps=3200):
    rng=np.random.default_rng(seed)
    step,n,params,Q,R,uclip,state,obs,estimate,cost,stable=task_spec(task,rng,steps)
    nominal=params[0]
    Anom,Bnom=numeric_linearization(step,n,nominal)
    K_nom=infinite_lqr_gain(Anom,Bnom,Q,R)
    Ahat=Anom.copy(); Bhat=Bnom.copy()
    K_adapt=K_nom.copy() if K_nom is not None else np.zeros((1,n))
    K_mpc=finite_horizon_gain(Ahat,Bhat,Q,R,24)
    states=[]; actions=[]; costs=[]; stables=[]; shift_costs=[]
    prev_o=None; prev_v=None; est_prev=None
    schedule=[0,1,0,2,1,3,0,2]
    seg=steps//len(schedule)
    for t in range(steps):
        rid=schedule[min(len(schedule)-1,t//seg)]
        p=params[rid]
        o=obs(state,t)
        if prev_v is None: prev_v=np.zeros(2 if task=="cartpole" else 1)
        est,vel=estimate(o,prev_o,prev_v)
        if est_prev is not None:
            states.append(est_prev.copy()); actions.append(float(u_prev))
        if len(actions)>=80 and t%20==0 and controller in ("CORE","GENERIC","MPC"):
            w=min(320,len(actions))
            seq=np.asarray(states[-w:]+[est.copy()],float)
            act=np.asarray(actions[-w:],float)
            cut=max(50,int(.75*w))
            try:
                Anew,Bnew=fit_ab(seq[:cut+1],act[:cut],lam=.08)
                if controller=="CORE":
                    Anew=Anew.copy(); Anew[np.abs(Anew)<0.02]=0.0
                Xv=seq[cut:-1]; Uv=act[cut:]; Yv=seq[cut+1:]
                oldp=(Xv@Ahat.T)+(Uv[:,None]@Bhat.T)
                newp=(Xv@Anew.T)+(Uv[:,None]@Bnew.T)
                if float(np.mean((newp-Yv)**2)) <= float(np.mean((oldp-Yv)**2))*0.95:
                    alpha=.20
                    Ac=(1-alpha)*Ahat+alpha*Anew; Bc=(1-alpha)*Bhat+alpha*Bnew
                    if controller=="MPC":
                        Knew=finite_horizon_gain(Ac,Bc,Q,R,24)
                    else:
                        Knew=infinite_lqr_gain(Ac,Bc,Q,R)
                    if Knew is not None and np.max(np.abs(Knew))<120:
                        rho=max(abs(np.linalg.eigvals(Ac-Bc@Knew)))
                        if rho<1.08:
                            Ahat,Bhat=Ac,Bc
                            if controller=="MPC": K_mpc=Knew
                            else: K_adapt=Knew
            except Exception:
                pass
        if controller=="ORACLE":
            At,Bt=numeric_linearization(step,n,p)
            Ko=finite_horizon_gain(At,Bt,Q,R,24)
            u=float(-(Ko@state)[0]) if Ko is not None else 0.0
        elif controller=="FROZEN":
            u=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
        elif controller=="MPC":
            use=K_mpc if K_mpc is not None else K_nom
            u=float(-(use@est)[0]) if use is not None else 0.0
        else:
            ua=float(-(K_adapt@est)[0]) if K_adapt is not None else 0.0
            un=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
            # Same nominal safety anchor; CORE and GENERIC differ only in learned model structure.
            u=.70*un+.30*ua
        u=float(np.clip(u,-uclip,uclip))
        c=cost(state,u); costs.append(c); stables.append(stable(state))
        within=t%seg
        if rid!=0 and within<80: shift_costs.append(c)
        next_state=step(state,u,p)
        prev_o=o; prev_v=vel; est_prev=est.copy(); u_prev=u; state=next_state
    return {
        "mean_cost":float(np.mean(costs)),
        "postshift_cost":float(np.mean(shift_costs)) if shift_costs else float("nan"),
        "stable_fraction":float(np.mean(stables)),
    }

def eval_seed(seed,steps=3200):
    ctrls=["CORE","GENERIC","MPC","FROZEN","ORACLE"]
    tasks={}
    for i,task in enumerate(("cartpole","pendulum")):
        tasks[task]={c:run_task(seed+1000*i,task,c,steps) for c in ctrls}
    def ratio(a,b,key):
        vals=[tasks[t][a][key]/(tasks[t][b][key]+1e-12) for t in tasks]
        return float(np.mean(vals))
    return {
        "seed":seed,
        "tasks":tasks,
        "core_mpc_cost_ratio":ratio("CORE","MPC","mean_cost"),
        "core_mpc_postshift_ratio":ratio("CORE","MPC","postshift_cost"),
        "core_generic_cost_ratio":ratio("CORE","GENERIC","mean_cost"),
        "mpc_frozen_ratio":ratio("MPC","FROZEN","mean_cost"),
        "core_oracle_ratio":ratio("CORE","ORACLE","mean_cost"),
        "core_stable":float(np.mean([tasks[t]["CORE"]["stable_fraction"] for t in tasks])),
        "mpc_stable":float(np.mean([tasks[t]["MPC"]["stable_fraction"] for t in tasks])),
    }

def med(records,key): return float(np.median([r[key] for r in records]))

def adjudicate(records,criteria):
    m={k:med(records,k) for k in [
        "core_mpc_cost_ratio","core_mpc_postshift_ratio","core_generic_cost_ratio",
        "mpc_frozen_ratio","core_oracle_ratio","core_stable","mpc_stable"
    ]}
    p=criteria["primary_noninferiority"]["medians"]
    checks={
      "core_stable":m["core_stable"]>=p["core_stable_min"],
      "mpc_stable":m["mpc_stable"]>=p["mpc_stable_min"],
      "core_mpc_cost":m["core_mpc_cost_ratio"]<=p["core_mpc_cost_ratio_max"],
      "core_mpc_postshift":m["core_mpc_postshift_ratio"]<=p["core_mpc_postshift_ratio_max"],
      "core_generic_low":m["core_generic_cost_ratio"]>=p["core_generic_cost_ratio_min"],
      "core_generic_high":m["core_generic_cost_ratio"]<=p["core_generic_cost_ratio_max"],
    }
    sg=criteria["primary_noninferiority"]["seed_guard"]
    sg_count=sum(
      r["core_stable"]>=sg["core_stable_min"] and
      r["mpc_stable"]>=sg["mpc_stable_min"] and
      r["core_mpc_cost_ratio"]<=sg["core_mpc_cost_ratio_max"] and
      r["core_mpc_postshift_ratio"]<=sg["core_mpc_postshift_ratio_max"] and
      r["core_generic_cost_ratio"]>=sg["core_generic_cost_ratio_min"] and
      r["core_generic_cost_ratio"]<=sg["core_generic_cost_ratio_max"]
      for r in records)
    noninf=all(checks.values()) and sg_count>=criteria["primary_noninferiority"]["seed_guard_required"]
    a=criteria["exclusive_core_advantage"]
    adv_check=m["core_mpc_cost_ratio"]<=a["core_mpc_cost_ratio_max"]
    adv_sg=sum(r["core_mpc_cost_ratio"]<=a["seed_guard_ratio_max"] for r in records)
    adv=adv_check and adv_sg>=a["seed_guard_required"]
    return {
      "medians":m,
      "noninferiority_checks":checks,
      "noninferiority_seed_guard_count":int(sg_count),
      "external_classical_control_noninferiority_pass":bool(noninf),
      "exclusive_core_advantage_median_check":bool(adv_check),
      "exclusive_core_advantage_seed_guard_count":int(adv_sg),
      "exclusive_core_advantage_pass":bool(adv),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--criteria",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    criteria=json.loads(Path(args.criteria).read_text())
    dev=[eval_seed(s,criteria["trajectory_steps"]) for s in criteria["development_seeds"]]
    confirm=[eval_seed(s,criteria["trajectory_steps"]) for s in criteria["confirmatory_seeds"]]
    out={
      "campaign":criteria["campaign"],
      "criteria_status":criteria["status"],
      "development":{"records":dev,"adjudication":adjudicate(dev,criteria)},
      "confirmatory":{"records":confirm,"adjudication":adjudicate(confirm,criteria)},
      "boundaries":criteria["fixed_boundaries"],
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["confirmatory"]["adjudication"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
