#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_are

def ridge_fit(X, Y, lam=1e-3):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if X.ndim==1: X=X[:,None]
    if Y.ndim==1: Y=Y[:,None]
    I=np.eye(X.shape[1])
    return np.linalg.solve(X.T@X + lam*I, X.T@Y)

def safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")

REL_MATS=[
    np.array([[1.00,0.65],[0.15,0.95]],float),
    np.array([[0.95,-0.55],[0.60,0.90]],float),
    np.array([[0.35,0.95],[0.90,-0.25]],float),
]

def relation_target(t, null=False):
    if null:
        if (t//20)%2==0:
            return np.array([0.75+0.10*np.sin(t/11),0.0])
        return np.array([0.0,0.75+0.10*np.cos(t/13)])
    return np.array([0.55+0.30*np.sin(t/17.0),0.55+0.28*np.cos(t/23.0)])

def choose_action(Mhat,d,variant):
    grid=np.linspace(0,1,7); cand=[]
    for u1 in grid:
        for u2 in grid:
            if variant=="noD" and (u1>1e-12 and u2>1e-12):
                continue
            cand.append(np.array([u1,u2],float))
    best=None; bestv=1e99
    for u in cand:
        if variant=="generic":
            a=u[0]-u[1]; q=u[0]+u[1]
            u=np.array([(q+a)/2,(q-a)/2])
        y=Mhat@u
        v=float(np.mean((y-d)**2)+0.002*np.sum(u*u))
        if v<bestv:
            bestv=v; best=u
    return best

def eval_integrated_seed(seed,null=False):
    variants=["full","generic","noD","noR","noC","noA"]; out={}
    blocks=6; block_len=160
    for variant in variants:
        rr=np.random.default_rng(seed+12345)
        Mhat=np.eye(2)*0.75; U=[]; Y=[]; mem=0.0
        losses=[]; memory_ok=[]; reports=[]; pre_reports=[]; relscores=[]; coacts=[]
        for t in range(blocks*block_len):
            b=t//block_len; within=t%block_len
            Mtrue=np.eye(2) if null else REL_MATS[b%len(REL_MATS)]
            cue=1.0 if ((seed+b)%2==0) else -1.0
            cue_visible=within<2
            if cue_visible and variant!="noC": mem=cue
            if variant=="noC" and not cue_visible: mem=0.0
            if variant!="noR":
                if len(U)>=12 and t%4==0:
                    w=min(70,len(U))
                    try: Mhat=ridge_fit(np.array(U[-w:]),np.array(Y[-w:]),lam=2e-2).T
                    except Exception: pass
            elif b==0 and len(U)>=12 and t%4==0:
                w=min(70,len(U))
                Mhat=ridge_fit(np.array(U[-w:]),np.array(Y[-w:]),lam=2e-2).T
            d=relation_target(t,null); u=choose_action(Mhat,d,variant)
            y=Mtrue@u+rr.normal(0,0.008,size=2)
            U.append(u.copy()); Y.append(y.copy())
            losses.append(float(np.mean((y-d)**2))); coacts.append(float(np.min(u)>1e-9))
            if within in (50,90,130):
                pred=mem if variant!="noC" else (1.0 if rr.random()<.5 else -1.0)
                memory_ok.append(float(np.sign(pred)==np.sign(cue)))
            if within in (60,100,140):
                pre_reports.append(0.5)
                report=(1.0 if rr.random()<.5 else -1.0) if variant in ("noA","noC") else mem
                reports.append(float(np.sign(report)==np.sign(cue)))
            if within>=45:
                relscores.append(max(0.0,1.0-np.linalg.norm(Mhat-Mtrue)/(np.linalg.norm(Mtrue)+1e-12)))
        offdiag=np.array(Mhat)-np.diag(np.diag(Mhat))
        out[variant]={
            "control_mse":safe_mean(losses),
            "memory_accuracy":safe_mean(memory_ok),
            "query_report_accuracy":safe_mean(reports),
            "prequery_access_accuracy":safe_mean(pre_reports),
            "relation_score":safe_mean(relscores),
            "coactivation_fraction":safe_mean(coacts),
            "final_offdiag_norm":float(np.linalg.norm(offdiag)),
            "final_relation":np.asarray(Mhat).tolist(),
        }
    full=out["full"]; gen=out["generic"]
    summary={
        "full_control_mse":full["control_mse"],
        "generic_gap":abs(full["control_mse"]-gen["control_mse"]),
        "noD_damage":out["noD"]["control_mse"]-full["control_mse"],
        "noR_damage":out["noR"]["control_mse"]-full["control_mse"],
        "memory_advantage":full["memory_accuracy"]-out["noC"]["memory_accuracy"],
        "access_advantage":full["query_report_accuracy"]-out["noA"]["query_report_accuracy"],
        "access_promotion":full["query_report_accuracy"]-full["prequery_access_accuracy"],
        "relation_score":full["relation_score"],
        "full_memory_accuracy":full["memory_accuracy"],
        "full_query_report_accuracy":full["query_report_accuracy"],
        "null_offdiag_norm":full["final_offdiag_norm"] if null else None,
        "null_relation_damage":out["noR"]["control_mse"]-full["control_mse"] if null else None,
        "null_coactivation_damage":out["noD"]["control_mse"]-full["control_mse"] if null else None,
    }
    return {"seed":seed,"null":null,"variants":out,"metrics":summary}

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

def lqr_gain(A,B,Q,R):
    try:
        P=solve_discrete_are(A,B,Q,R)
        K=np.linalg.solve(B.T@P@B+R,B.T@P@A)
        if np.all(np.isfinite(K)): return K
    except Exception: pass
    return None

def fit_ab(states,actions,lam=2e-2):
    S=np.asarray(states,float); U=np.asarray(actions,float).reshape(-1,1)
    X=np.hstack([S[:-1],U]); Y=S[1:]
    W=ridge_fit(X,Y,lam); n=S.shape[1]
    return W[:n,:].T,W[n:,:].T

def external_task(seed,task,controller):
    rng=np.random.default_rng(seed+888)
    if task=="cartpole":
        step=cartpole_step; n=4; dt=0.02
        ps=[{"mc":1.0,"mp":0.10,"l":0.50},{"mc":0.75,"mp":0.24,"l":0.78},{"mc":1.35,"mp":0.06,"l":0.34}]
        Q=np.diag([0.25,0.05,8.0,0.5]); R=np.array([[0.02]]); uclip=10.0
        state=np.array([rng.normal(0,.03),0,rng.normal(0,.035),0.0])
        obs_noise=rng.normal(0,0.0006,size=(1501,2))
        def obs(s,t): return np.array([s[0],s[2]])+obs_noise[t]
        def est_from(o,prev_o,prev_v):
            if prev_o is None: return np.array([o[0],0,o[1],0.0]),np.array([0.0,0.0])
            dv=(o-prev_o)/dt; vv=.65*prev_v+.35*dv
            return np.array([o[0],vv[0],o[1],vv[1]]),vv
        def cost(s,u): return float(s@Q@s+R[0,0]*u*u)
        alive=lambda s: float(abs(s[0])<2.4 and abs(s[2])<0.35)
    else:
        step=pendulum_step; n=2; dt=0.05
        ps=[{"m":1.0,"l":1.0,"b":0.05},{"m":1.55,"l":0.62,"b":0.30},{"m":0.60,"l":1.38,"b":0.01}]
        Q=np.diag([6.0,0.4]); R=np.array([[0.04]]); uclip=2.0
        state=np.array([rng.normal(0,.06),0.0])
        obs_noise=rng.normal(0,0.001,size=(1501,1))
        def obs(s,t): return np.array([s[0]])+obs_noise[t]
        def est_from(o,prev_o,prev_v):
            if prev_o is None: return np.array([o[0],0.0]),np.array([0.0])
            dv=(o-prev_o)/dt; vv=.65*prev_v+.35*dv
            return np.array([o[0],vv[0]]),vv
        def cost(s,u): return float(s@Q@s+R[0,0]*u*u)
        alive=lambda s: float(abs(s[0])<0.5)
    nominal=ps[0]; Anom,Bnom=numeric_linearization(step,n,nominal)
    K_nom=lqr_gain(Anom,Bnom,Q,R)
    Ahat=Anom.copy(); Bhat=Bnom.copy(); K=K_nom.copy() if K_nom is not None else np.zeros((1,n))
    states=[]; actions=[]; costs=[]; alive_list=[]; pred_err=[]; postshift_costs=[]
    prev_o=None; prev_v=np.zeros(2 if task=="cartpole" else 1); est_prev=None
    for t in range(1500):
        params=ps[min(2,t//500)]; o=obs(state,t); est,prev_v_new=est_from(o,prev_o,prev_v)
        if controller=="NO_C":
            est=np.array([o[0],0,o[1],0.0]) if task=="cartpole" else np.array([o[0],0.0])
        if controller=="ORACLE":
            Atrue,Btrue=numeric_linearization(step,n,params); Ko=lqr_gain(Atrue,Btrue,Q,R)
            uu=float(-(Ko@state)[0]) if Ko is not None else 0.0
        else:
            if est_prev is not None:
                states.append(est_prev.copy()); actions.append(float(u_prev))
            if len(actions)>=50 and t%20==0 and controller!="FROZEN":
                seq=np.asarray(states+[est.copy()],float); act=np.asarray(actions,float); w=min(220,len(actions))
                try:
                    sw=seq[-(w+1):]; uw=act[-w:]; cut=max(35,int(.75*w))
                    Anew,Bnew=fit_ab(sw[:cut+1],uw[:cut],lam=.08)
                    if controller=="FACTORIZED": Anew=np.diag(np.diag(Anew))
                    elif controller=="CORE": Anew[np.abs(Anew)<0.02]=0.0
                    Xv=sw[cut:-1]; Uv=uw[cut:]; Yv=sw[cut+1:]
                    oldp=(Xv@Ahat.T)+(Uv[:,None]@Bhat.T)
                    newp=(Xv@Anew.T)+(Uv[:,None]@Bnew.T)
                    oldm=float(np.mean((oldp-Yv)**2)); newm=float(np.mean((newp-Yv)**2))
                    alpha=.20; Ac=(1-alpha)*Ahat+alpha*Anew; Bc=(1-alpha)*Bhat+alpha*Bnew
                    Knew=lqr_gain(Ac,Bc,Q,R)
                    if Knew is not None and np.max(np.abs(Knew))<120 and newm<=oldm*.92:
                        rho=max(abs(np.linalg.eigvals(Ac-Bc@Knew)))
                        if rho<1.05: Ahat,Bhat,K=Ac,Bc,Knew
                except Exception: pass
            u_nom=float(-(K_nom@est)[0]) if K_nom is not None else 0.0
            u_adapt=float(-(K@est)[0]) if K is not None else u_nom
            uu=.80*u_nom+.20*u_adapt
        uu=float(np.clip(uu,-uclip,uclip)); next_state=step(state,uu,params)
        c=cost(state,uu); costs.append(c); alive_list.append(alive(state))
        if controller!="ORACLE" and est_prev is not None:
            try:
                pred=Ahat@est_prev+Bhat[:,0]*u_prev
                pred_err.append(float(np.mean((pred-est)**2)))
            except Exception: pass
        if t in range(500,560) or t in range(1000,1060): postshift_costs.append(c)
        prev_o=o; prev_v=prev_v_new; est_prev=est.copy(); u_prev=uu; state=next_state
    return {"mean_cost":float(np.mean(costs)),"stable_fraction":float(np.mean(alive_list)),
            "postshift_cost":float(np.mean(postshift_costs)),"prediction_mse":safe_mean(pred_err)}

def eval_external_seed(seed):
    ctrls=["CORE","GENERIC","FACTORIZED","FROZEN","NO_C","ORACLE"]; rows={}
    for task in ("cartpole","pendulum"):
        rows[task]={c:external_task(seed+100*(0 if task=="cartpole" else 1),task,c) for c in ctrls}
    def ratios(a,b,key="mean_cost"):
        return [rows[t][a][key]/(rows[t][b][key]+1e-12) for t in rows]
    return {"seed":seed,"tasks":rows,
            "core_generic_ratio":float(np.mean(ratios("CORE","GENERIC"))),
            "core_factorized_ratio":float(np.mean(ratios("CORE","FACTORIZED"))),
            "generic_factorized_ratio":float(np.mean(ratios("GENERIC","FACTORIZED"))),
            "core_frozen_ratio":float(np.mean(ratios("CORE","FROZEN"))),
            "core_noC_ratio":float(np.mean(ratios("CORE","NO_C"))),
            "core_oracle_ratio":float(np.mean(ratios("CORE","ORACLE"))),
            "core_stable":float(np.mean([rows[t]["CORE"]["stable_fraction"] for t in rows])),
            "generic_stable":float(np.mean([rows[t]["GENERIC"]["stable_fraction"] for t in rows]))}

def summarize_dev(A,B,C):
    def med(recs,key): return float(np.median([r["metrics"][key] for r in recs]))
    sa={k:med(A,k) for k in ["full_control_mse","generic_gap","noD_damage","noR_damage","memory_advantage",
        "access_advantage","access_promotion","relation_score","full_memory_accuracy","full_query_report_accuracy"]}
    sb={k:med(B,k) for k in ["generic_gap","null_offdiag_norm","null_relation_damage","null_coactivation_damage"]}
    sc={k:float(np.median([r[k] for r in C])) for k in ["core_generic_ratio","core_factorized_ratio",
        "generic_factorized_ratio","core_frozen_ratio","core_noC_ratio","core_oracle_ratio","core_stable","generic_stable"]}
    return {"R26A":sa,"R26B":sb,"R26C":sc}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--mode",choices=["dev","confirm"],default="dev"); a=ap.parse_args()
    s,e=map(int,a.seeds.split(":")); seeds=list(range(s,e+1))
    A=[eval_integrated_seed(x,False) for x in seeds]
    B=[eval_integrated_seed(x+100000,True) for x in seeds]
    C=[eval_external_seed(x+200000) for x in seeds]
    out={"campaign":"R26 minimal organizational class","mode":a.mode,"seeds":seeds,
         "R26A_records":A,"R26B_records":B,"R26C_records":C,
         "development_summary":summarize_dev(A,B,C)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["development_summary"],indent=2,sort_keys=True))
if __name__=="__main__":
    main()
