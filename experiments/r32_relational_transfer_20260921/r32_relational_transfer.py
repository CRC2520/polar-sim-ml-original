#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

N=6
B=np.eye(N)*0.65
CONTROL_REG=0.03
BLOCK_LEN=220
BLOCKS=4
WINDOW=100
UPDATE_EVERY=5
BURN=45

DEV_FAMILIES=["sparse","modular"]
CONFIRM_FAMILIES=["dense_lowrank","ring","random_dag","skew"]
STRICT_HELDOUT_FAMILIES=["random_dag","skew"]
CONDITIONS=["diagonal","gain_shift","static_relational","relational_shift"]
CONTROLLERS=["CORE","GENERIC","NO_R","FACTORIZED","FROZEN","ORACLE"]

def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")

def offdiag(M):
    M=np.asarray(M,float)
    return M-np.diag(np.diag(M))

def stabilize(A,maxrho=0.88):
    A=np.asarray(A,float)
    rho=float(max(abs(np.linalg.eigvals(A))))
    if rho<=maxrho:
        return A
    d=np.diag(np.diag(A)); w=A-d
    lo,hi=0.0,1.0
    for _ in range(48):
        m=(lo+hi)/2
        r=float(max(abs(np.linalg.eigvals(d+m*w))))
        if r<=maxrho: lo=m
        else: hi=m
    return d+lo*w

def gen_offdiag(rng,family,n=N):
    W=np.zeros((n,n),float)
    if family=="sparse":
        cand=[(i,j) for i in range(n) for j in range(n) if i!=j]
        rng.shuffle(cand)
        for i,j in cand[:8]:
            W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.07,0.14)
    elif family=="modular":
        for mod in [range(0,3),range(3,6)]:
            for i in mod:
                for j in mod:
                    if i!=j and rng.random()<0.65:
                        W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.05,0.12)
        for _ in range(3):
            i=int(rng.integers(0,3)); j=int(rng.integers(3,6))
            if rng.random()<0.5: i,j=j,i
            W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.04,0.09)
    elif family=="dense_lowrank":
        a=rng.normal(size=n); b=rng.normal(size=n)
        W=np.outer(a,b); np.fill_diagonal(W,0.0)
        W=W/(np.linalg.norm(W)+1e-12)*0.40
    elif family=="ring":
        for i in range(n):
            W[i,(i-1)%n]=rng.choice([-1.0,1.0])*rng.uniform(0.07,0.13)
            if rng.random()<0.70:
                W[i,(i+2)%n]=rng.choice([-1.0,1.0])*rng.uniform(0.04,0.09)
    elif family=="random_dag":
        order=rng.permutation(n)
        for aidx in range(n):
            for bidx in range(aidx+1,n):
                if rng.random()<0.45:
                    i=int(order[bidx]); j=int(order[aidx])
                    W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.05,0.13)
    elif family=="skew":
        raw=rng.normal(size=(n,n))
        W=raw-raw.T; np.fill_diagonal(W,0.0)
        W=W/(np.linalg.norm(W)+1e-12)*0.38
    else:
        raise ValueError(family)
    return W

def base_matrix(seed,family):
    rng=np.random.default_rng(seed)
    diag=0.52+rng.uniform(-0.035,0.035,N)
    return stabilize(np.diag(diag)+gen_offdiag(rng,family))

def matrices_for(seed,family,condition):
    A0=base_matrix(seed,family); d=np.diag(A0).copy()
    if condition=="static_relational":
        return [A0.copy() for _ in range(BLOCKS)]
    if condition=="diagonal":
        return [np.diag(d+s) for s in [-0.025,0.0,0.025,0.05]]
    if condition=="gain_shift":
        w=offdiag(A0)
        return [stabilize(np.diag(d+s)+w) for s in [0.0,0.05,-0.04,0.07]]
    if condition=="relational_shift":
        out=[A0]; norm0=float(np.linalg.norm(offdiag(A0)))
        for b in range(1,BLOCKS):
            rr=np.random.default_rng(seed+10000*b+77)
            w=gen_offdiag(rr,family); nw=float(np.linalg.norm(w))
            if nw>1e-12 and norm0>1e-12: w*=norm0/nw
            out.append(stabilize(np.diag(d)+w))
        return out
    raise ValueError(condition)

def target_seq(seed,T):
    rng=np.random.default_rng(seed+123)
    phases=rng.uniform(0.0,2.0*np.pi,N)
    freqs=np.linspace(23.0,41.0,N)
    t=np.arange(T)
    return np.stack([
        0.32*np.sin(t/freqs[i]+phases[i])+
        0.08*np.sin(t/9.0+phases[(i+2)%N])
        for i in range(N)
    ],axis=1)

def fit_A(X,U,Y,lam=0.05):
    X=np.asarray(X,float); U=np.asarray(U,float); Y=np.asarray(Y,float)
    R=Y-U@B.T
    W=np.linalg.solve(X.T@X+lam*np.eye(N),X.T@R)
    return W.T

def choose_u(Ahat,x,target):
    rhs=target-Ahat@x
    u=np.linalg.solve(B.T@B+CONTROL_REG*np.eye(N),B.T@rhs)
    return np.clip(u,-1.4,1.4)

def rel_err(Ahat,Atrue):
    den=float(np.linalg.norm(offdiag(Atrue)))
    if den<1e-12: return float("nan")
    return float(np.linalg.norm(offdiag(Ahat-Atrue))/den)

def wrong_matrix(Ahat,shift):
    d=np.diag(np.diag(Ahat)); w=offdiag(Ahat)
    wr=np.roll(w,shift=shift,axis=1); np.fill_diagonal(wr,0.0)
    nw=float(np.linalg.norm(wr)); no=float(np.linalg.norm(w))
    if nw>1e-12: wr*=no/nw
    return d+wr

def run_controller(seed,family,condition,controller):
    T=BLOCK_LEN*BLOCKS
    mats=matrices_for(seed,family,condition)
    targets=target_seq(seed,T)
    rng=np.random.default_rng(seed+2024)
    proc=rng.normal(0.0,0.004,size=(T,N))
    obs=rng.normal(0.0,0.002,size=(T+1,N))
    x=rng.normal(0.0,0.03,size=N)
    X=[]; U=[]; Y=[]
    Ahat=np.eye(N)*0.52; base_off=None
    costs=[]; post=[]; stable=[]; errors=[]
    for t in range(T):
        block=t//BLOCK_LEN; within=t%BLOCK_LEN
        Atrue=mats[block]; xo=x+obs[t]
        if controller!="ORACLE" and len(X)>=30 and t%UPDATE_EVERY==0:
            w=min(WINDOW,len(X))
            try:
                Afull=fit_A(np.array(X[-w:]),np.array(U[-w:]),np.array(Y[-w:]))
                if controller=="FACTORIZED":
                    Anew=np.diag(np.diag(Afull))
                elif controller=="NO_R":
                    if block==0: Anew=Afull
                    else:
                        if base_off is None: base_off=offdiag(Ahat).copy()
                        Anew=np.diag(np.diag(Afull))+base_off
                elif controller=="FROZEN":
                    Anew=Afull if block==0 else Ahat
                elif controller=="CORE":
                    Anew=Afull.copy(); Anew[np.abs(Anew)<0.012]=0.0
                elif controller=="GENERIC":
                    Anew=Afull
                else:
                    raise ValueError(controller)
                if controller!="FROZEN" or block==0:
                    Ahat=0.35*Anew+0.65*Ahat
            except Exception:
                pass
        if block==1 and within==0 and base_off is None:
            base_off=offdiag(Ahat).copy()
        Ause=Atrue if controller=="ORACLE" else Ahat
        u=choose_u(Ause,xo,targets[t])
        xnext=Atrue@x+B@u+proc[t]
        yobs=xnext+obs[t+1]
        X.append(xo.copy()); U.append(u.copy()); Y.append(yobs.copy())
        c=float(np.mean((xnext-targets[t])**2)+0.01*np.mean(u*u))
        costs.append(c); stable.append(float(np.linalg.norm(xnext)<2.0))
        if block>=1: post.append(c)
        if block>=1 and within>=BURN and controller!="ORACLE":
            e=rel_err(Ahat,Atrue)
            if np.isfinite(e): errors.append(e)
        x=xnext
    return {
        "mean_cost":mean(costs),
        "post_cost":mean(post),
        "stable_fraction":mean(stable),
        "relation_error":mean(errors)
    }

def run_core_intervention(seed,family,condition):
    T=BLOCK_LEN*BLOCKS
    mats=matrices_for(seed,family,condition)
    targets=target_seq(seed,T)
    rng=np.random.default_rng(seed+2024)
    proc=rng.normal(0.0,0.004,size=(T,N))
    obs=rng.normal(0.0,0.002,size=(T+1,N))
    x=rng.normal(0.0,0.03,size=N)
    X=[]; U=[]; Y=[]
    Ahat=np.eye(N)*0.52
    frozen_off=offdiag(Ahat).copy()
    delta=[]; wrong=[]; rescue=[]
    for t in range(T):
        block=t//BLOCK_LEN; within=t%BLOCK_LEN
        Atrue=mats[block]; xo=x+obs[t]
        if within==0 and block>=1:
            frozen_off=offdiag(Ahat).copy()
            frozen_off[np.abs(frozen_off)<0.020]=0.0
        if len(X)>=30 and t%UPDATE_EVERY==0:
            w=min(WINDOW,len(X))
            try:
                Afull=fit_A(np.array(X[-w:]),np.array(U[-w:]),np.array(Y[-w:]))
                Anew=Afull.copy(); Anew[np.abs(Anew)<0.012]=0.0
                Ahat=0.35*Anew+0.65*Ahat
            except Exception:
                pass
        u=choose_u(Ahat,xo,targets[t])
        xnext=Atrue@x+B@u+proc[t]
        yobs=xnext+obs[t+1]
        X.append(xo.copy()); U.append(u.copy()); Y.append(yobs.copy())
        if block>=1 and within>=BURN:
            AnoR=np.diag(np.diag(Ahat))+frozen_off
            un=choose_u(AnoR,xo,targets[t]); uc=choose_u(Ahat,xo,targets[t])
            yn=Atrue@x+B@un; yc=Atrue@x+B@uc
            cn=float(np.mean((yn-targets[t])**2)+0.01*np.mean(un*un))
            cc=float(np.mean((yc-targets[t])**2)+0.01*np.mean(uc*uc))
            delta.append(cn-cc)
            if condition=="relational_shift":
                Aw=wrong_matrix(Ahat,(block%3)+1)
                uw=choose_u(Aw,xo,targets[t]); uo=choose_u(Atrue,xo,targets[t])
                yw=Atrue@x+B@uw; yo=Atrue@x+B@uo
                cw=float(np.mean((yw-targets[t])**2)+0.01*np.mean(uw*uw))
                co=float(np.mean((yo-targets[t])**2)+0.01*np.mean(uo*uo))
                wrong.append(cw-cc); rescue.append(cw-co)
        x=xnext
    return {
        "delta_R_cf":mean(delta),
        "wrong_relation_damage":mean(wrong),
        "correct_relation_rescue":mean(rescue)
    }

def eval_seed(seed,mode):
    families=DEV_FAMILIES if mode=="dev" else CONFIRM_FAMILIES
    details={}; interventions={}
    for fi,fam in enumerate(families):
        details[fam]={}; interventions[fam]={}
        for ci,cond in enumerate(CONDITIONS):
            s=seed+1000*fi+100*ci
            details[fam][cond]={c:run_controller(s,fam,cond,c) for c in CONTROLLERS}
            interventions[fam][cond]=run_core_intervention(s,fam,cond)
    def dint(fams,cond):
        return float(np.mean([interventions[f][cond]["delta_R_cf"] for f in fams]))
    def ratio(fams,cond,a,b):
        return float(np.mean([
            details[f][cond][a]["post_cost"]/(details[f][cond][b]["post_cost"]+1e-12)
            for f in fams
        ]))
    def recovery(fams):
        return float(np.mean([
            details[f]["relational_shift"]["NO_R"]["relation_error"]-
            details[f]["relational_shift"]["CORE"]["relation_error"]
            for f in fams
        ]))
    d={c:dint(families,c) for c in CONDITIONS}
    rec={
        "seed":seed,
        "families":families,
        "delta_R_diagonal":d["diagonal"],
        "delta_R_gain_shift":d["gain_shift"],
        "delta_R_static_relational":d["static_relational"],
        "delta_R_relational_shift":d["relational_shift"],
        "specificity_margin":float(d["relational_shift"]-max(d["diagonal"],d["gain_shift"],d["static_relational"])),
        "core_generic_ratio_rel":ratio(families,"relational_shift","CORE","GENERIC"),
        "core_factorized_ratio_rel":ratio(families,"relational_shift","CORE","FACTORIZED"),
        "core_frozen_ratio_rel":ratio(families,"relational_shift","CORE","FROZEN"),
        "relation_recovery_gain":recovery(families),
        "wrong_relation_damage":float(np.mean([
            interventions[f]["relational_shift"]["wrong_relation_damage"] for f in families
        ])),
        "correct_relation_rescue":float(np.mean([
            interventions[f]["relational_shift"]["correct_relation_rescue"] for f in families
        ])),
        "core_stable":float(np.mean([
            details[f][c]["CORE"]["stable_fraction"] for f in families for c in CONDITIONS
        ])),
        "details":details,
        "interventions":interventions
    }
    if mode=="confirm":
        hf=STRICT_HELDOUT_FAMILIES
        hd={c:dint(hf,c) for c in CONDITIONS}
        rec.update({
            "heldout_delta_R_relational_shift":hd["relational_shift"],
            "heldout_specificity_margin":float(hd["relational_shift"]-max(hd["diagonal"],hd["gain_shift"],hd["static_relational"])),
            "heldout_core_generic_ratio_rel":ratio(hf,"relational_shift","CORE","GENERIC"),
            "heldout_core_factorized_ratio_rel":ratio(hf,"relational_shift","CORE","FACTORIZED"),
            "heldout_relation_recovery_gain":recovery(hf)
        })
    return rec

def summarize(records):
    keys=[
        "delta_R_diagonal","delta_R_gain_shift","delta_R_static_relational",
        "delta_R_relational_shift","specificity_margin","core_generic_ratio_rel",
        "core_factorized_ratio_rel","core_frozen_ratio_rel","relation_recovery_gain",
        "wrong_relation_damage","correct_relation_rescue","core_stable"
    ]
    if "heldout_specificity_margin" in records[0]:
        keys += [
            "heldout_delta_R_relational_shift","heldout_specificity_margin",
            "heldout_core_generic_ratio_rel","heldout_core_factorized_ratio_rel",
            "heldout_relation_recovery_gain"
        ]
    return {k:float(np.median([r[k] for r in records])) for k in keys}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",required=True)
    ap.add_argument("--mode",choices=["dev","confirm"],required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    lo,hi=map(int,a.seeds.split(":"))
    seeds=list(range(lo,hi+1))
    records=[eval_seed(s,a.mode) for s in seeds]
    out={
        "campaign":"R32 strong relational transfer / relation-specific intervention",
        "mode":a.mode,
        "development_families":DEV_FAMILIES,
        "confirmatory_families":CONFIRM_FAMILIES,
        "strict_heldout_families":STRICT_HELDOUT_FAMILIES,
        "seeds":seeds,
        "records":records,
        "medians":summarize(records)
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["medians"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
