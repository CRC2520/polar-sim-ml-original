#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

N=6
B=np.eye(N,dtype=float)*0.65
CONTROL_REG=0.03
SEQ=96
BLOCKS=3
BLOCK_LEN=SEQ//BLOCKS
TRAIN_EPISODES=160
EPOCHS=30
BATCH=16
HIDDEN=32
WINDOW=8
ADAPT_BURN=8
TRAIN_FAMILIES=["sparse","modular","dense_lowrank"]
DEV_EVAL_FAMILIES=["ring"]
CONFIRM_EVAL_FAMILIES=["random_dag","skew"]
DEV_ARCHS=["CURRENT_MLP","RNN","GRU"]
CONFIRM_ARCHS=["CURRENT_MLP","RNN","GRU","WINDOW_MLP","LSTM"]
STRICT_HELDOUT_ARCHS=["WINDOW_MLP","LSTM"]
CONDITIONS=["diagonal","gain_shift","static_relational","relational_shift"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def offdiag(A):
    A=np.asarray(A,float)
    return A-np.diag(np.diag(A))

def stabilize(A,maxrho=0.90):
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
        for i,j in cand[:9]:
            W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.08,0.16)
    elif family=="modular":
        for mod in [range(0,3),range(3,6)]:
            for i in mod:
                for j in mod:
                    if i!=j and rng.random()<0.75:
                        W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.06,0.14)
        for _ in range(4):
            i=int(rng.integers(0,3)); j=int(rng.integers(3,6))
            if rng.random()<0.5: i,j=j,i
            W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.05,0.10)
    elif family=="dense_lowrank":
        a=rng.normal(size=n); b=rng.normal(size=n)
        W=np.outer(a,b); np.fill_diagonal(W,0.0)
        W=W/(np.linalg.norm(W)+1e-12)*0.48
    elif family=="ring":
        for i in range(n):
            W[i,(i-1)%n]=rng.choice([-1.0,1.0])*rng.uniform(0.09,0.16)
            W[i,(i+1)%n]=rng.choice([-1.0,1.0])*rng.uniform(0.05,0.11)
            if rng.random()<0.65:
                W[i,(i+2)%n]=rng.choice([-1.0,1.0])*rng.uniform(0.04,0.09)
    elif family=="random_dag":
        order=rng.permutation(n)
        for aidx in range(n):
            for bidx in range(aidx+1,n):
                if rng.random()<0.52:
                    i=int(order[bidx]); j=int(order[aidx])
                    W[i,j]=rng.choice([-1.0,1.0])*rng.uniform(0.07,0.15)
    elif family=="skew":
        raw=rng.normal(size=(n,n))
        W=raw-raw.T; np.fill_diagonal(W,0.0)
        W=W/(np.linalg.norm(W)+1e-12)*0.46
    else:
        raise ValueError(family)
    return W

def base_matrix(seed,family):
    rng=np.random.default_rng(seed)
    d=0.50+rng.uniform(-0.03,0.03,N)
    return stabilize(np.diag(d)+gen_offdiag(rng,family))

def matrices(seed,family,condition):
    A0=base_matrix(seed,family)
    d=np.diag(A0).copy()
    if condition=="diagonal":
        return [np.diag(d+s) for s in [-0.03,0.02,0.055]]
    if condition=="gain_shift":
        w=offdiag(A0)
        return [stabilize(np.diag(d+s)+w) for s in [0.0,0.055,-0.045]]
    if condition=="static_relational":
        return [A0.copy() for _ in range(BLOCKS)]
    if condition=="relational_shift":
        out=[A0]
        norm0=float(np.linalg.norm(offdiag(A0)))
        for b in range(1,BLOCKS):
            rr=np.random.default_rng(seed+9173*b+71)
            w=gen_offdiag(rr,family)
            nw=float(np.linalg.norm(w))
            if nw>1e-12 and norm0>1e-12:
                w*=norm0/nw
            out.append(stabilize(np.diag(d)+w))
        return out
    raise ValueError(condition)

def target_sequence(seed,T=SEQ):
    rng=np.random.default_rng(seed+41)
    phases=rng.uniform(0,2*np.pi,N)
    freq=np.linspace(14.0,29.0,N)
    t=np.arange(T)
    return np.stack([
        0.34*np.sin(t/freq[i]+phases[i])+
        0.09*np.cos(t/7.0+phases[(i+1)%N])
        for i in range(N)
    ],axis=1)

def choose_oracle(A,x,target):
    rhs=target-A@x
    u=np.linalg.solve(B.T@B+CONTROL_REG*np.eye(N),B.T@rhs)
    return np.clip(u,-1.5,1.5)

def make_episode(seed,family,condition):
    As=matrices(seed,family,condition)
    target=target_sequence(seed)
    rng=np.random.default_rng(seed+991)
    x=rng.normal(0,0.035,N)
    prev_u=np.zeros(N)
    tokens=[]; y=[]; rel=[]; xs=[]
    for t in range(SEQ):
        b=t//BLOCK_LEN
        A=As[b]
        obs=x+rng.normal(0,0.003,N)
        token=np.concatenate([obs,prev_u])
        drift=A@x
        tokens.append(token)
        y.append(drift)
        rel.append(offdiag(A).reshape(-1))
        xs.append(x.copy())
        # independent exploratory behavior; no learner receives the true A
        beh=0.72*prev_u+0.28*rng.normal(0,0.55,N)
        beh=np.clip(beh,-1.0,1.0)
        x=A@x+B@beh+rng.normal(0,0.006,N)
        prev_u=beh
    return {
      "tokens":np.asarray(tokens,np.float32),
      "drift":np.asarray(y,np.float32),
      "relations":np.asarray(rel,np.float32),
      "states":np.asarray(xs,np.float32),
      "matrices":np.asarray([As[t//BLOCK_LEN] for t in range(SEQ)],np.float32),
      "family":family,
      "condition":condition
    }

class CurrentMLP(nn.Module):
    def __init__(self,inp=12,h=HIDDEN):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(inp,64),nn.Tanh(),nn.Linear(64,h),nn.Tanh())
        self.head=nn.Linear(h,N)
    def forward_seq(self,x):
        z=self.body(x)
        return self.head(z),z
    def reset_predictions(self,x):
        return self.forward_seq(x)[0]

class WindowMLP(nn.Module):
    def __init__(self,inp=12,h=HIDDEN,k=WINDOW):
        super().__init__()
        self.inp=inp; self.k=k
        self.body=nn.Sequential(nn.Linear(inp*k,80),nn.Tanh(),nn.Linear(80,h),nn.Tanh())
        self.head=nn.Linear(h,N)
    def windows(self,x,reset=False):
        Bn,T,D=x.shape
        if reset:
            out=torch.zeros((Bn,T,self.k,D),dtype=x.dtype,device=x.device)
            out[:,:,-1,:]=x
            return out.reshape(Bn,T,self.k*D)
        pad=torch.zeros((Bn,self.k-1,D),dtype=x.dtype,device=x.device)
        xp=torch.cat([pad,x],dim=1)
        ws=[]
        for t in range(T):
            ws.append(xp[:,t:t+self.k,:].reshape(Bn,-1))
        return torch.stack(ws,dim=1)
    def forward_seq(self,x):
        z=self.body(self.windows(x,False))
        return self.head(z),z
    def reset_predictions(self,x):
        z=self.body(self.windows(x,True))
        return self.head(z)

class RecurrentAgent(nn.Module):
    def __init__(self,kind,inp=12,h=HIDDEN):
        super().__init__()
        self.kind=kind
        if kind=="RNN":
            self.core=nn.RNN(inp,h,batch_first=True,nonlinearity="tanh")
        elif kind=="GRU":
            self.core=nn.GRU(inp,h,batch_first=True)
        elif kind=="LSTM":
            self.core=nn.LSTM(inp,h,batch_first=True)
        else:
            raise ValueError(kind)
        self.head=nn.Linear(h,N)
    def forward_seq(self,x):
        z,_=self.core(x)
        return self.head(z),z
    def reset_predictions(self,x):
        Bn,T,D=x.shape
        flat=x.reshape(Bn*T,1,D)
        z,_=self.core(flat)
        return self.head(z[:,-1,:]).reshape(Bn,T,N)

def make_model(kind,seed):
    set_seed(seed)
    if kind=="CURRENT_MLP": return CurrentMLP()
    if kind=="WINDOW_MLP": return WindowMLP()
    return RecurrentAgent(kind)

def make_training_set(seed):
    rng=np.random.default_rng(seed)
    eps=[]
    for i in range(TRAIN_EPISODES):
        fam=TRAIN_FAMILIES[int(rng.integers(0,len(TRAIN_FAMILIES)))]
        cond=CONDITIONS[int(rng.integers(0,len(CONDITIONS)))]
        eps.append(make_episode(seed*1000+100+i,fam,cond))
    X=torch.tensor(np.stack([e["tokens"] for e in eps]))
    Y=torch.tensor(np.stack([e["drift"] for e in eps]))
    return X,Y

def train_model(seed,kind,X,Y):
    model=make_model(kind,seed+hash(kind)%10000)
    opt=torch.optim.Adam(model.parameters(),lr=0.0025,weight_decay=1e-5)
    lossfn=nn.MSELoss()
    gen=torch.Generator().manual_seed(seed+77)
    n=X.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm=torch.randperm(n,generator=gen)
        for s in range(0,n,BATCH):
            idx=perm[s:s+BATCH]
            pred,_=model.forward_seq(X[idx])
            mask=torch.tensor([(t%BLOCK_LEN)>=ADAPT_BURN for t in range(SEQ)],dtype=torch.bool)
            loss=lossfn(pred[:,mask,:],Y[idx][:,mask,:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step()
    return model.eval()

def mse(a,b):
    return float(np.mean((np.asarray(a)-np.asarray(b))**2))

def fixed_eval(model,episode):
    x=torch.tensor(episode["tokens"])[None,:,:]
    y=episode["drift"]
    with torch.no_grad():
        pred,z=model.forward_seq(x)
        reset=model.reset_predictions(x)
    pred=pred[0].numpy(); z=z[0].numpy(); reset=reset[0].numpy()
    mask=np.array([(t%BLOCK_LEN)>=ADAPT_BURN for t in range(SEQ)],dtype=bool)
    intact=mse(pred[mask],y[mask])
    reset_mse=mse(reset[mask],y[mask])
    return {
      "intact_mse":intact,
      "reset_mse":reset_mse,
      "history_damage":reset_mse-intact,
      "pred":pred,
      "repr":z
    }

def linear_r2(X,Y,lam=1e-3):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    n=len(X); cut=max(10,int(0.65*n))
    Xm=X[:cut].mean(0,keepdims=True); Xs=X[:cut].std(0,keepdims=True)+1e-6
    Xn=(X-Xm)/Xs
    Xa=np.hstack([Xn,np.ones((n,1))])
    I=np.eye(Xa.shape[1]); I[-1,-1]=0
    W=np.linalg.solve(Xa[:cut].T@Xa[:cut]+lam*I,Xa[:cut].T@Y[:cut])
    yp=Xa[cut:]@W
    yt=Y[cut:]
    den=float(np.sum((yt-yt.mean(0,keepdims=True))**2))+1e-12
    return float(1.0-np.sum((yp-yt)**2)/den)

def d_rank1_damage(model,z,y):
    z=np.asarray(z,float); y=np.asarray(y,float)
    n=len(z); cut=max(10,int(0.65*n))
    mu=z[:cut].mean(0,keepdims=True)
    _,_,vt=np.linalg.svd(z[:cut]-mu,full_matrices=False)
    pc=vt[:1].T
    zr=mu+(z-mu)@pc@pc.T
    with torch.no_grad():
        pr=model.head(torch.tensor(zr,dtype=torch.float32)).numpy()
        pi=model.head(torch.tensor(z,dtype=torch.float32)).numpy()
    return float(mse(pr[cut:],y[cut:])-mse(pi[cut:],y[cut:]))

def relation_decode_r2(z,relations):
    # remove diagonal positions that are always zero
    mask=np.array([i!=j for i in range(N) for j in range(N)])
    return linear_r2(z,np.asarray(relations)[:,mask],lam=0.02)

def state_from_prefix(model,kind,prefix):
    x=torch.tensor(prefix,dtype=torch.float32)[None,:,:]
    if kind=="CURRENT_MLP":
        return None
    if kind=="WINDOW_MLP":
        arr=np.asarray(prefix[-(WINDOW-1):],np.float32)
        return arr
    with torch.no_grad():
        _,state=model.core(x)
    return state

def predict_from_state(model,kind,state,token):
    tok=torch.tensor(token,dtype=torch.float32).view(1,1,-1)
    with torch.no_grad():
        if kind=="CURRENT_MLP":
            p,_=model.forward_seq(tok)
            return p[0,0].numpy()
        if kind=="WINDOW_MLP":
            hist=np.asarray(state,np.float32)
            full=np.vstack([hist,np.asarray(token,np.float32)])
            if len(full)<WINDOW:
                pad=np.zeros((WINDOW-len(full),full.shape[1]),np.float32)
                full=np.vstack([pad,full])
            full=full[-WINDOW:]
            z=model.body(torch.tensor(full.reshape(1,1,-1)))
            return model.head(z)[0,0].numpy()
        z,newstate=model.core(tok,state)
        return model.head(z)[0,0].numpy()

def relation_transplant_damage(model,kind,epA,epB):
    points=[38,45,54,70,78,86]
    ds=[]
    for t in points:
        sa=state_from_prefix(model,kind,epA["tokens"][:t])
        sb=state_from_prefix(model,kind,epB["tokens"][:t])
        token=epA["tokens"][t]
        pa=predict_from_state(model,kind,sa,token)
        pb=predict_from_state(model,kind,sb,token)
        y=epA["drift"][t]
        ds.append(mse(pb,y)-mse(pa,y))
    return float(np.mean(ds))

def rollout(model,kind,seed,family):
    cond="relational_shift"
    As=matrices(seed,family,cond)
    target=target_sequence(seed)
    rng=np.random.default_rng(seed+404)
    x=rng.normal(0,0.035,N)
    prev_u=np.zeros(N)
    tokens=[]
    costs=[]; adapt_costs=[]; stable=[]
    for t in range(SEQ):
        A=As[t//BLOCK_LEN]
        token=np.concatenate([x+rng.normal(0,0.003,N),prev_u]).astype(np.float32)
        tokens.append(token)
        if kind=="CURRENT_MLP":
            prefix=np.asarray([token],np.float32)
        else:
            prefix=np.asarray(tokens,np.float32)
        with torch.no_grad():
            p,_=model.forward_seq(torch.tensor(prefix)[None,:,:])
        drift=p[0,-1].numpy()
        rhs=target[t]-drift
        u=np.linalg.solve(B.T@B+CONTROL_REG*np.eye(N),B.T@rhs)
        u=np.clip(u,-1.5,1.5)
        xnext=A@x+B@u+rng.normal(0,0.006,N)
        cc=float(np.mean((xnext-target[t])**2)+0.01*np.mean(u*u))
        costs.append(cc)
        if (t%BLOCK_LEN)>=ADAPT_BURN: adapt_costs.append(cc)
        stable.append(float(np.linalg.norm(xnext)<2.5))
        x=xnext; prev_u=u
    return float(np.mean(costs)),float(np.mean(adapt_costs)),float(np.mean(stable))

def oracle_rollout(seed,family):
    As=matrices(seed,family,"relational_shift")
    target=target_sequence(seed)
    rng=np.random.default_rng(seed+404)
    x=rng.normal(0,0.035,N)
    costs=[]; adapt_costs=[]
    for t in range(SEQ):
        A=As[t//BLOCK_LEN]
        u=choose_oracle(A,x,target[t])
        xnext=A@x+B@u+rng.normal(0,0.006,N)
        cc=float(np.mean((xnext-target[t])**2)+0.01*np.mean(u*u))
        costs.append(cc)
        if (t%BLOCK_LEN)>=ADAPT_BURN: adapt_costs.append(cc)
        x=xnext
    return float(np.mean(costs)),float(np.mean(adapt_costs))

def evaluate_arch(seed,kind,model,families,current_ref=None):
    per_family={}
    for fi,fam in enumerate(families):
        condres={}
        episodes={}
        for ci,cond in enumerate(CONDITIONS):
            ep=make_episode(seed*10000+fi*1000+ci*100+17,fam,cond)
            episodes[cond]=ep
            condres[cond]=fixed_eval(model,ep)
        rel=episodes["relational_shift"]
        z=condres["relational_shift"]["repr"]
        ddamage=d_rank1_damage(model,z,rel["drift"])
        r2=relation_decode_r2(z,rel["relations"])
        epB=make_episode(seed*10000+fi*1000+9999,fam,"relational_shift")
        transplant=relation_transplant_damage(model,kind,rel,epB)
        hist_rel=np.mean([
            condres["gain_shift"]["history_damage"],
            condres["static_relational"]["history_damage"],
            condres["relational_shift"]["history_damage"]
        ])
        a_spec=float(hist_rel-condres["diagonal"]["history_damage"])
        roll_cost,adapt_cost,roll_stable=rollout(model,kind,seed*10000+fi*1000+333,fam)
        o_cost,o_adapt_cost=oracle_rollout(seed*10000+fi*1000+333,fam)
        per_family[fam]={
          "drift_mse_rel":condres["relational_shift"]["intact_mse"],
          "D_rank1_damage":ddamage,
          "C_history_damage":float(hist_rel),
          "R_transplant_damage":transplant,
          "R_decode_r2":r2,
          "A_history_specificity":a_spec,
          "roll_cost":roll_cost,
          "adapt_cost":adapt_cost,
          "oracle_cost":o_cost,
          "oracle_adapt_cost":o_adapt_cost,
          "stable":roll_stable,
          "condition_history_damage":{c:float(condres[c]["history_damage"]) for c in CONDITIONS}
        }
    keys=["drift_mse_rel","D_rank1_damage","C_history_damage","R_transplant_damage",
          "R_decode_r2","A_history_specificity","roll_cost","adapt_cost","oracle_cost","oracle_adapt_cost","stable"]
    agg={k:float(np.mean([per_family[f][k] for f in families])) for k in keys}
    agg["per_family"]=per_family
    return agg

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    def ranks(a):
        order=np.argsort(a)
        r=np.empty(len(a),float); r[order]=np.arange(len(a),dtype=float)
        return r
    rx=ranks(x); ry=ranks(y)
    if np.std(rx)<1e-12 or np.std(ry)<1e-12:
        return 0.0
    return float(np.corrcoef(rx,ry)[0,1])

def eval_seed(seed,mode):
    archs=DEV_ARCHS if mode=="dev" else CONFIRM_ARCHS
    fams=DEV_EVAL_FAMILIES if mode=="dev" else CONFIRM_EVAL_FAMILIES
    X,Y=make_training_set(seed)
    models={}
    raw={}
    for ai,kind in enumerate(archs):
        model=train_model(seed+ai*100,kind,X,Y)
        models[kind]=model
        raw[kind]=evaluate_arch(seed+ai*10,kind,model,fams)

    cur=raw["CURRENT_MLP"]
    cur_cost=cur["adapt_cost"]; cur_mse=cur["drift_mse_rel"]
    rows={}
    for kind in archs:
        q=raw[kind]
        denom_cost=max(cur_cost-q["oracle_adapt_cost"],1e-4)
        control_gain=float((cur_cost-q["adapt_cost"])/denom_cost)
        prediction_gain=float((cur_mse-q["drift_mse_rel"])/max(cur_mse,1e-6))
        perf=float(0.5*control_gain+0.5*prediction_gain)
        rows[kind]={
          **q,
          "control_gain_vs_current":control_gain,
          "prediction_gain_vs_current":prediction_gain,
          "performance_score":perf
        }

    return {"seed":seed,"architectures":rows}

def summarize(records):
    archs=sorted(records[0]["architectures"].keys())
    out={}
    for a in archs:
        keys=["drift_mse_rel","D_rank1_damage","C_history_damage","R_transplant_damage",
              "R_decode_r2","A_history_specificity","roll_cost","adapt_cost","oracle_cost","oracle_adapt_cost","stable",
              "control_gain_vs_current","prediction_gain_vs_current","performance_score"]
        out[a]={k:float(np.median([r["architectures"][a][k] for r in records])) for k in keys}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",required=True)
    ap.add_argument("--mode",choices=["dev","confirm"],required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    lo,hi=map(int,a.seeds.split(":"))
    seeds=list(range(lo,hi+1))
    rec=[eval_seed(s,a.mode) for s in seeds]
    out={
      "campaign":"R33 mechanism emergence in generic learners",
      "mode":a.mode,
      "train_families":TRAIN_FAMILIES,
      "development_eval_families":DEV_EVAL_FAMILIES,
      "confirmatory_eval_families":CONFIRM_EVAL_FAMILIES,
      "development_architectures":DEV_ARCHS,
      "confirmatory_architectures":CONFIRM_ARCHS,
      "strict_heldout_architectures":STRICT_HELDOUT_ARCHS,
      "seeds":seeds,
      "records":rec,
      "medians":summarize(rec)
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["medians"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
