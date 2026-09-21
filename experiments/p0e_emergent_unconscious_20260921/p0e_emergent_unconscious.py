from __future__ import annotations
import math, random, json, argparse, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

torch.set_num_threads(2)

def unit(v):
    v=np.asarray(v,dtype=np.float32)
    return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-8)

def make_near(k,rng,sim=.86):
    n=rng.normal(size=k.shape).astype(np.float32)
    n=n-(n*k).sum(-1,keepdims=True)*k
    n=unit(n)
    return unit(sim*k+math.sqrt(1-sim**2)*n)

def orthogonal_other(k,rng):
    u=rng.normal(size=k.shape).astype(np.float32)
    u=u-(u*k).sum(-1,keepdims=True)*k
    return unit(u)

def gen_batch(seed,n=192,T=38,cue_dim=6):
    rng=np.random.default_rng(seed)
    k0=unit(rng.normal(size=(n,cue_dim))); k1=orthogonal_other(k0,rng)
    near0=make_near(k0,rng,.86); near1=make_near(k1,rng,.86)
    m0=rng.choice([-1.,1.],size=n); m1=rng.choice([-1.,1.],size=n)
    cond=rng.integers(0,3,size=n)
    t0=rng.integers(2,4,size=n); t1=t0+rng.integers(3,5,size=n)
    tq1=t1+rng.integers(3,5,size=n)
    tn=tq1+rng.integers(3,5,size=n)
    tq0=tn+rng.integers(4,6,size=n)
    tc=tq0+1; tc2=tq0+2; tp=tc2+rng.integers(4,6,size=n); tpq=tp+2
    D=cue_dim+5
    X=np.zeros((n,T,D),np.float32); X[:,:,:cue_dim]=rng.normal(0,.06,size=(n,T,cue_dim))
    X[:,:,cue_dim+3:]=rng.normal(0,.08,size=(n,T,2))
    at=np.zeros((n,T),np.float32); am=np.zeros((n,T),np.float32)
    wt=np.zeros((n,T),np.float32); wm=np.zeros((n,T),np.float32)
    for i in range(n):
        for t,k,m in [(t0[i],k0[i],m0[i]),(t1[i],k1[i],m1[i])]:
            X[i,t,:cue_dim]=k; X[i,t,cue_dim]=m; X[i,t,cue_dim+2]=1
            at[i,t]=(m>0); am[i,t]=1
        t=tq1[i]; X[i,t,:cue_dim]=k1[i]; X[i,t,cue_dim+1]=1
        wt[i,t]=(m1[i]>0); wm[i,t]=1
        t=tn[i]; X[i,t,:cue_dim]=near0[i]
        at[i,t]=(m0[i]>0); am[i,t]=1
        t=tq0[i]; X[i,t,:cue_dim]=k0[i]; X[i,t,cue_dim+1]=1
        wt[i,t]=(m0[i]>0); wm[i,t]=1
        if cond[i]==0:
            corr_key=k0[i]; cur=-m0[i]; new0=-m0[i]
        elif cond[i]==1:
            corr_key=k1[i]; cur=-m1[i]; new0=m0[i]
        else:
            corr_key=k0[i]; cur=m0[i]; new0=m0[i]
        for t in (tc[i],tc2[i]):
            X[i,t,:cue_dim]=corr_key; X[i,t,cue_dim]=cur; X[i,t,cue_dim+2]=1
            at[i,t]=(cur>0); am[i,t]=1
        t=tp[i]; X[i,t,:cue_dim]=near0[i]; at[i,t]=(new0>0); am[i,t]=3
        t=tpq[i]; X[i,t,:cue_dim]=k0[i]; X[i,t,cue_dim+1]=1; wt[i,t]=(new0>0); wm[i,t]=1
    meta=dict(k0=k0,k1=k1,near0=near0,near1=near1,m0=m0,m1=m1,cond=cond,t0=t0,t1=t1,tq1=tq1,tn=tn,tq0=tq0,tc=tc,tc2=tc2,tp=tp,tpq=tpq)
    arr=[torch.from_numpy(z) for z in (X,at,am,wt,wm)]
    return arr,meta

class SharedGRUAgent(nn.Module):
    def __init__(self,D=11,h=72,w=2,ws_noise=.58):
        super().__init__(); self.h=h; self.w=w; self.ws_noise=ws_noise
        self.gru=nn.GRU(D,h,batch_first=True)
        self.action=nn.Sequential(nn.Linear(h+D,64),nn.ReLU(),nn.Linear(64,1))
        self.ws_mu=nn.Sequential(nn.Linear(h+D,40),nn.Tanh(),nn.Linear(40,w))
        self.ws_report=nn.Linear(w,1)
        self.outcome_pred=nn.Sequential(nn.Linear(h+D,48),nn.ReLU(),nn.Linear(48,1))
    def forward(self,x,h0=None,noise=True):
        H,hn=self.gru(x,h0); Z=torch.cat([H,x],-1)
        act=self.action(Z).squeeze(-1)
        mu=self.ws_mu(Z)
        if noise:
            W=mu+torch.randn_like(mu)*self.ws_noise
        else: W=mu
        report=self.ws_report(W).squeeze(-1)
        pred=self.outcome_pred(Z).squeeze(-1)
        return act,report,pred,H,mu,W,hn

def train_model(seed,steps=1400,batch=160,lr=1.8e-3):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=SharedGRUAgent(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=3e-4)
    for step in range(steps):
        (X,at,am,wt,wm),meta=gen_batch(seed*100000+step,batch)
        act,report,pred,H,mu,W,_=model(X,noise=True)
        la=(F.binary_cross_entropy_with_logits(act,at,reduction='none')*am).sum()/(am.sum()+1e-6)
        lw=(F.binary_cross_entropy_with_logits(report,wt,reduction='none')*wm).sum()/(wm.sum()+1e-6)
        em=(X[:,:,8]>0).float(); et=(X[:,:,6]+1)/2
        lp=(F.binary_cross_entropy_with_logits(pred,et,reduction='none')*em).sum()/(em.sum()+1e-6)
        lib=0.030*(mu**2).mean()
        loss=la+0.95*lw+0.25*lp+lib
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model

def gather(A,tidx):
    return torch.stack([A[i,int(tidx[i])] for i in range(len(tidx))])

def sym_auc(y,s):
    a=roc_auc_score(y,s); return float(max(a,1-a))

@torch.no_grad()
def eval_model(model,seed,n=2400,ws_mc=8):
    (X,at,am,wt,wm),meta=gen_batch(seed,n)
    act,report,pred,H,mu,W,_=model(X,noise=False)
    points={}
    for name,t in [('acq',meta['t0']),('near',meta['tn']),('query',meta['tq0']),('post',meta['tp']),('postq',meta['tpq']),('corr',meta['tc'])]:
        points['h_'+name]=gather(H,t).cpu().numpy(); points['mu_'+name]=gather(mu,t).cpu().numpy(); points['x_'+name]=gather(X,t).cpu()
        points['act_'+name]=torch.sigmoid(gather(act,t)).cpu().numpy(); points['report_'+name]=torch.sigmoid(gather(report,t)).cpu().numpy()
    return points,meta,X,model

def probe_auc(A,y,tr,te,C=.5):
    clf=LogisticRegression(max_iter=500,C=C).fit(A[tr],y[tr]); s=clf.predict_proba(A[te])[:,1]
    return sym_auc(y[te],s),clf

def workspace_noisy_auc(model,mu,y,tr,te,rng_seed=0,noise=.35):
    rng=np.random.default_rng(rng_seed)
    A=mu + rng.normal(0,noise,size=mu.shape)
    return probe_auc(A,y,tr,te,C=.4)[0]

def current_only_auc(Xpt,y,tr,te):
    A=Xpt.numpy(); return probe_auc(A,y,tr,te,C=.5)[0]

def causal_lesion(model,hclf,H,Xpt,y):
    w=hclf.coef_[0].astype(np.float32); w=w/(np.linalg.norm(w)+1e-8)
    Ht=torch.from_numpy(H).float(); yt=torch.from_numpy(y.astype(np.float32)); Xpt=Xpt.float()
    def acc(Hx, Xin=Xpt):
        with torch.no_grad(): z=model.action(torch.cat([Hx,Xin],-1)).squeeze(-1)
        return float(((torch.sigmoid(z)>.5)==(yt>.5)).float().mean())
    base=acc(Ht)
    wt=torch.from_numpy(w); Hles=Ht-(Ht@wt).unsqueeze(1)*wt.unsqueeze(0)
    les=acc(Hles)
    rg=np.random.default_rng(881337); r=rg.normal(size=w.shape).astype(np.float32); r-=r.dot(w)*w; r/=np.linalg.norm(r)+1e-8
    rt=torch.from_numpy(r); Hr=Ht-(Ht@rt).unsqueeze(1)*rt.unsqueeze(0); rnd=acc(Hr)
    return base,les,rnd,w

def evaluate_seed(seed,steps=1400,n=1200):
    model=train_model(seed,steps=steps)
    pts,meta,X,_=eval_model(model,seed+700000,n=n)
    y=(meta['m0']>0).astype(int); idx=np.arange(n); tr=idx[:n//2]; te=idx[n//2:]
    h_auc,hclf=probe_auc(pts['h_near'],y,tr,te)
    ws_near=workspace_noisy_auc(model,pts['mu_near'],y,tr,te,rng_seed=seed+1)
    ws_query=workspace_noisy_auc(model,pts['mu_query'],y,tr,te,rng_seed=seed+2)
    cur_auc=current_only_auc(pts['x_near'],y,tr,te)
    near_acc=float(np.mean((pts['act_near'][te]>.5)==y[te]))
    query_acc=float(np.mean((pts['report_query'][te]>.5)==y[te]))
    new=np.where(meta['cond']==0,1-y,y); c=meta['cond'][te]
    post_acc=float(np.mean((pts['act_post'][te]>.5)==new[te])); postq=float(np.mean((pts['report_postq'][te]>.5)==new[te]))
    recon=float(np.mean(((pts['act_post'][te]>.5)==new[te])[c==0])); nore=float(np.mean(((pts['act_post'][te]>.5)==new[te])[c==1])); ronly=float(np.mean(((pts['act_post'][te]>.5)==new[te])[c==2]))
    base,les,rnd,w=causal_lesion(model,hclf,pts['h_near'][te],pts['x_near'][te],y[te])
    yac=y[te]
    acq_base,acq_les,acq_rnd,_=causal_lesion(model,hclf,pts['h_acq'][te],pts['x_acq'][te],yac)
    newy=new.astype(int); post_h_auc,_=probe_auc(pts['h_post'],newy,tr,te)
    return {
      'seed':seed,'hidden_memory_auc':h_auc,'workspace_near_auc':ws_near,'hidden_minus_workspace_auc':h_auc-ws_near,'workspace_query_auc':ws_query,'workspace_promotion_gain':ws_query-ws_near,
      'current_only_auc':cur_auc,'near_action_acc':near_acc,'query_report_acc':query_acc,
      'post_action_acc':post_acc,'post_query_acc':postq,'recon_update_acc':recon,'no_reactivation_preserve_acc':nore,'reactivation_no_contradiction_preserve_acc':ronly,
      'memory_lesion_action_drop':base-les,'random_lesion_action_drop':base-rnd,'memory_lesion_near_acc':les,
      'current_evidence_base_acc':acq_base,'current_evidence_memory_lesion_acc':acq_les,'current_evidence_lesion_drop':acq_base-acq_les,
      'post_hidden_updated_auc':post_h_auc
    }

PRIMARY_CRITERIA={
 'hidden_memory_auc':('>=',0.95),
 'current_only_auc':('<=',0.60),
 'hidden_minus_workspace_auc':('>=',0.20),
 'workspace_query_auc':('>=',0.95),
 'workspace_promotion_gain':('>=',0.20),
 'near_action_acc':('>=',0.95),
 'query_report_acc':('>=',0.95),
 'memory_lesion_action_drop':('>=',0.35),
 'memory_lesion_near_acc':('<=',0.65),
 'random_lesion_action_drop':('<=',0.05),
 'current_evidence_base_acc':('>=',0.95),
 'current_evidence_lesion_drop':('<=',0.05),
}
SEED_GUARD={
 'hidden_memory_auc':('>=',0.90), 'current_only_auc':('<=',0.65),
 'hidden_minus_workspace_auc':('>=',0.15), 'workspace_near_auc':('<=',0.82), 'workspace_query_auc':('>=',0.90),
 'workspace_promotion_gain':('>=',0.15), 'near_action_acc':('>=',0.90),
 'memory_lesion_action_drop':('>=',0.25), 'random_lesion_action_drop':('<=',0.08),
 'current_evidence_lesion_drop':('<=',0.08)
}

def _cmp(v,op,t):
    return (v>=t) if op=='>=' else (v<=t)

def summarize(records):
    keys=[k for k in records[0] if k!='seed']; med={k:float(np.median([r[k] for r in records])) for k in keys}
    median_checks={k:_cmp(med[k],*cond) for k,cond in PRIMARY_CRITERIA.items()}
    seed_guard=[]
    for r in records:
        seed_guard.append(all(_cmp(r[k],*cond) for k,cond in SEED_GUARD.items()))
    return {
      'medians':med,'primary_criteria':PRIMARY_CRITERIA,'median_checks':median_checks,
      'seed_guard_pass_count':int(sum(seed_guard)),'seed_guard':seed_guard,
      'global_pass':bool(all(median_checks.values()) and sum(seed_guard)>=9),
      'interpretation':'EMERGENT_FUNCTIONAL_SEPARATION_PASS' if (all(median_checks.values()) and sum(seed_guard)>=9) else 'EMERGENT_FUNCTIONAL_SEPARATION_NOT_CLOSED',
      'secondary_update_metrics':{k:med[k] for k in ['post_action_acc','post_query_acc','recon_update_acc','no_reactivation_preserve_acc','reactivation_no_contradiction_preserve_acc','post_hidden_updated_auc']}
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',required=True); ap.add_argument('--steps',type=int,default=1400); ap.add_argument('--output',required=True)
    a=ap.parse_args();
    if ':' in a.seeds:
        s,e=map(int,a.seeds.split(':')); seeds=list(range(s,e+1))
    else: seeds=[int(x) for x in a.seeds.split(',')]
    rec=[]
    for s in seeds:
        r=evaluate_seed(s,steps=a.steps); rec.append(r); print(s,{k:round(v,4) for k,v in r.items() if k!='seed'},flush=True)
    out={'campaign':'P0-E Emergent Latent Unconscious','seeds':seeds,'records':rec,'summary':summarize(rec)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('SUMMARY',json.dumps(out['summary'],indent=2,sort_keys=True))

if __name__=='__main__': main()
