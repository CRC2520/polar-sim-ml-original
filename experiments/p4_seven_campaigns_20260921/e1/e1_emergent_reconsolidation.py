#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)

def unit(v):
    v=np.asarray(v,np.float32); return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-8)
def orth(k,rng):
    u=rng.normal(size=k.shape).astype(np.float32); u-=(u*k).sum(-1,keepdims=True)*k; return unit(u)
def near(k,rng,sim=.86):
    u=orth(k,rng); return unit(sim*k+math.sqrt(1-sim**2)*u)

def gen(seed,n=96,T=18,cd=6):
    rng=np.random.default_rng(seed)
    k=unit(rng.normal(size=(n,cd))); q=near(k,rng,.86); d=orth(k,rng)
    y=rng.choice([-1.,1.],n)
    cond=rng.choice(4,size=n,p=[.4,.2,.2,.2])
    X=np.zeros((n,T,cd+4),np.float32); X[:,:,:cd]=rng.normal(0,.05,size=(n,T,cd))
    for i in range(n):
        X[i,1,:cd]=k[i]; X[i,1,cd]=y[i]; X[i,1,cd+1]=1
        if cond[i] in (0,2):
            X[i,9,:cd]=k[i]; X[i,10,:cd]=k[i]
        elif cond[i]==3:
            X[i,9,:cd]=d[i]; X[i,10,:cd]=d[i]
        corr=(-y[i] if cond[i] in (0,1,3) else y[i])
        X[i,11,cd]=corr; X[i,11,cd+1]=1
        X[i,12,cd]=corr; X[i,12,cd+1]=1
        X[i,16,:cd]=q[i]; X[i,16,cd+2]=1
    tf=np.where(cond==0,(y<0),(y>0)).astype(np.float32)
    return torch.from_numpy(X),torch.from_numpy(tf),dict(cond=cond,y=y)

class Agent(nn.Module):
    def __init__(self,D=10,h=128):
        super().__init__(); self.gru=nn.GRU(D,h,batch_first=True)
        self.act=nn.Sequential(nn.Linear(h+D,96),nn.ReLU(),nn.Linear(96,1))
        self.pred=nn.Sequential(nn.Linear(h+D,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):
        H,_=self.gru(x); z=torch.cat([H,x],-1)
        return self.act(z).squeeze(-1),self.pred(z).squeeze(-1),H

def train(seed,steps=3200,batch=64):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m=Agent(); opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=2e-4)
    for s in range(steps):
        X,tf,_=gen(seed*100000+s,batch)
        logits,pred,H=m(X)
        l=F.binary_cross_entropy_with_logits(logits[:,16],tf)
        outmask=X[:,:,7]; outt=(X[:,:,6]+1)/2
        lp=(F.binary_cross_entropy_with_logits(pred,outt,reduction='none')*outmask).sum()/(outmask.sum()+1e-6)
        loss=l+.12*lp
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m

def eval_seed(seed,steps=3200,n=1200):
    m=train(seed,steps)
    X,tf,meta=gen(seed+990000,n)
    with torch.no_grad(): logits,pred,H=m(X)
    pf=(torch.sigmoid(logits[:,16])>.5).numpy(); y=tf.numpy().astype(bool); c=meta['cond']
    r={'seed':seed,'overall':float(np.mean(pf==y))}
    for j,name in enumerate(['react_contradict','no_reactivation','react_same','distractor']):
        ix=c==j; r[name]=float(np.mean(pf[ix]==y[ix]))
    idx=np.arange(n); tr=idx[:n//2]; te=idx[n//2:]
    clf=LogisticRegression(max_iter=400,C=.5).fit(H[:,14].numpy()[tr],y.astype(int)[tr])
    r['updated_memory_auc']=float(roc_auc_score(y.astype(int)[te],clf.predict_proba(H[:,14].numpy()[te])[:,1]))
    return r

CRIT={'overall':.90,'react_contradict':.90,'no_reactivation':.88,'react_same':.95,'distractor':.82,'updated_memory_auc':.94}
GUARD={'react_contradict':.80,'no_reactivation':.78,'react_same':.90,'distractor':.70,'updated_memory_auc':.88}

def summarize(rec):
    med={k:float(np.median([x[k] for x in rec])) for k in CRIT}
    checks={k:med[k]>=v for k,v in CRIT.items()}
    guards=[all(x[k]>=v for k,v in GUARD.items()) for x in rec]
    return {'medians':med,'median_checks':checks,'guard_pass_count':int(sum(guards)),
            'global_pass':bool(all(checks.values()) and sum(guards)>=9)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args()
    if ':' in a.seeds:
        s,e=map(int,a.seeds.split(':')); seeds=list(range(s,e+1))
    else: seeds=[int(x) for x in a.seeds.split(',')]
    rec=[eval_seed(s) for s in seeds]
    out={'experiment':'E1 Emergent Reconsolidation','seeds':seeds,'criteria':CRIT,'guard':GUARD,
         'records':rec,'summary':summarize(rec)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out['summary'],indent=2,sort_keys=True))
if __name__=='__main__': main()
