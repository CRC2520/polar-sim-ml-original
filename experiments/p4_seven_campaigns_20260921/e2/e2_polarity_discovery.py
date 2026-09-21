#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(2)

def orth_cols(rng,d,k):
    q,_=np.linalg.qr(rng.normal(size=(d,k)))
    return q[:,:k].astype(np.float32)

class AxisNet(nn.Module):
    def __init__(self,d=6,k=3):
        super().__init__(); self.enc=nn.Linear(d,k,bias=False); self.dec=nn.Linear(k,d,bias=False)
    def forward(self,a): return self.dec(torch.tanh(self.enc(a)))

def data(seed,n=5000,d=6,k=3,noise=.02):
    rng=np.random.default_rng(seed); U=orth_cols(rng,d,k); V=orth_cols(rng,d,k)
    s=np.array([1.7,1.25,.9],np.float32); A=rng.normal(0,.8,size=(n,d)).astype(np.float32)
    Y=np.tanh((A@U)*s)@V.T+rng.normal(0,noise,size=(n,d)).astype(np.float32)
    return A,Y,U,V,s

def train(seed,steps=1200,n=5000):
    torch.manual_seed(seed); np.random.seed(seed)
    A,Y,U,V,s=data(seed+10000,n); At=torch.from_numpy(A); Yt=torch.from_numpy(Y)
    m=AxisNet(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        idx=rng.integers(0,n,size=256); loss=F.mse_loss(m(At[idx]),Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return m,U,V,s

def true_y(a,U,V,s): return np.tanh((np.asarray(a)@U)*s)@V.T

def eval_seed(seed):
    m,U,V,s=train(seed); W=m.enc.weight.detach().numpy(); Wn=W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-8)
    C=np.abs(Wn@U); ri,cj=linear_sum_assignment(-C); vals=C[ri,cj]
    rng=np.random.default_rng(seed+30000)
    Ah=rng.normal(0,.8,size=(2000,6)).astype(np.float32); Yh=np.tanh((Ah@U)*s)@V.T+rng.normal(0,.02,size=(2000,6))
    with torch.no_grad(): ph=m(torch.from_numpy(Ah)).numpy()
    mse=float(np.mean((ph-Yh)**2)); bcos=[]; ratios=[]
    for r,j in zip(ri,cj):
        d=Wn[r]; cand=[1.2*d,-1.2*d]
        with torch.no_grad(): preds=[m(torch.from_numpy(a.astype(np.float32))[None,:]).numpy()[0] for a in cand]
        bcos.append(float(np.dot(preds[0],preds[1])/(np.linalg.norm(preds[0])*np.linalg.norm(preds[1])+1e-8)))
        for sg in (-1,1):
            target=sg*V[:,j]; ps=[float(np.dot(p,target)) for p in preds]; ch=int(np.argmax(ps))
            ts=[float(np.dot(true_y(a,U,V,s),target)) for a in cand]; ratios.append(ts[ch]/(max(ts)+1e-8))
    R=rng.normal(size=(3,6)); R/=np.linalg.norm(R,axis=1,keepdims=True)
    Cr=np.abs(R@U); rr,cc=linear_sum_assignment(-Cr); rand=float(np.mean(Cr[rr,cc]))
    return {'seed':seed,'mean_axis_alignment':float(np.mean(vals)),'min_axis_alignment':float(np.min(vals)),
            'heldout_mse':mse,'bipolar_cosine':float(np.mean(bcos)),'context_control_ratio':float(np.mean(ratios)),
            'random_axis_alignment':rand,'alignment_gain':float(np.mean(vals)-rand)}

def summarize(rec):
    med={k:float(np.median([r[k] for r in rec])) for k in rec[0] if k!='seed'}
    checks={'mean_axis_alignment':med['mean_axis_alignment']>=.98,'min_axis_alignment':med['min_axis_alignment']>=.95,
            'heldout_mse':med['heldout_mse']<=.003,'bipolar_cosine':med['bipolar_cosine']<=-.98,
            'context_control_ratio':med['context_control_ratio']>=.98,'alignment_gain':med['alignment_gain']>=.25}
    guards=[r['mean_axis_alignment']>=.95 and r['min_axis_alignment']>=.90 and r['alignment_gain']>=.18 and r['heldout_mse']<=.006 for r in rec]
    return {'medians':med,'median_checks':checks,'guard_pass_count':int(sum(guards)),
            'global_pass':bool(all(checks.values()) and sum(guards)>=9)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    s,e=map(int,a.seeds.split(':')); seeds=list(range(s,e+1)); rec=[eval_seed(x) for x in seeds]
    out={'experiment':'E2 Polarity Discovery','seeds':seeds,'records':rec,'summary':summarize(rec)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
