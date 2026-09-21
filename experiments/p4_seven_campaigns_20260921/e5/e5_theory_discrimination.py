#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
def make_theory_data(seed,n=5000):
    rng=np.random.default_rng(seed)
    mem=rng.choice([-1.,1.],n).astype(np.float32)
    sens=rng.choice([-1.,1.],n).astype(np.float32)
    q=rng.integers(0,2,n).astype(np.float32)
    nuisance=rng.normal(0,1,size=(n,5)).astype(np.float32)
    h=np.column_stack([mem+rng.normal(0,.18,n),sens+rng.normal(0,.18,n),q,nuisance]).astype(np.float32)
    action=(mem>0).astype(np.float32)
    report=(mem>0).astype(np.float32)
    conf=(mem==sens).astype(np.float32)
    return h,action,report,conf,q

class TheoryRef2(nn.Module):
    def __init__(self,kind,hid=32,w=16,m=16):
        super().__init__(); self.kind=kind
        self.local=nn.Sequential(nn.Linear(8,hid),nn.Tanh(),nn.Linear(hid,hid),nn.Tanh())
        self.workspace=nn.Sequential(nn.Linear(hid,w),nn.Tanh())
        self.meta=nn.Sequential(nn.Linear(hid,m),nn.Tanh())
        def nl(inp): return nn.Sequential(nn.Linear(inp,16),nn.ReLU(),nn.Linear(16,1))
        if kind=='POLAR':
            self.action=nl(hid); self.report=nl(w); self.conf=nl(m)
        elif kind=='GNW':
            self.action=nl(w); self.report=nl(w); self.conf=nl(w)
        else:
            self.action=nl(hid); self.report=nl(m); self.conf=nl(m)
    def forward(self,x,lesion=None):
        h=self.local(x); w=self.workspace(h); meta=self.meta(h)
        if lesion=='workspace': w=torch.zeros_like(w)
        if lesion=='meta': meta=torch.zeros_like(meta)
        if self.kind=='POLAR': a=self.action(h); r=self.report(w); c=self.conf(meta)
        elif self.kind=='GNW': a=self.action(w); r=self.report(w); c=self.conf(w)
        else: a=self.action(h); r=self.report(meta); c=self.conf(meta)
        return a.squeeze(-1),r.squeeze(-1),c.squeeze(-1)
def train_theory2(seed,kind,steps=700):
    torch.manual_seed(seed)
    X,a,r,c,q=make_theory_data(seed+1000,5000); xt=torch.from_numpy(X)
    at=torch.from_numpy(a); rt=torch.from_numpy(r); ct=torch.from_numpy(c); qt=torch.from_numpy(q)
    m=TheoryRef2(kind); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=2e-4)
    rng=np.random.default_rng(seed)
    for st in range(steps):
        ix=rng.integers(0,len(X),256); al,rl,cl=m(xt[ix])
        la=F.binary_cross_entropy_with_logits(al,at[ix]); mask=qt[ix]
        lr=(F.binary_cross_entropy_with_logits(rl,rt[ix],reduction='none')*mask).sum()/(mask.sum()+1e-6)
        lc=F.binary_cross_entropy_with_logits(cl,ct[ix])
        loss=la+lr+.7*lc; opt.zero_grad(); loss.backward(); opt.step()
    return m
def theory_metrics2(seed,kind):
    m=train_theory2(seed,kind)
    X,a,r,c,q=make_theory_data(seed+9999,2500); xt=torch.from_numpy(X); targets=[a,r,c]
    def score(lesion=None):
        with torch.no_grad(): outs=m(xt,lesion)
        res=[]
        for i,(logit,t) in enumerate(zip(outs,targets)):
            pr=(torch.sigmoid(logit).numpy()>.5)
            if i==1:
                mask=q==1; res.append(float(np.mean(pr[mask]==(t[mask]>.5))))
            else: res.append(float(np.mean(pr==(t>.5))))
        return np.array(res)
    intact=score(); ws=score('workspace'); meta=score('meta')
    return {'intact_action':float(intact[0]),'intact_report':float(intact[1]),'intact_conf':float(intact[2]),
            'ws_action_drop':float(intact[0]-ws[0]),'ws_report_drop':float(intact[1]-ws[1]),'ws_conf_drop':float(intact[2]-ws[2]),
            'meta_action_drop':float(intact[0]-meta[0]),'meta_report_drop':float(intact[1]-meta[1]),'meta_conf_drop':float(intact[2]-meta[2])}

def sig_pass(kind,r):
    intact=min(r['intact_action'],r['intact_report'],r['intact_conf'])>=.98
    if kind=='POLAR':
        sig=(r['ws_action_drop']<=.05 and r['ws_report_drop']>=.40 and r['ws_conf_drop']<=.05 and
             r['meta_action_drop']<=.05 and r['meta_report_drop']<=.05 and r['meta_conf_drop']>=.40)
    elif kind=='GNW':
        sig=(r['ws_action_drop']>=.40 and r['ws_report_drop']>=.40 and r['ws_conf_drop']>=.40 and
             r['meta_action_drop']<=.05 and r['meta_report_drop']<=.05 and r['meta_conf_drop']<=.05)
    else:
        sig=(r['ws_action_drop']<=.05 and r['ws_report_drop']<=.05 and r['ws_conf_drop']<=.05 and
             r['meta_action_drop']<=.05 and r['meta_report_drop']>=.40 and r['meta_conf_drop']>=.40)
    return intact and sig

def summarize(records):
    counts={k:sum(sig_pass(k,r[k]) for r in records) for k in ['POLAR','GNW','HOT']}
    return {'signature_pass_counts':counts,'global_pass':all(v>=9 for v in counts.values())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    s,e=map(int,a.seeds.split(':')); seeds=list(range(s,e+1)); rec=[]
    for seed in seeds:
        rec.append({'seed':seed,**{k:theory_metrics2(seed,k) for k in ['POLAR','GNW','HOT']}})
    out={'experiment':'E5 Theory-Discriminating Consciousness Reference Implementations','seeds':seeds,'records':rec,'summary':summarize(rec)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
