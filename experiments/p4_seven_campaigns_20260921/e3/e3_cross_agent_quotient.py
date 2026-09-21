#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
torch.set_num_threads(2)

class BaseAgent(nn.Module):
    def __init__(self,inp=10,h=40,out=5):
        super().__init__(); self.fc1=nn.Linear(inp,h); self.fc2=nn.Linear(h,h); self.head=nn.Linear(h,out)
    def forward(self,x,ret_h=False):
        h=torch.tanh(self.fc1(x)); h=torch.tanh(self.fc2(h)); y=self.head(h)
        return (y,h) if ret_h else y

class BaseAgentHet(nn.Module):
    def __init__(self,inp=10,h=56,out=5):
        super().__init__(); self.fc1=nn.Linear(inp,h); self.fc2=nn.Linear(h,h); self.fc3=nn.Linear(h,h); self.head=nn.Linear(h,out)
    def forward(self,x,ret_h=False):
        h=F.gelu(self.fc1(x)); h=F.gelu(self.fc2(h)); h=torch.tanh(self.fc3(h)); y=self.head(h)
        return (y,h) if ret_h else y

class QuotientNetVar(nn.Module):
    def __init__(self,h,d=4,out=5):
        super().__init__(); self.enc=nn.Linear(h,d,bias=False)
        self.dec=nn.Sequential(nn.Linear(d,24),nn.Tanh(),nn.Linear(24,out))
    def forward(self,h):
        q=self.enc(h); return self.dec(q),q

def make_qdata(seed,n=5000):
    rng=np.random.default_rng(seed)
    z=rng.uniform(-1,1,size=(n,4)).astype(np.float32)
    # 10-D observations with nonlinear mix, fixed per dataset
    A=rng.normal(size=(4,8)).astype(np.float32)
    lin=z@A
    extra=np.column_stack([z[:,0]*z[:,1], z[:,2]*z[:,3]]).astype(np.float32)
    x=np.concatenate([lin,extra],axis=1)+rng.normal(0,.03,size=(n,10)).astype(np.float32)
    # Functional endpoints (continuous) + combined action
    y=np.column_stack([
        z[:,0], z[:,1], z[:,2], z[:,3],
        np.tanh(z[:,0]+.7*z[:,1]*z[:,3])
    ]).astype(np.float32)
    return x,y,z

def train_base(seed,x,y,steps=700):
    torch.manual_seed(seed)
    m=BaseAgent(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); rng=np.random.default_rng(seed)
    for s in range(steps):
        idx=rng.integers(0,len(x),size=256)
        pred=m(xt[idx]); loss=F.mse_loss(pred,yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_base_het(seed,x,y,steps=600):
    torch.manual_seed(seed)
    m=BaseAgentHet(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); rng=np.random.default_rng(seed)
    for s in range(steps):
        idx=rng.integers(0,len(x),size=256)
        loss=F.mse_loss(m(xt[idx]),yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_qv(seed,H,Y,d=4,steps=600):
    torch.manual_seed(seed+777)
    qn=QuotientNetVar(H.shape[1],d); opt=torch.optim.AdamW(qn.parameters(),lr=3e-3,weight_decay=1e-4)
    ht=torch.from_numpy(H); yt=torch.from_numpy(Y); rng=np.random.default_rng(seed+55)
    for s in range(steps):
        idx=rng.integers(0,len(H),size=256)
        pr,q=qn(ht[idx]); loss=F.mse_loss(pr,yt[idx])+1e-4*(q**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return qn

def eval_q_pair2(seed,steps_base=550,steps_q=550):
    x,y,z=make_qdata(seed+10000,6000); tr=np.arange(0,4000); te=np.arange(4000,6000)
    a=train_base(seed,x[tr],y[tr],steps_base); b=train_base_het(seed+100000,x[tr],y[tr],steps_base)
    with torch.no_grad():
        _,Ha=a(torch.from_numpy(x),True); _,Hb=b(torch.from_numpy(x),True)
    Ha=Ha.numpy(); Hb=Hb.numpy()
    q4a=train_qv(seed,Ha[tr],y[tr],4,steps_q); q4b=train_qv(seed+1,Hb[tr],y[tr],4,steps_q); q3a=train_qv(seed+2,Ha[tr],y[tr],3,steps_q)
    with torch.no_grad():
        pa,Qa=q4a(torch.from_numpy(Ha)); pb,Qb=q4b(torch.from_numpy(Hb)); p3,Q3=q3a(torch.from_numpy(Ha))
    Qa=Qa.numpy(); Qb=Qb.numpy()
    mse4=float(np.mean((pa.numpy()[te]-y[te])**2)); mse3=float(np.mean((p3.numpy()[te]-y[te])**2))
    fit=te[:64]; tst=te[64:]
    R,_=orthogonal_procrustes(Qb[fit]-Qb[fit].mean(0),Qa[fit]-Qa[fit].mean(0))
    qb_al=(Qb[tst]-Qb[fit].mean(0))@R+Qa[fit].mean(0)
    with torch.no_grad(): transfer=q4a.dec(torch.from_numpy(qb_al.astype(np.float32))).numpy()
    transfer_mse=float(np.mean((transfer-y[tst])**2))
    cors=[np.corrcoef(Qa[tst][:,j],qb_al[:,j])[0,1] for j in range(4)]
    align=float(np.mean(np.abs(cors)))
    # small-sample linear raw alignment B hidden -> A hidden
    reg=Ridge(alpha=1.0).fit(Hb[fit],Ha[fit])
    hb_map=reg.predict(Hb[tst]).astype(np.float32)
    with torch.no_grad(): rawtrans=a.head(torch.from_numpy(hb_map)).numpy()
    raw_mse=float(np.mean((rawtrans-y[tst])**2))
    rng=np.random.default_rng(seed+999); perm=rng.permutation(len(tst))
    with torch.no_grad(): pp=q4a.dec(torch.from_numpy(Qa[tst][perm].astype(np.float32))).numpy()
    perm_damage=float(np.mean((pp-y[tst])**2)-mse4)
    noise=rng.normal(0,.05*Qa[tst].std(0),size=Qa[tst].shape).astype(np.float32)
    with torch.no_grad(): pn=q4a.dec(torch.from_numpy((Qa[tst]+noise).astype(np.float32))).numpy()
    noise_damage=float(np.mean((pn-y[tst])**2)-mse4)
    return {'mse_d4':mse4,'mse_d3':mse3,'minimality_gap':mse3-mse4,'cross_agent_alignment':align,
            'cross_agent_transfer_mse':transfer_mse,'raw_hidden_transfer_mse':raw_mse,
            'transfer_gain':raw_mse-transfer_mse,'perm_damage':perm_damage,'noise_damage':noise_damage}

CRIT={'mse_d4':.005,'minimality_gap':.05,'cross_agent_alignment':.97,'cross_agent_transfer_mse':.03,'perm_damage':.50,'noise_damage':.005}
def summarize(rec):
    med={k:float(np.median([r[k] for r in rec])) for k in rec[0] if k!='seed'}
    checks={'mse_d4':med['mse_d4']<=.005,'minimality_gap':med['minimality_gap']>=.05,
      'cross_agent_alignment':med['cross_agent_alignment']>=.97,'cross_agent_transfer_mse':med['cross_agent_transfer_mse']<=.03,
      'perm_damage':med['perm_damage']>=.50,'noise_damage':med['noise_damage']<=.005}
    guards=[r['mse_d4']<=.008 and r['minimality_gap']>=.04 and r['cross_agent_alignment']>=.94 and r['cross_agent_transfer_mse']<=.05 and r['perm_damage']>=.40 for r in rec]
    return {'medians':med,'median_checks':checks,'guard_pass_count':int(sum(guards)),'global_pass':bool(all(checks.values()) and sum(guards)>=9)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    s,e=map(int,a.seeds.split(':')); seeds=list(range(s,e+1)); rec=[]
    for x in seeds:
        r=eval_q_pair2(x); r['seed']=x; rec.append(r)
    out={'experiment':'E3 Cross-Agent Causal Quotient','seeds':seeds,'records':rec,'summary':summarize(rec)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
