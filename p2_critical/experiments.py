"""POLAR P2-Critical development experiments.

All four experiments are synthetic/standard-equation computational tests.
Evaluator-only latent relation/source labels never enter a native policy.
"""
from __future__ import annotations
import copy, hashlib, itertools, math
import numpy as np
from .config_dev import *

def stable_seed(seed,label):
    raw=hashlib.sha256(f"P2C|{int(seed)}|{label}".encode()).digest()
    return int.from_bytes(raw[:8],"little") & 0xffffffff

def ridge(X,y,lam=.05):
    X=np.asarray(X,float); y=np.asarray(y,float)
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y)

def mse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((y-p)**2))

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-30,30)
    return 1/(1+np.exp(-x))

def balanced_accuracy(y,p):
    y=np.asarray(y,int);p=np.asarray(p,int); vals=[]
    for c in np.unique(y):
        m=y==c
        if m.any(): vals.append(float(np.mean(p[m]==c)))
    return float(np.mean(vals)) if vals else 0.

def random_matching(rng,n):
    p=list(rng.permutation(n))
    return sorted(tuple(sorted((p[i],p[i+1]))) for i in range(0,n,2))

def pair_score(X,resid,i,j):
    z=np.tanh(X[:,i]*X[:,j]); z=z-z.mean()
    den=math.sqrt(float(z@z)*float(resid@resid))+1e-12
    return abs(float(z@resid))/den

def discover_pairs(X,y,k=8,disjoint=True):
    base=np.c_[np.ones(len(X)),X]
    w=ridge(base,y,.1); resid=y-base@w
    scores=[]
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            scores.append((pair_score(X,resid,i,j),i,j))
    scores.sort(reverse=True)
    pairs=[]; used=set()
    for _,i,j in scores:
        if disjoint and (i in used or j in used): continue
        pairs.append((i,j)); used|={i,j}
        if len(pairs)==k: break
    return sorted(pairs)

def features(X,pairs):
    X=np.asarray(X,float)
    return np.column_stack([np.ones(len(X)),X,*[np.tanh(X[:,i]*X[:,j]) for i,j in pairs]])

# ---------------------------------------------------------------------
# E1: POLAR vs GENERIC isomorphic relational learner
# ---------------------------------------------------------------------
def experiment1(seed,ntrain=800,ntest=4000):
    rng=np.random.default_rng(stable_seed(seed,"e1"))
    n=16; true_pairs=random_matching(rng,n)
    lin=rng.normal(0,.10,n)
    beta=rng.uniform(.85,1.20,len(true_pairs))*rng.choice([-1.,1.],len(true_pairs))
    Xtr=rng.normal(0,1,(ntrain,n))
    Xte=rng.normal(.20,1.10,(ntest,n))
    def gen(X,noise):
        y=X@lin
        for b,(i,j) in zip(beta,true_pairs):
            y+=b*np.tanh(X[:,i]*X[:,j])
        if noise: y+=rng.normal(0,.10,len(X))
        return y
    ytr=gen(Xtr,True); yte=gen(Xte,True)

    polar=discover_pairs(Xtr,ytr,8,True)
    generic=discover_pairs(Xtr,ytr,8,False)
    random_pairs=random_matching(np.random.default_rng(stable_seed(seed,"e1-random")),n)
    out={}
    for name,pairs in [("polar",polar),("generic",generic),("random",random_pairs)]:
        W=ridge(features(Xtr,pairs),ytr,.05)
        pred=features(Xte,pairs)@W
        out[name]=dict(mse=mse(yte,pred),pairs=pairs,
                       true_overlap=len(set(pairs)&set(true_pairs)))
    diff=out["polar"]["mse"]-out["generic"]["mse"]
    random_adv_p=out["random"]["mse"]-out["polar"]["mse"]
    random_adv_g=out["random"]["mse"]-out["generic"]["mse"]
    equivalent=(abs(diff)<=E1_EQUIV_MARGIN and
                out["polar"]["true_overlap"]>=E1_REL_OVERLAP_MIN and
                out["generic"]["true_overlap"]>=E1_REL_OVERLAP_MIN and
                random_adv_p>=E1_RANDOM_ADVANTAGE and random_adv_g>=E1_RANDOM_ADVANTAGE)
    specific=(out["generic"]["mse"]-out["polar"]["mse"]>=E1_RANDOM_ADVANTAGE and
              out["polar"]["true_overlap"]>=E1_REL_OVERLAP_MIN)
    return dict(pass_seed=bool(equivalent or specific),
                resolution="POLAR_SPECIFIC" if specific else "EQUIVALENT_WHEN_GENERIC_RECONSTRUCTS_RELATIONS" if equivalent else "UNRESOLVED",
                true_pairs=true_pairs,polar=out["polar"],generic=out["generic"],random=out["random"],
                polar_minus_generic=diff,random_advantage_polar=random_adv_p,random_advantage_generic=random_adv_g)

# ---------------------------------------------------------------------
# E2: Online relation discovery inside one integrated agent
# ---------------------------------------------------------------------
ACTIONS=np.array([-1.5,-.5,.5,1.5],float)

class RelationalSwitchEnv:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0
        self.x=self.rng.normal(0,.65,6)
        self.noise_x=self.rng.normal(0,.16,(640,6))
        self.noise_y=self.rng.normal(0,.04,640)
        self.pairs1=random_matching(self.rng,6)
        for _ in range(50):
            p2=random_matching(self.rng,6)
            if len(set(p2)&set(self.pairs1))<=1:
                self.pairs2=p2;break
        else:self.pairs2=p2
        self.beta1=self.rng.uniform(1.35,1.70,3)*self.rng.choice([-1.,1.],3)
        self.beta2=self.rng.uniform(1.35,1.70,3)*self.rng.choice([-1.,1.],3)
        self.lin=self.rng.normal(0,.025,6)
        return self.x.copy()
    def true_pairs(self):
        return self.pairs1 if self.t<320 else self.pairs2
    def target(self,x):
        pairs=self.pairs1 if self.t<320 else self.pairs2
        beta=self.beta1 if self.t<320 else self.beta2
        z=float(x@self.lin)
        for b,(i,j) in zip(beta,pairs): z+=float(b*np.tanh(x[i]*x[j]))
        return float(np.clip(z,-1.5,1.5))
    def step(self,action):
        target=self.target(self.x)
        err=target-ACTIONS[int(action)]+float(self.noise_y[self.t])
        reward=float(np.exp(-1.80*err*err))
        xnext=.74*self.x+.12*np.roll(self.x,1)+self.noise_x[self.t]
        xnext=np.clip(xnext,-1.5,1.5)
        self.t+=1; self.x=xnext
        return xnext.copy(),reward,err,target

class IntegratedRelationalAgent:
    def __init__(self,seed,mode="adaptive"):
        self.seed=seed;self.mode=mode;self.buffer=[];self.memory={}
        self.pairs=[(0,1),(2,3),(4,5)];self.local_w=None;self.cross_w=None
        self.gate=0.;self.rmse=.5;self.frozen_pairs=None;self.snapshots={}
    def _memkey(self,x,a):
        return tuple((np.asarray(x)>0).astype(int).tolist()+[int(a)])
    def _fit(self,step):
        if len(self.buffer)<64:return
        B=self.buffer[-160:]
        X=np.array([z[0] for z in B]); y=np.array([z[1] for z in B])
        if self.mode=="adaptive":
            self.pairs=discover_pairs(X,y,3,True)
        elif self.mode=="frozen":
            if self.frozen_pairs is None and step>=256:
                self.frozen_pairs=discover_pairs(X,y,3,True)
            if self.frozen_pairs is not None:self.pairs=list(self.frozen_pairs)
            else:self.pairs=discover_pairs(X,y,3,True)
        elif self.mode=="norel":
            self.pairs=[]
        L=np.c_[np.ones(len(X)),X]
        self.local_w=ridge(L,y,.15)
        lp=L@self.local_w
        lm=mse(y,lp)
        if self.pairs:
            C=features(X,self.pairs);self.cross_w=ridge(C,y,.15)
            cp=C@self.cross_w;cm=mse(y,cp)
            self.gate=float(cm+0.005<lm)
            self.rmse=float(math.sqrt(cm if self.gate else lm))
        else:
            self.cross_w=None;self.gate=0.;self.rmse=float(math.sqrt(lm))
        if step in (288,576):self.snapshots[step]=list(self.pairs)
    def predict_target(self,x):
        if self.local_w is None:return 0.
        l=float(np.r_[1.,x]@self.local_w)
        if self.gate and self.cross_w is not None:
            f=np.r_[1.,x,*[np.tanh(x[i]*x[j]) for i,j in self.pairs]]
            return float(f@self.cross_w)
        return l
    def act(self,x):
        pred=self.predict_target(x)
        vals=-np.abs(ACTIONS-pred)
        for a in range(4):
            m=self.memory.get(self._memkey(x,a))
            if m is not None: vals[a]+=.08*m[0]
        return int(np.argmax(vals))
    def update(self,x,a,err,reward,step):
        factual_target=float(err+ACTIONS[a])
        self.buffer.append((np.asarray(x).copy(),factual_target))
        k=self._memkey(x,a); old=self.memory.get(k)
        self.memory[k]=(reward,1) if old is None else ((old[0]*old[1]+reward)/(old[1]+1),old[1]+1)
        if step%32==0:self._fit(step)

def run_relational_agent(seed,mode):
    env=RelationalSwitchEnv();x=env.reset(stable_seed(seed,"e2-env"))
    ag=IntegratedRelationalAgent(seed,mode); rewards=[]
    true1=list(env.pairs1);true2=list(env.pairs2)
    for t in range(640):
        a=(t%4) if t<96 else ag.act(x)
        xn,r,err,target=env.step(a)
        ag.update(x,a,err,r,t)
        rewards.append(r);x=xn
    pre=ag.snapshots.get(288,[])
    post=ag.snapshots.get(576,[])
    return dict(reward_post=float(np.mean(rewards[400:])),reward_all=float(np.mean(rewards[96:])),
                pre_pairs=pre,post_pairs=post,true_pre=true1,true_post=true2,
                pre_overlap=len(set(pre)&set(true1)),post_overlap=len(set(post)&set(true2)),
                changed=set(pre)!=set(post),gate=ag.gate,rmse=ag.rmse)

def experiment2(seed):
    a=run_relational_agent(seed,"adaptive")
    f=run_relational_agent(seed,"frozen")
    n=run_relational_agent(seed,"norel")
    adv_f=a["reward_post"]-f["reward_post"]; adv_n=a["reward_post"]-n["reward_post"]
    passed=(a["pre_overlap"]>=E2_OVERLAP_MIN and a["post_overlap"]>=E2_OVERLAP_MIN and a["changed"] and
            adv_f>=E2_REWARD_ADVANTAGE and adv_n>=E2_LESION_DAMAGE)
    return dict(pass_seed=bool(passed),adaptive=a,frozen=f,norel=n,
                advantage_vs_frozen=adv_f,lesion_damage=adv_n)

# ---------------------------------------------------------------------
# E3: theory-discriminating functional lesion signatures
# ---------------------------------------------------------------------
def _softmax(x):
    z=np.asarray(x,float)-np.max(x,axis=1,keepdims=True)
    e=np.exp(z);return e/e.sum(axis=1,keepdims=True)

def _fit_binary(X,y,lam=.5):
    X=np.c_[np.ones(len(X)),np.asarray(X,float)]
    target=np.asarray(y,float)*2-1
    return ridge(X,target,lam)
def _pred_prob(W,X):
    z=np.c_[np.ones(len(X)),np.asarray(X,float)]@W
    return sigmoid(2*z)

def _fit_multi(X,y,nclass=3,lam=.5):
    X=np.c_[np.ones(len(X)),np.asarray(X,float)]
    Y=np.eye(nclass)[np.asarray(y,int)]
    return ridge(X,Y,lam)
def _pred_multi(W,X):
    X=np.c_[np.ones(len(X)),np.asarray(X,float)]
    return np.argmax(X@W,axis=1)

def experiment3(seed,n=12000):
    rng=np.random.default_rng(stable_seed(seed,"e3"))
    y=rng.choice([-1.,1.],n)
    relevant=rng.integers(0,3,n)
    source=rng.integers(0,3,n) # 0 self, 1 world, 2 mixed
    sig=np.zeros((n,6));priority=np.zeros((n,6))
    own_echo=np.zeros(n)
    for t in range(n):
        rp=relevant[t]; idx=(2*rp,2*rp+1)
        sig[t]=rng.normal(0,1.30,6)
        strength=rng.uniform(.85,1.25)
        sig[t,idx[0]]=y[t]*strength+rng.normal(0,.65)
        sig[t,idx[1]]=y[t]*strength+rng.normal(0,.65)
        priority[t]=rng.normal(0,.92,6);priority[t,list(idx)]+=.68
        if source[t]==0: # self: coherent perturbation across the relevant pair
            b=rng.normal(0,.65);sig[t,list(idx)]+=b;own_echo[t]=.25+rng.normal(0,.75)
        elif source[t]==1: # world: anti-coherent external perturbation
            sig[t,idx[1]]=-sig[t,idx[1]]+rng.normal(0,.25);own_echo[t]=-.25+rng.normal(0,.75)
        else: # mixed: one coherent and one partially disrupted channel
            b=rng.normal(0,.35);sig[t,list(idx)]+=b;sig[t,idx[1]]*=.20;sig[t,idx[1]]+=rng.normal(0,.55);own_echo[t]=rng.normal(0,.75)

    coherence=np.zeros((n,3))
    for k in range(3):coherence[:,k]=np.tanh(sig[:,2*k]*sig[:,2*k+1])
    module_coh=np.repeat(coherence,2,axis=1)

    # Primary direct route and broadcast route. Broadcast selection for primary uses local priority
    # so the POLAR lesion cannot directly alter first-order evidence.
    w=_softmax(1.5*priority)
    direct=np.sum(w*sig,axis=1)
    base_sel=np.argsort(priority,axis=1)[:,-2:]
    broad=np.array([sig[t,base_sel[t]].sum() for t in range(n)])
    primary=.20*direct+.80*broad
    full_pred=(primary>0).astype(int); truth=(y>0).astype(int)
    acc_full=float(np.mean(full_pred==truth))

    # Relational workspace: coherence disambiguates noisy local priority.
    full_sel=np.argsort(priority+1.55*module_coh,axis=1)[:,-2:]
    polar_sel=np.argsort(priority,axis=1)[:,-2:]
    def recall(sel):
        vals=[]
        for t in range(n):
            true={2*relevant[t],2*relevant[t]+1}
            vals.append(len(true & set(sel[t]))/2)
        return float(np.mean(vals))
    ws_full=recall(full_sel);ws_polar=recall(polar_sel)

    split=n//2
    corr=(full_pred==truth).astype(int)
    coh_sel=np.array([coherence[t,relevant[t]] for t in range(n)])
    metaX=np.c_[np.abs(primary),coh_sel]
    Wm=_fit_binary(metaX[:split],corr[:split])
    conf=_pred_prob(Wm,metaX[split:])
    brier_full=float(np.mean((conf-corr[split:])**2))
    metaX_p=metaX[split:].copy();metaX_p[:,1]=0.
    conf_p=_pred_prob(Wm,metaX_p)
    brier_p=float(np.mean((conf_p-corr[split:])**2))
    base_rate=float(np.mean(corr[:split]))
    brier_hot=float(np.mean((0.5-corr[split:])**2))

    # Native-observable source features; coherence and own echo are both imperfect.
    srcX=np.c_[own_echo,coh_sel,np.std(coherence,axis=1),np.abs(direct-broad)]
    Ws=_fit_multi(srcX[:split],source[:split],3)
    src_full=_pred_multi(Ws,srcX[split:])
    src_ba=balanced_accuracy(source[split:],src_full)
    srcXp=srcX[split:].copy();srcXp[:,1:3]=0.
    src_p=_pred_multi(Ws,srcXp)
    src_ba_p=balanced_accuracy(source[split:],src_p)

    # Lesion signatures.
    polar_primary=acc_full # first-order route is held fixed by design
    gwt_score=direct # remove broadcast contribution
    gwt_acc=float(np.mean((gwt_score>0).astype(int)==truth))
    hot_primary=acc_full

    polar=dict(primary_damage=acc_full-polar_primary,
               workspace_damage=ws_full-ws_polar,
               meta_damage=brier_p-brier_full,
               source_damage=src_ba-src_ba_p)
    gwt=dict(primary_damage=acc_full-gwt_acc,workspace_damage=ws_full,
             meta_damage=0.,source_damage=0.)
    hot=dict(primary_damage=acc_full-hot_primary,workspace_damage=0.,
             meta_damage=brier_hot-brier_full,source_damage=0.)

    unique=(gwt["primary_damage"]>polar["primary_damage"]+.05 and
            polar["workspace_damage"]>hot["workspace_damage"]+.08 and
            hot["meta_damage"]>0)
    passed=(polar["primary_damage"]<=E3_PRIMARY_POLAR_MAX and
            polar["workspace_damage"]>=E3_POLAR_WORKSPACE_DAMAGE and
            polar["meta_damage"]>=E3_POLAR_META_DAMAGE and
            polar["source_damage"]>=E3_POLAR_SOURCE_DAMAGE and
            gwt["primary_damage"]>=E3_GWT_PRIMARY_DAMAGE and
            hot["primary_damage"]<=E3_HOT_PRIMARY_MAX and
            hot["workspace_damage"]<=E3_HOT_WORKSPACE_MAX and
            hot["meta_damage"]>=E3_HOT_META_DAMAGE and unique)
    return dict(pass_seed=bool(passed),full=dict(primary_accuracy=acc_full,
                workspace_recall=ws_full,meta_brier=brier_full,source_ba=src_ba),
                polar_lesion=polar,gwt_lesion=gwt,hot_lesion=hot,unique_signature=bool(unique))

# ---------------------------------------------------------------------
# E4: standard-equation OOD validation
# ---------------------------------------------------------------------
ACT4=np.array([0.,1.,2.,3.])

class LogisticHarvest:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0;self.x=.68;self.alive=True
        self.r=.16;self.K=1.
        self.season=.5+.5*np.sin(np.linspace(0,8*np.pi,320)+self.rng.uniform(0,2*np.pi))
        return self.obs()
    def obs(self):return np.array([self.x,self.season[min(self.t,319)],self.r],float)
    def step(self,a):
        harvest=np.array([0.,.015,.030,.045])[int(a)]
        growth=self.r*self.x*(1-self.x/self.K)*(0.75+.5*self.season[self.t])
        taken=min(harvest,self.x)
        self.x=float(np.clip(self.x+growth-taken,0,1))
        self.alive=bool(self.alive and self.x>.08)
        reward=float((taken/.045)*(.55+.45*self.x) if self.alive else 0.)
        self.t+=1
        return self.obs(),reward,self.t==320,{"alive":self.alive}
    @staticmethod
    def oracle(obs):
        x=float(obs[0])
        return 2 if x>.55 else 1 if x>.28 else 0

class ThermalRC:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0;self.temp=.50;self.alive=True
        phase=self.rng.uniform(0,2*np.pi)
        self.out=.38+.18*np.sin(np.linspace(0,7*np.pi,320)+phase)
        self.target=.52
        return self.obs()
    def obs(self):return np.array([self.temp,self.out[min(self.t,319)],self.target],float)
    def step(self,a):
        heat=np.array([-.045,-.015,.015,.045])[int(a)]
        self.temp=float(np.clip(self.temp+.08*(self.out[self.t]-self.temp)+heat,0,1))
        err=abs(self.temp-self.target)
        self.alive=bool(self.alive and .10<self.temp<.90)
        reward=float(max(0.,1-4*err-.04*abs(heat)/.045) if self.alive else 0.)
        self.t+=1
        return self.obs(),reward,self.t==320,{"alive":self.alive}
    @staticmethod
    def oracle(obs):
        temp,target=float(obs[0]),float(obs[2])
        delta=target-temp
        return 3 if delta>.04 else 2 if delta>0 else 0 if delta<-.04 else 1

class QueueService:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0;self.q=.12;self.alive=True
        self.arr=np.clip(self.rng.gamma(2.,.022,320),.005,.12)
        return self.obs()
    def obs(self):
        nxt=self.arr[min(self.t,319)]
        return np.array([min(1.,self.q/.60),min(1.,nxt/.12),.5],float)
    def step(self,a):
        service=np.array([.02,.05,.08,.11])[int(a)]
        arrival=float(self.arr[self.t]); served=min(self.q+arrival,service)
        self.q=max(0.,self.q+arrival-served)
        self.alive=bool(self.alive and self.q<.58)
        reward=float((1-min(1.,self.q/.60))*(.92-.08*a/3) if self.alive else 0.)
        self.t+=1
        return self.obs(),reward,self.t==320,{"alive":self.alive}
    @staticmethod
    def oracle(obs):
        q=float(obs[0])*.60; arr=float(obs[1])*.12
        need=q+arr
        vals=np.array([.02,.05,.08,.11])
        return int(np.argmin(np.where(vals>=need,vals-need,5+need-vals)))

def qphi(obs):
    r,e,d=np.asarray(obs,float)
    return np.array([1.,r,e,d,r*e,r*d,e*d,r*r,e*e,d*d],float)

class OutcomeAgent:
    def __init__(self):
        self.X=[[] for _ in range(4)];self.y=[[] for _ in range(4)]
        self.w=[None]*4
    def update(self,obs,a,reward):
        self.X[a].append(qphi(obs));self.y[a].append(float(reward))
        if len(self.y[a])>=8:
            X=np.asarray(self.X[a]);y=np.asarray(self.y[a])
            self.w[a]=ridge(X,y,.25)
    def act(self,obs):
        x=qphi(obs);pred=[]
        for a in range(4):
            pred.append(.25 if self.w[a] is None else float(x@self.w[a]))
        return int(np.argmax(pred))

def run_ood(cls,seed,agent=True,prefix=96):
    env=cls();obs=env.reset(seed);rr=[];aa=[];ag=OutcomeAgent()
    for t in range(320):
        act=(0,1,2,1)[t%4] if (agent and t<prefix) else ag.act(obs) if agent else cls.oracle(obs)
        nxt,r,_,info=env.step(act)
        if agent:ag.update(obs,act,r)
        if t>=prefix:
            rr.append(r);aa.append(float(info["alive"]))
        obs=nxt
    return dict(return_mean=float(np.mean(rr)),alive_fraction=float(np.mean(aa)))

def experiment4(seed):
    out={};ok=True
    for name,cls in [("logistic_harvest",LogisticHarvest),("thermal_rc",ThermalRC),("queue_service",QueueService)]:
        s=stable_seed(seed,"e4-"+name)
        oracle=run_ood(cls,s,False,96);full=run_ood(cls,s,True,96)
        ratio=full["return_mean"]/max(oracle["return_mean"],1e-9)
        passed=(oracle["alive_fraction"]>=E4_ORACLE_ALIVE_MIN and
                full["alive_fraction"]>=E4_ALIVE_MIN and ratio>=E4_RETURN_RATIO_MIN)
        out[name]=dict(oracle=oracle,agent=full,return_ratio=ratio,pass_task=bool(passed))
        ok&=passed
    return dict(pass_seed=bool(ok),tasks=out)

def run_seed(seed):
    return dict(seed=int(seed),E1=experiment1(seed),E2=experiment2(seed),
                E3=experiment3(seed),E4=experiment4(seed))
