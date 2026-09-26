"""R10 E4 — frozen cross-domain transfer to separately coded dynamics."""
from __future__ import annotations
import numpy as np
from .core import ridge_fit, ridge_predict
from .config import STEPS_E4_TRAIN, STEPS_E4_TEST

ACTIONS=np.array([0.,1/3,2/3,1.])
SOURCE=("buffer","maintenance")
HELDOUT=("queue","inertia")

def polar_features(obs,u):
    r,e,d=np.asarray(obs,float); u=float(u)
    return np.array([r,e,d,u,u-d,abs(u-d),r*u,e*u,(1-r)*u,(1-e)*u,d*u,u*u],float)

def generic_features(obs,u):
    r,e,d=np.asarray(obs,float); u=float(u)
    return np.array([r,e,d,u,r*r,e*e,d*d,u*u,r*e,r*d,e*d,u*(r+e+d)/3.],float)

class Domain:
    def __init__(self,name,seed):
        self.name=name; self.rng=np.random.default_rng(seed)
        self.reset()
    def reset(self):
        self.r=float(self.rng.uniform(.55,.85)); self.e=float(self.rng.uniform(.55,.85))
        self.d=float(self.rng.uniform(.2,.8)); self.momentum=0.; self.t=0; self.alive=True
        return self.obs()
    def obs(self): return np.array([self.r,self.e,self.d],float)
    def step(self,u):
        if not self.alive:
            self.t+=1
            return self.obs(),0.,False
        n=float(self.rng.normal(0,.006))
        self.d=float(np.clip(.82*self.d+.18*self.rng.uniform(.05,.95),0,1))
        if self.name=="buffer":
            self.r=float(np.clip(self.r+.085-.14*u-.055*self.d+n,0,1))
            self.e=float(np.clip(self.e+.018-.060*abs(u-self.d)-.025*max(0,.35-self.r),0,1))
            service=1.-abs(u-self.d)
        elif self.name=="maintenance":
            self.r=float(np.clip(self.r+.060-.110*u-.020*self.d+n,0,1))
            self.e=float(np.clip(self.e+.022-.085*(u-self.d)**2-.035*max(0,u-.72),0,1))
            service=1.-.75*abs(u-self.d)
        elif self.name=="queue":
            backlog=1.-self.r
            backlog=float(np.clip(backlog+.095*self.d-.125*u+n,0,1))
            self.r=1.-backlog
            self.e=float(np.clip(self.e+.016-.050*backlog-.018*u*u,0,1))
            service=float(np.clip(1.-backlog-.25*abs(u-self.d),0,1))
        elif self.name=="inertia":
            old=self.momentum
            self.momentum=.72*self.momentum+.28*u
            self.r=float(np.clip(self.r+.055-.10*self.momentum-.025*self.d+n,0,1))
            self.e=float(np.clip(self.e+.020-.070*(old-self.d)**2-.030*abs(self.momentum-old),0,1))
            service=float(np.clip(1.-.65*abs(old-self.d),0,1))
        else: raise ValueError(self.name)
        self.alive=bool(self.e>.05 and self.r>.03)
        reward=float(np.clip(service*(.55+.25*self.e+.20*self.r),0,1)) if self.alive else 0.
        self.t+=1
        return self.obs(),reward,self.alive

def collect(seed):
    rng=np.random.default_rng(seed)
    xp=[]; xg=[]; y=[]
    per=max(1,STEPS_E4_TRAIN//len(SOURCE))
    for j,name in enumerate(SOURCE):
        env=Domain(name,seed+100*j)
        obs=env.reset()
        for t in range(per):
            if not env.alive or t%220==0:
                obs=env.reset()
            u=float(ACTIONS[int(rng.integers(0,4))])
            nxt,reward,_=env.step(u)
            xp.append(polar_features(obs,u)); xg.append(generic_features(obs,u))
            y.append(np.r_[nxt,reward])
            obs=nxt
    return np.asarray(xp),np.asarray(xg),np.asarray(y)

def fit(seed):
    xp,xg,y=collect(seed)
    # Exactly equal coefficient count: 13 (intercept + 12) by four outputs.
    return ridge_fit(xp,y,.05),ridge_fit(xg,y,.05)

def evaluate(name,seed,wp,wg):
    envp=Domain(name,seed); envg=Domain(name,seed)
    rewards_p=[]; rewards_g=[]; alive_p=[]; alive_g=[]
    obs_p=envp.reset(); obs_g=envg.reset()
    for t in range(STEPS_E4_TEST):
        if t and t%200==0:
            obs_p=envp.reset(); obs_g=envg.reset()
        predp=np.vstack([ridge_predict(polar_features(obs_p,u)[None],wp)[0] for u in ACTIONS])
        predg=np.vstack([ridge_predict(generic_features(obs_g,u)[None],wg)[0] for u in ACTIONS])
        scorep=predp[:,3]+.15*predp[:,1]+.10*predp[:,0]
        scoreg=predg[:,3]+.15*predg[:,1]+.10*predg[:,0]
        up=float(ACTIONS[int(np.argmax(scorep))]); ug=float(ACTIONS[int(np.argmax(scoreg))])
        obs_p,rp,ap=envp.step(up); obs_g,rg,ag=envg.step(ug)
        rewards_p.append(rp); rewards_g.append(rg); alive_p.append(ap); alive_g.append(ag)
    return dict(polar_return=float(np.mean(rewards_p)),generic_return=float(np.mean(rewards_g)),
                return_advantage=float(np.mean(rewards_p)-np.mean(rewards_g)),
                polar_alive=float(np.mean(alive_p)),generic_alive=float(np.mean(alive_g)))

def run(seed):
    wp,wg=fit(seed+10)
    domains={name:evaluate(name,seed+1000+i,wp,wg) for i,name in enumerate(HELDOUT)}
    conditions=[]
    for m in domains.values():
        conditions.extend([m['polar_return']>=.65,m['polar_alive']>=.80,m['return_advantage']>=.02])
    return dict(seed=int(seed),parameter_count_each=int(wp.size),heldout=domains,
                pass_strong=bool(all(conditions)),
                scope="Two held-out families are separately coded within the same research program; parameters are frozen during transfer.")
