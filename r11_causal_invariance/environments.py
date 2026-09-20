"""R11 internally designed challenge environments.
They are not independent external validation.
"""
from __future__ import annotations
import numpy as np
from r9_completion.agent import encode_poles
ACTIONS=np.array([0.,.04,.08,.12],float)

class RelationalEnv:
    """Dynamics whose action utility is sparse in the declared pair basis."""
    def __init__(self,delay=False):
        self.delay=bool(delay)
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.r=.72; self.e=.74; self.d=.45
        self.hist=np.tile(np.r_[[self.r,self.e,self.d],0.],(16,1))
        self.noise=self.rng.normal(0,.008,640)
        return self.obs()
    def obs(self): return np.clip([self.r,self.e,self.d],0,1)
    def step(self,action):
        obs=self.obs(); poles=encode_poles(obs,self.hist,visits=min(self.t,20))
        orientation=poles[:,0]-poles[:,1]
        switch=((self.t//(80 if self.delay else 55))%2)
        # sparse relation: only a few pair orientations determine desirable throughput.
        latent=.65*orientation[0]-.55*orientation[4]+.45*orientation[7]
        if switch: latent=-latent+.25*orientation[2]
        desired=int(np.clip(np.round(1.5+1.4*np.tanh(2*latent)),0,3))
        a=int(action); flow=ACTIONS[a]
        self.d=float(np.clip(.48+.25*np.sin((self.t+(20 if self.delay else 0))/31.)+.08*self.noise[self.t%640],.08,.92))
        self.r=float(np.clip(self.r+.018-.50*flow-.012*self.d,0,1))
        action_quality=np.exp(-.85*(a-desired)**2)
        self.e=float(np.clip(self.e+.010+.025*action_quality-.022*(1-action_quality)-.006*self.d,0,1))
        self.alive=bool(self.alive and self.e>.08 and self.r>.05)
        reward=float(action_quality*(.55+.45*self.e) if self.alive else 0.)
        self.hist=np.concatenate((self.hist[1:],np.r_[obs,float(a)][None]),axis=0)
        self.t+=1
        return self.obs(),reward,self.t>=320,dict(alive=self.alive,desired=desired)

class AttributionTransferEnv:
    """Viable repeated-context source/time/counterfactual environment."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.r=.82; self.e=.84; self.d=.45; self.last_world=-99
        idx=np.arange(640)
        phase=int(self.rng.integers(0,11))
        self.world=((idx+phase)%11==0)|((idx+phase)%37==0)
        self.world_mag=np.where(((idx+phase)//11)%2==0,.045,-.045)*self.world
        self.dem=np.clip(.45+.30*np.sin(2*np.pi*(idx+phase)/64.)+.08*np.sin(2*np.pi*idx/17.),.08,.92)
        return self.obs()
    def obs(self): return np.clip([self.r,self.e,self.d],0,1)
    def step(self,action):
        before=self.obs(); a=int(action); flow=ACTIONS[a]
        oldd=self.d; self.d=float(self.dem[self.t])
        # action effect and world effect are simultaneously present often enough to form 3 source classes.
        self_vec=np.array([-.24*flow,.070*(flow/.12)-.020*(flow/.12)**2,0.])
        shock=float(self.world_mag[self.t])
        world_vec=np.array([.55*shock,.75*shock,self.d-oldd])
        self.r=float(np.clip(self.r+.020+self_vec[0]+world_vec[0],0,1))
        self.e=float(np.clip(self.e+.004+self_vec[1]+world_vec[1]-.006*self.d,0,1))
        self.alive=bool(self.alive and self.r>.06 and self.e>.12)
        desired=.018+.095*self.d
        service=min(1.,flow/max(desired,1e-6))
        # reward has an interaction that makes relational prediction useful.
        balance=1.-min(1.,abs(self.r-self.e))
        reward=float(service*(.55+.30*self.e+.15*balance)*(1-.08*flow/.12) if self.alive else 0.)
        sm=float(np.linalg.norm(self_vec[:2])); wm=float(np.linalg.norm(world_vec[:2]))
        if wm>1.35*max(sm,1e-9): src=1
        elif sm>1.35*max(wm,1e-9): src=0
        else: src=2
        if self.world[self.t]: self.last_world=self.t
        age=self.t-self.last_world; agebin=0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3
        self.t+=1
        return self.obs(),reward,self.t>=640,dict(alive=self.alive,source=src,agebin=agebin,
            self_strength=sm,world_strength=wm,before=before.tolist())
