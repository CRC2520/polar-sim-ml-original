"""POLAR P3 development experiments."""
from __future__ import annotations
import copy, hashlib, math
import numpy as np
from .config_dev import *

def stable_seed(seed,label):
    raw=hashlib.sha256(f"P3|{int(seed)}|{label}".encode()).digest()
    return int.from_bytes(raw[:8],"little") & 0xffffffff

def ridge(X,y,lam=.2):
    X=np.asarray(X,float);y=np.asarray(y,float)
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y)

# -------------------------------------------------------------------
# A. Prolonged autonomy / continual learning
# -------------------------------------------------------------------
ACTIONS=np.array([-1.2,-.4,.4,1.2],float)

class ContinualRegimeEnv:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0
        self.block=0; self.block_len=600
        self.schedule=[0,1,2,3,0,2,1,3]
        self.centers=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        self.w=np.array([
            [ .9,-.2,.4],
            [-.4, .8,.3],
            [ .3, .2,-.9],
            [-.7,-.4,.8],
        ])
        self.bias=np.array([-.2,.25,-.15,.10])
        self.x=self.rng.normal(0,.55,3)
        self.noise=self.rng.normal(0,.12,(len(self.schedule)*self.block_len,3))
        self.cnoise=self.rng.normal(0,.10,(len(self.schedule)*self.block_len,2))
        return self.obs()
    def regime(self):
        return self.schedule[min(len(self.schedule)-1,self.t//self.block_len)]
    def obs(self):
        r=self.regime(); cue=self.centers[r]+self.cnoise[min(self.t,len(self.cnoise)-1)]
        return np.r_[self.x,cue]
    def step(self,a):
        r=self.regime()
        target=float(np.clip(self.x@self.w[r]+self.bias[r],-1.2,1.2))
        err=target-ACTIONS[int(a)]
        reward=float(np.exp(-1.35*err*err))
        self.x=.78*self.x+self.noise[self.t]
        self.x=np.clip(self.x,-1.4,1.4)
        self.t+=1
        done=self.t>=len(self.schedule)*self.block_len
        return self.obs() if not done else np.zeros(5),reward,done,{"regime":r}

class ContextMemoryAgent:
    def __init__(self,persistent=True):
        self.persistent=persistent; self.models={}; self.buffers={};self.last_key=None
    def key(self,obs):
        c=np.asarray(obs)[3:5]
        return tuple((c>0).astype(int).tolist())
    def phi(self,obs):
        x=np.asarray(obs)[:3]
        return np.r_[1.,x,x*x]
    def act(self,obs):
        k=self.key(obs);W=self.models.get(k)
        if W is None:return 1
        pred=self.phi(obs)@W
        return int(np.argmin(np.abs(ACTIONS-pred)))
    def update(self,obs,a,reward):
        k=self.key(obs)
        # reward alone plus chosen action gives approximate target around selected action;
        # factual sign from action relative to local state creates a stable own-history estimator.
        x=self.phi(obs)
        pseudo=ACTIONS[a]
        self.buffers.setdefault(k,[]).append((x,pseudo,reward))
        b=self.buffers[k][-160:]
        if len(b)>=24:
            X=np.array([z[0] for z in b]); yy=np.array([z[1] for z in b])
            ww=np.array([max(.05,z[2]) for z in b])
            Xw=X*np.sqrt(ww)[:,None]; yw=yy*np.sqrt(ww)
            self.models[k]=ridge(Xw,yw,.35)
    def maybe_reset(self,obs):
        k=self.key(obs)
        if self.last_key is not None and k!=self.last_key and not self.persistent:
            self.models={};self.buffers={}
        self.last_key=k

def experiment_a(seed):
    env=ContinualRegimeEnv();obs=env.reset(stable_seed(seed,"A"))
    full=ContextMemoryAgent(True);reset=ContextMemoryAgent(False)
    rewards={"full":[],"reset":[]}
    # run matched separate envs
    def run(agent):
        e=ContinualRegimeEnv();o=e.reset(stable_seed(seed,"A")); blocks=[]
        for b in range(8):
            rr=[]
            for t in range(600):
                agent.maybe_reset(o)
                a=(t%4) if t<48 and b==0 else agent.act(o)
                no,r,d,info=e.step(a);agent.update(o,a,r);o=no;rr.append(r)
            blocks.append(float(np.mean(rr)))
        return blocks
    fb=run(full); rb=run(reset)
    # recurrence gain compares second visit of each regime with its first visit
    rec_pairs=[(0,4),(1,6),(2,5),(3,7)]
    recurrence=np.mean([fb[j]-fb[i] for i,j in rec_pairs])
    reset_recurrence=np.mean([rb[j]-rb[i] for i,j in rec_pairs])
    retention=np.mean([fb[j]/max(fb[i],1e-9) for i,j in rec_pairs])
    passed=(recurrence-reset_recurrence>=A_RECURRENT_GAIN and
            np.mean(fb[-2:])>=A_FINAL_REWARD and retention>=A_RETENTION)
    return dict(pass_seed=bool(passed),full_blocks=fb,reset_blocks=rb,
                recurrent_gain=float(recurrence-reset_recurrence),
                final_reward=float(np.mean(fb[-2:])),retention=float(retention))

# -------------------------------------------------------------------
# B. Autobiographical self
# -------------------------------------------------------------------
class AutobiographicalEnv:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0
        self.skill=np.clip(self.rng.normal([.35,.55,.75,.45],.06),.15,.9)
        self.fatigue=.15
        self.tasks=self.rng.integers(0,4,1600)
        self.diff=np.clip(self.rng.normal(.55,.18,1600),.15,.9)
        return self.obs()
    def obs(self):
        return np.array([self.tasks[self.t]/3,self.diff[self.t],self.fatigue],float)
    def expected(self,a):
        task=self.tasks[self.t];difficulty=self.diff[self.t]
        cap=self.skill[int(a)]*(1-.55*self.fatigue)
        return float(np.exp(-5*(difficulty-cap)**2))
    def step(self,a):
        exp=self.expected(a)
        r=float(np.clip(exp+self.rng.normal(0,.025),0,1))
        effort=(a+1)/4
        self.fatigue=float(np.clip(.90*self.fatigue+.07*effort-.04*(a==0),0,1))
        # own action history changes persistent competence slowly
        self.skill[int(a)]=float(np.clip(self.skill[int(a)]+.010*(r-.55),.10,.95))
        self.t+=1
        done=self.t>=1600
        return self.obs() if not done else np.zeros(3),r,done,{}

class SelfMemory:
    def __init__(self):
        self.count=np.ones((4,4))*2
        self.mean=np.ones((4,4))*.5
    def taskbin(self,obs):return int(np.clip(round(float(obs[0])*3),0,3))
    def predict(self,obs,a):
        return float(self.mean[self.taskbin(obs),a])
    def update(self,obs,a,r):
        k=self.taskbin(obs);n=self.count[k,a]
        self.mean[k,a]=(self.mean[k,a]*n+r)/(n+1);self.count[k,a]=n+1
    def act(self,obs):return int(np.argmax([self.predict(obs,a) for a in range(4)]))

def run_self(seed,memory=None,stateless=False):
    e=AutobiographicalEnv();o=e.reset(stable_seed(seed,"B-env"))
    m=SelfMemory() if memory is None else copy.deepcopy(memory)
    rr=[];err=[]
    for t in range(1600):
        a=t%4 if t<128 else (1 if stateless else m.act(o))
        pred=.5 if stateless else m.predict(o,a)
        no,r,d,_=e.step(a)
        if t>=128: rr.append(r);err.append(abs(pred-r))
        if not stateless:m.update(o,a,r)
        o=no
    return dict(reward=float(np.mean(rr)),mae=float(np.mean(err)),memory=m)

def experiment_b(seed):
    own=run_self(seed)
    stat=run_self(seed,stateless=True)
    donor=run_self(seed+1000003)["memory"]
    transplanted=run_self(seed,memory=donor)
    pred_gain=stat["mae"]-own["mae"]
    reward_gain=own["reward"]-stat["reward"]
    transplant_damage=own["reward"]-transplanted["reward"]
    passed=(pred_gain>=B_PRED_GAIN and reward_gain>=B_REWARD_GAIN and transplant_damage>=B_TRANSPLANT_DAMAGE)
    return dict(pass_seed=bool(passed),own_reward=own["reward"],own_mae=own["mae"],
                stateless_reward=stat["reward"],stateless_mae=stat["mae"],
                transplanted_reward=transplanted["reward"],
                prediction_gain=float(pred_gain),reward_gain=float(reward_gain),
                transplant_damage=float(transplant_damage))

# -------------------------------------------------------------------
# C. Endogenous goal-priority adaptation
# -------------------------------------------------------------------
class GoalEnv:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0
        self.state=np.array([.78,.80,.76],float)
        self.shock=np.zeros((1200,3))
        for s in range(80,1200,120):
            k=int(self.rng.integers(0,3));self.shock[s:s+25,k]=.025
        return self.state.copy()
    def step(self,a):
        # 0/1/2 replenish a resource, 3 task action
        decay=np.array([.010,.009,.008])+self.shock[self.t]
        self.state-=decay
        if a<3:
            self.state[a]+=.045
            self.state[(a+1)%3]-=.006
        else:
            self.state-=.004
        self.state=np.clip(self.state,0,1)
        alive=float(np.all(self.state>.10))
        task=1.0 if a==3 else .15
        reward=float(alive*(.55*np.mean(self.state)+.45*task))
        self.t+=1
        return self.state.copy(),reward,self.t>=1200,{"alive":alive}

class GoalAgent:
    def __init__(self,adaptive):
        self.adaptive=adaptive;self.prev=None;self.w=np.ones(3)/3
    def act(self,s):
        s=np.asarray(s)
        if self.adaptive:
            deficit=np.clip(.65-s,0,1)
            trend=np.zeros(3) if self.prev is None else np.clip(self.prev-s,0,1)
            raw=.03+2.4*deficit+1.3*trend
            self.w=raw/raw.sum()
        vals=[]
        for a in range(4):
            ns=s.copy()
            if a<3:
                ns[a]+=.045;ns[(a+1)%3]-=.006
                utility=float(self.w@ns)
            else:
                ns-=.004;utility=float(self.w@ns+.18)
            vals.append(utility)
        self.prev=s.copy()
        return int(np.argmax(vals))

def run_goal(seed,adaptive):
    e=GoalEnv();s=e.reset(stable_seed(seed,"C"))
    ag=GoalAgent(adaptive);alive=[];rr=[];priority=[];recovery=[]
    low_streak=0
    for t in range(1200):
        a=ag.act(s);ns,r,d,info=e.step(a)
        alive.append(info["alive"]);rr.append(r)
        true_low=int(np.argmin(s));priority.append(int(np.argmax(ag.w))==true_low)
        if np.min(s)<.45:low_streak+=1
        elif low_streak:
            recovery.append(low_streak);low_streak=0
        s=ns
    return dict(alive=float(np.mean(alive)),reward=float(np.mean(rr)),
                priority_acc=float(np.mean(priority)),
                recovery=float(np.mean(recovery)) if recovery else 999.)

def experiment_c(seed):
    a=run_goal(seed,True);f=run_goal(seed,False)
    alive_gain=a["alive"]-f["alive"]
    rec_gain=(f["recovery"]-a["recovery"])/max(f["recovery"],1)
    passed=(alive_gain>=C_ALIVE_GAIN and a["priority_acc"]>=C_PRIORITY_ACC and rec_gain>=C_SHOCK_RECOVERY_GAIN)
    return dict(pass_seed=bool(passed),adaptive=a,fixed=f,
                alive_gain=float(alive_gain),recovery_gain=float(rec_gain))

# -------------------------------------------------------------------
# D. Individual -> population -> ecology
# -------------------------------------------------------------------
class CollectiveSystem:
    def __init__(self,seed,local=True,transmission=True):
        self.rng=np.random.default_rng(seed);self.N=30;self.local=local;self.transmission=transmission
        self.resource=.72
        self.energy=np.clip(self.rng.normal(.72,.06,self.N),.5,.9)
        self.theta=np.clip(self.rng.normal(.35,.08,self.N),.05,.8) # restraint
        self.last_resource=self.resource
    def step(self,t):
        if self.local:
            dr=self.resource-self.last_resource
            scarcity=max(0,.55-self.resource)
            need=np.clip(.45-self.energy,0,1)
            self.theta=np.clip(self.theta+.045*scarcity-.035*need-.10*dr,.02,.95)
        extraction=.010+.022*(1-self.theta)
        taken=np.minimum(extraction,self.resource/self.N)
        self.resource=max(0.,self.resource-float(np.sum(taken)))
        regen=.14*self.resource*(1-self.resource)
        self.resource=min(1.,self.resource+regen)
        self.energy=np.clip(self.energy+.75*taken-.012,0,1)
        alive=self.energy>.08
        fitness=self.energy+.25*self.resource
        if self.transmission and t%40==39:
            for i in range(self.N):
                j=int(self.rng.integers(0,self.N))
                if fitness[j]>fitness[i]+.03:
                    self.theta[i]=np.clip(.94*self.theta[j]+self.rng.normal(0,.015),.02,.95)
        self.last_resource=self.resource
        return float(np.mean(alive)),float(np.mean(self.theta)),float(self.resource)

def run_collective(seed,local,trans):
    s=CollectiveSystem(stable_seed(seed,f"D-{local}-{trans}"),local,trans)
    aa=[];tt=[];rr=[]
    for t in range(1600):
        a,th,r=s.step(t);aa.append(a);tt.append(th);rr.append(r)
    return dict(alive=float(np.mean(aa[-400:])),restraint=float(np.mean(tt[-400:])),
                resource=float(np.mean(rr[-400:])))

def experiment_d(seed):
    full=run_collective(seed,True,True)
    nolocal=run_collective(seed,False,True)
    notrans=run_collective(seed,True,False)
    rg=full["resource"]-nolocal["resource"]
    ag=full["alive"]-nolocal["alive"]
    ts=abs(full["restraint"]-notrans["restraint"])
    passed=(rg>=D_RESOURCE_GAIN and ag>=D_ALIVE_GAIN and ts>=D_TRANSMISSION_SHIFT)
    return dict(pass_seed=bool(passed),full=full,no_local=nolocal,no_transmission=notrans,
                resource_gain=float(rg),alive_gain=float(ag),transmission_shift=float(ts))

def run_seed(seed):
    return dict(seed=int(seed),A=experiment_a(seed),B=experiment_b(seed),
                C=experiment_c(seed),D=experiment_d(seed))
