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
        return self.obs() if not done else np.zeros(5),reward,done,{"regime":r,"error":float(err),"target":float(target)}

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
    def update(self,obs,a,reward,target):
        k=self.key(obs)
        # The factual signed action consequence identifies the local target.
        x=self.phi(obs)
        pseudo=float(target)
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
    full=ContextMemoryAgent(True);reset=ContextMemoryAgent(False)
    def run(agent):
        e=ContinualRegimeEnv();o=e.reset(stable_seed(seed,"A")); blocks=[];early=[]
        for b in range(8):
            rr=[]
            for t in range(600):
                agent.maybe_reset(o)
                a=(t%4) if t<48 and b==0 else agent.act(o)
                no,r,d,info=e.step(a);agent.update(o,a,r,info["target"]);o=no;rr.append(r)
            blocks.append(float(np.mean(rr)));early.append(float(np.mean(rr[:24])))
        return blocks,early
    fb,fe=run(full); rb,re=run(reset)
    rec_pairs=[(0,4),(1,6),(2,5),(3,7)]
    recurrence=np.mean([fe[j]-fe[i] for i,j in rec_pairs])
    reset_recurrence=np.mean([re[j]-re[i] for i,j in rec_pairs])
    retention=np.mean([fb[j]/max(fb[i],1e-9) for i,j in rec_pairs])
    passed=(recurrence-reset_recurrence>=A_RECURRENT_GAIN and
            np.mean(fb[-2:])>=A_FINAL_REWARD and retention>=A_RETENTION)
    return dict(pass_seed=bool(passed),full_blocks=fb,reset_blocks=rb,full_early=fe,reset_early=re,
                recurrent_gain=float(recurrence-reset_recurrence),
                final_reward=float(np.mean(fb[-2:])),retention=float(retention))

# -------------------------------------------------------------------
# B. Autobiographical self
# -------------------------------------------------------------------
class AutobiographicalEnv:
    def reset(self,identity_seed,experience_seed,complement=False,horizon=1200):
        irng=np.random.default_rng(identity_seed);self.rng=np.random.default_rng(experience_seed);self.t=0;self.horizon=horizon
        base=irng.uniform(.18,.95,(4,4))
        self.cap=np.clip(1.13-base if complement else base,.08,.98)
        self.fatigue=.12
        self.tasks=self.rng.integers(0,4,horizon)
        self.diff=np.clip(self.rng.normal(.62,.10,horizon),.35,.85)
        self.noise=self.rng.normal(0,.018,horizon)
        return self.obs()
    def obs(self):
        return np.array([self.tasks[self.t]/3,self.diff[self.t],self.fatigue],float)
    def expected(self,a):
        task=self.tasks[self.t];difficulty=self.diff[self.t]
        ability=self.cap[task,int(a)]*(1-.30*self.fatigue)
        return float(np.clip(.10+.90*ability*np.exp(-2.0*max(0,difficulty-ability)**2),0,1))
    def step(self,a):
        exp=self.expected(a);r=float(np.clip(exp+self.noise[self.t],0,1))
        effort=(a+1)/4;self.fatigue=float(np.clip(.91*self.fatigue+.055*effort-.025*(a==0),0,.9))
        self.t+=1;done=self.t>=self.horizon
        return self.obs() if not done else np.zeros(3),r,done,{}

class SelfMemory:
    def __init__(self):
        self.count=np.ones((4,4))*1.5;self.mean=np.ones((4,4))*.45
    def taskbin(self,obs):return int(np.clip(round(float(obs[0])*3),0,3))
    def predict(self,obs,a):return float(self.mean[self.taskbin(obs),a])
    def update(self,obs,a,r):
        k=self.taskbin(obs);n=self.count[k,a]
        self.mean[k,a]=(self.mean[k,a]*n+r)/(n+1);self.count[k,a]=n+1
    def act(self,obs):return int(np.argmax([self.predict(obs,a) for a in range(4)]))

def acquire_memory(seed,complement=False):
    e=AutobiographicalEnv();o=e.reset(stable_seed(seed,"B-identity"),stable_seed(seed,"B-acquire"),complement=complement,horizon=900)
    m=SelfMemory()
    for t in range(900):
        a=t%4
        no,r,d,_=e.step(a);m.update(o,a,r);o=no
    return m

def eval_memory(seed,memory=None,stateless=False,complement=False):
    e=AutobiographicalEnv();o=e.reset(stable_seed(seed,"B-identity"),stable_seed(seed,"B-eval"),complement=complement,horizon=600)
    m=SelfMemory() if memory is None else copy.deepcopy(memory)
    rr=[];err=[]
    for t in range(600):
        a=1 if stateless else m.act(o)
        pred=.45 if stateless else m.predict(o,a)
        no,r,d,_=e.step(a);rr.append(r);err.append(abs(pred-r));o=no
    return dict(reward=float(np.mean(rr)),mae=float(np.mean(err)))

def experiment_b(seed):
    own_mem=acquire_memory(seed,False)
    donor_mem=acquire_memory(seed,True)
    own=eval_memory(seed,own_mem,False,False)
    stat=eval_memory(seed,None,True,False)
    transplanted=eval_memory(seed,donor_mem,False,False)
    pred_gain=stat["mae"]-own["mae"]
    reward_gain=own["reward"]-stat["reward"]
    transplant_reward_damage=own["reward"]-transplanted["reward"]
    transplant_prediction_damage=transplanted["mae"]-own["mae"]
    passed=(pred_gain>=B_PRED_GAIN and
            transplant_prediction_damage>=B_TRANSPLANT_PRED_DAMAGE and
            transplant_reward_damage>=B_TRANSPLANT_REWARD_DAMAGE)
    return dict(pass_seed=bool(passed),own_reward=own["reward"],own_mae=own["mae"],
                stateless_reward=stat["reward"],stateless_mae=stat["mae"],
                transplanted_reward=transplanted["reward"],transplanted_mae=transplanted["mae"],
                prediction_gain=float(pred_gain),reward_gain=float(reward_gain),
                transplant_prediction_damage=float(transplant_prediction_damage),
                transplant_reward_damage=float(transplant_reward_damage))

# -------------------------------------------------------------------
# C. Endogenous goal-priority adaptation
# -------------------------------------------------------------------
class GoalEnv:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0
        self.state=np.array([.76,.78,.74],float)
        self.pulse=np.zeros((1200,3));self.shock_dim=np.full(1200,-1,int)
        for step in range(90,1200,120):
            k=int(self.rng.integers(0,3));self.pulse[step,k]=.30;self.shock_dim[step]=k
        return self.state.copy()
    def step(self,a):
        self.state-=np.array([.008,.008,.008])
        self.state-=self.pulse[self.t]
        if a<3:
            self.state[a]+=.075
            self.state[(a+1)%3]-=.003
        else:
            self.state-=.003
        self.state=np.clip(self.state,0,1)
        alive=float(np.all(self.state>.10))
        task=1.0 if a==3 else .12
        reward=float(alive*(.60*np.mean(self.state)+.40*task))
        info={"alive":alive,"shock_dim":int(self.shock_dim[self.t])}
        self.t+=1
        return self.state.copy(),reward,self.t>=1200,info

class GoalAgent:
    def __init__(self,adaptive):
        self.adaptive=adaptive;self.prev=None;self.w=np.ones(3)/3
    def act(self,s):
        s=np.asarray(s)
        if self.adaptive:
            deficit=np.clip(.66-s,0,1)
            trend=np.zeros(3) if self.prev is None else np.clip(self.prev-s,0,1)
            raw=.025+2.8*deficit+1.6*trend
            self.w=raw/raw.sum()
        vals=[]
        for a in range(4):
            ns=s.copy()
            if a<3:
                ns[a]+=.075;ns[(a+1)%3]-=.003
                utility=float(self.w@ns)
            else:
                ns-=.003;utility=float(self.w@ns+.025)
            vals.append(utility)
        self.prev=s.copy()
        return int(np.argmax(vals))

def run_goal(seed,adaptive):
    e=GoalEnv();s=e.reset(stable_seed(seed,"C"))
    ag=GoalAgent(adaptive);alive=[];rr=[];priority=[];recoveries=[];pending=[]
    for t in range(1200):
        a=ag.act(s);ns,r,d,info=e.step(a)
        alive.append(info["alive"]);rr.append(r)
        true_low=int(np.argmin(s));priority.append(int(np.argmax(ag.w))==true_low)
        if info["shock_dim"]>=0:
            pending.append([info["shock_dim"],0])
        nxt=[]
        for dim,age in pending:
            age+=1
            if ns[dim]>=.60: recoveries.append(age)
            elif age<120: nxt.append([dim,age])
            else: recoveries.append(120)
        pending=nxt
        s=ns
    recoveries += [120 for _ in pending]
    return dict(alive=float(np.mean(alive)),reward=float(np.mean(rr)),
                priority_acc=float(np.mean(priority)),
                recovery=float(np.mean(recoveries)) if recoveries else 120.)

def experiment_c(seed):
    a=run_goal(seed,True);f=run_goal(seed,False)
    reward_gain=a["reward"]-f["reward"]
    rec_gain=(f["recovery"]-a["recovery"])/max(f["recovery"],1)
    passed=(a["alive"]>=C_ALIVE_MIN and a["priority_acc"]>=C_PRIORITY_ACC and
            reward_gain>=C_REWARD_GAIN and rec_gain>=C_SHOCK_RECOVERY_GAIN)
    return dict(pass_seed=bool(passed),adaptive=a,fixed=f,
                alive_gain=float(a["alive"]-f["alive"]),reward_gain=float(reward_gain),
                recovery_gain=float(rec_gain))

# -------------------------------------------------------------------
# D. Individual -> population -> ecology
# -------------------------------------------------------------------
class CollectiveSystem:
    def __init__(self,seed,local=True,transmission=True):
        self.rng=np.random.default_rng(seed);self.N=30;self.local=local;self.transmission=transmission
        self.resource=.78
        self.energy=np.clip(self.rng.normal(.72,.06,self.N),.5,.9)
        self.theta=np.clip(self.rng.normal(.28,.08,self.N),.05,.65)
        informed=self.rng.random(self.N)<.38
        self.adapt_rate=np.where(informed,self.rng.uniform(.85,1.25,self.N),0.)
        self.last_resource=self.resource
    def step(self,t):
        if t in (320,720,1120): self.resource=max(.05,self.resource-.16)
        if self.local:
            dr=self.resource-self.last_resource
            scarcity=max(0,.58-self.resource)
            need=np.clip(.42-self.energy,0,1)
            self.theta=np.clip(self.theta+self.adapt_rate*(.085*scarcity-.020*need-.30*dr),.02,.95)
        extraction=.00035+.00085*(1-self.theta)
        taken=np.minimum(extraction,self.resource/self.N)
        self.resource=max(0.,self.resource-float(np.sum(taken)))
        regen=.14*self.resource*(1-self.resource)
        self.resource=min(1.,self.resource+regen)
        self.energy=np.clip(self.energy+22.0*taken-.012,0,1)
        alive=self.energy>.08
        # Explicit public-resource-aware transmission rule; not an emergent moral value.
        fitness=self.energy+.80*self.theta*self.resource
        if self.transmission and t%20==19:
            for i in range(self.N):
                cand=self.rng.choice(self.N,6,replace=False)
                j=int(cand[np.argmax(fitness[cand])])
                if fitness[j]>fitness[i]+.010:
                    self.theta[i]=np.clip(.97*self.theta[j]+self.rng.normal(0,.010),.02,.95)
        self.last_resource=self.resource
        return float(np.mean(alive)),float(np.mean(self.theta)),float(self.resource)

def run_collective(seed,local,trans):
    s=CollectiveSystem(stable_seed(seed,"D"),local,trans)
    aa=[];tt=[];rr=[]
    for t in range(1600):
        a,th,r=s.step(t);aa.append(a);tt.append(th);rr.append(r)
    return dict(alive=float(np.mean(aa[-400:])),restraint=float(np.mean(tt[-400:])),
                resource=float(np.mean(rr[-400:])))

def experiment_d(seed):
    full=run_collective(seed,True,True)
    nolocal=run_collective(seed,False,True)
    notrans=run_collective(seed,True,False)
    resource_gain=full["resource"]-nolocal["resource"]
    restraint_gain=full["restraint"]-nolocal["restraint"]
    transmission_shift=abs(full["restraint"]-notrans["restraint"])
    ecology_transmission_effect=abs(full["resource"]-notrans["resource"])
    passed=(full["alive"]>=D_FULL_ALIVE_MIN and
            resource_gain>=D_RESOURCE_GAIN and
            restraint_gain>=D_RESTRAINT_GAIN and
            transmission_shift>=D_TRANSMISSION_SHIFT and
            ecology_transmission_effect>=D_ECOLOGY_TRANSMISSION_EFFECT)
    return dict(pass_seed=bool(passed),full=full,no_local=nolocal,no_transmission=notrans,
                resource_gain=float(resource_gain),
                alive_gain=float(full["alive"]-nolocal["alive"]),
                restraint_gain=float(restraint_gain),
                transmission_shift=float(transmission_shift),
                ecology_transmission_effect=float(ecology_transmission_effect))

def run_seed(seed):
    return dict(seed=int(seed),A=experiment_a(seed),B=experiment_b(seed),
                C=experiment_c(seed),D=experiment_d(seed))
