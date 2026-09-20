"""Second development pass for R11 after pilot-1 diagnostics.

Changes are development-only:
- gate truth is factual utility (not a designer regime label),
- transfer domains include oracle feasibility controls,
- transfer permits a declared 48-step own-action identification prefix,
- integrated assay freezes model parameters after prefix while factual memory/ERR/source state continue.
"""
from __future__ import annotations
import copy, hashlib, math
import numpy as np
from r9_completion.environments import ScalarEnvironment
from .agent import CompactCausalAgent, ACT
from .config_dev import SOURCE_EPISODES,SOURCE_STEPS
from .benchmarks import stable_seed, relational_benchmark, _ridge, _gate_basis, _ba, _auc

ADAPT_PREFIX=48

def source_train_v2(seed):
    a=CompactCausalAgent(seed,"polar")
    rng=np.random.default_rng(stable_seed(seed,"source-v2"))
    for ep in range(SOURCE_EPISODES):
        env=ScalarEnvironment("ecology_train"); obs=env.reset(stable_seed(seed,f"src2-{ep}")); a.reset_episode()
        for t in range(SOURCE_STEPS):
            gate=float(rng.integers(0,2))
            p=a.prepare(obs,force_gate=gate)
            action=int(rng.integers(0,4)) if rng.random()<.25 else int(p["action"])
            nxt,r,_,_=env.step(action)
            a.complete(obs,action,nxt,r,p,learn_model=True,learn_memory=True,learn_gate=True)
            obs=nxt
    return a

# ---------- G3: gate learns realized utility, evaluator label is actual better route ----------

def gate_benchmark_v2(seed,ntrain=7000,ntest=3500):
    rng=np.random.default_rng(stable_seed(seed,"gate-utility-v2"))
    X=rng.normal(size=(ntrain+ntest,6))
    hidden=.9*X[:,0]*X[:,1]-.7*X[:,2]*X[:,3]+.55*X[:,4]*X[:,5]+.25*X[:,0]**2
    local=.35*X[:,0]-.2*X[:,3]+.1*X[:,5]
    corr=.85*np.tanh(X[:,1]-X[:,2]+.4*X[:,4])
    # Continuous mixture makes the utility boundary contextual and not identical to sign(hidden).
    alpha=np.tanh(hidden)
    y=local+alpha*corr+rng.normal(0,.05,len(X))
    off_loss=(y-local)**2
    on_loss=(y-(local+corr))**2
    truth=(on_loss<off_loss).astype(int)

    # Contextual bandit: only chosen route's factual loss is used for fitting.
    chosen=rng.integers(0,2,ntrain)
    factual=np.where(chosen.astype(bool),on_loss[:ntrain],off_loss[:ntrain])
    B=_gate_basis(X[:ntrain])
    models=[]
    for g in (0,1):
        m=chosen==g
        models.append(_ridge(B[m],factual[m],.35))
    Bt=_gate_basis(X[ntrain:])
    gate=((Bt@models[1])<(Bt@models[0])).astype(int)
    yt=y[ntrain:]; lt=local[ntrain:]; ct=lt+corr[ntrain:]
    pred=np.where(gate.astype(bool),ct,lt)
    mse=lambda z:float(np.mean((yt-z)**2))
    acc=float(np.mean(gate==truth[ntrain:]))
    best=min(mse(lt),mse(ct)); gain=best-mse(pred)

    Xp=X[ntrain:].copy(); Xp[:,[0,2,4]]=Xp[:,[2,4,0]]
    Bp=_gate_basis(Xp); pg=((Bp@models[1])<(Bp@models[0])).astype(int)
    pp=np.where(pg.astype(bool),ct,lt)
    return dict(accuracy=acc,gain=gain,permutation_damage=mse(pp)-mse(pred),
                adaptive_mse=mse(pred),off_mse=mse(lt),on_mse=mse(ct),
                true_on_fraction=float(np.mean(truth[ntrain:])),selected_on_fraction=float(np.mean(gate)))

# ---------- G4: two feasible independently coded transfer dynamics ----------

class CyclicBufferV2:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.80; self.health=.86; self.deficit=0.; self.pipe=[0.,0.]
        idx=np.arange(320)
        self.repl=np.clip(.035+.018*np.sin(2*np.pi*idx/37.)+self.rng.normal(0,.003,320),.008,.065)
        self.dem=np.clip(.048+.025*(np.sin(2*np.pi*idx/53.+.4)>0)+self.rng.normal(0,.003,320),.025,.085)
        return self.obs()
    def obs(self):
        return np.clip([self.reserve,self.health,self.dem[min(self.t,319)]/.09],0,1)
    def step(self,action):
        flow=float(ACT[int(action)]); demand=float(self.dem[self.t]); arrival=self.pipe.pop(0);self.pipe.append(float(self.repl[self.t]))
        dispatch=min(self.reserve+arrival,flow); served=min(dispatch,demand); unmet=max(0.,demand-served)
        self.reserve=np.clip(self.reserve+arrival-dispatch,0,1)
        self.deficit=.82*self.deficit+unmet
        self.health=np.clip(self.health+.004-.055*self.deficit-.020*max(0.,dispatch-.08),0,1)
        self.alive=bool(self.alive and self.health>.08 and (self.reserve>.015 or arrival>.015))
        service=served/max(demand,1e-9)
        reward=float(service*(.62+.38*self.health)*(1-.08*dispatch/.12) if self.alive else 0)
        self.t+=1
        return self.obs(),reward,self.t==320,dict(alive=self.alive)
    @staticmethod
    def oracle_action(obs):
        # Demand proxy is obs[2]*.09; choose smallest action meeting it.
        demand=float(obs[2])*.09
        return int(np.argmin(np.where(ACT+1e-9>=demand,ACT-demand,10+np.abs(ACT-demand))))

class RepairQueueV2:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0;self.alive=True
        self.reserve=.78;self.health=.86;self.queue=.06
        self.arr=np.clip(self.rng.gamma(1.7,.010,320),0,.055)
        self.supply=np.clip(self.rng.normal(.024,.005,320),.006,.042)
        return self.obs()
    def obs(self):return np.clip([self.reserve,self.health,self.queue/.40],0,1)
    def step(self,action):
        intensity=float(ACT[int(action)])
        self.queue=min(.65,self.queue+float(self.arr[self.t]))
        repair=min(self.queue,intensity,self.reserve)
        self.queue=max(0.,self.queue-repair)
        self.reserve=np.clip(self.reserve-repair+float(self.supply[self.t]),0,1)
        self.health=np.clip(self.health+.004+.018*repair/.12-.030*self.queue-.010*max(0.,intensity-.08)/.04,0,1)
        self.alive=bool(self.alive and self.health>.08 and self.queue<.58)
        reward=float((1-min(1.,self.queue/.40))*(.60+.40*self.health)*(1-.06*intensity/.12) if self.alive else 0)
        self.t+=1
        return self.obs(),reward,self.t==320,dict(alive=self.alive)
    @staticmethod
    def oracle_action(obs):
        q=float(obs[2])*.40
        target=.08 if q>.12 else .04 if q>.05 else 0.
        return int(np.argmin(np.abs(ACT-target)))

def oracle_run(cls,seed):
    env=cls();obs=env.reset(seed);rr=[];aa=[]
    for _ in range(320):
        act=cls.oracle_action(obs);obs,r,_,info=env.step(act);rr.append(r);aa.append(float(info["alive"]))
    return dict(return_mean=float(np.mean(rr)),alive_fraction=float(np.mean(aa)))

def transfer_run(agent,cls,seed,adapt):
    env=cls();obs=env.reset(seed);a=copy.deepcopy(agent);a.reset_episode()
    rng=np.random.default_rng(stable_seed(seed,"transfer-prefix-v2"))
    rr=[];alive=[]
    for t in range(320):
        p=a.prepare(obs,force_gate=float(rng.integers(0,2)) if t<ADAPT_PREFIX else None)
        # Controlled excitation avoids destructive extremes while covering actions 0/1/2 and occasional 3.
        if t<ADAPT_PREFIX:
            pattern=(0,1,2,1,0,2,1,3)
            action=pattern[t%len(pattern)]
        else: action=int(p["action"])
        nxt,r,_,info=env.step(action)
        a.complete(obs,action,nxt,r,p,learn_model=adapt,learn_memory=adapt,learn_gate=adapt)
        if t>=ADAPT_PREFIX:
            rr.append(r);alive.append(float(info["alive"]))
        obs=nxt
    return dict(return_mean=float(np.mean(rr)),alive_fraction=float(np.mean(alive)))

def transfer_benchmark_v2(seed):
    base=source_train_v2(seed);out={}
    for name,cls in (("cyclic_buffer_v2",CyclicBufferV2),("repair_queue_v2",RepairQueueV2)):
        s=stable_seed(seed,name)
        oracle=oracle_run(cls,s)
        adaptive=transfer_run(base,cls,s,True)
        frozen=transfer_run(base,cls,s,False)
        out[name]=dict(oracle=oracle,adaptive=adaptive,frozen=frozen,
                       reward_gain=adaptive["return_mean"]-frozen["return_mean"],
                       alive_gain=adaptive["alive_fraction"]-frozen["alive_fraction"])
    return out

# ---------- G5: viable integrated trajectory with frozen learned model after prefix ----------

class AttributionMemoryV4:
    """Recurring observable cues plus self/world perturbations; memory remains useful after model freeze."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0;self.alive=True
        self.reserve=.82;self.health=.86;self.demand=.32;self.last_world=-99
        idx=np.arange(640);phase=int(self.rng.integers(0,13))
        self.cue=((idx+phase)//16)%3
        self.world=((idx+phase)%13==0)|((idx+phase)%29==0)
        self.mag=.028*np.where(((idx+phase)//13)%2==0,1.,-1.)*self.world
        # repeated demand cues; each cue has a different action-reward/resource relation
        self.demand_levels=np.array([.24,.50,.78])
        return self.obs()
    def obs(self):return np.clip([self.reserve,self.health,self.demand],0,1)
    def step(self,action):
        flow=float(ACT[int(action)]); cue=int(self.cue[self.t]); self.demand=float(self.demand_levels[cue])
        world=np.array([.55*self.mag[self.t],.75*self.mag[self.t],0.])
        preferred=(.04,.08,.12)[cue]
        mismatch=abs(flow-preferred)/.12
        selfeff=np.array([-.17*flow+.010*(1-mismatch),.050*(flow/.12)-.016*(flow/.12)**2-.010*mismatch,0.])
        self.reserve=np.clip(self.reserve+.015+selfeff[0]+world[0],0,1)
        self.health=np.clip(self.health+.003+selfeff[1]+world[1]-.004*self.demand,0,1)
        self.alive=bool(self.alive and self.reserve>.06 and self.health>.12)
        reward=float((1-mismatch)*(.62+.38*self.health) if self.alive else 0)
        sm=float(np.linalg.norm(selfeff[:2]));wm=float(np.linalg.norm(world[:2]))
        source=1 if wm>1.35*max(sm,1e-8) else 0 if sm>1.35*max(wm,1e-8) else 2
        if self.world[self.t]:self.last_world=self.t
        age=self.t-self.last_world;agebin=0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3
        self.t+=1
        return self.obs(),reward,self.t==640,dict(alive=self.alive,source=source,agebin=agebin)

def integrated_run_v2(agent,seed,variant="full"):
    env=AttributionMemoryV4();obs=env.reset(seed);a=copy.deepcopy(agent);a.reset_episode()
    rng=np.random.default_rng(stable_seed(seed,"int-prefix-v2"))
    source_true=[];source_internal=[];age_true=[];age_internal=[];unc=[];err=[];cf=[];memerr=[];rr=[];alive=[]
    PREFIX=96
    for t in range(640):
        no_mem=variant=="noMemory";perm=variant=="permuted_content";no_cross=variant=="noCross"
        p=a.prepare(obs,force_gate=float(rng.integers(0,2)) if t<PREFIX else None,
                    no_memory=no_mem,permuted_content=perm,no_cross=no_cross)
        action=int(rng.integers(0,4)) if t<PREFIX else int(p["action"])
        # evaluator-only counterfactual outcomes
        cfr=[]
        for aa in range(4):
            ee=copy.deepcopy(env);cfr.append(float(ee.step(aa)[1]))
        nxt,r,_,info=env.step(action)
        pe=float(np.mean(np.abs(nxt-p["pred"][action,:3])))
        # During prefix: adapt models + memory. During scored phase: freeze model/gate;
        # factual memory and ERR/source/time state continue.
        a.complete(obs,action,nxt,r,p,learn_model=t<PREFIX,learn_memory=not no_mem,learn_gate=t<PREFIX)
        if t>=PREFIX:
            source_true.append(info["source"]);source_internal.append(a.workspace.last_source)
            ai=0 if a.workspace.world_age==0 else 1 if a.workspace.world_age<=2 else 2 if a.workspace.world_age<=7 else 3
            age_true.append(info["agebin"]);age_internal.append(ai)
            unc.append(p["uncertainty"]);err.append(pe)
            cf.append(int(np.argmax(p["pred"][:,3])==np.argmax(cfr)))
            m=a.workspace.recall(obs,action)
            if m is not None:memerr.append(float(np.mean(np.abs(m[:3]-nxt))))
            rr.append(r);alive.append(float(info["alive"]))
        obs=nxt
    return dict(return_mean=float(np.mean(rr)),alive_fraction=float(np.mean(alive)),
                source_true=np.asarray(source_true),source_internal=np.asarray(source_internal),
                age_true=np.asarray(age_true),age_internal=np.asarray(age_internal),
                uncertainty=np.asarray(unc),errors=np.asarray(err),cf=np.asarray(cf),
                memory_mae=float(np.mean(memerr)) if memerr else 1.)

def integrated_benchmark_v2(seed):
    base=source_train_v2(seed);s=stable_seed(seed,"integrated-v4")
    full=integrated_run_v2(base,s,"full");nm=integrated_run_v2(base,s,"noMemory")
    pc=integrated_run_v2(base,s,"permuted_content");nc=integrated_run_v2(base,s,"noCross")
    st,sp=full["source_true"],full["source_internal"]
    sb=_ba(st,sp);m=st!=2
    sw=_ba((st[m]==0).astype(int),(sp[m]==0).astype(int)) if m.any() else 0.
    tb=_ba(full["age_true"],full["age_internal"])
    threshold=float(np.median(full["errors"]))
    auc=_auc(full["uncertainty"],(full["errors"]>threshold).astype(int))
    cf=float(np.mean(full["cf"]))
    return dict(full_return=full["return_mean"],full_alive=full["alive_fraction"],
                self_world=sw,source=sb,time=tb,metacog_auc=auc,counterfactual=cf,
                memory_mae=full["memory_mae"],memory_damage=nm["memory_mae"]-full["memory_mae"],
                content_drop=full["return_mean"]-pc["return_mean"],
                cross_drop=full["return_mean"]-nc["return_mean"],
                noMemory_alive=nm["alive_fraction"],permuted_alive=pc["alive_fraction"],noCross_alive=nc["alive_fraction"])
