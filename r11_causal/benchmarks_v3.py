"""R11 third development pass: oracle-solvable transfer and causal integrated challenge."""
from __future__ import annotations
import copy, numpy as np
from .agent import ACT
from .benchmarks import stable_seed, relational_benchmark, _ba, _auc
from .benchmarks_v2 import source_train_v2, gate_benchmark_v2, RepairQueueV2, oracle_run, transfer_run

class CyclicBufferV3:
    """Independently coded cyclic resource task with verified feasible long-run policy."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.84; self.health=.88; self.deficit=0.; self.pipe=[0.,0.]
        idx=np.arange(320)
        self.repl=np.clip(.060+.012*np.sin(2*np.pi*idx/41.)+self.rng.normal(0,.0025,320),.038,.080)
        self.dem=np.clip(.045+.014*(np.sin(2*np.pi*idx/57.+.3)>0)+self.rng.normal(0,.0025,320),.030,.065)
        return self.obs()
    def obs(self): return np.clip([self.reserve,self.health,self.dem[min(self.t,319)]/.07],0,1)
    def step(self,action):
        flow=float(ACT[int(action)]); demand=float(self.dem[self.t]); arrival=self.pipe.pop(0);self.pipe.append(float(self.repl[self.t]))
        dispatch=min(self.reserve+arrival,flow);served=min(dispatch,demand);unmet=max(0.,demand-served)
        self.reserve=np.clip(self.reserve+arrival-dispatch,0,1)
        self.deficit=.75*self.deficit+unmet
        self.health=np.clip(self.health+.004-.050*self.deficit-.012*max(0.,dispatch-.08),0,1)
        self.alive=bool(self.alive and self.health>.08 and self.reserve>.03)
        service=served/max(demand,1e-9)
        reward=float(service*(.60+.40*self.health)*(1-.06*dispatch/.12) if self.alive else 0)
        self.t+=1
        return self.obs(),reward,self.t==320,dict(alive=self.alive)
    @staticmethod
    def oracle_action(obs):
        demand=float(obs[2])*.07
        feasible=np.where(ACT+1e-9>=demand,ACT-demand,10+np.abs(ACT-demand))
        return int(np.argmin(feasible))

def transfer_benchmark_v3(seed):
    base=source_train_v2(seed);out={}
    for name,cls in (("cyclic_buffer_v3",CyclicBufferV3),("repair_queue_v2",RepairQueueV2)):
        s=stable_seed(seed,name)
        oracle=oracle_run(cls,s); adaptive=transfer_run(base,cls,s,True); frozen=transfer_run(base,cls,s,False)
        out[name]=dict(oracle=oracle,adaptive=adaptive,frozen=frozen,
                       reward_gain=adaptive["return_mean"]-frozen["return_mean"],
                       alive_gain=adaptive["alive_fraction"]-frozen["alive_fraction"])
    return out

class IntegratedEnvV5:
    """Single viable trajectory requiring source attribution, nonlinear prediction and adaptive memory."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.t=0;self.alive=True
        self.reserve=.86;self.health=.89;self.demand=.24;self.last_world=-99
        idx=np.arange(640);phase=int(self.rng.integers(0,17))
        self.cue=((idx+phase)//14)%3
        self.strong=((idx+phase)%17==0)
        self.weak=((idx+phase)%11==0)&(~self.strong)
        self.sign=np.where(((idx+phase)//17)%2==0,1.,-1.)
        self.levels=np.array([.24,.50,.78])
        return self.obs()
    def obs(self):return np.clip([self.reserve,self.health,self.demand],0,1)
    def preferred(self):
        cue=int(self.cue[self.t])
        before=(.04,.08,.12); after=(.12,.04,.08)
        return (before if self.t<320 else after)[cue]
    def step(self,action):
        flow=float(ACT[int(action)]);cue=int(self.cue[self.t]);self.demand=float(self.levels[cue])
        pref=self.preferred();mismatch=abs(flow-pref)/.12
        mag=.040*self.sign[self.t] if self.strong[self.t] else .018*self.sign[self.t] if self.weak[self.t] else 0.
        world=np.array([.50*mag,.72*mag,0.])
        # Nonlinear state/action interaction: cross route has a genuine role.
        interaction=(self.reserve-.5)*(self.demand-.5)
        selfeff=np.array([-.15*flow+.014*(1-mismatch)+.018*interaction,
                          .052*(flow/.12)-.014*(flow/.12)**2-.010*mismatch-.012*interaction,0.])
        self.reserve=np.clip(self.reserve+.016+selfeff[0]+world[0],0,1)
        self.health=np.clip(self.health+.0035+selfeff[1]+world[1]-.0035*self.demand,0,1)
        self.alive=bool(self.alive and self.reserve>.07 and self.health>.13)
        # Reward strongly depends on recurrent cue-action mapping, changed at t=320.
        reward=float((1-mismatch)*(.58+.42*self.health)*(1-.10*abs(interaction)) if self.alive else 0.)
        if self.strong[self.t]:source=1
        elif self.weak[self.t]:source=2
        else:source=0
        if self.strong[self.t] or self.weak[self.t]:self.last_world=self.t
        age=self.t-self.last_world;agebin=0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3
        self.t+=1
        return self.obs(),reward,self.t==640,dict(alive=self.alive,source=source,agebin=agebin)

def integrated_run_v3(agent,seed,variant="full"):
    env=IntegratedEnvV5();obs=env.reset(seed);a=copy.deepcopy(agent);a.reset_episode()
    rng=np.random.default_rng(stable_seed(seed,"integrated-v5-prefix"))
    PREFIX=112
    st=[];sp=[];at=[];ap=[];unc=[];errs=[];cf=[];memerr=[];rr=[];alive=[]
    for t in range(640):
        no_mem=variant=="noMemory";perm=variant=="permuted_content";no_cross=variant=="noCross"
        p=a.prepare(obs,force_gate=float(rng.integers(0,2)) if t<PREFIX else None,
                    no_memory=no_mem,permuted_content=perm,no_cross=no_cross)
        # Balanced diagnostic exploration during prefix, then fully native action.
        action=int(rng.integers(0,4)) if t<PREFIX else int(p["action"])
        cfr=[]
        for aa in range(4):
            ee=copy.deepcopy(env);cfr.append(float(ee.step(aa)[1]))
        nxt,r,_,info=env.step(action)
        pe=float(np.mean(np.abs(nxt-p["pred"][action,:3])))
        a.complete(obs,action,nxt,r,p,learn_model=t<PREFIX,learn_memory=not no_mem,learn_gate=t<PREFIX)
        if t>=PREFIX:
            st.append(info["source"]);sp.append(a.workspace.last_source)
            ai=0 if a.workspace.world_age==0 else 1 if a.workspace.world_age<=2 else 2 if a.workspace.world_age<=7 else 3
            at.append(info["agebin"]);ap.append(ai);unc.append(p["uncertainty"]);errs.append(pe)
            cf.append(int(np.argmax(p["pred"][:,3])==np.argmax(cfr)))
            m=a.workspace.recall(obs,action)
            if m is not None:memerr.append(float(np.mean(np.abs(m[:3]-nxt))))
            rr.append(r);alive.append(float(info["alive"]))
        obs=nxt
    return dict(return_mean=float(np.mean(rr)),alive_fraction=float(np.mean(alive)),
                source_true=np.asarray(st),source_internal=np.asarray(sp),age_true=np.asarray(at),age_internal=np.asarray(ap),
                uncertainty=np.asarray(unc),errors=np.asarray(errs),cf=np.asarray(cf),
                memory_mae=float(np.mean(memerr)) if memerr else 1.)

def integrated_benchmark_v3(seed):
    base=source_train_v2(seed);s=stable_seed(seed,"integrated-v5")
    full=integrated_run_v3(base,s,"full");nm=integrated_run_v3(base,s,"noMemory")
    pc=integrated_run_v3(base,s,"permuted_content");nc=integrated_run_v3(base,s,"noCross")
    st,sp=full["source_true"],full["source_internal"];source_ba=_ba(st,sp)
    m=st!=2;sw=_ba((st[m]==0).astype(int),(sp[m]==0).astype(int)) if m.any() else 0.
    tb=_ba(full["age_true"],full["age_internal"]);thr=float(np.median(full["errors"]))
    auc=_auc(full["uncertainty"],(full["errors"]>thr).astype(int));cf=float(np.mean(full["cf"]))
    return dict(full_return=full["return_mean"],full_alive=full["alive_fraction"],
        self_world=sw,source=source_ba,time=tb,metacog_auc=auc,counterfactual=cf,
        memory_mae=full["memory_mae"],memory_damage=nm["memory_mae"]-full["memory_mae"],
        noMemory_return=nm["return_mean"],memory_return_drop=full["return_mean"]-nm["return_mean"],
        content_drop=full["return_mean"]-pc["return_mean"],cross_drop=full["return_mean"]-nc["return_mean"],
        noMemory_alive=nm["alive_fraction"],permuted_alive=pc["alive_fraction"],noCross_alive=nc["alive_fraction"])
