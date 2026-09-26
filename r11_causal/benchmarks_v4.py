"""R11 development pass v4: feasible transfer + targeted integrated lesions."""
from __future__ import annotations
import copy, numpy as np
from .agent import ACT
from .benchmarks import stable_seed, _ba, _auc
from .benchmarks_v2 import source_train_v2, RepairQueueV2, oracle_run, transfer_run

ADAPT_PREFIX=64

class CyclicBufferV4:
    """Cyclic resource-control family with an independently verified viable oracle."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.88; self.health=.90; self.deficit=0.
        idx=np.arange(320)
        self.inflow=np.clip(.078+.010*np.sin(2*np.pi*idx/43.)+self.rng.normal(0,.002,320),.055,.095)
        self.dem=np.clip(.047+.014*np.sin(2*np.pi*idx/59.+.7)+self.rng.normal(0,.0025,320),.028,.068)
        return self.obs()
    def obs(self):
        return np.clip([self.reserve,self.health,self.dem[min(self.t,319)]/.07],0,1)
    def step(self,action):
        flow=float(ACT[int(action)]); demand=float(self.dem[self.t])
        served=min(flow,demand,self.reserve)
        unmet=max(0.,demand-served)
        self.reserve=np.clip(self.reserve+float(self.inflow[self.t])-flow,0,1)
        self.deficit=.72*self.deficit+unmet
        overload=max(0.,flow-.08)
        self.health=np.clip(self.health+.006-.040*self.deficit-.018*overload/.04,0,1)
        self.alive=bool(self.alive and self.reserve>.04 and self.health>.10)
        service=served/max(demand,1e-9)
        reward=float(service*(.62+.38*self.health)*(1-.07*flow/.12) if self.alive else 0.)
        self.t+=1
        return self.obs(),reward,self.t==320,dict(alive=self.alive)
    @staticmethod
    def oracle_action(obs):
        demand=float(obs[2])*.07
        # smallest request that covers demand
        feasible=np.where(ACT+1e-9>=demand,ACT-demand,10+np.abs(ACT-demand))
        return int(np.argmin(feasible))

def transfer_benchmark_v4(seed):
    base=source_train_v2(seed); out={}
    for name,cls in (("cyclic_buffer_v4",CyclicBufferV4),("repair_queue_v2",RepairQueueV2)):
        s=stable_seed(seed,name)
        oracle=oracle_run(cls,s)
        adaptive=transfer_run(base,cls,s,True)
        frozen=transfer_run(base,cls,s,False)
        out[name]=dict(oracle=oracle,adaptive=adaptive,frozen=frozen,
                       reward_gain=adaptive["return_mean"]-frozen["return_mean"],
                       alive_gain=adaptive["alive_fraction"]-frozen["alive_fraction"])
    return out

class IntegratedEnvV6:
    """Viable single-agent challenge with evaluator-only physical source decomposition."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.88; self.health=.90; self.demand=.24
        idx=np.arange(640); phase=int(self.rng.integers(0,19))
        self.cue=((idx+phase)//12)%3
        self.world=((idx+phase)%13==0)|((idx+phase)%31==0)
        self.world_mag=.026*np.where(((idx+phase)//13)%2==0,1.,-1.)*self.world
        self.levels=np.array([.22,.50,.80])
        return self.obs()
    def obs(self):
        return np.clip([self.reserve,self.health,self.demand],0,1)
    def preferred(self):
        cue=int(self.cue[self.t])
        # mapping reverses once, requiring retained factual memory to relearn
        return ((.04,.08,.12) if self.t<320 else (.12,.04,.08))[cue]
    def step(self,action):
        flow=float(ACT[int(action)]); cue=int(self.cue[self.t]); self.demand=float(self.levels[cue])
        pref=self.preferred(); mismatch=abs(flow-pref)/.12
        world=np.array([.45*self.world_mag[self.t],.72*self.world_mag[self.t],0.])
        interaction=(self.reserve-.5)*(self.demand-.5)
        own=np.array([-.14*flow+.012*(1-mismatch)+.050*interaction,
                      .050*(flow/.12)-.013*(flow/.12)**2-.009*mismatch-.042*interaction,0.])
        before=self.obs().copy()
        self.reserve=np.clip(self.reserve+.018+own[0]+world[0],0,1)
        self.health=np.clip(self.health+.004+own[1]+world[1]-.003*self.demand,0,1)
        self.alive=bool(self.alive and self.reserve>.07 and self.health>.13)
        reward=float((1-mismatch)*(.60+.40*self.health)*(1-.10*abs(interaction)) if self.alive else 0.)
        # Evaluator-only decomposition, never returned to native policy.
        sm=float(np.linalg.norm(own[:2])); wm=float(np.linalg.norm(world[:2]))
        if wm>max(.010,1.20*sm): source=1
        elif sm>max(.008,1.20*wm): source=0
        else: source=2
        self.t+=1
        return self.obs(),reward,self.t==640,dict(alive=self.alive,source=source,
            own=own.tolist(),world=world.tolist(),before=before.tolist(),world_event=bool(wm>.010))

def _timebin(age):
    return 0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3

def integrated_run_v4(agent,seed,variant="full",diagnostic=False):
    env=IntegratedEnvV6(); obs=env.reset(seed); a=copy.deepcopy(agent); a.reset_episode()
    rng=np.random.default_rng(stable_seed(seed,"integrated-v6-prefix"))
    PREFIX=128
    source_true=[];source_internal=[];age_true=[];age_internal=[];unc=[];errors=[];cf_full=[]
    pred_mse_full=[];pred_mse_nomem=[];pred_mse_perm=[];pred_mse_nocross=[]
    cf_nomem=[];cf_perm=[];cf_nocross=[];rr=[];alive=[];postshift_rr=[]
    true_world_age=99
    for t in range(640):
        no_mem=variant=="noMemory";perm=variant=="permuted_content";no_cross=variant=="noCross"
        fg=float(rng.integers(0,2)) if t<PREFIX else None
        p=a.prepare(obs,force_gate=fg,no_memory=no_mem,permuted_content=perm,no_cross=no_cross)

        # Evaluator-only exact one-step counterfactuals for all actions.
        targets=[]; cfr=[]
        for aa in range(4):
            ee=copy.deepcopy(env)
            no,cr,_,ci=ee.step(aa)
            targets.append(np.r_[no,cr]); cfr.append(float(cr))
        targets=np.asarray(targets); cfr=np.asarray(cfr)

        # Same-state selective diagnostics use copied native state; no evaluator target is fed back.
        if t>=PREFIX and diagnostic:
            shadows=[]
            for kw in (dict(no_memory=True),dict(permuted_content=True),dict(no_cross=True)):
                sh=copy.deepcopy(a); shadows.append(sh.prepare(obs,**kw))
            pred_mse_full.append(float(np.mean((p["pred"]-targets)**2)))
            pred_mse_nomem.append(float(np.mean((shadows[0]["pred"]-targets)**2)))
            pred_mse_perm.append(float(np.mean((shadows[1]["pred"]-targets)**2)))
            pred_mse_nocross.append(float(np.mean((shadows[2]["pred"]-targets)**2)))
            best=int(np.argmax(cfr))
            cf_full.append(int(np.argmax(p["pred"][:,3])==best))
            cf_nomem.append(int(np.argmax(shadows[0]["pred"][:,3])==best))
            cf_perm.append(int(np.argmax(shadows[1]["pred"][:,3])==best))
            cf_nocross.append(int(np.argmax(shadows[2]["pred"][:,3])==best))

        action=int(rng.integers(0,4)) if t<PREFIX else int(p["action"])
        nxt,r,_,info=env.step(action)
        pe=float(np.mean(np.abs(nxt-p["pred"][action,:3])))
        a.complete(obs,action,nxt,r,p,learn_model=t<PREFIX,learn_memory=not no_mem,learn_gate=t<PREFIX)

        if t>=PREFIX:
            if info["world_event"]: true_world_age=0
            else: true_world_age+=1
            source_true.append(info["source"]);source_internal.append(a.workspace.last_source)
            age_true.append(_timebin(true_world_age));age_internal.append(_timebin(a.workspace.world_age))
            unc.append(p["uncertainty"]);errors.append(pe);rr.append(r);alive.append(float(info["alive"]))
            if t>=320: postshift_rr.append(r)
        obs=nxt
    return dict(return_mean=float(np.mean(rr)),postshift_return=float(np.mean(postshift_rr)),
                alive_fraction=float(np.mean(alive)),source_true=np.asarray(source_true),
                source_internal=np.asarray(source_internal),age_true=np.asarray(age_true),
                age_internal=np.asarray(age_internal),uncertainty=np.asarray(unc),
                errors=np.asarray(errors),
                cf=float(np.mean(cf_full)) if cf_full else 0.,
                cf_nomem=float(np.mean(cf_nomem)) if cf_nomem else 0.,
                cf_perm=float(np.mean(cf_perm)) if cf_perm else 0.,
                cf_nocross=float(np.mean(cf_nocross)) if cf_nocross else 0.,
                pred_mse=float(np.mean(pred_mse_full)) if pred_mse_full else 0.,
                pred_mse_nomem=float(np.mean(pred_mse_nomem)) if pred_mse_nomem else 0.,
                pred_mse_perm=float(np.mean(pred_mse_perm)) if pred_mse_perm else 0.,
                pred_mse_nocross=float(np.mean(pred_mse_nocross)) if pred_mse_nocross else 0.)

def integrated_benchmark_v4(seed):
    base=source_train_v2(seed); s=stable_seed(seed,"integrated-v6")
    full=integrated_run_v4(base,s,"full",diagnostic=True)
    nm=integrated_run_v4(base,s,"noMemory")
    pc=integrated_run_v4(base,s,"permuted_content")
    nc=integrated_run_v4(base,s,"noCross")
    st,sp=full["source_true"],full["source_internal"]
    source_ba=_ba(st,sp); m=st!=2
    sw=_ba((st[m]==0).astype(int),(sp[m]==0).astype(int)) if m.any() else 0.
    time_ba=_ba(full["age_true"],full["age_internal"])
    thr=float(np.median(full["errors"]))
    auc=_auc(full["uncertainty"],(full["errors"]>thr).astype(int))
    return dict(full_return=full["return_mean"],postshift_return=full["postshift_return"],
        full_alive=full["alive_fraction"],self_world=sw,source=source_ba,time=time_ba,
        metacog_auc=auc,counterfactual=full["cf"],
        memory_cf_damage=full["cf"]-full["cf_nomem"],
        content_cf_damage=full["cf"]-full["cf_perm"],
        cross_cf_damage=full["cf"]-full["cf_nocross"],
        memory_pred_damage=full["pred_mse_nomem"]-full["pred_mse"],
        content_pred_damage=full["pred_mse_perm"]-full["pred_mse"],
        cross_pred_damage=full["pred_mse_nocross"]-full["pred_mse"],
        memory_return_drop=full["postshift_return"]-nm["postshift_return"],
        content_return_drop=full["postshift_return"]-pc["postshift_return"],
        cross_return_drop=full["return_mean"]-nc["return_mean"],
        noMemory_alive=nm["alive_fraction"],permuted_alive=pc["alive_fraction"],noCross_alive=nc["alive_fraction"])
