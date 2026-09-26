"""Development benchmarks for R11 four-gap correction."""
from __future__ import annotations
import copy, hashlib, math
import numpy as np
from r9_completion.environments import ScalarEnvironment
from r10_discriminating.experiments import CyclicBufferEnv, RepairQueueEnv
from .agent import CompactCausalAgent, ACT
from .config_dev import *

def stable_seed(seed,label):
    return int.from_bytes(hashlib.sha256(f"R11|{seed}|{label}".encode()).digest()[:8],"little") & 0xffffffff

def source_train(seed, relation_mode="polar"):
    a=CompactCausalAgent(seed,relation_mode)
    rng=np.random.default_rng(stable_seed(seed,"source-policy"))
    for ep in range(SOURCE_EPISODES):
        env=ScalarEnvironment("ecology_train")
        obs=env.reset(stable_seed(seed,f"source-{ep}"))
        a.reset_episode()
        for t in range(SOURCE_STEPS):
            # Randomized gate exposure gives factual local-vs-cross gain across contexts.
            gate=float(rng.integers(0,2))
            p=a.prepare(obs,force_gate=gate)
            action=int(rng.integers(0,4)) if rng.random()<.22 else int(p["action"])
            nxt,r,done,info=env.step(action)
            a.complete(obs,action,nxt,r,p,learn=True,learn_gate=True)
            obs=nxt
    return a

def run_env(agent, env_cls, seed, *, adapt=True, prefix=0, variant="full", steps=320):
    env=env_cls(); obs=env.reset(seed); a=copy.deepcopy(agent); a.reset_episode()
    rng=np.random.default_rng(stable_seed(seed,"adapt-actions"))
    rewards=[]; alive=[]; prefix_rewards=[]; source_internal=[]; source_true=[]; ages=[]; age_true=[]
    unc=[]; errs=[]; cf_ok=[]; mem_mae=[]
    for t in range(steps):
        force_gate=None
        no_mem=variant=="noMemory"; perm=variant=="permuted_content"; no_cross=variant=="noCross"
        if t<prefix:
            force_gate=float(rng.integers(0,2))
        p=a.prepare(obs,force_gate=force_gate,permuted_content=perm,no_memory=no_mem,no_cross=no_cross)
        if t<prefix:
            action=int(rng.integers(0,4))
        else:
            action=int(p["action"])
        # evaluator-only counterfactual best action
        cfr=[]
        for aa in range(4):
            ee=copy.deepcopy(env)
            try: cfr.append(float(ee.step(aa)[1]))
            except Exception: cfr.append(-1e9)
        nxt,r,done,info=env.step(action)
        pred_err=float(np.mean(np.abs(nxt-p["pred"][action,:3])))
        a.complete(obs,action,nxt,r,p,learn=adapt,learn_gate=adapt)
        if t>=prefix:
            rewards.append(r); alive.append(float(info.get("alive",True)))
            unc.append(p["uncertainty"]); errs.append(pred_err)
            cf_ok.append(int(int(np.argmax(p["pred"][:,3]))==int(np.argmax(cfr))))
            if "source" in info:
                source_internal.append(a.workspace.last_source); source_true.append(info["source"])
            if "agebin" in info:
                ai=0 if a.workspace.world_age==0 else 1 if a.workspace.world_age<=2 else 2 if a.workspace.world_age<=7 else 3
                ages.append(ai); age_true.append(info["agebin"])
            recalled=a.workspace.recall(obs,action)
            if recalled is not None:
                mem_mae.append(float(np.mean(np.abs(recalled[:3]-nxt))))
        else: prefix_rewards.append(r)
        obs=nxt
    return dict(return_mean=float(np.mean(rewards)) if rewards else 0.,
                alive_fraction=float(np.mean(alive)) if alive else 0.,
                prefix_return=float(np.mean(prefix_rewards)) if prefix_rewards else 0.,
                uncertainty=np.asarray(unc),errors=np.asarray(errs),cf=np.asarray(cf_ok),
                source_internal=np.asarray(source_internal,int),source_true=np.asarray(source_true,int),
                age_internal=np.asarray(ages,int),age_true=np.asarray(age_true,int),
                memory_mae=float(np.mean(mem_mae)) if mem_mae else 1.0)

# ---------- Gap 2: learned relational structure ----------

def _ridge(X,y,lam=1e-3):
    X=np.asarray(X,float); y=np.asarray(y,float)
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y)

def _pair_features(X,pairs):
    return np.column_stack([np.ones(len(X)),X,*[X[:,i]*X[:,j] for i,j in pairs]])

def _random_matching(rng,n=16):
    p=list(rng.permutation(n)); return [tuple(sorted((p[i],p[i+1]))) for i in range(0,n,2)]

def _discover_matching(X,y):
    base=np.column_stack([np.ones(len(X)),X])
    w=_ridge(base,y,.02); resid=y-base@w
    scores=[]
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            z=X[:,i]*X[:,j]; z=z-z.mean()
            s=abs(float(z@resid))/(math.sqrt(float(z@z)*float(resid@resid))+1e-12)
            scores.append((s,i,j))
    used=set(); pairs=[]
    for _,i,j in sorted(scores,reverse=True):
        if i not in used and j not in used:
            pairs.append((i,j)); used|={i,j}
            if len(pairs)==X.shape[1]//2: break
    return sorted(pairs)

def relational_benchmark(seed,ntrain=3500,ntest=2500):
    rng=np.random.default_rng(stable_seed(seed,"relational"))
    true_pairs=_random_matching(rng,16)
    beta=rng.uniform(.65,1.05,8)*rng.choice([-1.,1.],8)
    lin=rng.normal(0,.12,16)
    Xtr=rng.normal(0,1,(ntrain,16))
    Xte=rng.normal(.15,.95,(ntest,16))  # mild distribution shift
    def target(X,noise=True):
        y=X@lin
        for b,(i,j) in zip(beta,true_pairs): y+=b*np.tanh(X[:,i]*X[:,j])
        if noise: y+=rng.normal(0,.12,len(X))
        return y
    ytr=target(Xtr,True); yte=target(Xte,True)
    discovered=_discover_matching(Xtr,ytr)
    generic=_random_matching(np.random.default_rng(stable_seed(seed,"generic-match")),16)
    # Same exact feature/parameter count and ridge training.
    Wp=_ridge(_pair_features(Xtr,discovered),ytr,.05)
    Wg=_ridge(_pair_features(Xtr,generic),ytr,.05)
    pp=_pair_features(Xte,discovered)@Wp; pg=_pair_features(Xte,generic)@Wg
    mse_p=float(np.mean((yte-pp)**2)); mse_g=float(np.mean((yte-pg)**2))
    recovered=len(set(discovered)&set(true_pairs))
    return dict(true_pairs=true_pairs,discovered=discovered,generic=generic,
                recovered=recovered,mse_relational=mse_p,mse_generic=mse_g,
                mse_gain=mse_g-mse_p)

# ---------- Gap 3: utility-trained adaptive gate ----------

def _gate_basis(X):
    X=np.asarray(X,float)
    return np.column_stack([np.ones(len(X)),X,X[:,0]*X[:,1],X[:,2]*X[:,3],
                            X[:,4]*X[:,5],X[:,0]**2,X[:,2]**2])

def gate_benchmark(seed,ntrain=6000,ntest=3000):
    rng=np.random.default_rng(stable_seed(seed,"gate-bandit"))
    X=rng.normal(size=(ntrain+ntest,6))
    regime=(.9*X[:,0]*X[:,1]-.7*X[:,2]*X[:,3]+.55*X[:,4]*X[:,5]+.25*X[:,0]**2>0)
    truth=regime.astype(int)
    local=.35*X[:,0]-.2*X[:,3]+.1*X[:,5]
    corr=.8*np.tanh(X[:,1]-X[:,2]+.4*X[:,4])
    y=local+(2*truth-1)*corr+rng.normal(0,.06,len(X))
    chosen=rng.integers(0,2,ntrain)
    pred_chosen=np.where(chosen.astype(bool),local[:ntrain]+corr[:ntrain],local[:ntrain])
    loss=(y[:ntrain]-pred_chosen)**2
    B=_gate_basis(X[:ntrain])
    models=[]
    for g in (0,1):
        m=chosen==g; models.append(_ridge(B[m],loss[m],.5))
    Bt=_gate_basis(X[ntrain:])
    l0=Bt@models[0]; l1=Bt@models[1]
    gate=(l1<l0).astype(int)
    lt=local[ntrain:]; ct=lt+corr[ntrain:]; yt=y[ntrain:]
    pred=np.where(gate.astype(bool),ct,lt)
    mse=lambda z:float(np.mean((yt-z)**2))
    acc=float(np.mean(gate==truth[ntrain:]))
    best=min(mse(lt),mse(ct)); gain=best-mse(pred)
    Xp=X[ntrain:].copy(); Xp[:,[0,2,4]]=Xp[:,[2,4,0]]
    Bp=_gate_basis(Xp); pg=((Bp@models[1])<(Bp@models[0])).astype(int)
    pp=np.where(pg.astype(bool),ct,lt); damage=mse(pp)-mse(pred)
    return dict(accuracy=acc,adaptive_mse=mse(pred),off_mse=mse(lt),on_mse=mse(ct),
                gain=gain,permutation_damage=damage)

# ---------- Gap 4: adaptive transfer ----------

def transfer_benchmark(seed):
    base=source_train(seed)
    out={}
    for name,cls in (("cyclic_buffer",CyclicBufferEnv),("repair_queue",RepairQueueEnv)):
        s=stable_seed(seed,name)
        adaptive=run_env(base,cls,s,adapt=True,prefix=TARGET_ADAPT_STEPS,steps=320)
        frozen=run_env(base,cls,s,adapt=False,prefix=TARGET_ADAPT_STEPS,steps=320)
        out[name]=dict(adaptive={k:v for k,v in adaptive.items() if not isinstance(v,np.ndarray)},
                       frozen={k:v for k,v in frozen.items() if not isinstance(v,np.ndarray)},
                       reward_gain=adaptive["return_mean"]-frozen["return_mean"],
                       alive_gain=adaptive["alive_fraction"]-frozen["alive_fraction"])
    return out

# ---------- Gap 5: single-agent integrated precursors ----------

class AttributionEnvV3:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.80; self.health=.84; self.demand=.45; self.last_world=-99
        idx=np.arange(640)
        phase=int(self.rng.integers(0,11))
        self.shock=((idx+phase)%11==0)|((idx+phase)%23==0)
        self.mag=.03*np.where(((idx+phase)//11)%2==0,1.,-1.)*self.shock
        self.dem=np.clip(.46+.23*np.sin(2*np.pi*idx/52.+phase*.11)+.05*np.sin(2*np.pi*idx/17),.1,.9)
        return self.obs()
    def obs(self):return np.clip([self.reserve,self.health,self.demand],0,1)
    def step(self,action):
        flow=ACT[int(action)]; oldd=self.demand; self.demand=float(self.dem[self.t])
        world=np.array([.5*self.mag[self.t],.8*self.mag[self.t],self.demand-oldd])
        selfeff=np.array([-.20*flow,.06*(flow/.12)-.015*(flow/.12)**2,0.])
        self.reserve=np.clip(self.reserve+.017+selfeff[0]+world[0],0,1)
        self.health=np.clip(self.health+.003+selfeff[1]+world[1]-.005*self.demand,0,1)
        self.alive=bool(self.alive and self.reserve>.06 and self.health>.12)
        required=.015+.085*self.demand
        service=min(1.,flow/max(required,1e-6))
        reward=float(service*(.65+.35*self.health)*(1-.08*flow/.12) if self.alive else 0)
        sm=float(np.linalg.norm(selfeff[:2])); wm=float(np.linalg.norm(world[:2]))
        source=1 if wm>1.35*max(sm,1e-8) else 0 if sm>1.35*max(wm,1e-8) else 2
        if self.shock[self.t]:self.last_world=self.t
        age=self.t-self.last_world; agebin=0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3
        self.t+=1
        return self.obs(),reward,self.t==640,dict(alive=self.alive,source=source,agebin=agebin)

def _ba(y,p):
    y=np.asarray(y);p=np.asarray(p); vals=[]
    for c in np.unique(y):
        m=y==c
        if m.any():vals.append(np.mean(p[m]==c))
    return float(np.mean(vals)) if vals else 0.

def _auc(score,label):
    score=np.asarray(score);label=np.asarray(label)
    pos=score[label==1];neg=score[label==0]
    if not len(pos) or not len(neg):return .5
    return float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))

def integrated_benchmark(seed):
    base=source_train(seed)
    s=stable_seed(seed,"integrated-v3")
    full=run_env(base,AttributionEnvV3,s,adapt=True,prefix=64,steps=640)
    nm=run_env(base,AttributionEnvV3,s,adapt=True,prefix=64,steps=640,variant="noMemory")
    pc=run_env(base,AttributionEnvV3,s,adapt=True,prefix=64,steps=640,variant="permuted_content")
    nc=run_env(base,AttributionEnvV3,s,adapt=True,prefix=64,steps=640,variant="noCross")
    st,sp=full["source_true"],full["source_internal"]
    source_ba=_ba(st,sp) if len(st) else 0.
    m=st!=2
    sw_ba=_ba((st[m]==0).astype(int),(sp[m]==0).astype(int)) if m.any() else 0.
    time_ba=_ba(full["age_true"],full["age_internal"]) if len(full["age_true"]) else 0.
    threshold=float(np.median(full["errors"])) if len(full["errors"]) else 0.
    met=_auc(full["uncertainty"],(full["errors"]>threshold).astype(int)) if len(full["errors"]) else .5
    cf=float(np.mean(full["cf"])) if len(full["cf"]) else 0.
    return dict(full_return=full["return_mean"],full_alive=full["alive_fraction"],
        self_world=sw_ba,source=source_ba,time=time_ba,metacog_auc=met,counterfactual=cf,
        memory_mae=full["memory_mae"],memory_damage=nm["memory_mae"]-full["memory_mae"],
        content_drop=full["return_mean"]-pc["return_mean"],
        cross_drop=full["return_mean"]-nc["return_mean"],
        noMemory_alive=nm["alive_fraction"],permuted_alive=pc["alive_fraction"],noCross_alive=nc["alive_fraction"])
