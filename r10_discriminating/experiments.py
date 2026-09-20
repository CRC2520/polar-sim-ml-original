"""POLAR R10: five discriminating experiments on the merged R9 realization.

This module imports R9 without modifying it.  New environments are deliberately
small and transparent.  Evaluator-only labels/counterfactuals never enter the
native policy.  See PREREG_R10.md for frozen criteria.
"""
from __future__ import annotations
import copy, hashlib, itertools, json, math
from pathlib import Path
import numpy as np

from r9_completion.agent import NativeAgent
from r9_completion.runner import rollout, concatenate, select_modes, tape_seed
from r9_completion.config import PROTOCOL
from .config import *

ACTIONS = np.array([0.0, .04, .08, .12], float)


def _finite(x):
    return bool(np.isfinite(np.asarray(x, float)).all())


class GenericIsomorphicAgent(NativeAgent):
    """Equal-information R9 agent with a fixed permutation of the 48 predictor coordinates."""
    def __init__(self, seed):
        super().__init__(seed, 'full')
        rng = np.random.default_rng(int(seed) ^ 0x51A9)
        self.iso_permutation = rng.permutation(48)

    def _pack(self, poles, tensions):
        base = np.r_[np.asarray(poles).ravel(), np.asarray(tensions).ravel()]
        return base[self.iso_permutation]


def _train_agent(seed, agent_cls=NativeAgent, kind='full'):
    agent = agent_cls(seed) if agent_cls is GenericIsomorphicAgent else agent_cls(seed, kind)
    traces = []
    for episode in range(PROTOCOL['training_episodes']):
        result = rollout(agent, 'ecology_train', 100+episode, learn=True,
                         epsilon=PROTOCOL['epsilon'], gate_override='constant_half',
                         collect_events=False)
        traces.append(result[0])
        if (episode+1) % PROTOCOL['fit_every_episodes'] == 0:
            agent.fit_model(concatenate(traces), episode+1)
    for episode in range(PROTOCOL['calibration_episodes']):
        result = rollout(agent, 'ecology_train', 200+episode,
                         epsilon=PROTOCOL['epsilon'], gate_override='constant_half',
                         collect_events=False)
        if episode == 0:
            agent.calibrate_uncertainty(result[0])
        else:
            agent.fit_gate(result[0])
    base = copy.deepcopy(agent)
    records = {}
    for mode in PROTOCOL['selection_candidates']:
        cand = copy.deepcopy(base)
        records[mode] = rollout(cand, 'ecology_train', 300, gate_override=mode,
                                collect_events=False)[1]
    agent.gate_mode, agent.constant_mode = select_modes(records)
    agent.selection_record = dict(candidates=records, selected=agent.gate_mode,
                                  constant_selected=agent.constant_mode)
    return agent


def _lesion_dynamic_state(agent, subset, obs):
    subset = set(subset)
    obs = np.asarray(obs, float)
    if 'HIST' not in subset:
        agent.history = np.tile(np.r_[obs, 0.], (16, 1))
        agent.previous_action = None
    if 'PRED' not in subset:
        agent.previous_prediction = None
        agent.previous_pole_prediction = None
        agent.previous_tension = np.zeros((8, 4))
    if 'ERR' not in subset:
        agent.recent_errors = np.zeros((16, 3))
    if 'MEM' not in subset:
        w = agent.workspace
        w.memory = {}
        w.recalled = {}
        w.pending = []
        w._history = [np.r_[obs, -1.].tolist()]
        w.previous_observation = None
        w.previous_cue = None
        w.previous_time = None
        w.previous_absolute_time = None
    if 'GOAL' not in subset:
        w = agent.workspace
        w.goal_primary = 'balanced'
        w.goal_weights = w.initial_goal_weights.copy()
        w.goal_pressures = np.zeros(3)
        w.goal_age = 0
        w.goal_revision = 0
        w.goal_candidate = None
        w.goal_candidate_age = 0


def _reference_trace(agent, domain, episode):
    return rollout(copy.deepcopy(agent), domain, episode, collect_events=False)


def _csd_one(agent, subset, domain='ecology_delay9', episode=851):
    from r9_completion.environments import ScalarEnvironment
    env = ScalarEnvironment(domain)
    tape_domain = 'ecology_train' if domain == 'ecology_delay9' else domain
    obs = env.reset(tape_seed(agent.seed, episode, tape_domain))
    full = copy.deepcopy(agent); full.reset_episode(obs)
    cand = copy.deepcopy(agent); cand.reset_episode(obs)
    shadow_agree=[]; gate_agree=[]; mem_err=[]; unc_err=[]
    full_rewards=[]; full_alive=[]; cand_rewards=[]; cand_alive=[]
    for step in range(PROTOCOL['steps']):
        pf = full.prepare(obs)
        shadow = copy.deepcopy(full)
        _lesion_dynamic_state(shadow, subset, obs)
        ps = shadow.prepare(obs)
        shadow_agree.append(int(ps['action']==pf['action']))
        gate_agree.append(int(float(ps['gate'])==float(pf['gate'])))
        mf = np.r_[pf['memory_energy'], pf['memory_resource']]
        ms = np.r_[ps['memory_energy'], ps['memory_resource']]
        mem_err.append(float(np.mean(np.abs(mf-ms))))
        unc_err.append(abs(float(pf['gate_context'][-1])-float(ps['gate_context'][-1])))

        # Reference executes on a copied environment state so the candidate owns the factual path.
        env_ref = copy.deepcopy(env)
        nf, rf, _, infof = env_ref.step(int(pf['action']))
        full.complete_transition(obs, int(pf['action']), nf, rf, pf, learn=False)
        full_rewards.append(rf); full_alive.append(float(infof['alive']))

        pc = cand.prepare(obs)
        action = int(pc['action'])
        nxt, reward, _, info = env.step(action)
        cand.complete_transition(obs, action, nxt, reward, pc, learn=False)
        _lesion_dynamic_state(cand, subset, nxt)
        cand_rewards.append(reward); cand_alive.append(float(info['alive']))
        obs = nxt
    return dict(subset=sorted(subset), action_agreement=float(np.mean(shadow_agree)),
                gate_agreement=float(np.mean(gate_agree)), memory_mae=float(np.mean(mem_err)),
                uncertainty_mae=float(np.mean(unc_err)),
                full_return=float(np.mean(full_rewards)), candidate_return=float(np.mean(cand_rewards)),
                full_alive=float(np.mean(full_alive)), candidate_alive=float(np.mean(cand_alive)),
                return_loss=float(np.mean(full_rewards)-np.mean(cand_rewards)),
                alive_loss=float(np.mean(full_alive)-np.mean(cand_alive)))


def _csd_pass(m):
    t=CSD_THRESHOLDS
    return (m['action_agreement']>=t['action_agreement'] and
            m['memory_mae']<=t['memory_mae'] and
            m['uncertainty_mae']<=t['uncertainty_mae'] and
            m['gate_agreement']>=t['gate_agreement'] and
            m['return_loss']<=t['return_loss'] and
            m['alive_loss']<=t['alive_loss'])


def select_csd_subset(pilot_agents):
    blocks=list(CSD_BLOCKS)
    ordered=[]
    for size in range(len(blocks)+1):
        ordered.extend(itertools.combinations(blocks,size))
    results={}
    for subset in ordered:
        key='+'.join(subset) if subset else 'EMPTY'
        vals=[_csd_one(a, subset, episode=851) for a in pilot_agents]
        results[key]=vals
        if all(_csd_pass(v) for v in vals):
            return tuple(subset), results
    return tuple(blocks), results


class CyclicBufferEnv:
    def reset(self, seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.72; self.health=.74; self.deficit=0.; self.pipe=[0.,0.,0.]
        self.repl=np.clip(.025+.018*np.sin(np.linspace(0,8*np.pi,STEPS))+
                          self.rng.normal(0,.004,STEPS),0,.06)
        self.dem=np.clip(.045+.028*(np.sin(np.linspace(0,5*np.pi,STEPS)+.6)>0)+
                         self.rng.normal(0,.004,STEPS),.015,.10)
        return self.obs()
    def obs(self):
        return np.clip([self.reserve,self.health,min(1.,self.dem[min(self.t,STEPS-1)]/.10)],0,1)
    def step(self, action):
        if self.t>=STEPS: raise RuntimeError('done')
        request=ACTIONS[int(action)]; demand=self.dem[self.t]
        arrival=self.pipe.pop(0); self.pipe.append(self.repl[self.t])
        throughput=min(self.reserve+arrival, request)
        served=min(throughput,demand); unmet=max(0.,demand-served)
        self.reserve=np.clip(self.reserve+arrival-throughput-.002*abs(math.sin(self.t*.17)),0,1)
        damage=.012+2.6*throughput**2+.22*unmet
        recovery=.006 if throughput<.04 else .001
        self.health=np.clip(self.health+recovery-damage,0,1)
        self.alive=bool(self.alive and self.health>.03 and self.reserve>=0)
        reward=float((served/max(demand,1e-6))*(1-.25*throughput/.12)*(self.health if self.alive else 0))
        self.t+=1
        return self.obs(), reward, self.t==STEPS, {'alive':self.alive,'source':0}

class RepairQueueEnv:
    def reset(self, seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.68; self.health=.78; self.queue=.08
        self.arrivals=np.clip(self.rng.gamma(1.8,.012,STEPS),0,.08)
        self.supply=np.clip(self.rng.normal(.018,.006,STEPS),0,.04)
        return self.obs()
    def obs(self):
        return np.clip([self.reserve,self.health,min(1.,self.queue/.45)],0,1)
    def step(self, action):
        if self.t>=STEPS: raise RuntimeError('done')
        intensity=ACTIONS[int(action)]
        self.queue=min(.7,self.queue+self.arrivals[self.t])
        repair=min(self.queue,intensity,self.reserve)
        self.queue=max(0.,self.queue-repair)
        self.reserve=np.clip(self.reserve-repair+self.supply[self.t],0,1)
        strain=.006+.18*self.queue+.9*max(0.,intensity-.08)**2
        self.health=np.clip(self.health+.012*repair/.12-strain,0,1)
        self.alive=bool(self.alive and self.health>.03 and self.queue<.62)
        reward=float((1-min(1.,self.queue/.45))*(.65+.35*self.health)*(1-.15*intensity/.12) if self.alive else 0.)
        self.t+=1
        return self.obs(), reward, self.t==STEPS, {'alive':self.alive,'source':0}


def _run_custom(agent, env_cls, seed, steps=STEPS, variant='full', collect=False):
    env=env_cls(); obs=env.reset(seed)
    a=copy.deepcopy(agent); a.reset_episode(obs,variant)
    rows=[]; rewards=[]; alive=[]
    for t in range(steps):
        p=a.prepare(obs,variant)
        action=int(p['action'])
        nxt,r,_,info=env.step(action)
        a.complete_transition(obs,action,nxt,r,p,learn=False)
        if collect:
            rows.append((obs.copy(),p,action,nxt.copy(),r,copy.deepcopy(info),copy.deepcopy(env)))
        rewards.append(r); alive.append(float(info['alive'])); obs=nxt
    return dict(return_mean=float(np.mean(rewards)),alive_fraction=float(np.mean(alive)),
                finite=_finite(rewards) and _finite(alive), rows=rows)


def experiment2(full, generic, seed):
    out={}
    ok=True
    for domain in ('ecology_train','ecology_delay9'):
        fm=rollout(copy.deepcopy(full),domain,872,collect_events=False)[1]
        gm=rollout(copy.deepcopy(generic),domain,872,collect_events=False)[1]
        dR=fm['reward_mean']-gm['reward_mean']; dA=fm['alive_fraction']-gm['alive_fraction']
        out[domain]=dict(full=fm,generic=gm,reward_diff=dR,alive_diff=dA)
        ok &= dR>=E2_REWARD_MARGIN and dA>=E2_ALIVE_MARGIN
    return dict(pass_seed=bool(ok),domains=out)


def experiment3(seed):
    rng=np.random.default_rng(seed^0xA631)
    ntrain,ntest=5000,2500
    X=rng.normal(size=(ntrain+ntest,6))
    need=(X[:,1]+.7*X[:,2]-.35*X[:,4]>0).astype(int)
    local=.4*X[:,0]-.25*X[:,3]
    correction=.75*np.tanh(X[:,1]-X[:,2]+.5*X[:,5])
    noise=rng.normal(0,.08,len(X))
    y=local+(2*need-1)*correction+noise
    local_pred=local
    cross_pred=local+correction
    gain=(y-local_pred)**2-(y-cross_pred)**2
    A=np.c_[np.ones(ntrain),X[:ntrain]]
    lam=.1*np.diag([0.]+[1.]*6)
    coef=np.linalg.solve(A.T@A+lam,A.T@gain[:ntrain])
    Xt=X[ntrain:]; yt=y[ntrain:]; lt=local_pred[ntrain:]; ct=cross_pred[ntrain:]
    score=np.c_[np.ones(ntest),Xt]@coef
    gate=(score>0).astype(int)
    pred=np.where(gate.astype(bool),ct,lt)
    pred_off=lt; pred_on=ct
    perm=Xt.copy(); perm[:,[1,2,4]]=perm[:,[2,4,1]]
    pgate=(np.c_[np.ones(ntest),perm]@coef>0).astype(int)
    ppred=np.where(pgate.astype(bool),ct,lt)
    mse=lambda z: float(np.mean((yt-z)**2))
    best_const=min(mse(pred_off),mse(pred_on))
    acc=float(np.mean(gate==need[ntrain:]))
    gain_mse=best_const-mse(pred)
    perm_damage=mse(ppred)-mse(pred)
    passed=acc>=E3_GATE_ACCURACY and gain_mse>=E3_MSE_GAIN and perm_damage>=E3_PERM_DAMAGE
    return dict(pass_seed=bool(passed),accuracy=acc,adaptive_mse=mse(pred),
                off_mse=mse(pred_off),on_mse=mse(pred_on),best_constant_mse=best_const,
                improvement=gain_mse,permuted_mse=mse(ppred),permutation_damage=perm_damage,
                coefficients=coef.tolist())


def experiment4(full,dense,seed):
    out={}; ok=True
    for name,env in [('cyclic_buffer',CyclicBufferEnv),('repair_queue',RepairQueueEnv)]:
        fs=_run_custom(full,env,seed^hash(name)&0xffffffff)
        ds=_run_custom(dense,env,seed^hash(name)&0xffffffff)
        dR=fs['return_mean']-ds['return_mean']; dA=fs['alive_fraction']-ds['alive_fraction']
        passed=(fs['alive_fraction']>=E4_ALIVE and fs['return_mean']>=E4_RETURN and
                dR>=E4_RETURN_NONINFERIOR and dA>=E4_ALIVE_NONINFERIOR)
        out[name]=dict(full={k:v for k,v in fs.items() if k!='rows'},
                       dense={k:v for k,v in ds.items() if k!='rows'},
                       reward_diff=dR,alive_diff=dA,pass_domain=bool(passed))
        ok &= passed
    return dict(pass_seed=bool(ok),domains=out)


class AttributionEnv:
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.72; self.health=.76; self.demand=.45; self.last_world=-99
        self.shock_flag=self.rng.random(ATTR_STEPS)<.14
        self.shock_mag=self.rng.choice([-1.,1.],ATTR_STEPS)*self.rng.uniform(.05,.13,ATTR_STEPS)*self.shock_flag
        self.demand_tape=np.clip(.48+.25*np.sin(np.linspace(0,9*np.pi,ATTR_STEPS))+
                                 self.rng.normal(0,.04,ATTR_STEPS),.08,.92)
        return self.obs()
    def obs(self):
        return np.clip([self.reserve,self.health,self.demand],0,1)
    def step(self,action):
        if self.t>=ATTR_STEPS: raise RuntimeError('done')
        flow=ACTIONS[int(action)]
        shock=float(self.shock_mag[self.t])
        self.demand=float(self.demand_tape[self.t])
        self_effect=np.array([-.55*flow, .10*flow-.035*(flow/.12)**2, 0.])
        world_effect=np.array([.35*shock, shock, self.demand-self.obs()[2]])
        before=self.obs().copy()
        self.reserve=np.clip(self.reserve+self_effect[0]+world_effect[0]+.012,0,1)
        self.health=np.clip(self.health+self_effect[1]+world_effect[1]-.004-.025*self.demand,0,1)
        self.alive=bool(self.alive and self.health>.025)
        served=min(1.,flow/max(.02,.12*self.demand))
        reward=float(served*(.55+.45*self.health)*(1-.12*flow/.12) if self.alive else 0.)
        smag=float(np.linalg.norm(self_effect[:2])); wmag=float(np.linalg.norm(world_effect[:2]))
        if wmag>1.25*smag: source=1
        elif smag>1.25*wmag: source=0
        else: source=2
        if self.shock_flag[self.t]: self.last_world=self.t
        age=self.t-self.last_world
        agebin=0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3
        self.t+=1
        return self.obs(),reward,self.t==ATTR_STEPS,dict(alive=self.alive,source=source,agebin=agebin,
             world_event=bool(self.shock_flag[self.t-1]),self_magnitude=smag,world_magnitude=wmag,before=before.tolist())


def _ridge_classifier(X,y,nclass,lam=1.):
    X=np.asarray(X,float); y=np.asarray(y,int)
    mu=X.mean(0); sd=X.std(0); sd=np.where(sd<1e-6,1.,sd)
    Z=(X-mu)/sd; A=np.c_[np.ones(len(Z)),Z]
    Y=np.eye(nclass)[y]
    reg=lam*np.eye(A.shape[1]); reg[0,0]=0.
    W=np.linalg.solve(A.T@A+reg,A.T@Y)
    return (mu,sd,W)


def _predict_classifier(model,X):
    mu,sd,W=model; Z=(np.asarray(X)-mu)/sd; return np.argmax(np.c_[np.ones(len(Z)),Z]@W,axis=1)


def _balanced_accuracy(y,p):
    vals=[]
    for c in np.unique(y):
        m=np.asarray(y)==c
        if m.any(): vals.append(np.mean(np.asarray(p)[m]==c))
    return float(np.mean(vals)) if vals else 0.


def _auc(scores,labels):
    scores=np.asarray(scores,float); labels=np.asarray(labels,int)
    pos=scores[labels==1]; neg=scores[labels==0]
    if len(pos)==0 or len(neg)==0: return .5
    # Mann-Whitney form, deterministic and tie-aware.
    return float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))


def _integrated_run(agent,seed,variant='full'):
    env=AttributionEnv(); obs=env.reset(seed)
    a=copy.deepcopy(agent); a.reset_episode(obs,variant)
    X=[]; source=[]; age=[]; unc=[]; errors=[]; cf=[]; mem_pred=[]; obs_hist=[]
    rewards=[]; alive=[]
    for t in range(ATTR_STEPS):
        p=a.prepare(obs,variant)
        # evaluator-only counterfactual action outcomes from exact copies
        cf_rewards=[]
        for act in range(4):
            ee=copy.deepcopy(env); cf_rewards.append(ee.step(act)[1])
        action=int(p['action'])
        nxt,r,_,info=env.step(action)
        a.complete_transition(obs,action,nxt,r,p,learn=False)
        feat=np.r_[p['features'],p['gate_context'],p['goals'],p['scores'],
                   p['memory_energy'],p['memory_resource']]
        X.append(feat); source.append(info['source']); age.append(info['agebin'])
        unc.append(float(p['gate_context'][-1]))
        errors.append(float(np.mean(np.abs(nxt-p['predictions'][action,:3]))))
        cf.append(int(np.argmax(p['predictions'][:,3])==np.argmax(cf_rewards)))
        mem_pred.append(float(p['memory_resource'][action])); obs_hist.append(nxt.copy())
        rewards.append(r); alive.append(float(info['alive'])); obs=nxt
    mem_targets=[]; mp=[]
    for t in range(ATTR_STEPS-12):
        mem_targets.append(.5*(obs_hist[t+3][0]+obs_hist[t+11][0])); mp.append(mem_pred[t])
    return dict(X=np.asarray(X),source=np.asarray(source),age=np.asarray(age),unc=np.asarray(unc),
                errors=np.asarray(errors),cf=np.asarray(cf),mem_mae=float(np.mean(np.abs(np.asarray(mp)-np.asarray(mem_targets)))),
                return_mean=float(np.mean(rewards)),alive_fraction=float(np.mean(alive)))


def experiment5(full,seed):
    base=_integrated_run(full,seed^0xC011,'full')
    split=ATTR_STEPS//2
    Xtr,Xte=base['X'][:split],base['X'][split:]
    src_tr,src_te=base['source'][:split],base['source'][split:]
    age_tr,age_te=base['age'][:split],base['age'][split:]
    source_model=_ridge_classifier(Xtr,src_tr,3)
    source_pred=_predict_classifier(source_model,Xte)
    source_ba=_balanced_accuracy(src_te,source_pred)
    mask_tr=src_tr!=2; mask_te=src_te!=2
    sw_model=_ridge_classifier(Xtr[mask_tr],(src_tr[mask_tr]==0).astype(int),2)
    sw_pred=_predict_classifier(sw_model,Xte[mask_te])
    sw_ba=_balanced_accuracy((src_te[mask_te]==0).astype(int),sw_pred)
    time_model=_ridge_classifier(Xtr,age_tr,4)
    time_pred=_predict_classifier(time_model,Xte)
    time_ba=_balanced_accuracy(age_te,time_pred)
    err_thr=float(np.median(base['errors'][:split]))
    auc=_auc(base['unc'][split:],(base['errors'][split:]>err_thr).astype(int))
    cf=float(np.mean(base['cf'][split:]))

    nm=_integrated_run(full,seed^0xC011,'noMemory')
    pc=_integrated_run(full,seed^0xC011,'permuted_content')
    nc=_integrated_run(full,seed^0xC011,'noCross')
    memory_damage=nm['mem_mae']-base['mem_mae']
    content_drop=base['return_mean']-pc['return_mean']
    cross_drop=base['return_mean']-nc['return_mean']
    alive_damages=[base['alive_fraction']-x['alive_fraction'] for x in (nm,pc,nc)]
    q=E5
    base_ok=(sw_ba>=q['self_world'] and source_ba>=q['source'] and time_ba>=q['time'] and
             auc>=q['auc'] and cf>=q['cf_action'])
    lesion_ok=(memory_damage>=q['memory_damage'] and content_drop>=q['content_return'] and
               cross_drop>=q['cross_return'] and max(abs(x) for x in alive_damages)<=q['max_alive_damage'])
    return dict(pass_seed=bool(base_ok and lesion_ok),base_conjunction=bool(base_ok),
                lesion_conjunction=bool(lesion_ok),self_world_balanced_accuracy=sw_ba,
                source_balanced_accuracy=source_ba,time_balanced_accuracy=time_ba,
                metacog_auc=auc,counterfactual_best_action_accuracy=cf,
                full_return=base['return_mean'],full_alive=base['alive_fraction'],
                memory_mae=base['mem_mae'],noMemory_memory_mae=nm['mem_mae'],
                memory_damage=memory_damage,permuted_content_return=pc['return_mean'],
                content_return_drop=content_drop,noCross_return=nc['return_mean'],
                cross_return_drop=cross_drop,alive_damages=alive_damages)


def summarize(records, selected_subset):
    exp_names=('E1','E2','E3','E4','E5')
    counts={k:sum(bool(r[k]['pass_seed']) for r in records) for k in exp_names}
    med={}
    def median(path):
        vals=[]
        for r in records:
            v=r
            for p in path: v=v[p]
            vals.append(float(v))
        return float(np.median(vals))
    med['E1_return_loss']=median(('E1','return_loss'))
    med['E2_ecology_train_reward_diff']=median(('E2','domains','ecology_train','reward_diff'))
    med['E3_accuracy']=median(('E3','accuracy'))
    med['E3_improvement']=median(('E3','improvement'))
    med['E4_cyclic_return']=median(('E4','domains','cyclic_buffer','full','return_mean'))
    med['E4_repair_return']=median(('E4','domains','repair_queue','full','return_mean'))
    med['E5_self_world']=median(('E5','self_world_balanced_accuracy'))
    med['E5_source']=median(('E5','source_balanced_accuracy'))
    med['E5_time']=median(('E5','time_balanced_accuracy'))
    med['E5_auc']=median(('E5','metacog_auc'))
    med['E5_cf']=median(('E5','counterfactual_best_action_accuracy'))
    return dict(selected_csd_subset=list(selected_subset),n=len(records),required=GLOBAL_REQUIRED,
                pass_counts=counts,verdicts={k:('PASS' if counts[k]>=GLOBAL_REQUIRED else 'FAIL') for k in exp_names},
                medians=med)


def run_all(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    # Pilot is used only for CSD subset selection.
    pilot_agents=[_train_agent(s) for s in PILOT_SEEDS]
    selected,pilot_search=select_csd_subset(pilot_agents)
    (output/'pilot_csd.json').write_text(json.dumps(dict(selected=list(selected),search=pilot_search),indent=2))

    records=[]
    for seed in FINAL_SEEDS:
        full=_train_agent(seed)
        dense=_train_agent(seed,NativeAgent,'dense')
        generic=_train_agent(seed,GenericIsomorphicAgent)
        e1=_csd_one(full,selected,episode=851); e1['pass_seed']=_csd_pass(e1)
        e2=experiment2(full,generic,seed)
        e3=experiment3(seed)
        e4=experiment4(full,dense,seed)
        e5=experiment5(full,seed)
        rec=dict(seed=seed,E1=e1,E2=e2,E3=e3,E4=e4,E5=e5,
                 full_gate_mode=full.gate_mode,generic_gate_mode=generic.gate_mode,
                 full_parameter_digest=full.parameter_digest(),dense_parameter_digest=dense.parameter_digest())
        records.append(rec)
        (output/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2))
    summary=summarize(records,selected)
    bundle=dict(protocol='PREREG_R10.md',base_r9='a0696148e568c94365a26311953430b082a3725b',
                pilot_seeds=list(PILOT_SEEDS),final_seeds=list(FINAL_SEEDS),summary=summary,records=records)
    (output/'RESULTS_R10.json').write_text(json.dumps(bundle,indent=2))
    lines=['# POLAR R10 — Five discriminating experiments','',
           f"Confirmatory agents: {len(records)}; global threshold: {GLOBAL_REQUIRED}/{len(records)}.",'',
           f"CSD subset selected on pilot only: **{'+'.join(selected) if selected else 'EMPTY'}**.",'',
           '| Experiment | Pass seeds | Verdict |','|---|---:|---|']
    for k in ('E1','E2','E3','E4','E5'):
        lines.append(f"| {k} | {summary['pass_counts'][k]}/{len(records)} | **{summary['verdicts'][k]}** |")
    lines+=['','## Selected medians','']
    for k,v in summary['medians'].items(): lines.append(f"- {k}: **{v:.6f}**")
    lines+=['','## Interpretation boundary',
            'PASS/FAIL refers only to the frozen operational criterion in this synthetic campaign.',
            'E5 concerns functional precursors and selective lesions; it is not a measurement of phenomenal consciousness.']
    (output/'REPORT_R10.md').write_text('\n'.join(lines)+'\n')
    # deterministic integrity digest
    h=hashlib.sha256()
    for p in sorted(output.glob('*.json'))+sorted(output.glob('*.md')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    (output/'SHA256.txt').write_text(h.hexdigest()+'  R10_OUTPUT_SET\n')
    return summary
