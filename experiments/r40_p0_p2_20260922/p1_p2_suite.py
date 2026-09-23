#!/usr/bin/env python3
"""Prospective finite-scope tests. No historical runner or threshold is overwritten."""
import argparse, copy, hashlib, importlib.util, json, math, os, pickle, time
from pathlib import Path
import numpy as np
from scipy import stats

HERE=Path(__file__).resolve().parent
REF=HERE.parent/'r36_r1_metacog_stabilization_20260922'
SOURCE=REF/'r36_r1_metacog.py'
b=SOURCE.read_bytes()
assert hashlib.sha1(b'blob '+str(len(b)).encode()+bytes([0])+b).hexdigest()=='99c6ac712e26abbbe7a1202709701e5af94c6ab4'
spec=importlib.util.spec_from_file_location('frozen_r36r1',SOURCE)
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
FREEZE=json.loads((REF/'R36_R1_CONFIRM_FREEZE.json').read_text())
N=3; T=1760; U=base.ACTIONS.copy(); CENTERS=base.CUE_CENTERS.copy()


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def rng(seed,label):
    h=hashlib.sha256(f'R40|{seed}|{label}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8],'little'))


def costs(A,B,x,target):
    x1=x@A.T+U@B.T
    x2=x1[:,None,:]@A.T+(U@B.T)[None,:,:]
    c1=np.mean((x1-target)**2,axis=1)+base.ACTION_PENALTY*np.mean(U*U,axis=1)
    c2=np.mean((x2-target)**2,axis=2)+base.ACTION_PENALTY*np.mean(U*U,axis=1)[None,:]
    return c1[:,None]+base.PLAN_DISCOUNT*c2


def fast_plan(self,x,target,key,A_override=None,B_override=None):
    m=self.model(key)
    A=m.A if A_override is None else A_override
    B=m.B if B_override is None else B_override
    return int(np.argmin(costs(A,B,x,target))//len(U))


def fast_model_cost(self,x,target,key,first_action):
    m=self.model(key)
    return float(costs(m.A,m.B,x,target)[int(first_action)].min())


# Verify numerical optimization before any scientific seed is evaluated.
original_plan=base.PersistentAgent.plan_action
original_cost=base.PersistentAgent.model_two_step_cost
for i in range(100):
    r=rng(12345+i,'instrument'); a=base.PersistentAgent(); k=(0,0); m=a.model(k)
    m.A=r.normal(0,.2,(3,3)); m.B=np.diag(r.uniform(.7,1.3,3))
    x=r.normal(size=3); g=r.normal(size=3)
    assert original_plan(a,x,g,k)==fast_plan(a,x,g,k)
    for u in range(len(U)):
        assert abs(original_cost(a,x,g,k,u)-fast_model_cost(a,x,g,k,u))<1e-10
base.PersistentAgent.plan_action=fast_plan
base.PersistentAgent.model_two_step_cost=fast_model_cost


class BayesCue(base.PersistentAgent):
    """Only adjustment: conventional robust HMM over the same four observed cue modes."""
    def __init__(self):
        super().__init__(); self.belief=np.ones(4)/4
    def update_cue(self,cue):
        prior=.985*self.belief+.015/4
        sq=((np.asarray(cue)[None,:]-CENTERS)/.72)**2
        log_l=-2.5*np.log1p(sq/4).sum(axis=1)
        l=np.exp(log_l-log_l.max())
        self.belief=prior*l; self.belief/=self.belief.sum()
        i=int(np.argmax(self.belief)); self.cue_state=CENTERS[i].copy()
        return base.cue_key(self.cue_state)


class World:
    def __init__(self,seed,stress=False,experience=None,other_profile=False):
        self.seed=seed; self.t=0; self.stress=stress
        world=rng(seed,'matrices'); schedule=rng(seed,'schedule'); goals=rng(seed,'independent_goals')
        self.As=[]
        for _ in range(4):
            a=world.normal(0,.11,(3,3)); np.fill_diagonal(a,world.uniform(.46,.60,3))
            rad=max(abs(np.linalg.eigvals(a)))
            if rad>.82: a*=.82/rad
            self.As.append(a)
        self.As=np.asarray(self.As)
        self.schedule=np.r_[schedule.permutation(4),schedule.permutation(4)]
        self.cue_map=schedule.permutation(4)
        self.goals=goals.uniform(-.6,.6,(8,3))
        gain=rng(seed,'body').uniform(.75,1.25,3)
        self.B=np.diag(2.-gain if other_profile else gain)
        e=seed if experience is None else experience
        self.x=rng(e,'initial').normal(0,.12,3)
        self.proc=rng(e,'process').normal(0,.012,(T,3))
        self.sensor=rng(e,'sensor').normal(0,.025 if stress else .005,(T+1,3))
        self.cnoise=rng(e,'cue_noise').normal(0,1.10 if stress else .72,(T+1,2))
        if stress:
            r=rng(e,'outliers'); bad=r.random(T+1)<.10
            self.cnoise[bad]+=r.normal(0,3.,(int(bad.sum()),2))
        r=rng(e,'shocks'); bad=r.random(T)<.02
        self.shock=np.zeros((T,3)); ix=np.flatnonzero(bad)
        if len(ix): self.shock[ix,r.integers(3,size=len(ix))]=r.choice([-1.,1.],size=len(ix))*r.uniform(.28,.40,len(ix))
    def block(self): return min(7,self.t//220)
    def reg(self): return int(self.schedule[self.block()])
    def target(self): return self.goals[self.block()].copy()
    def obs(self):
        return np.r_[self.x+self.sensor[self.t],CENTERS[self.cue_map[self.reg()]]+self.cnoise[self.t]]
    def step(self,a):
        reg=self.reg(); goal=self.target(); A=self.As[reg]; ext=bool(np.any(self.shock[self.t]))
        self.x=np.clip(A@self.x+self.B@U[a]+self.proc[self.t]+self.shock[self.t],-2.5,2.5)
        cost=float(np.mean((self.x-goal)**2)+base.ACTION_PENALTY*np.mean(U[a]**2))
        self.t+=1
        # Real terminal observation. Historical source used a zero pseudo-target at terminal.
        return self.obs(),cost,ext


def step_agent(agent,env,redact=True):
    obs=env.obs(); x=obs[:3].copy(); cue=obs[3:].copy(); target=env.target()
    action,key,conf,pred,use_plan,adv=agent.act(obs,target)
    nxt,cost,ext=env.step(action)
    # Oracle metadata stays in the evaluator, not active agent inputs.
    agent.update(x,cue,action,nxt[:3],False if redact else ext,key,conf,-1,-1)
    return action,pred,nxt,cost,ext,conf,key,use_plan


def warm(seed,kind='FROZEN',stress=False,experience=None,other=False,steps=880):
    env=World(seed,stress,experience,other)
    agent=base.PersistentAgent() if kind=='FROZEN' else BayesCue()
    for _ in range(steps): step_agent(agent,env)
    return agent,env


def classify(x,strict=True):
    t=FREEZE['criteria']
    keys={'D':'D_prediction_damage','C':'C_recurrent_context_gain','R':'R_relation_lesion_damage',
        'memory':'memory_reentry_gain','own_history':'own_history_transplant_damage','source':'source_balanced_accuracy',
        'time':'time_order_accuracy','meta_cal':'metacog_calibration_gap','meta_gate':'metacog_gate_gain',
        'planning':'planning_gain','stability':'stable_fraction','models':'models_learned'}
    out={}
    for name,k in keys.items():
        value=x['planning_candidate_gain_all'] if name=='planning' and strict else x[k]
        out[name]=bool(value>=t[k+'_min'])
    out['lifetime']=x['same_agent_steps']==1760 and x['memory_entries']==1760
    if not strict: out['invocation']=x['planning_invocation_rate']>=t['planning_invocation_rate_min']
    return out


def run_life(seed,kind,stress,outdir):
    ag=base.PersistentAgent() if kind=='FROZEN' else BayesCue(); env=World(seed,stress)
    donor,_=warm(seed,kind,stress,experience=seed+100000,other=True,steps=880)
    vals={k:[] for k in ['D','C','R','own','reentry','plan','invoked_plan','meta','conf','error','all_conf','all_error','cost','stable']}
    sources=[]; cold=[]; inv=[]; traces=[]
    for t in range(T):
        obs=env.obs(); x=obs[:3].copy(); cue=obs[3:].copy(); target=env.target(); block=env.block(); reg=env.reg()
        action,key,conf,pred,use_plan,adv=ag.act(obs,target)
        m=ag.model(key); fitted=m.fitted; visits=ag.key_visits[key]
        cur=ag.model(base.cue_key(cue)).predict(x,action)
        pd=np.repeat(m.A.mean(axis=1,keepdims=True),3,axis=1)@x+m.B@U[action]
        pr=np.diag(np.diag(m.A))@x+m.B@U[action]
        dm=donor.models.get(key); po=dm.predict(x,action) if dm is not None else .54*x+.9*U[action]
        if fitted and visits>55:
            ap=ag.plan_action(x,target,key); am=ag.myopic_action(x,target,key)
            tc=costs(env.As[reg],env.B,env.x,target).min(axis=1)
            gain=float(tc[am]-tc[ap]); vals['plan'].append(gain); inv.append(float(use_plan))
            if use_plan: vals['invoked_plan'].append(gain)
            vals['meta'].append(float(tc[am if use_plan else ap]-tc[ap if use_plan else am]))
        if t%220==0: cold=[]
        nxt,cost,ext=env.step(action); y=nxt[:3].copy(); err=base.mse(pred,y)
        if fitted and not ext:
            for k,z in [('D',pd),('C',cur),('R',pr),('own',po)]: vals[k].append(base.mse(z,y)-err)
        if fitted and visits>55:
            vals['all_conf'].append(conf); vals['all_error'].append(err)
            if not ext: vals['conf'].append(conf); vals['error'].append(err)
        if block>=4 and t%220<34 and not ext:
            vals['reentry'].append(base.mse(base.cold_predict(cold,x,action),y)-err)
        if not ext: cold.append((np.r_[x,U[action]],y.copy()))
        ag.update(x,cue,action,y,False,key,conf,-1,-1)
        sources.append((ext,ag.memory[-1]['source_pred']))
        vals['cost'].append(cost); vals['stable'].append(float(np.linalg.norm(env.x)<3.))
        traces.append(np.r_[t,obs,action,pred,y,cost,conf,ext,use_plan])
    def avg(k): return float(np.mean(vals[k])) if vals[k] else 0.
    def gap(ck,ek):
        c=np.asarray(vals[ck]); e=np.asarray(vals[ek])
        if not len(c): return 0.
        lo,hi=np.quantile(c,[.25,.75]); return float(e[c<=lo].mean()-e[c>=hi].mean())
    s=np.array(sources[:-80],dtype=bool)
    acc=.5*((s[s[:,0],1].mean() if s[:,0].any() else 0)+(~s[~s[:,0],1]).mean())
    result={'seed':seed,'agent':kind,'stress':stress,'D_prediction_damage':avg('D'),
        'C_recurrent_context_gain':avg('C'),'R_relation_lesion_damage':avg('R'),
        'own_history_transplant_damage':avg('own'),'memory_reentry_gain':avg('reentry'),
        'source_balanced_accuracy':float(acc),'time_order_accuracy':base.time_order_accuracy(ag.memory,seed),
        'metacog_calibration_gap':gap('conf','error'),'metacog_gap_including_shocks':gap('all_conf','all_error'),
        'metacog_gate_gain':avg('meta'),'planning_gain':avg('invoked_plan'),
        'planning_candidate_gain_all':avg('plan'),'planning_invocation_rate':float(np.mean(inv)) if inv else 0.,
        'mean_tracking_cost':avg('cost'),'stable_fraction':avg('stable'),
        'models_learned':sum(m.fitted for m in ag.models.values()),'same_agent_steps':ag.clock,'memory_entries':len(ag.memory)}
    result['strict_checks']=classify(result,True); result['invoked_checks']=classify(result,False)
    result['strict_joint']=all(result['strict_checks'].values()); result['invoked_joint']=all(result['invoked_checks'].values())
    p=outdir/'traces'/f'{seed}_{kind}_{int(stress)}.npz'; p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,trace=np.asarray(traces),As=env.As,B=env.B,schedule=env.schedule,goals=env.goals)
    return result


def p1(out):
    records=[]
    for split,seeds in [('development',range(2076001,2076005)),('confirmation',range(2077001,2077033))]:
        for s in seeds:
            for stress in [False,True]:
                for kind in ['FROZEN','BAYES_CUE']:
                    q=run_life(s,kind,stress,out); q['split']=split; records.append(q)
            print('P1_SEED_DONE',s,flush=True)
    groups={}; confirm=[x for x in records if x['split']=='confirmation']
    for stress in [False,True]:
        for kind in ['FROZEN','BAYES_CUE']:
            r=[x for x in confirm if x['stress']==stress and x['agent']==kind]
            groups[f'{kind}_{int(stress)}']={'strict_joint':sum(x['strict_joint'] for x in r),
                'invoked_joint':sum(x['invoked_joint'] for x in r),
                'strict_component_counts':{k:sum(x['strict_checks'][k] for x in r) for k in r[0]['strict_checks']},
                'mean_cost':float(np.mean([x['mean_tracking_cost'] for x in r]))}
        a=sorted([x for x in confirm if x['stress']==stress and x['agent']=='FROZEN'],key=lambda z:z['seed'])
        b=sorted([x for x in confirm if x['stress']==stress and x['agent']=='BAYES_CUE'],key=lambda z:z['seed'])
        diff=np.array([x['mean_tracking_cost']-y['mean_tracking_cost'] for x,y in zip(a,b)])
        boot=rng(990,'bootstrap').choice(diff,(4000,len(diff)),replace=True).mean(axis=1)
        groups[f'paired_improvement_{int(stress)}']={'mean':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    ok=all(groups[f'BAYES_CUE_{int(s)}']['strict_joint']>=24 for s in [False,True])
    result={'resolution':'P1_NEW_DOMAIN_JOINT_PASS' if ok else 'P1_NEW_DOMAIN_JOINT_FAIL',
        'required_joint':24,'denominator':32,'groups':groups,'records':records,
        'scope':'new internally generated domain; NOT E6b or external control confirmation',
        'planning_primary':'all eligible states, not only invoked states','time_scope':'explicit timestamp telemetry only'}
    dump(out/'P1_RESULTS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


def rollout(ag,env,n=96,redact=True):
    actions=[]; preds=[]; loss=[]
    for _ in range(n):
        a,p,y,c,e,conf,key,use=step_agent(ag,env,redact)
        actions.append(a); preds.append(p); loss.append(c)
    return np.array(actions),np.array(preds),float(np.mean(loss))


def profile_error(ag,env,seed):
    r=rng(seed,'profile_probes'); errors=[]
    for reg in range(4):
        key=base.cue_key(CENTERS[env.cue_map[reg]]); m=ag.model(key)
        for _ in range(64):
            x=r.uniform(-.6,.6,3); a=int(r.integers(len(U)))
            y=env.As[reg]@x+env.B@U[a]
            errors.append(base.mse(m.predict(x,a),y))
    return float(np.mean(errors))


def transplanted(original,donor,alpha):
    a=copy.deepcopy(original)
    for key,m in a.models.items():
        if key in donor.models: m.B=(1-alpha)*m.B+alpha*donor.models[key].B
    return a


def p2_seed(seed):
    original,env=warm(seed)
    same,_=warm(seed,experience=seed+100000)
    different,_=warm(seed,experience=seed+100000,other=True)
    pristine=copy.deepcopy(original); archival_bytes=len(pickle.dumps(pristine))
    anchor=rollout(copy.deepcopy(original),copy.deepcopy(env))
    variants={}
    a=copy.deepcopy(original); a.memory=[]; a.last_pred=None; a.last_key=None; a.last_conf=0.
    variants['archive_removed']=a
    a=copy.deepcopy(original)
    for row in a.memory: row['source_true']=False; row['regime']=-1; row['block']=-1
    variants['metadata_redacted']=a
    a=copy.deepcopy(original); a.cue_state[:]=0.; variants['recurrent_reset']=a
    variants['same_profile_B']=transplanted(original,same,1.)
    variants['different_profile_B']=transplanted(original,different,1.)
    a=copy.deepcopy(original); a.memory=copy.deepcopy(different.memory); variants['archive_transplant']=a
    variants['arithmetic_B_merge']=transplanted(original,different,.5)
    interventions={}
    compressed_bytes=len(pickle.dumps(variants['archive_removed']))
    for name,ag in variants.items():
        actions,preds,cost=rollout(ag,copy.deepcopy(env))
        interventions[name]={'action_change_fraction':float(np.mean(actions!=anchor[0])),
            'max_prediction_change':float(np.max(abs(preds-anchor[1]))),'tracking_cost_change':cost-anchor[2]}
    clone=rollout(copy.deepcopy(original),copy.deepcopy(env))
    grad=[{'alpha':float(a),'prediction_mse':profile_error(transplanted(original,different,a),env,seed)} for a in [0.,.25,.5,.75,1.]]
    own=profile_error(original,env,seed)
    same_e=profile_error(transplanted(original,same,1.),env,seed)
    other_e=profile_error(transplanted(original,different,1.),env,seed)
    # Single-step interventions are accompanied by closed-loop fork continuation above.
    return {'seed':seed,'clone_exact':bool(np.array_equal(clone[0],anchor[0]) and np.max(abs(clone[1]-anchor[1]))<1e-10),
        'interventions':interventions,'graded_B_replacement':grad,'own_prediction_mse':own,
        'same_profile_transplant_mse':same_e,'different_profile_transplant_mse':other_e,
        'profile_specificity_margin':other_e-same_e,
        'archival_state_bytes':archival_bytes,'pruned_state_bytes':compressed_bytes,
        'archival_query_entries_before':len(pristine.memory),'archival_query_entries_after_pruning':0,
        'full_capacity_reduction_pass':False,
        'action_state_reduction_pass':bool(interventions['archive_removed']['action_change_fraction']==0 and interventions['archive_removed']['max_prediction_change']<1e-10)}


def p2(out):
    records=[]
    for split,seeds in [('development',range(2086001,2086005)),('confirmation',range(2087001,2087033))]:
        for seed in seeds:
            q=p2_seed(seed); q['split']=split; records.append(q); print('P2_SEED_DONE',seed,flush=True)
    c=[x for x in records if x['split']=='confirmation']
    count=sum(x['profile_specificity_margin']>=1e-4 for x in c)
    result={'resolution':'P2_PROFILE_SPECIFICITY_PASS_LOCAL' if count>=24 else 'P2_PROFILE_SPECIFICITY_NOT_ESTABLISHED',
        'profile_specificity_count':count,'required':24,'denominator':32,
        'clone_exact_count':sum(x['clone_exact'] for x in c),
        'action_state_reduction_count':sum(x['action_state_reduction_pass'] for x in c),
        'full_capacity_reduction_count':0,'records':records,
        'identity_scope':'actuator-history specificity plus declared interventions, not phenomenal or universal identity',
        'minimality_scope':'action-preserving finite implementation reduction; archival report functionality is lost'}
    dump(out/'P2_RESULTS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


def e7(out):
    r=rng(2096001,'synthetic_instrument_only'); studies=1000; n=64
    common=r.normal(size=(studies,n,1)); independent=r.normal(size=(studies,n,4))
    null=np.sqrt(.4)*common+np.sqrt(.6)*independent
    def rejected(x):
        t=x.mean(axis=1)/(x.std(axis=1,ddof=1)/np.sqrt(n))
        p=2*stats.t.sf(abs(t),df=n-1)
        return p<.05/4
    nul=rejected(null); alt=rejected(null+.5)
    fwer=float(nul.any(axis=1).mean())
    result={'analysis_pipeline_status':'PASS_SYNTHETIC_INSTRUMENT' if fwer<=.08 else 'FAIL_SYNTHETIC_INSTRUMENT',
        'null_familywise_rejection_rate':fwer,'alternative_d':.5,'alternative_any_detection_power':float(alt.any(axis=1).mean()),
        'alternative_all_four_power':float(alt.all(axis=1).mean()),'simulation_studies':studies,'paired_units_per_study':n,
        'real_biological_rows':0,'biological_adjudication':'BLOCKED_REAL_DATA_PENDING','E7':'OPEN',
        'note':'Synthetic statistical calibration is not biological evidence or consciousness evidence.'}
    dump(out/'E7_INSTRUMENT.json',result)
    dump(out/'E6b_HANDOFF.json',{'E6b':'OPEN_SEPARATE_TEAM_REQUIRED','same_program_execution':True,
        'required':['separate investigator/team','implementation independent of frozen runner','prospective code/config freeze',
                    'new blinded evaluation seeds','all negative results retained','raw data and provenance'],
        'no_external_team_claimed':True})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['p1','p2','e7'],required=True)
    p.add_argument('--out',default='r40_results'); a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    dump(out/f'INSTRUMENT_{a.stage}.json',{'vectorized_planner_equivalence_cases':100,'pass':True,
        'reference_runner_blob':'99c6ac712e26abbbe7a1202709701e5af94c6ab4','terminal_actual_observation':True,
        'oracle_metadata_redacted_before_agent_update':True})
    {'p1':p1,'p2':p2,'e7':e7}[a.stage](out)
