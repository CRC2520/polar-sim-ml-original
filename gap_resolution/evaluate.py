"""Reproducible outcomes and capability probes. Final criteria are not retuned."""
from copy import deepcopy
import numpy as np
from .tasks import CELLS,MODES,FINAL_SEEDS,make_task,physical,run_trial,observation,jsonable
from integrated_polar import IntegratedAgent,AgentConfig
from integrated_polar.layers import CalibratedMonitor,SemanticAdapter,infer_effect_source
from integrated_polar.agent import IndependentResourceAgent,coordinate_offers
from integrated_polar.numerics import CoupledRLS

CLAIMS=[('coupled_prediction','diagonal_plan','loss'),('priorities','no_priorities','loss'),
        ('retained_memory','no_memory','return_loss'),('temporal_planning','myopic','post_change')]
CAPABILITY_SEEDS=list(range(951001,951025))


def interval(values,idx,upper=.9875):
    values=np.asarray(values,float)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('invalid paired sample')
    boot=values[idx].mean(axis=1)
    return {'mean':float(values.mean()),'low95':float(np.quantile(boot,.025)),'high95':float(np.quantile(boot,.975)),
            'upper_corrected':float(np.quantile(boot,upper))}


def decision(valid,claims):
    if set(claims)!={x[0] for x in CLAIMS}:raise ValueError('incomplete prespecified claims')
    if not valid:return 'invalid_evaluation'
    if all(claims.values()):return 'bounded_functional_improvements_supported'
    return 'engineering_verified_with_partial_empirical_support'


def analyze(rows,equivalence,seeds=FINAL_SEEDS):
    expected={(s,tuple(c),m) for s in seeds for c in CELLS for m in MODES}
    keys=[(r['seed'],tuple(r['cell']),r['mode']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('incomplete or duplicate experiment')
    metrics=list(rows[0]['metrics']);index=dict(zip(keys,rows))
    values={m:{v:np.array([np.mean([index[s,c,m]['metrics'][v] for c in CELLS]) for s in seeds]) for v in metrics} for m in MODES}
    idx=np.random.default_rng(941777).integers(0,len(seeds),(20000,len(seeds)))
    contrasts={};claims={}
    for name,other,metric in CLAIMS:
        effect=interval(values['full'][metric]-values[other][metric],idx)
        effect.update(comparator=other,metric=metric,required_upper_below=-.0002)
        claims[name]=effect['upper_corrected']<-.0002;effect['passes']=claims[name];contrasts[name]=effect
    hard=int(sum(r['metrics']['violations'] for r in rows if r['mode']=='full'))
    negative=interval(values['zero']['loss']-values['full']['loss'],idx)
    valid=bool(hard==0 and equivalence<=1e-8 and negative['low95']>0)
    return {'seeds':len(seeds),'trials':len(rows),'transitions':len(rows)*48,'calibration_transitions_per_trial':48,
            'status':'prospective_final' if list(seeds)==FINAL_SEEDS else 'development_only',
            'summary':{m:{k:float(v.mean()) for k,v in d.items()} for m,d in values.items()},
            'contrasts':contrasts,'claims':claims,'valid':valid,'hard_violations':hard,
            'generic_equivalent_max_action_difference':equivalence,'negative_control':negative,
            'decision':decision(valid,claims),
            'scope':'Conventional predictive-control realization of architectural functions; no unique polar advantage or consciousness inference.'}


def verify_record(record):
    task=make_task(record['seed'],record['cell']);state=np.full(4,.35)
    if len(record['frames'])!=48 or [f['t'] for f in record['frames']]!=list(range(48)):raise ValueError('missing/reordered transitions')
    losses=[];costs=[];violations=0
    for f in record['frames']:
        t=f['t'];u=np.array(f['action']);obs,target,w=observation(task,t,state)
        if jsonable(obs)!=f['observation']:raise ValueError('observation mismatch')
        y,valid=physical(task,state,u,t)
        if not np.allclose(y,f['effect'],rtol=0,atol=1e-12) or valid.tolist()!=f['valid_transition']:raise ValueError('physical effect mismatch')
        if target.tolist()!=f['evaluation_target'] or w.tolist()!=f['evaluation_priority']:raise ValueError('evaluation target mismatch')
        loss=float(w@((y-target)**2)/w.sum());cost=float(task['costs']@u/task['costs'].sum())
        if abs(loss-f['loss'])>1e-12 or abs(cost-f['cost'])>1e-12:raise ValueError('metric mismatch')
        mask=np.array(f['controller']['allowed'],bool)
        bad=bool(np.any(u< -1e-10) or np.any(u>1+1e-10) or np.any(abs(u[~mask])>1e-10) or task['costs']@u>task['budget']+1e-10)
        if bad!=f['violation']:raise ValueError('violation record mismatch')
        if f['controller']['action']!=f['action']:raise ValueError('controller/environment action mismatch')
        if f['controller']['feedback']['state']!=f['effect']:raise ValueError('feedback mismatch')
        losses.append(loss);costs.append(cost);violations+=bad;state=y
    switch=[i for start in (12,24,36) for i in range(start,start+4)]
    expected={'loss':float(np.mean(losses)),'post_change':float(np.mean(np.array(losses)[switch])),
              'return_loss':float(np.mean(losses[36:])),'cost':float(np.mean(costs)),'violations':int(violations)}
    for key,v in expected.items():
        if abs(record['metrics'][key]-v)>1e-12:raise ValueError('summary mismatch: '+key)
    return expected


def coordination_probe(seed,communication):
    rng=np.random.default_rng(seed);names=['local-A','local-B'];agents=[IndependentResourceAgent(n) for n in names]
    targets=np.array([rng.uniform(.7,.8),rng.uniform(.18,.28)]);weights=np.array([4.,1.])
    if seed%2:targets=targets[::-1];weights=weights[::-1]
    states=np.array([.35,.35]);frames=[]
    for t in range(16):
        offers=[a.offer(states[i],targets[i],weights[i]) for i,a in enumerate(agents)]
        allocation=coordinate_offers(offers,.65,communication=communication);actions=[];next_states=[];traces=[]
        for i,a in enumerate(agents):
            o={'state':np.array([states[i]]),'cue':'local','event':{'targets':[[targets[i]]]*3,'priorities':[[weights[i]]]*3,'version':t+1},'costs':np.ones(2)}
            u=a.act(o,allocation[names[i]]);y=np.clip(.9*states[i]+.03+np.array([.12,-.11])@u,0,1)
            a.controller.learn({'state':np.array([y]),'token':deepcopy(a.controller.pending['token'])})
            actions.append(u.tolist());next_states.append(float(y));traces.append({'actor':names[i],'own_model':a.controller.model.B.tolist(),'own_state':states[i],'own_target':targets[i]})
        spent=float(np.sum(actions));loss=float(weights@((np.array(next_states)-targets)**2)/weights.sum())
        frames.append({'t':t,'offers':offers,'allocations':allocation,'actions':actions,'effects':next_states,'loss':loss,'spent':spent,'local_controllers':traces})
        states=np.array(next_states)
    return {'seed':seed,'communication':communication,'loss':float(np.mean([x['loss'] for x in frames])),
            'violations':sum(x['spent']>.65+1e-10 for x in frames),'frames':frames}


def capability_probe(seed):
    rng=np.random.default_rng(seed);records={};correct=0;ambiguous=0
    model=CoupledRLS(2,4,ridge=1e-6,forgetting=1);B=rng.uniform(-.2,.2,(2,4))
    for _ in range(100):
        u=rng.uniform(0,1,4);model.update(u,B@u)
    source=[]
    for t in range(12):
        a=rng.uniform(0,1,4);b=rng.uniform(0,1,4);which=t%2
        effects={'A':model.predict(a),'B':model.predict(b)}
        z=B@(a if which==0 else b)+rng.normal(0,.001,2)
        found=infer_effect_source(z,effects,variance=1e-4)
        truth='A' if which==0 else 'B';correct+=found['source']==truth
        amb=infer_effect_source(B@a,{'A':model.predict(a),'B':model.predict(a)},variance=1e-4);ambiguous+=amb['source'] is None
        source.append({'candidate_actions':[a.tolist(),b.tolist()],'measured_response':z.tolist(),'true_actor_for_evaluation_only':truth,'inference':found,'ambiguous_inference':amb})
    adapter=SemanticAdapter(2,3);levels=rng.uniform(.1,.9,2);priority=rng.uniform(1,5,2)
    command=f'tank_1 = {levels[0]:.6f} @ {priority[0]:.6f}; tank_2 = {levels[1]:.6f} @ {priority[1]:.6f}'
    parsed=adapter.parse(command,'heldout')
    semantic_error=float(np.max(abs(np.array(parsed.targets)[0]-np.round(levels,6))))
    # Sign grounding is learned from physical input/output observations, starting at zero.
    sign_model=CoupledRLS(1,2,ridge=1e-6,forgetting=1)
    for _ in range(80):
        u=rng.uniform(0,1,2);sign_model.update(u,[.13*u[0]-.11*u[1]])
    norm_records=[]
    for approved in (False,True):
        a=IntegratedAgent(AgentConfig(tanks=1));o={'state':np.array([.7]),'cue':'policy','event':{'targets':[[.1]]*3,'priorities':[[1.]]*3},'budget':2.,'normative':{'protected':np.array([True]),'approved_drain':approved,'mandatory_min':[0.,.2]}}
        u=a.act(o);norm_records.append({'approved':approved,'action':u.tolist(),'rules':a.trace['E']})
    return {'seed':seed,'source_correct':int(correct),'source_trials':12,'ambiguous_abstentions':int(ambiguous),
            'source_records':source,'semantic_command':command,'semantic_targets':parsed.targets,'semantic_max_error':semantic_error,
            'learned_fill_drain':sign_model.B.tolist(),'normative_records':norm_records,
            'normative_pass':bool(norm_records[0]['action'][1]==0 and norm_records[1]['action'][1]>.01)}


def calibration_metrics(frames,monitor):
    p=np.array([f['probability'] for f in frames],float);y=np.array([f['success'] for f in frames],float)
    if not np.isfinite(p).all():raise ValueError('probabilities must precede outcomes')
    bins=np.minimum(9,(p*10).astype(int));ece=0.;reliability=[]
    for k in range(10):
        chosen=bins==k
        if chosen.any():
            conf=float(p[chosen].mean());rate=float(y[chosen].mean());count=int(chosen.sum())
            ece+=count/len(y)*abs(conf-rate);reliability.append({'bin':k,'count':count,'forecast':conf,'observed':rate})
    brier=float(np.mean((p-y)**2));base=float(np.mean((monitor.base-y)**2))
    return {'samples':len(y),'brier':brier,'constant_development_base_brier':base,'ece':float(ece),
            'reliability':reliability,'calibration_acceptance':bool(ece<=.15 and brier<=base+.02),
            'proposed_review_fraction':float(np.mean(p<.55)),
            'scope':'Forecast of this controller task-success event, conditional on fitted development bins; not self-awareness.'}
