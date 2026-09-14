"""Prospective B1-S v1.1 registries and utilities; no training at import."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE='29feb9f66eb5192616b75581ba6c947ea863c950'
EDGES=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
SPLITS={'qa','discovery-train','discovery-select','calibration-train','calibration-select','competitive-train','competitive-eval','causal','probe','bootstrap'}
PPO_COMMON=dict(lr=.0003,gamma=.99,rollout_steps=512,minibatch=128,epochs=4,gae_lambda=.95,clip=.2,value_coef=.5,max_grad_norm=.5,adam_eps=1e-5)
PPO_CANDIDATES=[dict(id='ppo-v1selected',lr=.0005,gamma=.98,rollout_steps=512,minibatch=128,epochs=4),dict(id='ppo-standard',lr=.0003,gamma=.99,rollout_steps=2048,minibatch=64,epochs=10),dict(id='ppo-conservative',lr=.0001,gamma=.995,rollout_steps=2048,minibatch=64,epochs=10)]
SAC_CANDIDATES=[dict(id='sac-v1selected',lr=.002,gamma=.95,train_freq=8,gradient_steps=1,learning_starts=1024,batch_size=128),dict(id='sac-standard',lr=.0003,gamma=.99,train_freq=1,gradient_steps=1,learning_starts=10000,batch_size=256),dict(id='sac-conservative',lr=.0001,gamma=.995,train_freq=1,gradient_steps=1,learning_starts=10000,batch_size=256)]
DELTA=130/30
EQUIV=DELTA/2
TCRIT_DF5=2.570581835636314


def plan():return json.loads((ROOT/'PLAN.json').read_text())

def commit():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(p)

def seed(split,*key):
    if split not in SPLITS:raise ValueError('Only registered DEVELOPMENT split identities are allowed')
    raw=json.dumps([plan()['namespace'],split,*key],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**63)

def sparse_masks():return [''.join(map(str,x)) for x in itertools.product((0,1),repeat=6) if sum(x)<=3]

def screening_registry():
    p=plan();rows=[]
    for mask in sparse_masks()+['111111']:
        for rep in range(p['independent_discovery_replicates']):
            rows.append(dict(id=f'discovery-{mask}-r{rep}',phase='discovery',rep=rep,kind='graph',mask=mask,steps=p['discovery_steps'],hp=PPO_COMMON.copy()))
    for kind,candidates in [('ppo',PPO_CANDIDATES),('sac',SAC_CANDIDATES)]:
        for candidate in candidates:
            hp={**PPO_COMMON,**candidate} if kind=='ppo' else candidate.copy()
            for rep in range(p['independent_calibration_replicates']):
                rows.append(dict(id=f'calibration-{candidate["id"]}-r{rep}',phase='calibration',rep=rep,kind=kind,mask=None,steps=p['calibration_steps'],hp=hp))
    return rows

def competition_registry(selected):
    p=plan();roles={
        'S-M0':dict(kind='graph',mask='000000',hp=PPO_COMMON.copy()),
        'S-M2-fixed':dict(kind='graph',mask='011001',hp=PPO_COMMON.copy()),
        'S-M3':dict(kind='graph',mask=selected['selected_support'],hp=PPO_COMMON.copy()),
        'S-M4':dict(kind='graph',mask='111111',hp=PPO_COMMON.copy()),
        'S-M5':dict(kind='ppo',mask=None,hp=selected['ppo_hp']),
        'S-M6':dict(kind='sac',mask=None,hp=selected['sac_hp'])}
    return [dict(id=f'competitive-{role}-r{rep}',role=role,phase='competitive',rep=rep,steps=p['competitive_steps'],**cfg) for role,cfg in roles.items() for rep in range(p['independent_competitive_replicates'])]

def preservation():
    def tree(ref):
        raw=subprocess.check_output(['git','ls-tree','-r','-z',ref])
        return {r.split(b'\t',1)[1]:r.split(b'\t',1)[0] for r in raw.split(b'\0') if r}
    before,after=tree(BASE),tree('HEAD')
    bad=[p.decode() for p,v in before.items() if after.get(p)!=v]
    if bad:raise RuntimeError('Frozen historical Git objects modified: '+repr(bad[:20]))
    return dict(status='PASS',base_commit=BASE,preserved_objects_and_modes=len(before),changed=[])

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,c-h),min(1,c+h)]

def jaccard(a,b):
    aa={i for i,x in enumerate(a) if x=='1'};bb={i for i,x in enumerate(b) if x=='1'}
    return len(aa&bb)/len(aa|bb) if aa|bb else 1.

def stability(winners):
    from collections import Counter
    counts=Counter(winners);n=len(winners)
    pairs=[jaccard(a,b) for a,b in itertools.combinations(winners,2)]
    nulls=[]
    for a,b in itertools.combinations(winners,2):
        lhs=[x for x in sparse_masks() if x.count('1')==a.count('1')]
        rhs=[x for x in sparse_masks() if x.count('1')==b.count('1')]
        nulls.append(sum(jaccard(x,y) for x in lhs for y in rhs)/(len(lhs)*len(rhs)))
    mean=sum(pairs)/len(pairs) if pairs else None
    return {'winners':winners,'counts':dict(counts),'n_independent_training_initializations':n,
            'edges':[{'edge':list(e),'count':sum(w[i]=='1' for w in winners),'frequency':sum(w[i]=='1' for w in winners)/n,'wilson_nominal_95':wilson(sum(w[i]=='1' for w in winners),n)} for i,e in enumerate(EDGES)],
            'pairwise_jaccard':[[jaccard(a,b) for b in winners] for a in winners],
            'mean_jaccard':mean,'conditional_edgecount_null_mean_jaccard':sum(nulls)/len(nulls) if nulls else None,
            'modal_fraction':max(counts.values())/n,
            'engineering_stability_diagnostic':bool(mean is not None and mean>=.6 and max(counts.values())/n>=2/3),
            'scope':'Descriptive engineering thresholds fixed before v1.1; Wilson intervals assume independent training selections and are marginal, not simultaneous. Count-conditioned null is a reference, not a catalogue or symmetry p-value.'}

def interval(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    if len(a)!=6 or not np.isfinite(a).all():raise ValueError('Expected six finite independent competitive bundle means')
    mean=float(a.mean());h=TCRIT_DF5*float(a.std(ddof=1))/math.sqrt(6);lo,hi=mean-h,mean+h
    label='INCONCLUSIVE'
    if hi < -DELTA:label='DEVELOPMENT_PATTERN_IMPROVED'
    elif lo > DELTA:label='DEVELOPMENT_PATTERN_WORSE'
    elif lo>=-EQUIV and hi<=EQUIV:label='DEVELOPMENT_PATTERN_WITHIN_MARGIN'
    return dict(mean=mean,nominal_interval=[lo,hi],by_training_rep=a.tolist(),n=6,classification=label,scope='Exploratory paired t interval df=5, not multiplicity-adjusted or confirmatory')

def self_test():
    r=screening_registry();assert len(r)==276 and len({x['id'] for x in r})==276
    assert sum(x['steps'] for x in r)==43253760
    assert len(sparse_masks())==42 and sparse_masks()[0]=='000000'
    assert len({seed(s,r,i) for s in SPLITS for r in range(6) for i in range(1000)})==len(SPLITS)*6000
    try:seed('final',0)
    except ValueError:pass
    else:raise AssertionError('Final namespace accepted')
    assert jaccard('000000','000000')==1 and jaccard('100000','010000')==0
    s=stability(['100000']*6);assert s['engineering_stability_diagnostic'] and s['edges'][0]['count']==6
    assert interval([-10]*6)['classification']=='DEVELOPMENT_PATTERN_IMPROVED'
    assert interval([0]*6)['classification']=='DEVELOPMENT_PATTERN_WITHIN_MARGIN'
    assert interval([10]*6)['classification']=='DEVELOPMENT_PATTERN_WORSE'
    assert interval([-100,100,-100,100,-100,100])['classification']=='INCONCLUSIVE'
    return dict(status='PASS',screening_fits=len(r),screening_steps=sum(x['steps'] for x in r),future_competitive_fits=36,future_competitive_steps=18874368,full_planned_steps=62128128,development_only=True)
