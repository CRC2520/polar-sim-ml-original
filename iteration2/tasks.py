"""Two declared synthetic physical domains, with distinct dynamics and units."""
from __future__ import annotations
from itertools import product
from time import perf_counter
from copy import deepcopy
import numpy as np
from .controller import CausalAgent,Config,MODES

N=3;M=6;H=5;STEPS=48;CALIBRATION=64
CELLS=list(product(('water','thermal'),('weak','strong'),('fixed','change')))
DEVELOPMENT_SEEDS=list(range(971001,971007))
FINAL_SEEDS=list(range(981001,981025))
CAPABILITY_SEEDS=list(range(991001,991025))


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def task(seed,cell):
    if tuple(cell) not in CELLS:raise ValueError('unknown task cell')
    domain,coupling,regime=cell;rng=np.random.default_rng(seed)
    F=np.eye(N)*(.91 if domain=='water' else .92)
    if domain=='thermal':
        F=np.eye(N)*.90+(.035/2)*(np.ones((N,N))-np.eye(N))
    a=.5 if domain=='water' else .7
    B=np.zeros((2,N,M));cross=.006 if coupling=='weak' else .055
    for i in range(N):
        B[0,i,2*i]=rng.uniform(.1,.14);B[0,i,2*i+1]=-rng.uniform(.1,.14)
        for j in range(N):
            if i!=j:
                sign=-1 if domain=='water' else 1
                B[0,i,2*j]=sign*cross*rng.uniform(.7,1.3)
                B[0,i,2*j+1]=-sign*cross*rng.uniform(.7,1.3)
    B[1]=B[0]*rng.uniform(.55,1.25,(1,M)) if regime=='change' else B[0]
    base=np.array([[.72,.28,.62],[.28,.72,.38],[.62,.38,.72]])+rng.uniform(-.025,.025,(3,N))
    priority=np.array([[4.,1.,2.],[1.,4.,2.],[2.,1.,4.]])
    schedule=np.array([0]*12+[1]*12+[2]*12+[0]*(12+H))
    targets=base[schedule];priorities=priority[schedule]
    total=STEPS+H
    if domain=='water':drift=np.full((total,N),.0405)
    else:
        ambient=.45+.12*np.sin(np.arange(total)*.09)
        drift=(1-F.sum(axis=1))[None,:]*ambient[:,None]
    budgets=np.array([1.5 if t%12<8 else .7 for t in range(total)],float)
    noise=rng.normal(0,.0015,(STEPS,N));cal_noise=rng.normal(0,.0015,(CALIBRATION,N))
    masks=np.ones((STEPS+1,N),bool)
    for t in range(1,STEPS+1):
        if 4<=t%12<=8:masks[t,(np.arange(N)+seed)%2==0]=False
    return {'seed':int(seed),'cell':list(cell),'F':F,'B':B,'a':a,'targets':targets,'priorities':priorities,'base_goals':base,
            'drift':drift,'budgets':budgets,'noise':noise,'masks':masks,'costs':rng.uniform(.9,1.1,M),
            'cal_actions':rng.uniform(0,.8,(CALIBRATION,M)),'cal_noise':cal_noise}


def transition(tk,x,held,u,t):
    h=tk['a']*held+(1-tk['a'])*np.asarray(u)
    B=tk['B'][int(t>=24)]
    raw=tk['F']@x+tk['drift'][t]+B@h+tk['noise'][t]
    return np.clip(raw,0,1),h,(raw>0)&(raw<1)


def calibrate(agent,tk):
    x=np.full(N,.5);h=np.zeros(M);records=[]
    for t,u in enumerate(tk['cal_actions']):
        h=tk['a']*h+(1-tk['a'])*u
        raw=tk['F']@x+tk['drift'][0]+tk['B'][0]@h+tk['cal_noise'][t]
        y=np.clip(raw,0,1);valid=(raw>0)&(raw<1)
        agent.model.update(h,y-tk['F']@x-tk['drift'][0],valid)
        records.append({'state':x.tolist(),'executed':u.tolist(),'actuator':h.tolist(),'effect':y.tolist(),'usable':valid.tolist()});x=y
    # The evaluation episode is physically reset; only learned coefficients carry over.
    return records


def observation(tk,t,state):
    mask=tk['masks'][t];episode=t//12;idx=[0,1,2,0][episode];cue=['A','B','C','A'][episode]
    allowed=np.ones(M,bool)
    if 24<=t<30:allowed[0]=False
    protect=np.zeros(N,bool)
    if 12<=t<18:protect[1]=True
    return {'state':np.where(mask,state,0.),'observed':mask.copy(),'cue':cue,
            'goal':tk['base_goals'][idx] if t<36 else None,
            'priorities':tk['priorities'][t].copy(),
            'forecast':tk['targets'][t:t+H].copy() if t<36 else None,
            'F':tk['F'].copy(),'actuator_retention':tk['a'],'drift_forecast':tk['drift'][t:t+H].copy(),
            'budget_forecast':tk['budgets'][t:t+H].copy(),'costs':tk['costs'].copy(),'allowed':allowed,
            'normative':{'protected':protect,'approved_drain':False}}


def summarize(frames):
    if [f['t'] for f in frames]!=list(range(STEPS)):raise ValueError('incomplete trajectory')
    loss=np.array([f['loss'] for f in frames]);indices=[i for s in (12,24,36) for i in range(s-3,s+3)]
    hidden=[i for i,f in enumerate(frames) if not all(f['observation']['observed'])]
    return {'loss':float(loss.mean()),'switch_loss':float(loss[indices].mean()),'hidden_loss':float(loss[hidden].mean()),
            'return_loss':float(loss[36:].mean()),'cost':float(np.mean([f['cost'] for f in frames])),
            'violations':sum(f['violation'] for f in frames),'intervention_fraction':float(np.mean([f['intervened'] for f in frames])),
            'runtime_seconds':sum(f['runtime_seconds'] for f in frames),
            'state_clip_fraction':float(np.mean([not all(f['physical_valid']) for f in frames]))}


def run_trial(seed,cell,mode,*,monitor=None,meta=None,retain=True):
    tk=task(seed,cell);cfg=Config(mode=mode,**(meta or {}));agent=CausalAgent(cfg,monitor)
    calibration=calibrate(agent,tk);initial=agent.snapshot();state=np.full(N,.5);held=np.zeros(M);frames=[]
    for t in range(STEPS):
        o=observation(tk,t,state);start=perf_counter();u=agent.act(o);acttime=perf_counter()-start
        y,nextheld,valid=transition(tk,state,held,u,t)
        mask=tk['masks'][t+1]
        feedback={'state':np.where(mask,y,0.),'observed':mask,'valid_transition':valid&mask,'token':deepcopy(agent.pending['token'])}
        start=perf_counter();agent.learn(feedback);elapsed=acttime+perf_counter()-start
        w=tk['priorities'][t];target=tk['targets'][t];loss=float(w@((y-target)**2)/w.sum())
        allowed=np.array(agent.trace['allowed'],bool);minimum=np.array(agent.trace['minimum']);cost=float(tk['costs']@u/tk['costs'].sum())
        violation=bool(np.any(u<minimum-1e-10) or np.any(u>1+1e-10) or np.any(abs(u[~allowed])>1e-10) or tk['costs']@u>tk['budgets'][t]+1e-10)
        f={'t':t,'observation':jsonable(o),'action':u.tolist(),'effect':y.tolist(),'physical_valid':valid.tolist(),
           'feedback':jsonable(feedback),'evaluation_target':target.tolist(),'evaluation_priority':w.tolist(),'loss':loss,'cost':cost,
           'violation':violation,'intervened':agent.trace['meta_decision'] not in ('base','inactive'),
           'score':agent.trace['base_score_before_feedback'],'probability':agent.trace['base_success_probability'],
           'success':bool(loss<=.003),'runtime_seconds':elapsed}
        if retain:f['controller']=jsonable(agent.trace)
        frames.append(f);state=y;held=nextheld
    return {'seed':int(seed),'cell':list(cell),'mode':mode,'meta_config':meta or {},'calibration':calibration if retain else None,
            'initial':jsonable(initial) if retain else None,'frames':frames,'metrics':summarize(frames),
            'final':jsonable(agent.snapshot()) if retain else None}
