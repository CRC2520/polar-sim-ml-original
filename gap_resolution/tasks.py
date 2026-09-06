"""Dynamical tanks: observed state and commands; inaccessible true effect matrix."""
from copy import deepcopy
import numpy as np
from integrated_polar import IntegratedAgent,AgentConfig

MODES=['full','diagonal_plan','myopic','no_priorities','no_memory','no_broadcast','generic_equivalent','no_internal','no_learning','reactive','zero']
CELLS=[(family,budget) for family in ('fixed','change') for budget in (.6,2.)]
DEV_SEEDS=list(range(921001,921007));FINAL_SEEDS=list(range(931001,931025))
H=3;STEPS=48;N=4;M=8


def make_task(seed,cell):
    family,budget=cell
    if tuple(cell) not in CELLS:raise ValueError('invalid task cell')
    rng=np.random.default_rng(int(seed))
    B=np.zeros((2,N,M))
    for s in range(2):
        for i in range(N):
            B[s,i,2*i]=rng.uniform(.09,.14);B[s,i,2*i+1]=-rng.uniform(.09,.14)
            for j in range(N):
                if i!=j:
                    B[s,i,2*j]=rng.uniform(-.025,.025);B[s,i,2*j+1]=rng.uniform(-.025,.025)
    if family=='fixed':B[1]=B[0]
    targets=rng.uniform(.15,.8,(3,N));weights=rng.choice([1.,2.,4.],(3,N))
    # Ensure priority and forgotten-cue tests are not all symmetric by construction.
    targets[0]=np.array([.75,.2,.7,.25])+rng.uniform(-.03,.03,N)
    weights[0]=[4,1,2,1]
    calibration=rng.uniform(0,.65,(48,M))
    noise=rng.normal(0,.001,(STEPS,N));calnoise=rng.normal(0,.001,(48,N))
    return {'seed':int(seed),'cell':list(cell),'B':B,'targets':targets,'weights':weights,'budget':float(budget),
            'calibration':calibration,'noise':noise,'calnoise':calnoise,'costs':rng.uniform(.9,1.1,M)}


def physical(task,state,action,t,*,calibration=False):
    B=task['B'][0 if calibration or t<24 else 1]
    noise=task['calnoise'][t] if calibration else task['noise'][t]
    raw=.9*state+.03+B@action+noise
    return np.clip(raw,0,1),(raw>0)&(raw<1)


def calibrate_model(agent,task):
    state=np.full(N,.35);records=[]
    for t,u in enumerate(task['calibration']):
        y,valid=physical(task,state,u,t,calibration=True)
        if agent.config.mode!='no_learning':agent.model.update(u,y-.9*state-.03,valid)
        records.append({'state':state.tolist(),'own_action':u.tolist(),'next_state':y.tolist(),'valid':valid.tolist()})
        state=y
    return records


def observation(task,t,state):
    episode=t//12;phase=[0,1,2,0][episode];cue=['A','B','C','A'][episode]
    allowed=np.ones(M,bool)
    if 24<=t<36:allowed[0]=False
    protected=np.zeros(N,bool)
    if 12<=t<24:protected[1]=True
    obs={'state':state.copy(),'cue':cue,'rho':.9,'drift':np.full(N,.03),'costs':task['costs'].copy(),
         'budget':task['budget'],'allowed':allowed,'normative':{'protected':protected.tolist(),'approved_drain':False}}
    # Convert to bool ndarray for the typed policy interface.
    obs['normative']['protected']=protected
    if episode<3:
        obs['event']={'targets':np.tile(task['targets'][phase],(H,1)),
                      'priorities':np.tile(task['weights'][phase],(H,1)),
                      'identity':f'{cue}-{t}','version':t+1}
    return obs,task['targets'][phase],task['weights'][phase]


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.float64,np.float32)):return float(x)
    if isinstance(x,(np.int64,np.int32)):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x


def compact_trace(trace):
    tr=deepcopy(trace)
    for name in ['planning','intended_planning']:
        if name in tr:
            for key in ['jacobian','directed_messages','K_effective','W_effective']:
                tr[name].pop(key,None)
    return jsonable(tr)


def run_trial(seed,cell,mode,monitor=None,*,retain=True):
    task=make_task(seed,cell);agent=IntegratedAgent(AgentConfig(tanks=N,horizon=H,mode=mode),monitor)
    calibration=calibrate_model(agent,task);initial=agent.snapshot();state=np.full(N,.35);frames=[]
    for t in range(STEPS):
        obs,target,weight=observation(task,t,state)
        u=agent.act(obs);token=deepcopy(agent.pending['token']);y,valid=physical(task,state,u,t)
        agent.learn({'state':y,'valid_transition':valid,'token':token})
        loss=float(np.sum(weight*(y-target)**2)/weight.sum())
        cost=float(task['costs']@u/task['costs'].sum())
        mask=np.array(agent.trace['allowed'])
        violation=bool(np.any(u< -1e-10) or np.any(u>1+1e-10) or np.any(abs(u[~mask])>1e-10) or task['costs']@u>task['budget']+1e-10)
        frame={'t':t,'observation':jsonable(obs),'evaluation_target':target.tolist(),'evaluation_priority':weight.tolist(),
               'action':u.tolist(),'effect':y.tolist(),'valid_transition':valid.tolist(),'loss':loss,'cost':cost,'violation':violation,
               'pre_feedback_score':agent.trace['risk_score_before_feedback'],
               'probability':agent.trace['success_probability_before_feedback'],'success':bool(loss<=.003)}
        if retain:frame['controller']=compact_trace(agent.trace)
        frames.append(frame);state=y
    losses=np.array([f['loss'] for f in frames]);switch=[i for start in (12,24,36) for i in range(start,start+4)]
    metrics={'loss':float(losses.mean()),'post_change':float(losses[switch].mean()),'return_loss':float(losses[36:].mean()),
             'cost':float(np.mean([f['cost'] for f in frames])),'violations':sum(f['violation'] for f in frames),
             'max_internal_action_separation':float(np.max(np.abs(agent.p-agent.previous_action))),
             'off_pair_coefficient_norm':float(np.linalg.norm(np.where(__import__('integrated_polar.numerics',fromlist=['own_pair_mask']).own_pair_mask(N),0,agent.model.B)))}
    return {'seed':seed,'cell':list(cell),'mode':mode,'calibration':calibration if retain else None,
            'initial':initial if retain else None,'metrics':metrics,'frames':frames,'final':jsonable(agent.snapshot()) if retain else None}
