"""Scientific task and controller definitions for Study 3; freeze before final."""
from itertools import product
from time import perf_counter
import numpy as np
from polar.model import ContextualPolarModel, ModelConfig
from network_tension import NetworkTensionModel

STEPS = 64
SHAPE = (1, 4, 2)
DEV_SEEDS = list(range(801001, 801007))
FINAL_SEEDS = list(range(901001, 901041))
CELLS = list(product(('linear', 'congestion'), ('aligned', 'rewired', 'absent'), ('ample', 'scarce')))
MODES = ['full','no_K','no_W','no_WK','retuned_zero','rewired','reverse',
         'generic_nonlinear','generic_equivalent','signed_intensity','gradient','zero_negative']
TUNED = ['full','rewired','reverse','generic_nonlinear','retuned_zero','gradient']
SOURCES = {'aligned': np.array([3,0,1,2]), 'rewired': np.array([2,0,3,1]),
           'reverse': np.array([1,2,3,0])}


def candidates(mode):
    gains = (-.2,0.,.2) if mode not in ('retuned_zero','gradient') else (0.,)
    return [dict(eta=e,a=a,b=b) for e,a,b in product((.35,.65,1.),gains,gains)]


def templates(topology='aligned'):
    src=SOURCES[topology]; W=np.zeros((8,8));K=np.zeros((8,4))
    for i,j in enumerate(src):
        for s in range(2):
            W[2*i+s,2*j+s]=1.; W[2*i+s,2*j+1-s]=-.6; K[2*i+s,j]=1.
    return W,K


class GenericController(ContextualPolarModel):
    """Same inputs and scaffold; independent feature/gradient proposal implementation."""
    def __init__(self,config,W,K,mode):
        super().__init__(config); self.W=W;self.K=K;self.mode=mode;self.chi=np.zeros((1,4))
        self.feature_trace={}

    def set_incompatibility(self,chi):
        value=np.asarray(chi,float)
        if self._pending_feedback:raise RuntimeError('pending feedback')
        if value.shape!=(1,4) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
            raise ValueError('invalid incompatibility')
        self.chi=value.copy()

    def propose_action(self,effective_target,weights,horizon):
        p=self.q; G=self._planning_gain; eta=np.clip(self.config.step_size*horizon,0,1)
        base=super().propose_action(effective_target,weights,horizon)
        if self.mode=='gradient':
            out=p+eta*G*(effective_target-G*p)/max(1.,float(np.max(G*G)))
            self.feature_trace={'rule':'local_gradient'}
            return out
        if self.mode=='zero_negative':
            self.feature_trace={'rule':'zero_negative'}
            return np.zeros_like(p)
        v=np.clip(effective_target/G,0,1)
        if self.mode=='generic_nonlinear':
            features=np.mean(np.tanh(2*(v-p)),axis=-1)+self.chi*np.mean(p,axis=-1)
        else:
            # Independent unnamed algebraic control: same function, no claimed novelty.
            features=np.mean(np.abs(v-p),axis=-1)+self.chi*np.prod(p,axis=-1)
        state=(self.W@p.ravel()).reshape(SHAPE)
        feature=(self.K@features.ravel()).reshape(SHAPE)
        self.feature_trace={'features':features.tolist(),'state_term':state.tolist(),
                            'feature_term':feature.tolist(),'base':base.tolist()}
        return base+eta*(state+feature)


def build(mode,selected,seed):
    if mode not in MODES:raise ValueError('unknown mode')
    source=mode if mode in TUNED else 'full'
    pars=dict(selected[source]);a,b=pars['a'],pars['b']
    if mode in ('no_K','no_WK'):b=0.
    if mode in ('no_W','no_WK'):a=0.
    if mode in ('retuned_zero','gradient','zero_negative'):a=b=0.
    top=mode if mode in ('rewired','reverse') else 'aligned'
    W,K=templates(top); W=a*W; K=b*K
    cfg=ModelConfig(agents=1,types=4,seed=int(seed),step_size=pars['eta'],
                    representation='signed_intensity' if mode=='signed_intensity' else 'dual_pole')
    if mode in ('generic_nonlinear','generic_equivalent','gradient','zero_negative'):
        m=GenericController(cfg,W,K,mode)
    else:
        m=NetworkTensionModel(cfg,state_coupling=W,tension_coupling=K)
    m.study_parameters={'selected_from':source,'eta':pars['eta'],'a':a,'b':b,'topology':top,
                        'W':W.tolist(),'K':K.tolist(),
                        'independently_tuned_scalars':1 if mode in ('retuned_zero','gradient') else 3,
                        'nominal_network_coefficients':0 if mode in ('retuned_zero','gradient','zero_negative') else 24,
                        'nonzero_coefficients':int(np.count_nonzero(W)+np.count_nonzero(K))}
    return m


def environment(seed,cell):
    family,topology,resource=cell
    if tuple(cell) not in CELLS:raise ValueError('unknown cell')
    rng=np.random.default_rng(int(seed))
    patterns=rng.uniform(.15,.75,(3,)+SHAPE)
    phases=np.repeat([0,1,2,0],16)
    target=np.clip(patterns[phases]+.03*np.sin(np.arange(STEPS)*.4)[:,None,None,None],0,1)
    costs0=rng.uniform(.8,1.2,(4,)+SHAPE)
    weights0=rng.uniform(.5,1.5,(4,)+SHAPE)
    chi0=rng.choice([0.,.5,1.],(4,1,4))
    costs=np.repeat(costs0,16,axis=0);weights=np.repeat(weights0,16,axis=0)
    chi=np.repeat(chi0,16,axis=0)
    gains=np.repeat(rng.uniform(.8,1.2,(2,)+SHAPE),32,axis=0)
    noise=rng.normal(0,.005,(STEPS,)+SHAPE)
    allowed=np.ones((STEPS,)+SHAPE,bool)
    allowed.reshape(STEPS,8)[32:48,int(seed)%8]=False
    budget=costs.reshape(STEPS,8).sum(axis=1)*(1 if resource=='ample' else .4)
    return dict(seed=int(seed),cell=list(cell),target=target,costs=costs,weights=weights,chi=chi,
                gains=gains,noise=noise,allowed=allowed,budget=budget)


def transition(env,t,u):
    u=np.asarray(u,float)
    if u.shape!=SHAPE or not np.isfinite(u).all():raise ValueError('invalid action')
    y=env['gains'][t]*u
    family,topology,_=env['cell']
    if topology!='absent':
        src=SOURCES[topology]; sent=u[:,src,:]
        y=y+.22*(sent-.6*sent[...,::-1])
        if family=='congestion':
            y=y-.35*(env['chi'][t][:,src]*np.prod(sent,axis=-1))[...,None]
    return np.clip(y+env['noise'][t],0,1)


def metrics(frames):
    if len(frames)!=STEPS or [x['t'] for x in frames]!=list(range(STEPS)):
        raise ValueError('incomplete or unordered trajectory')
    losses=[];costs=[];violations=0;saturation=[]
    for f in frames:
        u=np.asarray(f['action'],float);y=np.asarray(f['effect'],float)
        o=f['observation'];d=np.asarray(o['target']);w=np.asarray(o['weights']);c=np.asarray(o['costs']);allowed=np.asarray(o['allowed'],bool)
        if any(a.shape!=SHAPE or not np.isfinite(a).all() for a in (u,y,d,w,c)):
            raise ValueError('invalid trajectory arrays')
        losses.append(float(np.sum(w*(y-d)**2)/np.sum(w)))
        costs.append(float(np.sum(c*u)/np.sum(c)))
        violations+=int(np.any(u< -1e-10) or np.any(u>1+1e-10) or np.any(np.abs(u[~allowed])>1e-10) or np.sum(c*u)>o['budget']+1e-10)
        saturation.append(float(np.mean((u<=1e-10)|(u>=1-1e-10))))
    switch=[t for start in (16,32,48) for t in range(start,start+4)]
    return dict(loss=float(np.mean(losses)),cost=float(np.mean(costs)),
                post_switch_loss=float(np.mean(np.array(losses)[switch])),return_loss=float(np.mean(losses[48:])),
                saturation=float(np.mean(saturation)),violations=violations)


def trial(mode,selected,env,retain=True):
    m=build(mode,selected,env['seed']);frames=[];elapsed=0.;initial=m.snapshot()
    for t in range(STEPS):
        o={k:env[k][t] for k in ('target','weights','costs','allowed')}
        o.update(budget=float(env['budget'][t]),cue=int(t//16),observed=True)
        start=perf_counter();m.set_incompatibility(env['chi'][t]);u=m.act(o)
        elapsed+=perf_counter()-start
        y=transition(env,t,u)
        start=perf_counter();m.learn({'effect':y});elapsed+=perf_counter()-start
        observed={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o.items()}
        observed['chi']=env['chi'][t].tolist()
        f={'t':t,'observation':observed,'action':u.tolist(),'effect':y.tolist()}
        if retain:
            tr=m.last_trace
            net=tr.get('network',getattr(m,'feature_trace',{}))
            net={k:v for k,v in net.items() if k not in ('state_coupling','tension_coupling','state_edges','tension_edges')}
            f['internal']={k:tr[k] for k in ('q_before','proposal','planning_gain','effective_target')}
            f['internal']['network_or_feature']=net
        frames.append(f);m.traces.clear()
    result=metrics(frames);result['runtime_seconds']=elapsed
    return {'mode':mode,'seed':env['seed'],'cell':env['cell'],'parameters':m.study_parameters,
            'metrics':result,'initial_state':initial,'final_state':m.snapshot(),'frames':frames}
