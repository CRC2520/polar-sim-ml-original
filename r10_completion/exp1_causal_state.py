"""R10 E1 — feedback-stable causal-state discovery."""
from __future__ import annotations
import numpy as np
from .core import sigmoid, pca_fit, pca_project, brier
from .config import E1_DIMS, STEPS_E1_TRAIN, STEPS_E1_VALID, STEPS_E1_TEST

def _tape(seed, steps):
    rng=np.random.default_rng(seed)
    world=np.empty(steps,dtype=float); rel=np.empty(steps,dtype=int)
    w=1. if rng.random()<.5 else -1.; r=int(rng.random()<.5)
    for t in range(steps):
        if rng.random()<.025: w=-w
        if rng.random()<.04: r=1-r
        world[t]=w; rel[t]=r
    return dict(
        world=world, rel=rel,
        task=np.where(rng.random(steps)<.5,-1.,1.),
        route_noise=rng.normal(0,.12,steps),
        e1_small=rng.normal(0,.22,steps), e1_large=rng.normal(0,1.0,steps),
        e2_small=rng.normal(0,.22,steps), e2_large=rng.normal(0,1.0,steps),
        self_noise=rng.normal(0,.05,steps),
        world_event=(rng.random(steps)<.035)*rng.choice([-1.,1.],steps),
        event_noise=rng.normal(0,.08,steps),
        competence_noise=rng.normal(0,.006,steps),
    )

def _derived(h):
    h=np.asarray(h,float).copy()
    h[8]=h[0]*h[1]
    h[9]=h[5]*(1.-h[3])
    h[10]=h[4]*(1.-h[4])
    h[11]=np.tanh(h[0]+.5*h[5]-.5*h[7])
    return h

def initial_state():
    h=np.zeros(12,float)
    h[1]=.7; h[2]=.5; h[3]=1.; h[4]=.5; h[7]=.5
    return _derived(h)

def readout(h):
    h=np.asarray(h,float)
    margin=1.25*h[0]*h[6]+.65*h[5]-.75*(.5-h[1])+.45*(h[4]-.5)+.35*(.5-h[7])
    action=int(margin>0)
    confidence=float(sigmoid(3.2*abs(margin)-1.8*h[7]))
    return action,float(h[4]),confidence,float(h[5]),float(margin)

def _update(h, *, world, rel, task, route_noise, e1s,e1l,e2s,e2l,
            selfcue,event,prev_action):
    h=np.asarray(h,float).copy()
    route_cue=np.clip(rel+route_noise,0,1)
    h[4]=np.clip(.82*h[4]+.18*sigmoid(7*(route_cue-.5)),0,1)
    e1=world+(e1s if rel else e1l)
    e2=world+(e2l if rel else e2s)
    evidence=h[4]*e1+(1-h[4])*e2
    h[0]=np.clip(.78*h[0]+.22*np.tanh(evidence),-1,1)
    h[1]=np.clip(.85*h[1]+.15*selfcue,0,1)
    self_signal=float(prev_action==1)
    src_obs=sigmoid(5*(abs(event)*self_signal-.10))
    h[2]=np.clip(.85*h[2]+.15*src_obs,0,1)
    h[3]=0. if prev_action==1 else min(1.,h[3]+.035)
    if abs(event)>.12:
        h[5]=np.clip(.78*h[5]+.22*np.tanh(2*event),-1,1)
    h[6]=float(task)
    disagreement=min(1.,abs(e1-e2)/3.)
    h[7]=np.clip(.85*h[7]+.15*disagreement,0,1)
    return _derived(h)

def rollout(seed,steps,projector=None):
    tape=_tape(seed,steps)
    h=initial_state(); competence=.72; prev_action=0
    states=[]; actions=[]; routes=[]; conf=[]; memories=[]; rewards=[]; labels=[]
    prev_event=0.
    for t in range(steps):
        event=float(tape['world_event'][t] + (.18 if prev_action==1 else 0.) + tape['event_noise'][t])
        selfcue=np.clip(competence+tape['self_noise'][t],0,1)
        h=_update(h, world=tape['world'][t], rel=tape['rel'][t], task=tape['task'][t],
                  route_noise=tape['route_noise'][t],
                  e1s=tape['e1_small'][t], e1l=tape['e1_large'][t],
                  e2s=tape['e2_small'][t], e2l=tape['e2_large'][t],
                  selfcue=selfcue,event=event,prev_action=prev_action)
        if projector is not None:
            h=np.asarray(projector(h[None]))[0]
            h[:8]=np.r_[np.clip(h[0],-1,1),np.clip(h[1:5],0,1),
                        np.clip(h[5],-1,1),np.clip(h[6],-1,1),np.clip(h[7],0,1)]
            h=_derived(h)
        a,rp,cf,mem,_=readout(h)
        optimal=int(tape['world'][t]*tape['task'][t] + .35*(competence-.5) > 0)
        reward=1.0 if a==optimal else 0.0
        if a==1 and competence<.35: reward=max(0.,reward-.15)
        competence=np.clip(.985*competence+.018*(1-a)-.028*a+tape['competence_noise'][t],.1,1.)
        states.append(h.copy()); actions.append(a); routes.append(rp); conf.append(cf)
        memories.append(mem); rewards.append(reward); labels.append(tape['rel'][t])
        prev_action=a; prev_event=event
    return {k:np.asarray(v) for k,v in dict(states=states,actions=actions,routes=routes,
        confidence=conf,memory=memories,reward=rewards,route_label=labels).items()}

def _offline_metrics(run,mean,basis):
    rec=pca_project(run['states'],mean,basis)
    ro=np.array([readout(x) for x in run['states']],dtype=float)
    rr=np.array([readout(x) for x in rec],dtype=float)
    return dict(
        action_agreement=float(np.mean(ro[:,0]==rr[:,0])),
        routing_mae=float(np.mean(abs(ro[:,1]-rr[:,1]))),
        confidence_mae=float(np.mean(abs(ro[:,2]-rr[:,2]))),
        memory_mae=float(np.mean(abs(ro[:,3]-rr[:,3]))),
    )

def run(seed):
    train=rollout(seed+11,STEPS_E1_TRAIN)
    valid=rollout(seed+22,STEPS_E1_VALID)
    candidates={}
    chosen=None
    for d in E1_DIMS:
        mean,basis=pca_fit(train['states'],d)
        m=_offline_metrics(valid,mean,basis)
        candidates[d]=m
        if chosen is None and m['action_agreement']>=.98 and m['routing_mae']<=.05 and m['confidence_mae']<=.05 and m['memory_mae']<=.05:
            chosen=(d,mean,basis)
    if chosen is None:
        d=E1_DIMS[-1]; mean,basis=pca_fit(train['states'],d); chosen=(d,mean,basis)
    d,mean,basis=chosen
    projector=lambda x:pca_project(x,mean,basis)
    full=rollout(seed+33,STEPS_E1_TEST)
    comp=rollout(seed+33,STEPS_E1_TEST,projector=projector)
    correctness_full=(full['reward']>.5).astype(float)
    correctness_comp=(comp['reward']>.5).astype(float)
    route_full=np.mean((full['routes']>.5)==full['route_label'])
    route_comp=np.mean((comp['routes']>.5)==comp['route_label'])
    metrics=dict(
        selected_dimension=int(d),
        action_agreement=float(np.mean(full['actions']==comp['actions'])),
        routing_accuracy_full=float(route_full),
        routing_accuracy_compressed=float(route_comp),
        routing_accuracy_loss=float(route_full-route_comp),
        brier_full=brier(correctness_full,full['confidence']),
        brier_compressed=brier(correctness_comp,comp['confidence']),
        confidence_brier_increase=float(brier(correctness_comp,comp['confidence'])-brier(correctness_full,full['confidence'])),
        memory_mae=float(np.mean(abs(full['memory']-comp['memory']))),
        reward_full=float(full['reward'].mean()),
        reward_compressed=float(comp['reward'].mean()),
        reward_loss=float(full['reward'].mean()-comp['reward'].mean()),
    )
    passed=(d<=8 and metrics['action_agreement']>=.95 and metrics['routing_accuracy_loss']<=.03
            and metrics['confidence_brier_increase']<=.01 and metrics['memory_mae']<=.08
            and metrics['reward_loss']<=.03)
    return dict(seed=int(seed),selected_dimension=int(d),validation={str(k):v for k,v in candidates.items()},
                confirmatory=metrics,pass_strong=bool(passed))
