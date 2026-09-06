"""One integrated lifecycle with distinct belief, commitment, intention and action."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,asdict
import numpy as np
from integrated_polar.numerics import CoupledRLS,array,own_pair_mask
from integrated_polar.layers import Content,ContentWorkspace,GatedMemory,MotivationalGoals,NormativePolicy,digest
from .planning import plan,operators,project_rows,REGULARIZATION

MODES=('full','diagonal','myopic','no_forecast','observer_hold','no_intention','no_memory',
       'no_priorities','utility_meta','legacy_review','generic_equivalent','zero')


@dataclass(frozen=True)
class Config:
    n:int=3
    horizon:int=5
    mode:str='full'
    actor:str='controller'
    beta:float=.25
    margin:float=.00001
    max_interventions:int=12
    def __post_init__(self):
        if self.n<1 or self.horizon<1 or self.mode not in MODES or self.beta<0 or self.margin<0 or self.max_interventions<0:raise ValueError('invalid controller configuration')


class CausalAgent:
    """Reuses original content/memory/norm interfaces; old studies remain unchanged.

    A state estimator is not the polar continuity state p. Their lesions are
    separate, so any benefit of state prediction cannot be credited to p.
    """
    def __init__(self,config=None,monitor=None):
        self.config=config or Config();self.n=self.config.n;self.m=2*self.n
        initial=np.zeros((self.n,self.m))
        for i in range(self.n):initial[i,2*i:2*i+2]=[.1,-.1]
        self.model=CoupledRLS(self.n,self.m,initial=initial,forgetting=.99,ridge=.005)
        self.C=ContentWorkspace();self.U=GatedMemory();self.goals=MotivationalGoals(self.n,self.config.horizon);self.E=NormativePolicy()
        self.p=np.zeros(self.m);self.intention=np.zeros(self.m);self.action=np.zeros(self.m);self.held=np.zeros(self.m)
        self.belief=None;self.last_seen=None;self.age=np.zeros(self.n,int);self.step=0;self.pending=None;self.trace={}
        self.monitor=monitor;self.interventions=0;self.last_intervention=-3
    def _targets(self,o):
        cue=o['cue'];H=self.config.horizon;candidates=[]
        if o.get('goal') is not None:
            goal=array(o['goal'],(self.n,),'goal',0,1);w=array(o['priorities'],(self.n,),'priorities',1e-12)
            candidates.append(Content(f'{cue}-{self.step}',cue,tuple(map(tuple,np.tile(goal,(H,1)))),
                                      tuple(map(tuple,np.tile(w,(H,1)))),'operator',2.,self.step+1))
        else:
            recalled=self.U.recall(cue,gate=self.config.mode!='no_memory' and o.get('memory_gate',True))
            if recalled is not None:candidates.append(recalled)
        # Genuine bounded competition: a valid lower-salience maintenance proposal.
        maintenance=np.full((H,self.n),.5)
        candidates.append(Content(f'maintenance-{self.step}',cue,tuple(map(tuple,maintenance)),
                                  tuple(map(tuple,np.ones_like(maintenance))),'operator',.1,self.step+1))
        # Invalid provenance must never override an accepted control source.
        candidates.append(Content('untrusted-distractor',cue,tuple(map(tuple,1-maintenance)),
                                  tuple(map(tuple,np.ones_like(maintenance))),'external-unverified',100.,self.step+1))
        delivered=self.C.broadcast(candidates)
        if self.config.mode!='no_memory' and o.get('goal') is not None:self.U.store(delivered['memory'])
        r,w=self.goals.consume(delivered['planner'],cue)
        if o.get('forecast') is not None:
            # Current and future supplied targets are inputs, never private evaluator targets.
            supplied=array(o['forecast'],(H,self.n),'goal forecast',0,1)
            if self.config.mode!='no_forecast':r=supplied
        if self.config.mode=='no_priorities':w=np.ones_like(w)
        return r,w,delivered
    def act(self,o):
        if self.pending is not None:raise RuntimeError('feedback required before next own action')
        if not isinstance(o,dict) or not isinstance(o.get('cue'),str):raise ValueError('typed observation/cue required')
        mask=np.asarray(o['observed']);raw=array(o['state'],(self.n,),'sensor packet')
        if mask.dtype!=np.bool_ or mask.shape!=(self.n,):raise ValueError('sensor mask')
        if self.belief is None:
            if not mask.all():raise ValueError('first state must be fully observed')
            self.belief=raw.copy();self.last_seen=raw.copy()
        self.last_seen=np.where(mask,raw,self.last_seen)
        estimated=self.last_seen if self.config.mode=='observer_hold' else self.belief
        y=np.where(mask,raw,estimated);self.age=np.where(mask,0,self.age+1)
        F=array(o['F'],(self.n,self.n),'public autonomous dynamics');a=float(o['actuator_retention'])
        if not 0<=a<1:raise ValueError('actuator retention')
        D=array(o['drift_forecast'],(self.config.horizon,self.n),'public disturbance forecast')
        costs=array(o['costs'],(self.m,),'costs',1e-12);bud=array(o['budget_forecast'],(self.config.horizon,),'resource schedule',0)
        permissions=np.asarray(o['allowed'])
        if permissions.dtype!=np.bool_ or permissions.shape!=(self.m,):raise ValueError('permissions')
        allowed,lower,ethics=self.E.compile(self.n,y,o.get('normative',{}),permissions,float(bud[0]),costs)
        self.U.tick();r,w,delivered=self._targets(o)
        H=1 if self.config.mode=='myopic' else self.config.horizon;r=r[:H];w=w[:H];D=D[:H];bud=bud[:H]
        if self.config.mode=='no_intention':self.p.fill(0.)
        p_before=self.p.copy();B=self.model.B.copy()
        useB=np.where(own_pair_mask(self.n),B,0.) if self.config.mode=='diagonal' else B
        if ethics['halt']:
            uplan=np.zeros((H,self.m));planning={'plan':uplan.tolist(),'tracking_objective':0.,'halt':True};candidates=[]
        else:
            uplan,planning=plan(F,useB,a,D,y,self.held,r,w,self.p,costs,bud,allowed,lower=lower,
                               generic=self.config.mode=='generic_equivalent')
            candidates=[('base',uplan)]
        # Independent one-step unconstrained proposal; never directly executed.
        be=(1-a)*B;baseline=F@y+D[0]+a*B@self.held;qw=w[0]/w[0].sum()
        wish=np.linalg.solve((be.T*qw)@be+.001*np.eye(self.m),be.T@(qw*(r[0]-baseline))+.001*self.p)
        self.intention=np.clip(wish,0,1)
        prediction=F@y+D[0]+B@(a*self.held+(1-a)*uplan[0])
        score=float(w[0]@((prediction-r[0])**2)/w[0].sum()+np.mean(self.model.predictive_variance(a*self.held+(1-a)*uplan[0])))
        probability=None if self.monitor is None else self.monitor.probability(score)
        decision='base';candidate_scores={};benefit=0.
        if self.config.mode=='utility_meta' and not ethics['halt']:
            diagonal,_=plan(F,np.where(own_pair_mask(self.n),B,0.),a,D,y,self.held,r,w,self.p,costs,bud,allowed,lower=lower)
            hold=project_rows(np.tile(self.action,(H,1)),costs,bud,allowed,lower)
            candidates.extend([('diagonal',diagonal),('hold',hold)])
            A,offset=operators(F,B,a,D,y,self.held);q=w.ravel()/w.sum()
            for name,c in candidates:
                pred=(A@c.ravel()+offset).reshape(H,self.n)
                next_held=a*self.held+(1-a)*c[0]
                uncertainty=float(np.mean(self.model.predictive_variance(next_held)))*(1+float(self.age.mean()))
                candidate_scores[name]=float(np.sum(q*(pred-r).ravel()**2)+REGULARIZATION*np.mean(c*c)+self.config.beta*uncertainty)
            best=min(candidate_scores,key=candidate_scores.get);benefit=candidate_scores['base']-candidate_scores[best]
            eligible=self.interventions<self.config.max_interventions and self.step-self.last_intervention>=2
            if best!='base' and benefit>self.config.margin and eligible:
                uplan=dict(candidates)[best];decision=best;self.interventions+=1;self.last_intervention=self.step
        elif self.config.mode=='legacy_review' and probability is not None and probability<.55:
            uplan=np.tile(lower,(H,1));decision='legacy_minimum';self.interventions+=1
        if self.config.mode=='zero':uplan=np.tile(lower,(H,1));decision='inactive'
        u=uplan[0].copy();predicted=F@y+D[0]+B@(a*self.held+(1-a)*u)
        self.p=np.clip(.65*self.p+.35*self.intention,0,1)
        token={'actor':self.config.actor,'sequence':self.step,'action_hash':digest(u.tolist())}
        self.pending={'token':token,'previous_belief':y.copy(),'mask':mask.copy(),'F':F.copy(),'drift':D[0].copy(),
                      'held':a*self.held+(1-a)*u,'prediction':predicted.copy(),'action':u.copy()}
        self.belief=predicted.copy();self.held=self.pending['held'].copy();self.action=u.copy()
        self.trace={'step':self.step,'mode':self.config.mode,'C':deepcopy(self.C.events[-1]),
                    'U_last_event':deepcopy(self.U.events[-1]) if self.U.events else None,
                    'selected_content':None if delivered['planner'] is None else delivered['planner'].__dict__,
                    'belief_before':y.tolist(),'sensor_age':self.age.tolist(),'p_before':p_before.tolist(),'p_after':self.p.tolist(),
                    'intention':self.intention.tolist(),'executed':u.tolist(),'held':self.held.tolist(),
                    'B_before':B.tolist(),'planning_B':useB.tolist(),'target_used':r.tolist(),'priority_used':w.tolist(),
                    'planning':planning,'E':ethics,'allowed':allowed.tolist(),'minimum':lower.tolist(),'token':token,
                    'predicted_state_before_feedback':predicted.tolist(),'base_score_before_feedback':score,
                    'base_success_probability':probability,'meta_decision':decision,'predicted_utility_gain':benefit,
                    'candidate_scores':candidate_scores,'candidate_first_actions':{name:c[0].tolist() for name,c in candidates},
                    'intervention_count':self.interventions,'feedback':None}
        self.step+=1;return u
    def learn(self,f):
        if self.pending is None:raise RuntimeError('no pending action')
        p=self.pending
        if f.get('token')!=p['token']:raise ValueError('feedback actor/sequence/action mismatch')
        y=array(f['state'],(self.n,),'feedback sensor');mask=np.asarray(f['observed']);valid=np.asarray(f['valid_transition'])
        if any(a.dtype!=np.bool_ or a.shape!=(self.n,) for a in (mask,valid)):raise ValueError('feedback mask')
        known_previous=~((p['F']!=0)&(~p['mask'])[None,:]).any(axis=1)
        usable=mask&valid&known_previous
        response=y-p['F']@p['previous_belief']-p['drift']
        update=self.model.update(p['held'],response,usable)
        self.belief=np.where(mask,y,p['prediction'])
        if self.last_seen is not None:self.last_seen=np.where(mask,y,self.last_seen)
        self.trace['feedback']={'state':np.where(mask,y,0.).tolist(),'observed':mask.tolist(),'usable_rows':usable.tolist(),
                                'update':update,'B_after':self.model.B.tolist()}
        self.pending=None
    def snapshot(self):
        return {'configuration':asdict(self.config),'step':self.step,'B_state':self.model.snapshot(),
                'belief':None if self.belief is None else self.belief.tolist(),'last_seen':None if self.last_seen is None else self.last_seen.tolist(),
                'p':self.p.tolist(),'intention':self.intention.tolist(),'action':self.action.tolist(),'held':self.held.tolist(),
                'memory':self.U.snapshot(),'goals':self.goals.targets.tolist(),'interventions':self.interventions}
