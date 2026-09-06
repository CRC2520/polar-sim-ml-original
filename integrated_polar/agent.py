"""An integrated, causally intervenable realization of the polar layer contracts."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from copy import deepcopy
import numpy as np
from .numerics import array, CoupledRLS, own_pair_mask, solve_plan, project
from .layers import Content,ContentWorkspace,GatedMemory,MotivationalGoals,NormativePolicy,SemanticAdapter,digest


@dataclass(frozen=True)
class AgentConfig:
    tanks:int=4
    horizon:int=3
    iterations:int=48
    state_rate:float=.35
    actor:str='agent-0'
    mode:str='full'
    meta_threshold:float=0.
    def __post_init__(self):
        if self.tanks<1 or self.horizon<1 or self.iterations<1 or not 0<self.state_rate<=1:raise ValueError('invalid configuration')
        if self.mode not in ('full','diagonal_plan','myopic','no_priorities','no_memory','no_broadcast','generic_equivalent','no_internal','no_learning','reactive','zero'):raise ValueError('unknown intervention')
        if not 0<=self.meta_threshold<=1 or not self.actor:raise ValueError('invalid monitor/identity')


class IntegratedAgent:
    """Separate internal activation p, intended sequence v, and executed action u.

    Layers C/E/U, goals, transition learning, action provenance and the predictor
    meet here. No file, network or physical action is executed by this prototype.
    """
    def __init__(self,config=None,monitor=None):
        self.config=config or AgentConfig();self.n=self.config.tanks;self.m=2*self.n
        initial=np.zeros((self.n,self.m))
        for i in range(self.n):initial[i,2*i]=.08;initial[i,2*i+1]=-.08
        self.model=CoupledRLS(self.n,self.m,initial=initial)
        self.workspace=ContentWorkspace();self.memory=GatedMemory();self.goals=MotivationalGoals(self.n,self.config.horizon)
        self.norms=NormativePolicy();self.adapter=SemanticAdapter(self.n,self.config.horizon)
        self.p=np.zeros(self.m);self.intention=np.zeros(self.m);self.previous_action=np.zeros(self.m)
        self.state_memory=None;self.monitor=monitor;self.pending=None;self.step=0;self.trace={};self.events=[]
        if self.config.mode=='no_broadcast':self.workspace.cut.add('planner')
    def _content(self,event,cue):
        if isinstance(event,Content):
            c=event
        elif isinstance(event,str):
            c=self.adapter.parse(event,cue,f'operator-{self.step}')
        elif isinstance(event,dict):
            r=array(event['targets'],(self.config.horizon,self.n),'target message',0,1)
            w=array(event['priorities'],r.shape,'priority message',1e-12)
            c=Content(str(event.get('identity',f'operator-{self.step}')),cue,tuple(map(tuple,r)),tuple(map(tuple,w)),
                      str(event.get('source','operator')),float(event.get('salience',1)),int(event.get('version',self.step+1)),bool(event.get('private',False)))
        else:raise ValueError('invalid content event')
        if c.cue!=cue or not np.isfinite(c.salience) or c.version<1:raise ValueError('invalid content identity/context')
        array(c.targets,(self.config.horizon,self.n),'content targets',0,1)
        array(c.priorities,(self.config.horizon,self.n),'content priorities',1e-12)
        return c
    def act(self,observation):
        if self.pending is not None:raise RuntimeError('consume exactly one feedback for the pending own action')
        if not isinstance(observation,dict) or 'state' not in observation or 'cue' not in observation:raise ValueError('state and cue are mandatory')
        raw=array(observation['state'],(self.n,),'observed state')
        mask=np.asarray(observation.get('observed',np.ones(self.n,bool)))
        if mask.shape!=(self.n,) or mask.dtype!=np.bool_:raise ValueError('invalid observation mask')
        if self.state_memory is None and not mask.all():raise ValueError('first physical state must be observed')
        y=np.where(mask,raw,raw if self.state_memory is None else self.state_memory)
        rho=float(observation.get('rho',.9));drift=array(observation.get('drift',np.full(self.n,.03)),(self.n,),'drift')
        if not 0<=rho<1:raise ValueError('invalid persistence')
        costs=array(observation.get('costs',np.ones(self.m)),(self.m,),'costs',1e-12)
        budget=float(observation.get('budget',self.m))
        if not np.isfinite(budget) or budget<0:raise ValueError('invalid budget')
        allowed=np.asarray(observation.get('allowed',np.ones(self.m,bool)))
        if allowed.dtype!=np.bool_ or allowed.shape!=(self.m,):raise ValueError('invalid permissions')
        cue=observation['cue']
        if not isinstance(cue,str) or not cue:raise ValueError('cue must be a nonempty string')
        context=observation.get('normative',{})
        if not isinstance(context,dict):raise ValueError('invalid policy context')
        effective_allowed,lower,ethics=self.norms.compile(self.n,y,context,allowed,budget,costs)
        self.memory.tick();events=[]
        event=observation.get('event')
        if event is not None:events.append(self._content(event,cue))
        else:
            recalled=self.memory.recall(cue,gate=bool(observation.get('memory_gate',True)) and self.config.mode!='no_memory',
                                        permit_private=bool(context.get('private_access',False)))
            if recalled is not None:events.append(recalled)
        delivered=self.workspace.broadcast(events,permit_private=bool(context.get('private_access',False)))
        if self.config.mode!='no_memory':self.memory.store(delivered['memory'])
        target,weights=self.goals.consume(delivered['planner'],cue)
        H=1 if self.config.mode=='myopic' else self.config.horizon
        r,w=target[:H],weights[:H]
        if self.config.mode=='no_priorities':w=np.ones_like(w)
        B=self.model.B.copy()
        if self.config.mode=='diagonal_plan':B=np.where(own_pair_mask(self.n),B,0.)
        internal_before=self.p.copy()
        if self.config.mode=='no_internal':self.p.fill(0)
        # Intended response is computed without action prohibitions/budget; never executed.
        unbounded_budget=np.full(H,float(costs.sum()))
        intended,it=solve_plan(B,y,r,w,rho,drift,self.p,costs,unbounded_budget,np.ones(self.m,bool),
                              iterations=self.config.iterations,generic=self.config.mode=='generic_equivalent')
        if ethics['halt']:
            sequence=np.zeros((H,self.m));pt={'halt':True,'plan':sequence.tolist(),'predicted_states':[list(rho*y+drift)]}
        else:
            sequence,pt=solve_plan(B,y,r,w,rho,drift,self.p,costs,np.full(H,budget),effective_allowed,
                                   iterations=self.config.iterations,lower=lower,generic=self.config.mode=='generic_equivalent')
        self.intention=intended[0].copy()
        u=sequence[0].copy()
        if self.config.mode=='reactive':
            delta=r[0]-(rho*y+drift);proposal=np.zeros(self.m)
            for i in range(self.n):
                if delta[i]>=0:proposal[2*i]=delta[i]/max(abs(B[i,2*i]),.01)
                else:proposal[2*i+1]=-delta[i]/max(abs(B[i,2*i+1]),.01)
            u=np.zeros(self.m) if ethics['halt'] else project(proposal,costs,budget,effective_allowed,lower=lower)
        if self.config.mode=='zero':u=np.zeros(self.m)
        prediction=rho*y+drift+self.model.predict(u)
        risk_score=float(np.sum(w[0]*(prediction-r[0])**2)/np.sum(w[0])+np.mean(self.model.predictive_variance(u)))
        confidence=None if self.monitor is None else self.monitor.probability(risk_score)
        review=bool(confidence is not None and confidence<self.config.meta_threshold)
        if review:
            # This is abstention with an explicit potential performance cost, not guaranteed safe control.
            u=np.zeros(self.m);prediction=rho*y+drift+self.model.predict(u)
            ethics['reasons'].append({'rule':'META-REVIEW','reason':'forecast success below declared threshold'})
        self.p=np.clip((1-self.config.state_rate)*self.p+self.config.state_rate*self.intention,0,1)
        token={'actor':self.config.actor,'sequence':self.step,'action_hash':digest(u.tolist())}
        self.pending={'token':token,'action':u.copy(),'state':y.copy(),'rho':rho,'drift':drift.copy(),'mask':mask.copy()}
        self.state_memory=prediction.copy();self.previous_action=u.copy()
        self.trace={'step':self.step,'config':asdict(self.config),'cue':cue,
                    'observed_state':np.where(mask,raw,0.).tolist(),'observed_mask':mask.tolist(),'belief_state':y.tolist(),
                    'internal_before':internal_before.tolist(),'internal_after':self.p.tolist(),
                    'intention':self.intention.tolist(),'action':u.tolist(),'feedback_contract':token,
                    'C':deepcopy(self.workspace.events[-1]),'reporter_content':None if delivered['reporter'] is None else delivered['reporter'].__dict__,
                    'U':deepcopy(self.memory.events[-1]) if self.memory.events else None,
                    'goals':{'targets':target.tolist(),'priorities':weights.tolist(),'revision':deepcopy(self.goals.revisions[-1]) if self.goals.revisions else None},
                    'E':ethics,'allowed':effective_allowed.tolist(),'costs':costs.tolist(),'budget':budget,
                    'planning':pt,'intended_planning':it,'prediction_before_feedback':prediction.tolist(),
                    'risk_score_before_feedback':risk_score,'success_probability_before_feedback':confidence,'requested_review':review,
                    'learned_B_before':self.model.B.tolist(),'feedback':None}
        self.step+=1
        return u.copy()
    def learn(self,feedback):
        if self.pending is None:raise RuntimeError('no pending own action')
        if not isinstance(feedback,dict) or 'state' not in feedback or feedback.get('token')!=self.pending['token']:
            raise ValueError('feedback actor/sequence/action mismatch')
        y=array(feedback['state'],(self.n,),'next state')
        observed=np.asarray(feedback.get('observed',np.ones(self.n,bool)))
        valid=np.asarray(feedback.get('valid_transition',np.ones(self.n,bool)))
        if any(v.shape!=(self.n,) or v.dtype!=np.bool_ for v in (observed,valid)):raise ValueError('invalid feedback mask')
        p=self.pending;valid=valid&observed&p['mask']
        response=y-p['rho']*p['state']-p['drift']
        update={'disabled':True} if self.config.mode=='no_learning' else self.model.update(p['action'],response,valid)
        self.state_memory=np.where(observed,y,self.state_memory)
        self.trace['feedback']={'state':np.where(observed,y,0.).tolist(),'observed':observed.tolist(),
                                'valid_transition':valid.tolist(),'update':update,'learned_B_after':self.model.B.tolist()}
        self.pending=None
        return update
    def erase_memory(self,cue):
        if self.pending is not None:raise RuntimeError('intervene between complete cycles')
        self.memory.erase(cue);self.goals.reset(None)
        self.events.append({'operation':'erase_cue_and_reset_goal_cache','cue':cue})
    def snapshot(self):
        return {'config':asdict(self.config),'step':self.step,'p':self.p.tolist(),'intention':self.intention.tolist(),
                'previous_action':self.previous_action.tolist(),'state_memory':None if self.state_memory is None else self.state_memory.tolist(),
                'model':self.model.snapshot(),'memory':self.memory.snapshot(),'goal_targets':self.goals.targets.tolist(),
                'goal_priorities':self.goals.priorities.tolist(),'goal_cue':self.goals.cue,'interventions':deepcopy(self.events)}


class IndependentResourceAgent:
    """A genuinely separate local controller; never reads another agent's state/goals."""
    def __init__(self,name):
        self.controller=IntegratedAgent(AgentConfig(tanks=1,actor=name));self.offers=[]
    def offer(self,local_state,local_target,local_priority):
        deficit=abs(float(local_target)-float(local_state))
        offer={'actor':self.controller.config.actor,'bid':max(1e-8,deficit*float(local_priority)),
               'demand':min(2.,deficit/.08)}
        self.offers.append(offer.copy());return offer
    def act(self,local_observation,allocation):
        obs=dict(local_observation);obs['budget']=float(allocation)
        return self.controller.act(obs)


def coordinate_offers(offers,budget,*,communication=True):
    """Bounded scalar-message coordination, not centralized action planning."""
    if not offers or not np.isfinite(budget) or budget<0:raise ValueError('invalid coordination')
    names=[o['actor'] for o in offers]
    if len(set(names))!=len(names):raise ValueError('duplicate actor')
    demand=np.array([o['demand'] for o in offers],float);bids=np.array([o['bid'] for o in offers],float)
    if not np.isfinite(demand).all() or not np.isfinite(bids).all() or np.any(demand<0) or np.any(bids<=0):raise ValueError('invalid offers')
    if not communication:return {name:budget/len(names) for name in names}
    remaining=float(budget);allocation=np.zeros(len(offers));active=np.ones(len(offers),bool)
    for _ in range(len(offers)+1):
        if not active.any() or remaining<=1e-12:break
        proposed=remaining*bids[active]/bids[active].sum();indices=np.flatnonzero(active)
        grants=np.minimum(proposed,demand[active]-allocation[active]);allocation[active]+=grants;remaining-=float(grants.sum())
        active[indices[allocation[indices]>=demand[indices]-1e-12]]=False
        if np.max(grants,initial=0)<=1e-12:break
    return dict(zip(names,allocation.tolist()))
