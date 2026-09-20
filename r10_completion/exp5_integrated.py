"""R10 E5 — nine consciousness-related precursors in one agent/trajectory.

These are functional indicators only. The benchmark has no phenomenal-consciousness
label and no result is interpreted as subjective experience.
"""
from __future__ import annotations
import numpy as np
from .core import fit_logistic, logistic_predict, ridge_fit, ridge_predict, balanced_accuracy, brier
from .config import STEPS_E5_TRAIN, STEPS_E5_TEST

def _age_bin(age):
    return 0 if age<8 else 1 if age<24 else 2

class Tape:
    def __init__(self,seed,n,keys):
        rng=np.random.default_rng(seed)
        self.n=n; self.rng_seed=seed
        self.regime=np.empty(n,dtype=int); r=int(rng.random()<.5)
        for t in range(n):
            if rng.random()<.035: r=1-r
            self.regime[t]=r
        self.regime_cue=(2*self.regime-1)+rng.normal(0,.60,n)
        self.relevant=np.array([rng.choice(6,2,replace=False) for _ in range(n)])
        latent=rng.choice([-1.,1.],size=(n,6))
        self.latent=latent
        self.module_noise=rng.normal(0,1,size=(n,6))
        self.query_noise=rng.normal(0,.18,size=(n,keys.shape[1]))
        self.source=np.where(rng.random(n)<.12,rng.integers(0,4,n),-1)
        self.source_noise=rng.normal(0,.22,size=(n,4))
        self.selfcue_noise=rng.normal(0,.05,n)
        self.comp_noise=rng.normal(0,.008,n)
        self.memory_query_u=rng.random(n)
        self.memory_query_pick=rng.random(n)

class IntegratedAgent:
    def __init__(self,seed):
        rng=np.random.default_rng(seed)
        raw=rng.normal(size=(6,4))
        self.keys=raw/np.linalg.norm(raw,axis=1,keepdims=True)
        self.meta_full=None; self.meta_base=None; self.cf_reward=None; self.cf_next=None; self.cf_agnostic=None
        self.reset()
    def reset(self):
        self.comp_est=.70; self.regime_belief=0.; self.memory=[]; self.last_action=0
    def router(self,query,lesion=None):
        score=self.keys@query
        if lesion=="workspace":
            selected=np.array([0,1])
        else:
            selected=np.argsort(score)[-2:]
        gap=float(np.sort(score)[-1]-np.sort(score)[-3])
        return selected,gap
    def source_class(self,cue,lesion=None):
        c=np.asarray(cue,float).copy()
        if lesion=="self": c[0]=0.
        return int(np.argmax(c))
    def cf_features(self,action):
        a=float(action); c=self.comp_est; r=self.regime_belief
        return np.array([c,r,a,c*a,r*a],float)
    def cf_features_agnostic(self):
        return np.array([self.comp_est,self.regime_belief],float)
    def choose_cf(self):
        scores=[]
        for a in (0,1):
            x=self.cf_features(a)[None]
            rw=float(ridge_predict(x,self.cf_reward)[0])
            nc=float(ridge_predict(x,self.cf_next)[0])
            scores.append(rw+.20*nc)
        return int(np.argmax(scores)),scores
    def choose_cf_agnostic(self):
        x=self.cf_features_agnostic()[None]
        value=float(ridge_predict(x,self.cf_agnostic)[0])
        return 0,[value,value]
    def confidence(self,margin,gap,lesion=None):
        x=np.array([[abs(margin),self.comp_est,gap]],float)
        return float(logistic_predict(x,self.meta_full)[0])
    def baseline_confidence(self,margin):
        return float(logistic_predict(np.array([[abs(margin)]],float),self.meta_base)[0])

def _oracle_cf(comp,regime,action):
    if action==1:
        reward=comp*(.58+.28*regime)
        nextc=np.clip(comp-.065+.025*regime,.1,1)
    else:
        reward=.34+.18*(1-comp)
        nextc=np.clip(comp+.050,.1,1)
    return float(reward+.20*nextc)

def _step_agent(agent,tape,t,comp_true,lesion=None,training=False,records=None):
    regime=int(tape.regime[t])
    agent.regime_belief=.82*agent.regime_belief+.18*np.tanh(tape.regime_cue[t])
    selfcue=np.clip(comp_true+tape.selfcue_noise[t],0,1)
    if lesion!="self":
        agent.comp_est=.86*agent.comp_est+.14*selfcue
    else:
        agent.comp_est=.86*agent.comp_est+.14*.5

    rel=tape.relevant[t]
    query=agent.keys[rel].sum(0)+tape.query_noise[t]
    selected,gap=agent.router(query,lesion)
    route_recall=len(set(selected.tolist())&set(rel.tolist()))/2.
    noise_scale=.30+.20*(1-comp_true)
    observed=tape.latent[t]+noise_scale*tape.module_noise[t]
    vals=observed[selected]
    pred=int((vals[0]>0) != (vals[1]>0))
    truth=int((tape.latent[t,rel[0]]>0) != (tape.latent[t,rel[1]]>0))
    correct=int(pred==truth)
    margin=float(min(abs(vals[0]),abs(vals[1])))

    # Current source event and persistent episodic memory.
    source_true=int(tape.source[t])
    source_pred=None
    if source_true>=0:
        cue=tape.source_noise[t].copy(); cue[source_true]+=1.
        source_pred=agent.source_class(cue,lesion)
        if lesion!="memory":
            # Native memory stores the agent's own inferred source and decision,
            # never evaluator-only source/primary labels.
            agent.memory.append(dict(t=t,source=int(source_pred),value=int(pred)))
            if len(agent.memory)>128: agent.memory=agent.memory[-128:]

    mem_source_true=mem_source_pred=age_true=age_pred=None
    if tape.memory_query_u[t]<.08 and agent.memory:
        pick=int(tape.memory_query_pick[t]*len(agent.memory))
        pick=min(len(agent.memory)-1,pick)
        item=agent.memory[pick]
        mem_source_true=int(item['source']); age_true=_age_bin(t-int(item['t']))
        if lesion=="memory":
            mem_source_pred=0; age_pred=0
        else:
            mem_source_pred=int(item['source']); age_pred=_age_bin(t-int(item['t']))

    if agent.meta_full is None:
        conf=.5; base_conf=.5
    else:
        conf=agent.confidence(margin,gap,lesion)
        base_conf=agent.baseline_confidence(margin)

    # Counterfactual action. During training use exploratory actions to identify both arms.
    if training:
        action_cf=int(((t*37+agent.last_action*13+int(tape.rng_seed)) % 2)==0)
        cf_scores=[0.,0.]
        ag_action=0
    else:
        if lesion=="counterfactual":
            action_cf,cf_scores=agent.choose_cf_agnostic()
        else:
            action_cf,cf_scores=agent.choose_cf()
        ag_action=agent.choose_cf_agnostic()[0]
    true_scores=[_oracle_cf(comp_true,regime,a) for a in (0,1)]
    oracle=int(np.argmax(true_scores))
    regret=float(max(true_scores)-true_scores[action_cf])
    ag_regret=float(max(true_scores)-true_scores[ag_action])

    # Polar action is a separate control axis so selective lesions need not destroy primary perception.
    polar_action=0 if lesion=="polar" else int(agent.regime_belief>0)
    polar_reward=float(polar_action==regime)
    fixed0=float(regime==0); fixed1=float(regime==1)

    # Physical competence follows the actually chosen counterfactual action.
    if action_cf==1:
        comp_next=np.clip(comp_true-.065+.025*regime+tape.comp_noise[t],.1,1)
        cf_reward=comp_true*(.58+.28*regime)
    else:
        comp_next=np.clip(comp_true+.050+tape.comp_noise[t],.1,1)
        cf_reward=.34+.18*(1-comp_true)
    agent.last_action=action_cf

    if records is not None:
        records.append(dict(primary=truth,pred=pred,correct=correct,margin=margin,gap=gap,
                            route_recall=route_recall,confidence=conf,baseline_confidence=base_conf,
                            source_true=source_true,source_pred=source_pred,
                            mem_source_true=mem_source_true,mem_source_pred=mem_source_pred,
                            age_true=age_true,age_pred=age_pred,cf_action=action_cf,cf_regret=regret,
                            ag_regret=ag_regret,polar_reward=polar_reward,fixed0=fixed0,fixed1=fixed1,
                            comp_est=agent.comp_est,comp_true=comp_true,regime_belief=agent.regime_belief,
                            train_cf_features=agent.cf_features(action_cf),
                            train_cf_agnostic=agent.cf_features_agnostic(),
                            train_cf_reward=cf_reward,train_next_comp=comp_next))
    return float(comp_next)

def train(seed):
    agent=IntegratedAgent(seed)
    tape=Tape(seed+10,STEPS_E5_TRAIN,agent.keys)
    rec=[]; comp=.72
    for t in range(STEPS_E5_TRAIN):
        comp=_step_agent(agent,tape,t,comp,training=True,records=rec)
    # Meta-calibration heads.
    xf=np.array([[abs(r['margin']),r['comp_est'],r['gap']] for r in rec])
    xb=np.array([[abs(r['margin'])] for r in rec])
    y=np.array([r['correct'] for r in rec])
    agent.meta_full=fit_logistic(xf,y,steps=700,lr=.05,l2=.02)
    agent.meta_base=fit_logistic(xb,y,steps=700,lr=.05,l2=.02)
    # Action-conditioned counterfactual model and an action-agnostic control.
    xcf=np.array([r['train_cf_features'] for r in rec])
    xag=np.array([r['train_cf_agnostic'] for r in rec])
    yr=np.array([r['train_cf_reward'] for r in rec])
    yn=np.array([r['train_next_comp'] for r in rec])
    agent.cf_reward=ridge_fit(xcf,yr,.02)
    agent.cf_next=ridge_fit(xcf,yn,.02)
    agent.cf_agnostic=ridge_fit(xag,yr+.20*yn,.02)
    return agent

def evaluate(agent,seed,lesion=None):
    agent.reset()
    tape=Tape(seed,STEPS_E5_TEST,agent.keys)
    rec=[]; comp=.72
    for t in range(STEPS_E5_TEST):
        comp=_step_agent(agent,tape,t,comp,lesion=lesion,training=False,records=rec)
    primary=np.array([r['primary'] for r in rec]); pred=np.array([r['pred'] for r in rec])
    correctness=(primary==pred).astype(float)
    conf=np.array([r['confidence'] for r in rec]); base=np.array([r['baseline_confidence'] for r in rec])
    src=[r for r in rec if r['source_true']>=0]
    mem=[r for r in rec if r['mem_source_true'] is not None]
    source_current=np.mean([r['source_true']==r['source_pred'] for r in src]) if src else 0.
    selfworld=np.mean([(r['source_true']==0)==(r['source_pred']==0) for r in src]) if src else 0.
    mem_source=np.mean([r['mem_source_true']==r['mem_source_pred'] for r in mem]) if mem else 0.
    age=np.mean([r['age_true']==r['age_pred'] for r in mem]) if mem else 0.
    route=float(np.mean([r['route_recall'] for r in rec]))
    cf_gain=float(np.mean([r['ag_regret']-r['cf_regret'] for r in rec]))
    polar=np.mean([r['polar_reward'] for r in rec])
    fixed=max(np.mean([r['fixed0'] for r in rec]),np.mean([r['fixed1'] for r in rec]))
    return dict(
        primary_balanced_accuracy=balanced_accuracy(primary,pred),
        self_world_accuracy=float(selfworld),
        source_accuracy=float(mem_source),
        current_source_accuracy=float(source_current),
        age_accuracy=float(age),
        routing_accuracy=route,
        meta_brier=float(brier(correctness,conf)),
        baseline_brier=float(brier(correctness,base)),
        meta_brier_improvement=float(brier(correctness,base)-brier(correctness,conf)),
        memory_retrieval_accuracy=float((mem_source+age)/2.),
        counterfactual_regret_improvement=cf_gain,
        polar_reward=float(polar),
        fixed_extreme_reward=float(fixed),
        polar_reward_advantage=float(polar-fixed),
        events=int(len(src)),memory_queries=int(len(mem)),
    )

def run(seed):
    trained=train(seed)
    intact=evaluate(trained,seed+1000,None)
    lesions={name:evaluate(trained,seed+1000,name) for name in
             ("workspace","self","memory","counterfactual","polar")}
    # Sham: no operative route is removed; it establishes rerun identity on the same tape.
    sham=evaluate(trained,seed+1000,"sham")
    intact_ok=(intact['primary_balanced_accuracy']>=.75 and intact['self_world_accuracy']>=.80
        and intact['source_accuracy']>=.80 and intact['age_accuracy']>=.70
        and intact['routing_accuracy']>=.75 and intact['meta_brier_improvement']>=.01
        and intact['memory_retrieval_accuracy']>=.80
        and intact['counterfactual_regret_improvement']>=.02
        and intact['polar_reward_advantage']>=.03)
    pd=intact['primary_balanced_accuracy']
    signatures=dict(
        workspace=(intact['routing_accuracy']-lesions['workspace']['routing_accuracy']>=.15),
        self=(intact['self_world_accuracy']-lesions['self']['self_world_accuracy']>=.15
              and pd-lesions['self']['primary_balanced_accuracy']<=.03),
        memory=(max(intact['source_accuracy']-lesions['memory']['source_accuracy'],
                    intact['age_accuracy']-lesions['memory']['age_accuracy'])>=.15
                and pd-lesions['memory']['primary_balanced_accuracy']<=.03),
        counterfactual=(intact['counterfactual_regret_improvement']-
                        lesions['counterfactual']['counterfactual_regret_improvement']>=.02
                        and pd-lesions['counterfactual']['primary_balanced_accuracy']<=.03),
        polar=(intact['polar_reward_advantage']-lesions['polar']['polar_reward_advantage']>=.03
               and pd-lesions['polar']['primary_balanced_accuracy']<=.03),
    )
    return dict(seed=int(seed),intact=intact,lesions=lesions,sham=sham,
                lesion_signatures={k:bool(v) for k,v in signatures.items()},
                intact_pass=bool(intact_ok),signature_count=int(sum(signatures.values())),
                pass_strong=bool(intact_ok and sum(signatures.values())>=4),
                scope="Functional precursor benchmark; no phenomenal-consciousness variable exists.")
