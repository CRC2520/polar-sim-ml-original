"""Content, memory, normative, semantic and metacognitive layers.

Names C/E/U identify operational architectural roles, not phenomenology or
clinical/ethical competence. Every layer has a logged, intervenable interface.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from copy import deepcopy
import hashlib
import json
import re
import numpy as np
from .numerics import array, project


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Content:
    identity: str
    cue: str
    targets: tuple
    priorities: tuple
    source: str='operator'
    salience: float=1.
    version: int=1
    private: bool=False


class ContentWorkspace:
    """Selective, copied content to independent consumers; not a budget broadcast."""
    def __init__(self):
        self.cut=set();self.events=[]
    def broadcast(self, candidates, *, permit_private=False):
        valid=[c for c in candidates if c.source in ('operator','memory') and (not c.private or permit_private)]
        selected=max(valid,key=lambda c:(c.salience,c.version,c.identity)) if valid else None
        consumers={}
        for name in ('planner','memory','reporter'):
            consumers[name]=None if selected is None or name in self.cut else deepcopy(selected)
        self.events.append({'selected':None if selected is None else selected.identity,
                            'content_hash':None if selected is None else digest(selected.__dict__),
                            'delivered':[k for k,v in consumers.items() if v is not None],
                            'cuts':sorted(self.cut),'candidate_count':len(candidates)})
        return consumers


class GatedMemory:
    def __init__(self, retention=.999):
        if not 0<retention<=1:raise ValueError('invalid memory retention')
        self.retention=retention;self.records={};self.events=[]
    def tick(self):
        for rec in self.records.values():rec['confidence']*=self.retention
    def store(self, content):
        if content is None or content.source!='operator':return
        old=self.records.get(content.cue)
        if old and content.version < old['content'].version:raise ValueError('stale memory revision')
        if old and content.version==old['content'].version and content!=old['content']:
            raise ValueError('conflicting same-version memory')
        self.records[content.cue]={'content':deepcopy(content),'confidence':1.}
        self.events.append({'operation':'store','cue':content.cue,'hash':digest(content.__dict__)})
    def recall(self,cue,*,gate=True,permit_private=False):
        rec=self.records.get(cue)
        ok=bool(gate and rec and rec['confidence']>=.2 and (not rec['content'].private or permit_private))
        self.events.append({'operation':'release','cue':cue,'gate':bool(gate),'released':ok})
        if not ok:return None
        c=rec['content'];return Content(c.identity,c.cue,c.targets,c.priorities,'memory',c.salience,c.version,c.private)
    def erase(self,cue):
        existed=self.records.pop(cue,None) is not None
        self.events.append({'operation':'erase','cue':cue,'existed':existed})
    def reconsolidate(self,cue,targets,*,authorized,source='operator'):
        if not authorized or source!='operator':raise PermissionError('authorized correction required')
        if cue not in self.records:raise KeyError(cue)
        old=self.records[cue]['content']
        new=Content(old.identity+'-revision',cue,tuple(tuple(r) for r in targets),old.priorities,
                    source,old.salience,old.version+1,old.private)
        self.store(new)
        self.events.append({'operation':'reconsolidate','old_hash':digest(old.__dict__),'new_hash':digest(new.__dict__)})
    def snapshot(self):
        return {k:{'content':v['content'].__dict__,'confidence':v['confidence']} for k,v in self.records.items()}


class MotivationalGoals:
    """Persistent target commitments; supplied goals, not autonomous value invention."""
    def __init__(self,n,horizon):
        self.n=n;self.horizon=horizon;self.cue=None;self.revisions=[];self.reset(None)
    def reset(self,cue):
        self.cue=cue;self.targets=np.full((self.horizon,self.n),.5);self.priorities=np.ones_like(self.targets)
    def consume(self,content,cue):
        if self.cue!=cue:self.reset(cue)
        if content is not None:
            targets=array(content.targets,(self.horizon,self.n),'content targets',0,1)
            priorities=array(content.priorities,(self.horizon,self.n),'content priorities',1e-12)
            self.targets,self.priorities=targets,priorities
            self.revisions.append({'cue':cue,'content':content.identity,'version':content.version,'source':content.source})
        return self.targets.copy(),self.priorities.copy()


class NormativePolicy:
    """A small explicit policy language: permissions, protected drain, obligations.

    Highest precedence: external permissions and protected-resource prohibition.
    Lower-precedence action requests cannot override these. Infeasible mandatory
    requests halt with reasons. This is institutional rule compliance, not morality.
    """
    version='institutional-demo-1'
    def compile(self,n,state,context,external_allowed,budget,costs):
        m=2*n;allowed=np.asarray(external_allowed)
        if allowed.shape!=(m,) or allowed.dtype!=np.bool_:raise ValueError('bad action permissions')
        allowed=allowed.copy();reasons=[]
        protected=np.asarray(context.get('protected',np.zeros(n,bool)))
        if protected.shape!=(n,) or protected.dtype!=np.bool_:raise ValueError('bad protection flags')
        authorized=context.get('approved_drain',False)
        if not isinstance(authorized,bool):raise ValueError('approval must be explicit boolean')
        for i in np.flatnonzero(protected):
            if not authorized:
                allowed[2*i+1]=False;reasons.append({'rule':'E-PROTECT','channel':int(2*i+1),'reason':'protected drain lacks authorization'})
        lower=array(context.get('mandatory_min',np.zeros(m)),(m,),'mandatory minimum',0,1)
        conflict=(lower>0)&~allowed
        for j in np.flatnonzero(conflict):
            reasons.append({'rule':'E-PRECEDENCE','channel':int(j),'reason':'prohibition overrides lower-priority request'})
        lower[~allowed]=0
        infeasible=float(np.asarray(costs)@lower)>budget+1e-12
        if infeasible:reasons.append({'rule':'E-INFEASIBLE','reason':'mandatory requests exceed resource budget; human review required'})
        return allowed,lower,{'policy_version':self.version,'reasons':reasons,'halt':infeasible}


class SemanticAdapter:
    """Grounded operational vocabulary: tank_i level, fill/drain, dimensionless units.

    This is a strict command grammar, not general language or emotional semantics.
    """
    def __init__(self,n,horizon):self.n=n;self.H=horizon
    def parse(self,text,cue,identity='command'):
        if not isinstance(text,str):raise ValueError('text required')
        parts=[x.strip() for x in text.split(';')];seen=set();target=np.zeros(self.n);weight=np.ones(self.n)
        for part in parts:
            match=re.fullmatch(r'tank_(\d+)\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*@\s*(\d+(?:\.\d+)?)',part)
            if not match:raise ValueError('expected tank_i = level @ priority')
            i=int(match[1])-1
            if i not in range(self.n) or i in seen:raise ValueError('unknown or duplicate tank')
            seen.add(i);target[i]=float(match[2]);weight[i]=float(match[3])
        if len(seen)!=self.n or np.any(weight<=0):raise ValueError('all tanks and positive priorities required')
        return Content(identity,cue,tuple(tuple(target) for _ in range(self.H)),tuple(tuple(weight) for _ in range(self.H)))
    def schema(self):
        return [{'pair':i,'state':f'tank_{i+1}_level','units':'fraction of capacity',
                 'positive_channel':f'fill_{i+1}','negative_channel':f'drain_{i+1}',
                 'meaning':'independent valve intensities; coactivation allowed with resource cost'} for i in range(self.n)]


class CalibratedMonitor:
    """Empirical success-frequency calibration, fitted ONLY on development episodes.

    score is a prediction made before feedback. Calibration validity is evaluated,
    never assumed from the class name. Histogram bins may be constant when evidence
    is insufficient. Abstention is an explicit policy with a reported coverage cost.
    """
    def __init__(self):self.edges=None;self.probabilities=None;self.base=.5
    def fit(self,scores,successes):
        x=np.asarray(scores,float);y=np.asarray(successes,float)
        if x.ndim!=1 or y.shape!=x.shape or len(x)<20 or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('invalid development calibration data')
        self.edges=np.unique(np.quantile(x,[.2,.4,.6,.8]))
        bins=np.searchsorted(self.edges,x,side='right');self.base=float((y.sum()+1)/(len(y)+2))
        self.probabilities=np.array([(y[bins==k].sum()+1)/(np.sum(bins==k)+2) for k in range(len(self.edges)+1)])
        return self
    def probability(self,score):
        if not np.isfinite(score):raise ValueError('nonfinite score')
        if self.edges is None:raise RuntimeError('monitor not fitted on development data')
        return float(self.probabilities[np.searchsorted(self.edges,score,side='right')])
    def snapshot(self):
        return {'edges':None if self.edges is None else self.edges.tolist(),
                'probabilities':None if self.probabilities is None else self.probabilities.tolist(),'base':self.base}
    @classmethod
    def restore(cls,snapshot):
        obj=cls();obj.edges=np.asarray(snapshot['edges'],float);obj.probabilities=np.asarray(snapshot['probabilities'],float);obj.base=float(snapshot['base'])
        if obj.probabilities.shape!=(len(obj.edges)+1,) or not np.isfinite(obj.probabilities).all() or np.any((obj.probabilities<0)|(obj.probabilities>1)):raise ValueError('bad calibrator')
        return obj


def infer_effect_source(response,candidates,*,variance=1e-4,minimum_margin=4.):
    """Competing causal-source predictions, conditional on supplied candidate actions.

    No source label is used in the score. Identical or equally plausible effects
    return unknown. This does not solve unobserved actors or adversarial forgery.
    """
    z=np.asarray(response,float)
    if z.ndim!=1 or not np.isfinite(z).all() or variance<=0 or not candidates:raise ValueError('invalid attribution problem')
    scores={name:float(np.sum((z-array(pred,z.shape,'candidate effect'))**2)/variance) for name,pred in candidates.items()}
    ranked=sorted(scores,key=scores.get)
    margin=float(scores[ranked[1]]-scores[ranked[0]]) if len(ranked)>1 else 0.
    return {'source':ranked[0] if len(ranked)>1 and margin>minimum_margin else None,'scores':scores,'margin':margin,'status':'conditional_attribution' if margin>minimum_margin else 'ambiguous'}
