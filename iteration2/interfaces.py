"""Typed content, normative contracts and freshness-aware inter-agent messages."""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
from integrated_polar.layers import Content,ContentWorkspace,GatedMemory,MotivationalGoals,NormativePolicy,digest
from integrated_polar.agent import coordinate_offers

DOMAINS={
    'water':{'state':'fraction of tank capacity','positive':'fill valve','negative':'drain valve','range':(0.,1.)},
    'thermal':{'state':'normalized zone temperature; physical T=16+16*x degC','positive':'heater','negative':'cooler','range':(0.,1.)}}


class DomainAdapter:
    """Strict physical command grammar. Not a natural-language understanding model."""
    def __init__(self,domain,n,horizon):
        if domain not in DOMAINS or n<1 or horizon<1:raise ValueError('unknown domain/dimensions')
        self.domain,self.n,self.horizon=domain,n,horizon
    def parse(self,text,cue):
        if not isinstance(text,str):raise ValueError('text command required')
        prefix='tank' if self.domain=='water' else 'zone';target=np.zeros(self.n);priority=np.zeros(self.n);seen=set()
        for segment in text.split(';'):
            match=re.fullmatch(prefix+r'_(\d+)\s*=\s*([\d.]+)\s*@\s*([\d.]+)',segment.strip())
            if match is None:raise ValueError('invalid domain-specific command')
            i=int(match[1])-1
            if i<0 or i>=self.n or i in seen:raise ValueError('unknown or duplicate channel')
            seen.add(i);v=float(match[2]);w=float(match[3])
            if self.domain=='thermal':v=(v-16.)/16.
            if not np.isfinite([v,w]).all() or not 0<=v<=1 or w<=0:raise ValueError('invalid units/range/priority')
            target[i]=v;priority[i]=w
        if len(seen)!=self.n:raise ValueError('incomplete command')
        return Content('parsed-'+digest(text),cue,tuple(map(tuple,np.tile(target,(self.horizon,1)))),
                       tuple(map(tuple,np.tile(priority,(self.horizon,1)))) )


@dataclass(frozen=True)
class Offer:
    actor:str
    sequence:int
    timestamp:int
    bid:float
    demand:float


class FreshCoordinator:
    """Reject stale, duplicate, out-of-order or malformed offers, then allocate.

    Claimed actor identity is NOT authenticated. This checks protocol consistency,
    not adversarial sender authentication or truthfulness of declared preferences.
    """
    def __init__(self,actors,ttl=2):
        if len(set(actors))!=len(actors) or not actors or ttl<0:raise ValueError('invalid actors/TTL')
        self.actors=tuple(actors);self.ttl=ttl;self.latest={};self.highwater={};self.events=[]
    def receive(self,packets,now):
        decisions=[]
        for p in packets:
            reason='accepted'
            if p.actor not in self.actors:reason='unknown_actor'
            elif not isinstance(p.timestamp,int) or p.timestamp>now or now-p.timestamp>self.ttl:reason='expired_or_future'
            elif not isinstance(p.sequence,int) or p.sequence<=self.highwater.get(p.actor,-1):reason='duplicate_or_reordered'
            elif not np.isfinite([p.bid,p.demand]).all() or p.bid<=0 or not 0<=p.demand<=2:reason='invalid_offer'
            if reason=='accepted':self.latest[p.actor]=p;self.highwater[p.actor]=p.sequence
            decisions.append({'actor':p.actor,'sequence':p.sequence,'decision':reason})
        self.events.extend(decisions);return decisions
    def allocate(self,budget,now,mode='fresh'):
        if mode not in ('fresh','naive','equal'):raise ValueError('unknown coordination mode')
        if not np.isfinite(budget) or budget<0:raise ValueError('invalid shared resource')
        if mode=='equal':return {a:budget/len(self.actors) for a in self.actors}
        valid={a:p for a,p in self.latest.items() if mode=='naive' or now-p.timestamp<=self.ttl}
        offers=[{'actor':a,'bid':valid[a].bid if a in valid else .1,'demand':valid[a].demand if a in valid else 1.} for a in self.actors]
        return coordinate_offers(offers,budget)
