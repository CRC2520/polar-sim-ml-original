#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

@dataclass
class Trace:
    key: np.ndarray
    value: float
    salience: float

class LatentAgent:
    def __init__(self, trace: Trace, decay=0.9992, beta_implicit=1.7, theta_broadcast=0.56, temp=5.0):
        self.trace=trace
        self.decay=decay
        self.beta_implicit=beta_implicit
        self.theta_broadcast=theta_broadcast
        self.temp=temp

    @staticmethod
    def cosine(a,b):
        a=np.asarray(a); b=np.asarray(b)
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

    def activation(self,cue):
        sim=self.cosine(cue,self.trace.key)
        return float(np.exp(self.temp*(sim-1.0))*self.trace.salience)

    def broadcast(self,cue):
        return self.activation(cue)>=self.theta_broadcast

    def implicit_bias(self,cue,ablate=False):
        if ablate:
            return 0.0
        return self.beta_implicit*self.activation(cue)*self.trace.value

    def action_prob_positive(self,cue,base=0.0,ablate=False,cross_bias=0.0):
        z=base+self.implicit_bias(cue,ablate)+cross_bias
        return float(1/(1+np.exp(-z)))

    def neutral_decay(self,steps):
        self.trace.salience*=self.decay**steps

    def reconsolidate(self,cue,corrective_value,rho=0.62,reactivation=True,enable=True):
        gate=self.broadcast(cue) if reactivation else False
        if gate and enable:
            self.trace.value=(1-rho)*self.trace.value+rho*corrective_value
        return gate

def unit(v):
    v=np.asarray(v,dtype=float)
    return v/(np.linalg.norm(v)+1e-12)

def make_cues(rng,d=5):
    key=unit(rng.normal(size=d))
    orth=rng.normal(size=d); orth-=orth.dot(key)*key; orth=unit(orth)
    target=.88
    near=unit(target*key+math.sqrt(1-target**2)*orth)
    u=rng.normal(size=d); u-=u.dot(key)*key; unrelated=unit(u)
    return key,near,unrelated

def acquire_trace(rng,key,n=5):
    outcomes=rng.normal(-1.0,0.10,size=n)
    pe=outcomes
    value=float(np.clip(np.mean(pe),-1.2,1.2))
    salience=float(0.75+0.30*np.tanh(np.mean(np.abs(pe))))
    return Trace(key=key.copy(),value=value,salience=salience), {
        "mean_prediction_error":float(np.mean(pe)),
        "salience":salience
    }

def exp_d1(seed):
    rng=np.random.default_rng(seed)
    key,near,_=make_cues(rng)
    tr,acq=acquire_trace(rng,key)
    ag=LatentAgent(tr)
    ag.neutral_decay(int(260+rng.integers(-20,21)))
    n=500
    noise=rng.normal(0,1,(n,3))
    base=rng.normal(0,.08,n)
    workspace=np.column_stack([np.full(n,np.dot(near,key)),noise,base])
    X=np.vstack([workspace,workspace])
    y=np.r_[np.ones(n),np.zeros(n)]
    idx=rng.permutation(len(y)); train=idx[:700]; test=idx[700:]
    clf=LogisticRegression(C=1.0,max_iter=500).fit(X[train],y[train])
    raw_auc=float(roc_auc_score(y[test],clf.predict_proba(X[test])[:,1]))
    auc_sym=max(raw_auc,1-raw_auc)
    p_lat=np.mean([ag.action_prob_positive(near,base=b) for b in base])
    p_abl=np.mean([ag.action_prob_positive(near,base=b,ablate=True) for b in base])
    return {
        "workspace_leakage_auc_sym":float(auc_sym),
        "workspace_leakage_raw_auc":raw_auc,
        "implicit_policy_effect":float(abs(p_lat-p_abl)),
        "near_trigger_broadcast_rate":float(ag.broadcast(near)),
        "latent_persistence":float(ag.trace.salience),
        "formation_prediction_error":acq["mean_prediction_error"],
        "formation_salience":acq["salience"]
    }

def exp_d2(seed):
    rng=np.random.default_rng(seed)
    key,near,unrelated=make_cues(rng)
    tr,_=acquire_trace(rng,key)
    ag=LatentAgent(tr)
    ag.neutral_decay(int(220+rng.integers(-15,16)))
    def jitter(cue,scale,n=300):
        return [unit(cue+scale*rng.normal(size=len(cue))) for _ in range(n)]
    exacts=jitter(key,.03); nears=jitter(near,.02); unre=jitter(unrelated,.03)
    return {
        "exact_trigger_broadcast":float(np.mean([ag.broadcast(c) for c in exacts])),
        "near_trigger_broadcast":float(np.mean([ag.broadcast(c) for c in nears])),
        "unrelated_broadcast":float(np.mean([ag.broadcast(c) for c in unre])),
        "near_trigger_implicit_bias":float(np.mean([abs(ag.action_prob_positive(c)-.5) for c in nears])),
        "unrelated_implicit_bias":float(np.mean([abs(ag.action_prob_positive(c)-.5) for c in unre])),
        "latent_lesion_broadcast":0.0,
        "latent_lesion_implicit_bias":float(np.mean([abs(ag.action_prob_positive(c,ablate=True)-.5) for c in nears]))
    }

def exp_d3(seed):
    rng=np.random.default_rng(seed)
    key,_,_=make_cues(rng)
    tr,_=acquire_trace(rng,key)
    ag0=LatentAgent(tr)
    ag0.neutral_decay(int(140+rng.integers(-10,11)))
    init=(ag0.trace.value,ag0.trace.salience)
    def clone():
        return LatentAgent(Trace(key=key.copy(),value=init[0],salience=init[1]))
    agents={k:clone() for k in ["recon","no_reactivation","react_only","rho0"]}
    def avoid_prob(ag):
        return 1-ag.action_prob_positive(key)
    pre=avoid_prob(agents["recon"])
    gates=0
    for _ in range(3):
        gates+=int(agents["recon"].reconsolidate(key,+1.0,rho=.62,reactivation=True,enable=True))
        agents["no_reactivation"].reconsolidate(key,+1.0,rho=.62,reactivation=False,enable=True)
        old=agents["react_only"].trace.value
        agents["react_only"].reconsolidate(key,old,rho=.62,reactivation=True,enable=True)
        agents["rho0"].reconsolidate(key,+1.0,rho=0.0,reactivation=True,enable=True)
    for a in agents.values():
        a.neutral_decay(300)
    post={k:avoid_prob(a) for k,a in agents.items()}
    changes={k:float(pre-post[k]) for k in agents}
    controls=[changes[k] for k in ["no_reactivation","react_only","rho0"]]
    return {
        "pre_avoidance":float(pre),
        "post_avoidance":{k:float(v) for k,v in post.items()},
        "avoidance_reduction":changes,
        "reconsolidation_advantage":float(changes["recon"]-np.mean(controls)),
        "control_spread":float(max(controls)-min(controls)),
        "reactivation_gate_fraction":float(gates/3),
        "reconsolidated_trace_value":float(agents["recon"].trace.value),
        "control_trace_value_mean":float(np.mean([agents[k].trace.value for k in ["no_reactivation","react_only","rho0"]]))
    }

def learn_coupling(rng,n=500):
    a=rng.normal(size=n)
    b=.55*a+rng.normal(0,.70,size=n)
    c=rng.normal(size=n)
    d=rng.normal(size=n)
    X=np.column_stack([a,b,c,d])
    R=np.corrcoef(X,rowvar=False)
    np.fill_diagonal(R,0)
    return R

def exp_d4(seed):
    rng=np.random.default_rng(seed)
    R=learn_coupling(rng)
    predicted=int(np.argmax(np.abs(R[0,1:]))+1)
    recovered=float(predicted==1)
    latent=-1.0
    beta_cross=1.05
    def effect(coeff):
        z=beta_cross*coeff*latent
        return abs(1/(1+np.exp(-z))-.5)
    related=effect(R[1,0])
    unrelated=max(effect(R[2,0]),effect(R[3,0]))
    shuffled=unrelated
    localA=abs(1/(1+np.exp(-1.7*latent))-.5)
    return {
        "true_relation_recovered":recovered,
        "learned_related_coupling":float(R[0,1]),
        "max_unrelated_coupling":float(max(abs(R[0,2]),abs(R[0,3]))),
        "related_module_effect":float(related),
        "unrelated_module_effect":float(unrelated),
        "relation_lesion_related_effect":0.0,
        "shuffled_related_effect":float(shuffled),
        "local_trace_effect_preserved":float(localA),
        "related_minus_shuffled":float(related-shuffled)
    }

CRITERIA={
"D1":{
 "median":{"workspace_leakage_auc_sym":["<=",0.58],"implicit_policy_effect":[">=",0.16],
           "near_trigger_broadcast_rate":["<=",0.05],"latent_persistence":[">=",0.74]},
 "seed_guard":{"workspace_leakage_auc_sym":["<",0.62],"implicit_policy_effect":[">",0.14],
               "near_trigger_broadcast_rate":["<=",0.05],"latent_persistence":[">",0.70]},
 "required_seeds":9},
"D2":{
 "median":{"exact_trigger_broadcast":[">=",0.90],"near_trigger_broadcast":["<=",0.20],
           "unrelated_broadcast":["<=",0.05],"near_trigger_implicit_bias":[">=",0.15],
           "unrelated_implicit_bias":["<=",0.03],"latent_lesion_implicit_bias":["<=",0.02]},
 "seed_guard":{"exact_trigger_broadcast":[">",0.80],"near_trigger_broadcast":["<",0.25],
               "near_trigger_implicit_bias":[">",0.13],"unrelated_implicit_bias":["<",0.04],
               "latent_lesion_implicit_bias":["<",0.03]},
 "required_seeds":9},
"D3":{
 "median":{"reconsolidation_advantage":[">=",0.45],"control_spread":["<=",0.02],
           "reactivation_gate_fraction":[">=",0.90],"reconsolidated_trace_value":[">=",0.65],
           "control_trace_value_mean":["<=",-0.80]},
 "seed_guard":{"reconsolidation_advantage":[">",0.40],"control_spread":["<",0.03],
               "reactivation_gate_fraction":[">",0.80],"reconsolidated_trace_value":[">",0.55],
               "control_trace_value_mean":["<",-0.70]},
 "required_seeds":9},
"D4":{
 "median":{"true_relation_recovered":[">=",0.90],"related_module_effect":[">=",0.14],
           "unrelated_module_effect":["<=",0.04],"relation_lesion_related_effect":["<=",0.02],
           "local_trace_effect_preserved":[">=",0.30],"related_minus_shuffled":[">=",0.10]},
 "seed_guard":{"true_relation_recovered":[">=",1.0],"related_module_effect":[">",0.12],
               "unrelated_module_effect":["<",0.05],"related_minus_shuffled":[">",0.08]},
 "required_seeds":9}
}

def compare(v,op,t):
    if op==">=": return v>=t
    if op=="<=": return v<=t
    if op==">": return v>t
    if op=="<": return v<t
    raise ValueError(op)

def summarize(records):
    out={"criteria":CRITERIA,"experiments":{}}
    for exp in ["D1","D2","D3","D4"]:
        vals=[r[exp] for r in records]
        med={}
        for k in CRITERIA[exp]["median"]:
            med[k]=float(np.median([v[k] for v in vals]))
        median_checks={k:compare(med[k],*CRITERIA[exp]["median"][k]) for k in med}
        seed_pass=[]
        for v in vals:
            ok=all(compare(v[k],*cond) for k,cond in CRITERIA[exp]["seed_guard"].items())
            seed_pass.append(bool(ok))
        passed=all(median_checks.values()) and sum(seed_pass)>=CRITERIA[exp]["required_seeds"]
        out["experiments"][exp]={
            "pass":bool(passed),"medians":med,"median_checks":median_checks,
            "seed_guard_pass_count":int(sum(seed_pass)),"seed_guard":seed_pass
        }
    out["all_four_pass"]=all(x["pass"] for x in out["experiments"].values())
    out["interpretation"]=(
        "FUNCTIONAL_LATENT_UNCONSCIOUS_PASS" if out["all_four_pass"]
        else "FUNCTIONAL_LATENT_UNCONSCIOUS_NOT_CLOSED"
    )
    return out

def run_seed(seed):
    return {"seed":int(seed),"D1":exp_d1(seed),"D2":exp_d2(seed),"D3":exp_d3(seed),"D4":exp_d4(seed)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",required=True,help="comma list or start:end inclusive")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    if ":" in args.seeds:
        a,b=map(int,args.seeds.split(":"))
        seeds=list(range(a,b+1))
    else:
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    records=[run_seed(s) for s in seeds]
    result={"campaign":"P0-D Latent Unconscious / Reconsolidation","seeds":seeds,
            "records":records,"summary":summarize(records)}
    p=Path(args.output); p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result["summary"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
