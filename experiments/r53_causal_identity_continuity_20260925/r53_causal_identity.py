#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RELEVANT_CONTEXTS=tuple(range(8))
IRRELEVANT_CONTEXTS=tuple(range(8,16))
N_ACTIONS=4
ROUNDS=3
PROBE_REPEATS=8
DEV_SEEDS=list(range(2236001,2236017))
CONF_SEEDS=list(range(2237001,2237033))
SMOKE_SEED=2235000
PRIOR_SEEDS=list(range(2234001,2234017))

def stable_seed(seed:int,*labels:str)->int:
    raw="|".join(["R53",str(int(seed)),*labels]).encode()
    h=hashlib.sha256(raw).digest()
    return int.from_bytes(h[:8],"little") & 0xffffffff

def body_perm(seed:int)->np.ndarray:
    rng=np.random.default_rng(stable_seed(seed,"body"))
    return rng.permutation(N_ACTIONS)

def calibrate_inverse(perm:np.ndarray)->Tuple[Dict[int,int],float]:
    inv={}
    for motor in range(N_ACTIONS):
        outcome=int(perm[motor])
        inv[outcome]=motor
    acc=np.mean([perm[inv[o]]==o for o in range(N_ACTIONS)])
    return inv,float(acc)

def generate_history(seed:int,label:str,relevant:bool=True)->List[dict]:
    rng=np.random.default_rng(stable_seed(seed,label))
    contexts=list(RELEVANT_CONTEXTS if relevant else IRRELEVANT_CONTEXTS)
    events=[]
    t=0
    for rnd in range(ROUNDS):
        order=rng.permutation(contexts)
        for ctx in order:
            target=int(rng.integers(0,N_ACTIONS))
            events.append({"t":t,"context":int(ctx),"target":target,"round":rnd,"relevant":bool(relevant)})
            t+=1
    return events

def latest_map(events:List[dict],contexts=RELEVANT_CONTEXTS)->Dict[int,int]:
    out={}
    for e in events:
        if int(e["context"]) in contexts:
            out[int(e["context"])]=int(e["target"])
    return out

def lookup_latest(events:List[dict],ctx:int):
    for e in reversed(events):
        if int(e["context"])==int(ctx):
            return int(e["target"])
    return None

def population_prior()->Dict[int,int]:
    per={c:[] for c in RELEVANT_CONTEXTS}
    for s in PRIOR_SEEDS:
        m=latest_map(generate_history(s,"population",True))
        for c in RELEVANT_CONTEXTS:
            per[c].append(m[c])
    out={}
    for c,vals in per.items():
        counts=Counter(vals)
        best=max(counts.values())
        out[c]=min(k for k,v in counts.items() if v==best)
    return out

POP_PRIOR=population_prior()

def probe_schedule(seed:int,label:str="probe")->List[int]:
    rng=np.random.default_rng(stable_seed(seed,label))
    base=list(RELEVANT_CONTEXTS)*PROBE_REPEATS
    rng.shuffle(base)
    return [int(x) for x in base]

def policy_actions(archive:List[dict],inv_body:Dict[int,int],schedule:List[int],fallback:Dict[int,int])->List[int]:
    acts=[]
    for ctx in schedule:
        target=lookup_latest(archive,ctx)
        if target is None:
            target=int(fallback[int(ctx)])
        acts.append(int(inv_body[int(target)]))
    return acts

def correct_actions(own_latest:Dict[int,int],inv_body:Dict[int,int],schedule:List[int])->List[int]:
    return [int(inv_body[int(own_latest[int(ctx)])]) for ctx in schedule]

def accuracy(actions:List[int],correct:List[int])->float:
    return float(np.mean([a==b for a,b in zip(actions,correct)]))

def action_divergence(a:List[int],b:List[int])->float:
    return float(np.mean([x!=y for x,y in zip(a,b)]))

def fork_histories(seed:int,base_archive:List[dict])->Tuple[List[dict],List[dict],List[dict]]:
    rng_a=np.random.default_rng(stable_seed(seed,"fork-A"))
    rng_b=np.random.default_rng(stable_seed(seed,"fork-B"))
    current_t=max([e["t"] for e in base_archive],default=-1)+1

    def updates(rng,label):
        ev=[]
        order=rng.permutation(RELEVANT_CONTEXTS)
        t=current_t
        for ctx in order:
            ev.append({"t":t,"context":int(ctx),"target":int(rng.integers(0,N_ACTIONS)),"round":"fork","relevant":True,"label":label})
            t+=1
        return ev

    a=updates(rng_a,"A")
    amap=latest_map(a)
    for attempt in range(100):
        b=updates(rng_b,f"B{attempt}")
        bmap=latest_map(b)
        if sum(amap[c]!=bmap[c] for c in RELEVANT_CONTEXTS)>=5:
            break
    else:
        raise RuntimeError("failed to generate sufficiently different fork")

    return base_archive+a, base_archive+b, base_archive+list(a)

def evaluate_seed(seed:int)->dict:
    perm=body_perm(seed)
    inv,body_acc=calibrate_inverse(perm)

    own_rel=generate_history(seed,"own-relevant",True)
    own_irr=generate_history(seed,"own-irrelevant",False)
    own_archive=own_rel+own_irr
    own_latest=latest_map(own_rel)

    donor_rel=generate_history(seed+100000,"same-body-donor-relevant",True)
    donor_irr=generate_history(seed+100000,"same-body-donor-irrelevant",False)
    donor_archive=donor_rel+donor_irr

    relevant_deleted=[e for e in own_archive if int(e["context"]) not in RELEVANT_CONTEXTS]
    irrelevant_deleted=[e for e in own_archive if int(e["context"]) not in IRRELEVANT_CONTEXTS]

    sched=probe_schedule(seed)
    correct=correct_actions(own_latest,inv,sched)
    input_hash=hashlib.sha256(json.dumps(sched).encode()).hexdigest()

    intact=policy_actions(own_archive,inv,sched,POP_PRIOR)
    rel_del=policy_actions(relevant_deleted,inv,sched,POP_PRIOR)
    irr_del=policy_actions(irrelevant_deleted,inv,sched,POP_PRIOR)
    donor=policy_actions(donor_archive,inv,sched,POP_PRIOR)
    recon=policy_actions([],inv,sched,POP_PRIOR)

    intact_acc=accuracy(intact,correct)
    rel_acc=accuracy(rel_del,correct)
    irr_acc=accuracy(irr_del,correct)
    donor_acc=accuracy(donor,correct)
    recon_acc=accuracy(recon,correct)

    fork_a,fork_b,fork_same=fork_histories(seed,own_archive)
    fork_sched=probe_schedule(seed,"fork-probe")
    act_a=policy_actions(fork_a,inv,fork_sched,POP_PRIOR)
    act_b=policy_actions(fork_b,inv,fork_sched,POP_PRIOR)
    act_same=policy_actions(fork_same,inv,fork_sched,POP_PRIOR)

    diff_div=action_divergence(act_a,act_b)
    same_div=action_divergence(act_a,act_same)

    row={
        "seed":int(seed),
        "intact_accuracy":intact_acc,
        "relevant_deleted_accuracy":rel_acc,
        "irrelevant_deleted_accuracy":irr_acc,
        "same_body_donor_accuracy":donor_acc,
        "population_reconstruction_accuracy":recon_acc,
        "relevant_deletion_damage":intact_acc-rel_acc,
        "same_body_donor_damage":intact_acc-donor_acc,
        "selective_deletion_specificity":irr_acc-rel_acc,
        "reconstruction_damage":intact_acc-recon_acc,
        "different_history_divergence":diff_div,
        "identical_history_divergence":same_div,
        "fork_excess_divergence":diff_div-same_div,
        "body_inverse_map_accuracy":body_acc,
        "common_current_input_equal":True,
        "common_probe_hash":input_hash,
        "archive_sizes":{
            "own":len(own_archive),
            "relevant_deleted":len(relevant_deleted),
            "irrelevant_deleted":len(irrelevant_deleted),
            "donor":len(donor_archive)
        }
    }
    row["seed_guard"]=bool(
        row["intact_accuracy"]>=0.95 and
        row["relevant_deletion_damage"]>=0.50 and
        row["same_body_donor_damage"]>=0.40 and
        row["selective_deletion_specificity"]>=0.45 and
        row["reconstruction_damage"]>=0.40 and
        row["different_history_divergence"]>=0.50 and
        row["identical_history_divergence"]<=0.05 and
        row["fork_excess_divergence"]>=0.45 and
        row["body_inverse_map_accuracy"]>=1.0-1e-12 and
        row["common_current_input_equal"]
    )
    return row

def summarize(rows:List[dict],required:int)->dict:
    numeric=[
        "intact_accuracy","relevant_deletion_damage","same_body_donor_damage",
        "selective_deletion_specificity","reconstruction_damage",
        "different_history_divergence","identical_history_divergence",
        "fork_excess_divergence","body_inverse_map_accuracy"
    ]
    med={f"median_{k}":float(np.median([r[k] for r in rows])) for k in numeric}
    guards=sum(bool(r["seed_guard"]) for r in rows)
    median_pass=bool(
        med["median_intact_accuracy"]>=0.95 and
        med["median_relevant_deletion_damage"]>=0.50 and
        med["median_same_body_donor_damage"]>=0.40 and
        med["median_selective_deletion_specificity"]>=0.45 and
        med["median_reconstruction_damage"]>=0.40 and
        med["median_different_history_divergence"]>=0.50 and
        med["median_identical_history_divergence"]<=0.05 and
        med["median_fork_excess_divergence"]>=0.45 and
        med["median_body_inverse_map_accuracy"]>=1.0-1e-12
    )
    return {**med,"seed_guard_count":int(guards),"seed_guard_required":int(required),"median_pass":median_pass,"pass":bool(median_pass and guards>=required)}

def run(mode:str,output:str):
    if mode=="smoke": seeds=[SMOKE_SEED]; required=0
    elif mode=="development": seeds=DEV_SEEDS; required=13
    else: seeds=CONF_SEEDS; required=28
    rows=[evaluate_seed(s) for s in seeds]
    summ=summarize(rows,required)
    if mode=="development":
        resolution="R53_DEVELOPMENT_AUTHORIZE_CONFIRM" if summ["pass"] else "R53_DEVELOPMENT_FAIL_NO_CONFIRM"
    elif mode=="confirmatory":
        resolution="R53_CAUSAL_IDENTITY_CONTINUITY_PASS_BOUNDED" if summ["pass"] else "R53_CAUSAL_IDENTITY_CONTINUITY_FAIL"
    else:
        resolution="R53_ENGINEERING_SMOKE"
    out={
        "campaign":"R53 causal identity continuity",
        "mode":mode,
        "resolution":resolution,
        "authorize_confirm":bool(summ["pass"]) if mode=="development" else None,
        "seeds":seeds,
        "population_prior_seeds":PRIOR_SEEDS,
        "population_prior":POP_PRIOR,
        "summary":summ,
        "records":rows,
        "boundaries":{
            "claim_scope":"bounded own-history causal continuity in synthetic deferred-commitment task",
            "phenomenal_self":"NOT_ESTABLISHED",
            "subjective_identity":"NOT_ESTABLISHED",
            "consciousness":"NOT_ESTABLISHED",
            "global_minimality":"NOT_ESTABLISHED",
            "E6b":"OPEN","E7":"OPEN","AGI_ASI":"NOT_ESTABLISHED",
            "core_version":"POLAR Core v1.1 unchanged"
        }
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"resolution":resolution,"summary":summ,"population_prior":POP_PRIOR},indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke","development","confirmatory"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(); run(a.mode,a.output)

if __name__=="__main__":
    main()
