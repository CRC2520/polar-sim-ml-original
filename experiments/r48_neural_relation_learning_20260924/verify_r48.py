#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

DEV=[2196001,2196002,2196003]
CONF=[2197001,2197002,2197003]
ISO=1e-12

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def dev_ok(r):
    return (
        r["intact_accuracy"]>=0.90 and
        r["d_effect"]>=0.25 and
        r["c_effect"]>=0.25 and
        r["r_effect"]>=0.20 and
        r["iso_gap"]<=ISO and
        r["iso_action_agreement"]>=1.0-ISO
    )

def conf_ok(r):
    return (
        r["intact_accuracy"]>=0.80 and
        r["d_effect"]>=0.15 and
        r["c_effect"]>=0.30 and
        r["r_effect"]>=0.15 and
        r["iso_gap"]<=ISO and
        r["iso_action_agreement"]>=1.0-ISO
    )

def verify_dev(path):
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R48 neural relation discovery from reward","campaign")
    req(d["phase"]=="development","development phase")
    req(d["training_seeds"]==DEV,"development seeds")
    rows=sorted(d["checkpoints"],key=lambda x:int(x["train_seed"]))
    req([int(r["train_seed"]) for r in rows]==DEV,"checkpoint seeds")
    req(all(r["source"]["commit"]=="e397e5eac9965f9963d18c9f455cd1983bca14fb" for r in rows),"external source commit")
    req(all(r["source"]["environment"]=="CountRecallMedium" for r in rows),"development environment")
    req(all(r["arch"]=="LSTM" for r in rows),"development LSTM only")
    req(all(bool(r["eligible"])==dev_ok(r) for r in rows),"development eligibility recomputes")
    n=sum(dev_ok(r) for r in rows)
    req(d["eligible_checkpoints"]==n,"development eligible count")
    req(d["authorize_confirm"]==(n>=2),"development authorization")
    exp="R48_DEVELOPMENT_AUTHORIZE_CONFIRM" if n>=2 else "R48_DEVELOPMENT_FAIL_NO_CONFIRM"
    req(d["resolution"]==exp,"development resolution")
    return n>=2

def verify_confirm(path,authorized):
    req(authorized,"confirmatory opened only after authorized development")
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R48 neural relation discovery from reward","campaign")
    req(d["phase"]=="confirmatory","confirmatory phase")
    confirms=[]
    for arch in ("LSTM","GRU"):
        s=d["architectures"][arch]
        rows=sorted(s["checkpoints"],key=lambda x:int(x["train_seed"]))
        req([int(r["train_seed"]) for r in rows]==CONF,f"{arch} confirm seeds")
        req(all(r["source"]["environment"]=="CountRecallHard" for r in rows),f"{arch} hard environment")
        req(all(r["arch"]==arch for r in rows),f"{arch} architecture identity")
        req(all(bool(r["eligible"])==conf_ok(r) for r in rows),f"{arch} eligibility recomputes")
        n=sum(conf_ok(r) for r in rows)
        req(s["eligible_checkpoints"]==n,f"{arch} eligible count")
        req(s["confirm"]==(n>=2),f"{arch} confirm rule")
        confirms.append(n>=2)
    n=sum(confirms)
    exp=("R48_NEURAL_RELATION_LEARNABILITY_PASS" if n==2 else
         "R48_NEURAL_RELATION_LEARNABILITY_PARTIAL" if n==1 else
         "R48_NEURAL_RELATION_LEARNABILITY_FAIL")
    req(d["resolution"]==exp,"final R48 resolution")
    req(d["boundaries"]["cross_domain_transport"]=="NOT_ESTABLISHED","cross-domain boundary")
    req(d["boundaries"]["global_minimality"]=="OPEN","minimality open")
    req(d["boundaries"]["E6b"]=="OPEN","E6b open")
    req(d["boundaries"]["E7"]=="OPEN","E7 open")
    req(d["boundaries"]["POLAR_superiority"]=="NOT_ESTABLISHED","no superiority")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","no consciousness claim")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True)
    p.add_argument("--confirmatory")
    a=p.parse_args()
    authorized=verify_dev(a.development)
    if a.confirmatory:
        verify_confirm(a.confirmatory,authorized)
    print("R48 integrity verification COMPLETE")

if __name__=="__main__":
    main()
