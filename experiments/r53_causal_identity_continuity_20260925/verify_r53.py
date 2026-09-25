#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

DEV=list(range(2236001,2236017))
CONF=list(range(2237001,2237033))

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def guard(r):
    return bool(
        r["intact_accuracy"]>=0.95 and
        r["relevant_deletion_damage"]>=0.50 and
        r["same_body_donor_damage"]>=0.40 and
        r["selective_deletion_specificity"]>=0.45 and
        r["reconstruction_damage"]>=0.40 and
        r["different_history_divergence"]>=0.50 and
        r["identical_history_divergence"]<=0.05 and
        r["fork_excess_divergence"]>=0.45 and
        r["body_inverse_map_accuracy"]>=1.0-1e-12 and
        bool(r["common_current_input_equal"])
    )

def verify(path,mode):
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R53 causal identity continuity","campaign")
    req(d["mode"]==mode,"mode")
    seeds=DEV if mode=="development" else CONF
    required=13 if mode=="development" else 28
    req(d["seeds"]==seeds,"frozen seeds")
    counts=sum(guard(r) for r in d["records"])
    req(all(bool(r["seed_guard"])==guard(r) for r in d["records"]),"seed guards recompute")
    req(d["summary"]["seed_guard_count"]==counts,"guard count")
    req(d["summary"]["seed_guard_required"]==required,"guard requirement")
    fields=[
        "intact_accuracy","relevant_deletion_damage","same_body_donor_damage",
        "selective_deletion_specificity","reconstruction_damage",
        "different_history_divergence","identical_history_divergence",
        "fork_excess_divergence","body_inverse_map_accuracy"
    ]
    med={}
    for k in fields:
        val=float(np.median([r[k] for r in d["records"]]))
        req(abs(val-d["summary"][f"median_{k}"])<=1e-12,f"median {k}")
        med[k]=val
    median_pass=bool(
        med["intact_accuracy"]>=0.95 and
        med["relevant_deletion_damage"]>=0.50 and
        med["same_body_donor_damage"]>=0.40 and
        med["selective_deletion_specificity"]>=0.45 and
        med["reconstruction_damage"]>=0.40 and
        med["different_history_divergence"]>=0.50 and
        med["identical_history_divergence"]<=0.05 and
        med["fork_excess_divergence"]>=0.45 and
        med["body_inverse_map_accuracy"]>=1.0-1e-12
    )
    req(bool(d["summary"]["median_pass"])==median_pass,"median pass")
    passed=median_pass and counts>=required
    req(bool(d["summary"]["pass"])==passed,"panel pass")
    exp=("R53_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R53_DEVELOPMENT_FAIL_NO_CONFIRM") if mode=="development" else ("R53_CAUSAL_IDENTITY_CONTINUITY_PASS_BOUNDED" if passed else "R53_CAUSAL_IDENTITY_CONTINUITY_FAIL")
    req(d["resolution"]==exp,"resolution")
    req(d["boundaries"]["phenomenal_self"]=="NOT_ESTABLISHED","phenomenal-self boundary")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")
    req(d["boundaries"]["E6b"]=="OPEN","E6b open")
    req(d["boundaries"]["E7"]=="OPEN","E7 open")
    return passed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True)
    p.add_argument("--confirmatory")
    a=p.parse_args()
    auth=verify(a.development,"development")
    if a.confirmatory:
        req(auth,"confirmation only after development pass")
        verify(a.confirmatory,"confirmatory")
    print("R53 integrity verification COMPLETE")

if __name__=="__main__":
    main()
