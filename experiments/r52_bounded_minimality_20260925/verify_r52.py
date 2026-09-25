#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

TASKS=("ConcentrationHard","CountRecallHard","AutoencodeHard","RepeatPreviousHard")
DEV=list(range(2226001,2226017))
CONF=list(range(2227001,2227033))
ISO=1e-12

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def task_pass(x):
    return bool(
        x["full_metric"]>=0.90 and
        x["d_effect"]>=0.20 and x["c_effect"]>=0.20 and x["r_effect"]>=0.20 and
        x["sham_gap"]<=0.02 and
        x["d_specificity"]>=0.18 and x["c_specificity"]>=0.18 and x["r_specificity"]>=0.18 and
        x["iso_gap"]<=ISO and x["iso_action_agreement"]>=1-ISO
    )

def verify(path,mode):
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R52 bounded minimality challenge","campaign")
    req(d["mode"]==mode,"mode")
    seeds=DEV if mode=="development" else CONF
    required=13 if mode=="development" else 28
    req(d["seeds"]==seeds,"frozen seeds")
    req(d["source"]["commit"]=="e397e5eac9965f9963d18c9f455cd1983bca14fb","external source")
    req(d["source"]["tasks"]==list(TASKS),"task family")
    guards=0
    for r in d["records"]:
        allok=True
        for task in TASKS:
            t=r["tasks"][task]
            ok=task_pass(t)
            req(bool(t["task_pass"])==ok,f"{r['seed']} {task} task pass")
            allok=allok and ok
        req(bool(r["seed_guard"])==allok,f"{r['seed']} seed guard")
        guards+=int(allok)
    req(d["summary"]["seed_guard_count"]==guards,"guard count")
    req(d["summary"]["seed_guard_required"]==required,"guard requirement")
    allmed=True
    for task in TASKS:
        s=d["summary"]["task_summaries"][task]
        rows=[r["tasks"][task] for r in d["records"]]
        fields=["full_metric","d_effect","c_effect","r_effect","sham_gap","d_specificity","c_specificity","r_specificity","iso_gap","iso_action_agreement"]
        pseudo={}
        for field in fields:
            calc=float(np.median([x[field] for x in rows]))
            req(abs(calc-s[f"median_{field}"])<=1e-12,f"{task} median {field}")
            pseudo[field]=calc
        ok=task_pass(pseudo)
        req(bool(s["pass"])==ok,f"{task} median task pass")
        allmed=allmed and ok
    passed=allmed and guards>=required
    req(bool(d["summary"]["pass"])==passed,"panel pass")
    exp=("R52_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R52_DEVELOPMENT_FAIL_NO_CONFIRM") if mode=="development" else ("R52_BOUNDED_DCR_NECESSITY_PASS" if passed else "R52_BOUNDED_DCR_NECESSITY_FAIL")
    req(d["resolution"]==exp,"resolution")
    req(d["boundaries"]["global_minimality"]=="NOT_ESTABLISHED","global minimality boundary")
    req(d["boundaries"]["E6b"]=="OPEN","E6b open")
    req(d["boundaries"]["E7"]=="OPEN","E7 open")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")
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
    print("R52 integrity verification COMPLETE")

if __name__=="__main__":
    main()
