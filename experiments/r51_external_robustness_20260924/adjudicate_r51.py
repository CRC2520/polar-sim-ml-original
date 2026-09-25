#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

TASKS=("autoencode","repeat_previous")
MILD=("NOISE_05","OCCLUSION_10","STALE_05")
ISO=1e-12

def task_guard(t):
    if t["CLEAN"]["full_accuracy"] < 0.98: return False
    for p in MILD:
        if t[p]["full_accuracy"] < 0.82: return False
    if t["SEMANTIC_SHIFT_HALF"]["full_accuracy"] < 0.45: return False

    for k in ("d_effect","c_effect","r_effect"):
        if t["CLEAN"][k] < 0.55: return False
        for p in MILD:
            if t[p][k] < 0.35: return False
        if t["SEMANTIC_SHIFT_HALF"][k] < 0.15: return False

    if t["CLEAN"]["iso_gap"] > ISO: return False
    if t["CLEAN"]["iso_action_agreement"] < 1.0-ISO: return False
    return True

def seed_guard(r):
    return all(task_guard(r["tasks"][task]) for task in TASKS)

def median_summary(rows):
    out={}
    for task in TASKS:
        out[task]={}
        for p in ("CLEAN","NOISE_05","OCCLUSION_10","STALE_05","SEMANTIC_SHIFT_HALF"):
            vals=[r["tasks"][task][p] for r in rows]
            item={}
            for k in ("full_accuracy","d_effect","c_effect","r_effect"):
                item[k]=float(np.median([v[k] for v in vals]))
            if p=="CLEAN":
                item["iso_gap"]=float(np.median([v["iso_gap"] for v in vals]))
                item["iso_action_agreement"]=float(np.median([v["iso_action_agreement"] for v in vals]))
            out[task][p]=item
    return out

def median_pass(m):
    fake={"tasks":m}
    return all(task_guard(fake["tasks"][task]) for task in TASKS)

def adjudicate(input_path,phase,output_path):
    d=json.loads(Path(input_path).read_text())
    rows=d["records"]
    expected=list(range(2216001,2216017)) if phase=="development" else list(range(2217001,2217033))
    required=13 if phase=="development" else 28
    if [int(r["seed"]) for r in rows] != expected:
        raise SystemExit("R51 seed mismatch")

    enriched=[]
    for r in rows:
        rr=dict(r)
        rr["seed_guard"]=seed_guard(r)
        enriched.append(rr)
    n=sum(bool(r["seed_guard"]) for r in enriched)
    med=median_summary(rows)
    med_ok=median_pass(med)
    passed=bool(n>=required and med_ok)

    if phase=="development":
        resolution="R51_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R51_DEVELOPMENT_FAIL_NO_CONFIRM"
    else:
        resolution="R51_EXTERNAL_ROBUSTNESS_BATTERY_PASS" if passed else "R51_EXTERNAL_ROBUSTNESS_BATTERY_FAIL"

    out={
        "campaign":"R51 fresh external robustness battery",
        "phase":phase,
        "source":d["source"],
        "seeds":expected,
        "required_seed_guards":required,
        "seed_guard_count":n,
        "median_summary":med,
        "median_criteria_pass":med_ok,
        "pass":passed,
        "authorize_confirm": passed if phase=="development" else None,
        "resolution":resolution,
        "records":enriched,
        "boundaries":d["boundaries"],
    }
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "resolution":resolution,
        "seed_guard_count":n,
        "required_seed_guards":required,
        "median_criteria_pass":med_ok,
        "median_summary":med
    },indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--phase",choices=["development","confirmatory"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    adjudicate(a.input,a.phase,a.output)

if __name__=="__main__":
    main()
