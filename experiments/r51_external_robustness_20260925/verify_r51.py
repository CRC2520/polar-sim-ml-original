#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

TASKS=("ConcentrationHard","CountRecallHard")
PERTS=("CLEAN","SUBSTITUTE10","STALE_BURST","MIXED")
DEV=list(range(2216001,2216013))
CONF=list(range(2217001,2217033))
ISO=1e-12
CLEAN={"full_metric":0.90,"d_effect":0.20,"c_effect":0.30,"r_effect":0.20}
PERT={"full_metric":0.55,"robustness_ratio":0.55,"d_effect":0.10,"c_effect":0.15,"r_effect":0.10}

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def cell_pass(c,pert):
    if pert=="CLEAN":
        return bool(c["full_metric"]>=CLEAN["full_metric"] and c["d_effect"]>=CLEAN["d_effect"] and c["c_effect"]>=CLEAN["c_effect"] and c["r_effect"]>=CLEAN["r_effect"] and c["iso_gap"]<=ISO and c["iso_action_agreement"]>=1-ISO)
    return bool(c["full_metric"]>=PERT["full_metric"] and c["robustness_ratio"]>=PERT["robustness_ratio"] and c["d_effect"]>=PERT["d_effect"] and c["c_effect"]>=PERT["c_effect"] and c["r_effect"]>=PERT["r_effect"] and c["iso_gap"]<=ISO and c["iso_action_agreement"]>=1-ISO)

def verify(path,mode):
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R51 fresh external robustness battery","campaign")
    req(d["mode"]==mode,"mode")
    seeds=DEV if mode=="development" else CONF
    required=10 if mode=="development" else 26
    req(d["seeds"]==seeds,"frozen seeds")
    req(d["source"]["commit"]=="e397e5eac9965f9963d18c9f455cd1983bca14fb","external source")
    req(d["source"]["tasks"]==list(TASKS),"task identities")
    guard=0
    for r in d["records"]:
        all_cells=True
        for task in TASKS:
            for pert in PERTS:
                c=r["cells"][f"{task}:{pert}"]
                ok=cell_pass(c,pert)
                req(bool(c["cell_pass"])==ok,f"{r['seed']} {task} {pert} cell")
                all_cells = all_cells and ok
        req(bool(r["seed_guard"])==all_cells,f"{r['seed']} seed guard")
        guard += int(all_cells)
    req(d["summary"]["seed_guard_count"]==guard,"seed guard count")
    req(d["summary"]["seed_guard_required"]==required,"seed guard requirement")
    all_medians=True
    for task in TASKS:
        for pert in PERTS:
            key=f"{task}:{pert}"
            s=d["summary"]["cell_summaries"][key]
            rows=[r["cells"][key] for r in d["records"]]
            fields=["full_metric","robustness_ratio","d_effect","c_effect","r_effect","iso_gap","iso_action_agreement"]
            for field in fields:
                calc=float(np.median([x[field] for x in rows]))
                req(abs(calc-s[f"median_{field}"])<=1e-12,f"{key} median {field}")
            pseudo={
                "full_metric":s["median_full_metric"],
                "robustness_ratio":s["median_robustness_ratio"],
                "d_effect":s["median_d_effect"],
                "c_effect":s["median_c_effect"],
                "r_effect":s["median_r_effect"],
                "iso_gap":s["median_iso_gap"],
                "iso_action_agreement":s["median_iso_action_agreement"],
            }
            ok=cell_pass(pseudo,pert)
            req(bool(s["pass"])==ok,f"{key} median cell pass")
            all_medians = all_medians and ok
    passed=all_medians and guard>=required
    req(bool(d["summary"]["pass"])==passed,"panel pass")
    exp=("R51_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R51_DEVELOPMENT_FAIL_NO_CONFIRM") if mode=="development" else ("R51_EXTERNAL_ROBUSTNESS_PASS_SAME_PROGRAM" if passed else "R51_EXTERNAL_ROBUSTNESS_FAIL")
    req(d["resolution"]==exp,"resolution")
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
    print("R51 integrity verification COMPLETE")

if __name__=="__main__":
    main()
