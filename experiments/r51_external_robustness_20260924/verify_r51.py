#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def verify(path,phase):
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R51 fresh external robustness battery","campaign")
    req(d["phase"]==phase,"phase")
    expected=list(range(2216001,2216017)) if phase=="development" else list(range(2217001,2217033))
    required=13 if phase=="development" else 28
    req(d["seeds"]==expected,"frozen seed set")
    req(d["required_seed_guards"]==required,"frozen guard count")
    n=sum(bool(r["seed_guard"]) for r in d["records"])
    req(n==d["seed_guard_count"],"seed guard count recomputes")
    passed=(n>=required and bool(d["median_criteria_pass"]))
    req(bool(d["pass"])==passed,"panel pass recomputes")
    exp=("R51_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R51_DEVELOPMENT_FAIL_NO_CONFIRM") if phase=="development" else ("R51_EXTERNAL_ROBUSTNESS_BATTERY_PASS" if passed else "R51_EXTERNAL_ROBUSTNESS_BATTERY_FAIL")
    req(d["resolution"]==exp,"resolution recomputes")
    req(d["source"]["commit"]=="e397e5eac9965f9963d18c9f455cd1983bca14fb","external source")
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
        req(auth,"confirmatory opened only after authorized development")
        verify(a.confirmatory,"confirmatory")
    print("R51 integrity verification COMPLETE")

if __name__=="__main__":
    main()
