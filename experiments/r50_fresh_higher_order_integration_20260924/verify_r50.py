#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

COMPONENTS=[
"D","C","R","memory","own_history","source","time","metacog_calibration",
"metacog_gate","planning","planning_invocation","stability","same_lifetime","models"
]

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def verify_file(path,phase):
    d=json.loads(Path(path).read_text())
    req(d["campaign"]=="R50 fresh higher-order integration in persistent agent","campaign identity")
    req(d["phase"]==phase,"phase identity")
    expected=list(range(2206001,2206007)) if phase=="development" else list(range(2207001,2207017))
    required=4 if phase=="development" else 12
    req(d["seeds"]==expected,"frozen seeds")
    req(d["required_seed_count"]==required,"required pass count")
    counts={k:0 for k in COMPONENTS+["joint"]}
    for r in d["records"]:
        for k,v in r["passes"].items():
            counts[k]+=int(bool(v))
    req(counts==d["component_counts"],"component counts recompute")
    passed=(all(counts[k]>=required for k in COMPONENTS) and counts["joint"]>=required
            and d["medians"]["stable_fraction"]>=0.995 and d["medians"]["models_learned"]>=5)
    req(bool(d["pass"])==passed,"panel pass recomputes")
    exp=("R50_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R50_DEVELOPMENT_FAIL_NO_CONFIRM") if phase=="development" else ("R50_FRESH_HIGHER_ORDER_INTEGRATION_PASS" if passed else "R50_FRESH_HIGHER_ORDER_INTEGRATION_FAIL")
    req(d["resolution"]==exp,"resolution recomputes")
    req(d["boundaries"]["E6b"]=="OPEN","E6b remains open")
    req(d["boundaries"]["E7"]=="OPEN","E7 remains open")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")
    return passed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True)
    p.add_argument("--confirmatory")
    a=p.parse_args()
    auth=verify_file(a.development,"development")
    if a.confirmatory:
        req(auth,"confirmation only after authorized development")
        verify_file(a.confirmatory,"confirmatory")
    print("R50 integrity verification COMPLETE")

if __name__=="__main__":
    main()
