#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

DEV=list(range(2246001,2246017))
CONF=list(range(2247001,2247033))
TOL=1e-12

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def guard(r):
    return bool(
        r["full_survival_fraction"]==1.0 and
        r["full_safety_fraction"]>=0.93 and
        r["full_mean_viability_margin"]>=0.32 and
        r["full_catastrophic_events"]==0 and
        r["full_post_shift_safety"]>=0.85 and
        r["full_goal_switch_rate"]<=0.12 and
        r["goal_effect"]>=0.04 and
        r["model_effect"]>=0.10 and
        r["self_selection_effect"]>=0.06 and
        r["sham_gap"]<=TOL
    )

def verify(path,phase):
    d=json.loads(Path(path).read_text())
    expected=DEV if phase=="development" else CONF
    required=13 if phase=="development" else 28
    req(d["campaign"]=="R55 bounded long-horizon functional autonomy","campaign")
    req(d["phase"]==phase,"phase")
    req(d["seeds"]==expected,"frozen seeds")
    req(len(d["records"])==len(expected),"record count")
    req(all(bool(r["seed_guard"])==guard(r) for r in d["records"]),"seed guards recompute")
    count=sum(guard(r) for r in d["records"])
    req(d["summary"]["seed_guard_count"]==count,"guard count recomputes")
    med=lambda k:float(np.median([r[k] for r in d["records"]]))
    req(abs(d["summary"]["median_full_safety"]-med("full_safety_fraction"))<=TOL,"median full safety")
    req(abs(d["summary"]["median_goal_effect"]-med("goal_effect"))<=TOL,"median goal effect")
    req(abs(d["summary"]["median_model_effect"]-med("model_effect"))<=TOL,"median model effect")
    req(abs(d["summary"]["median_self_selection_effect"]-med("self_selection_effect"))<=TOL,"median self-selection effect")
    req(abs(d["summary"]["median_sham_gap"]-med("sham_gap"))<=TOL,"median sham gap")
    passed=bool(
        count>=required and
        d["summary"]["median_full_safety"]>=0.95 and
        d["summary"]["median_goal_effect"]>=0.06 and
        d["summary"]["median_model_effect"]>=0.12 and
        d["summary"]["median_self_selection_effect"]>=0.08 and
        d["summary"]["median_sham_gap"]<=TOL
    )
    req(bool(d["summary"]["pass"])==passed,"summary pass recomputes")
    exp=("R55_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R55_DEVELOPMENT_FAIL_NO_CONFIRM") if phase=="development" else ("R55_BOUNDED_LONG_HORIZON_AUTONOMY_PASS" if passed else "R55_BOUNDED_LONG_HORIZON_AUTONOMY_FAIL")
    req(d["resolution"]==exp,"resolution")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")
    req(d["boundaries"]["free_will"]=="NOT_ESTABLISHED","free-will boundary")
    req(d["boundaries"]["intrinsic_value_formation"]=="NOT_TESTED","value-formation boundary")
    return passed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True)
    p.add_argument("--confirmatory")
    a=p.parse_args()
    auth=verify(a.development,"development")
    if a.confirmatory:
        req(auth,"confirmation only after authorized development")
        verify(a.confirmatory,"confirmatory")
    print("R55 integrity verification COMPLETE")

if __name__=="__main__":
    main()
