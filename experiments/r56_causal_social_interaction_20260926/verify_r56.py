#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

DEV=list(range(2256001,2256017))
CONF=list(range(2257001,2257033))
TOL=1e-12

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def guard(r):
    return bool(
        r["full_mean_reward"]>=0.76 and
        r["full_revisit_reward"]>=0.70 and
        r["full_responsive_contrast"]>=0.30 and
        r["full_responsive_cooperation"]>=0.65 and
        r["full_p3_cooperation"]<=0.35 and
        r["identity_effect"]>=0.05 and
        r["history_effect"]>=0.04 and
        r["relation_effect"]>=0.07 and
        r["long_horizon_effect"]>=0.18 and
        r["sham_gap"]<=TOL
    )

def verify(path,phase):
    d=json.loads(Path(path).read_text())
    expected=DEV if phase=="development" else CONF
    required=13 if phase=="development" else 28
    req(d["campaign"]=="R56 bounded causal social interaction","campaign")
    req(d["phase"]==phase,"phase")
    req(d["seeds"]==expected,"frozen seeds")
    req(all(bool(r["seed_guard"])==guard(r) for r in d["records"]),"seed guards recompute")
    count=sum(guard(r) for r in d["records"])
    req(d["summary"]["seed_guard_count"]==count,"guard count")
    med=lambda k:float(np.median([r[k] for r in d["records"]]))
    checks={
        "median_full_reward":"full_mean_reward",
        "median_identity_effect":"identity_effect",
        "median_history_effect":"history_effect",
        "median_relation_effect":"relation_effect",
        "median_long_horizon_effect":"long_horizon_effect",
        "median_sham_gap":"sham_gap"
    }
    for sk,rk in checks.items():
        req(abs(d["summary"][sk]-med(rk))<=TOL,sk)
    passed=bool(
        count>=required and
        d["summary"]["median_full_reward"]>=0.78 and
        d["summary"]["median_identity_effect"]>=0.06 and
        d["summary"]["median_history_effect"]>=0.05 and
        d["summary"]["median_relation_effect"]>=0.08 and
        d["summary"]["median_long_horizon_effect"]>=0.20 and
        d["summary"]["median_sham_gap"]<=TOL
    )
    req(bool(d["summary"]["pass"])==passed,"summary pass")
    exp=("R56_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R56_DEVELOPMENT_FAIL_NO_CONFIRM") if phase=="development" else ("R56_BOUNDED_CAUSAL_SOCIAL_INTERACTION_PASS" if passed else "R56_BOUNDED_CAUSAL_SOCIAL_INTERACTION_FAIL")
    req(d["resolution"]==exp,"resolution")
    req(d["boundaries"]["empathy"]=="NOT_ESTABLISHED","empathy boundary")
    req(d["boundaries"]["moral_agency"]=="NOT_ESTABLISHED","moral-agency boundary")
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
    print("R56 integrity verification COMPLETE")

if __name__=="__main__":
    main()
