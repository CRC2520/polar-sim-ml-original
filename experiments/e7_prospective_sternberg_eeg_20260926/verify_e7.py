#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

DEV=[f"{x:02d}" for x in range(4,20)]
CONF=[f"{x:02d}" for x in range(21,53)]

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def counts(rows):
    valid=sum(bool(r["scientifically_valid"]) for r in rows)
    D=sum(bool(r.get("D") and r["D"]["pass"]) for r in rows)
    C=sum(bool(r.get("C") and r["C"]["pass"]) for r in rows)
    R=sum(bool(r.get("R") and r["R"]["pass"]) for r in rows)
    joint=sum(bool(r.get("joint_pass")) for r in rows)
    med={}
    vr=[r for r in rows if r["scientifically_valid"]]
    for key in ("D","C","R"):
        vals=[float(r[key]["specificity"]) for r in vr if r.get(key)]
        med[f"{key}_median_specificity"]=float(np.median(vals)) if vals else float("nan")
    return valid,D,C,R,joint,med

def verify_dev(path):
    d=json.loads(Path(path).read_text())
    req(d["phase"]=="development","development phase")
    req(d["subjects"]==DEV,"development subjects")
    valid,D,C,R,joint,med=counts(d["records"])
    req(d["scientifically_valid_subjects"]==valid,"valid subject count")
    req((d["D_pass_count"],d["C_pass_count"],d["R_pass_count"],d["joint_pass_count"])==(D,C,R,joint),"D/C/R/joint counts")
    invalid=valid<14
    passed=(not invalid and joint>=10 and D>=11 and C>=11 and R>=10)
    exp="E7_DEVELOPMENT_INVALID" if invalid else "E7_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "E7_DEVELOPMENT_FAIL_NO_CONFIRM"
    req(d["resolution"]==exp,"development resolution")
    req(bool(d["authorize_confirm"])==passed,"development authorization")
    return passed

def verify_conf(path,authorized):
    req(authorized,"confirmation opened only after authorized development")
    d=json.loads(Path(path).read_text())
    req(d["phase"]=="confirmatory","confirmatory phase")
    req(d["subjects"]==CONF,"confirmatory subjects")
    valid,D,C,R,joint,med=counts(d["records"])
    req(d["scientifically_valid_subjects"]==valid,"valid subject count")
    req((d["D_pass_count"],d["C_pass_count"],d["R_pass_count"],d["joint_pass_count"])==(D,C,R,joint),"D/C/R/joint counts")
    valid_ok=valid>=28; d_ok=D>=22; c_ok=C>=22; r_ok=R>=21; joint_ok=joint>=21
    specs=all(np.isfinite(v) and v>0 for v in med.values())
    full=valid_ok and d_ok and c_ok and r_ok and joint_ok and specs
    n=sum([d_ok,c_ok,r_ok])
    exp="E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_INVALID" if not valid_ok else "E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PASS" if full else "E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PARTIAL" if n==2 else "E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_FAIL"
    req(d["resolution"]==exp,"confirmatory resolution")
    req(d["boundaries"]["causal_neural_necessity"]=="NOT_ESTABLISHED","no causal-neural overclaim")
    req(d["boundaries"]["E6b"]=="OPEN","E6b open")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True)
    p.add_argument("--confirmatory")
    a=p.parse_args()
    auth=verify_dev(a.development)
    if a.confirmatory: verify_conf(a.confirmatory,auth)
    print("E7 integrity verification COMPLETE")

if __name__=="__main__":
    main()
