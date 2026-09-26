#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

DEV=[f"{x:03d}" for x in range(1,9)]
CONF=["009","010","011","012","014","015","016","017","018","019","020","021","022","023","024"]

def req(v,msg):
    if not v: raise AssertionError(msg)
    print("PASS:",msg)

def recount(rows):
    valid=sum(bool(r["scientifically_valid"]) for r in rows)
    D=sum(bool(r.get("D") and r["D"]["pass"]) for r in rows)
    C=sum(bool(r.get("C") and r["C"]["pass"]) for r in rows)
    R=sum(bool(r.get("R") and r["R"]["pass"]) for r in rows)
    joint=sum(bool(r.get("joint_pass")) for r in rows)
    med={}
    vr=[r for r in rows if r["scientifically_valid"]]
    for k in ("D","C","R"):
        vals=[float(r[k]["specificity"]) for r in vr if r.get(k)]
        med[f"{k}_median_specificity"]=float(np.median(vals)) if vals else float("nan")
    return valid,D,C,R,joint,med

def verify_dev(path):
    d=json.loads(Path(path).read_text())
    req(d["phase"]=="development","development phase")
    req(d["subjects"]==DEV,"development subjects")
    valid,D,C,R,joint,med=recount(d["records"])
    req((valid,D,C,R,joint)==(d["scientifically_valid_subjects"],d["D_pass_count"],d["C_pass_count"],d["R_pass_count"],d["joint_pass_count"]),"development counts")
    invalid=valid<7
    passed=(not invalid and D>=6 and C>=6 and R>=6 and joint>=5 and all(np.isfinite(v) and v>0 for v in med.values()))
    exp="E7_R1_DEVELOPMENT_INVALID" if invalid else "E7_R1_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "E7_R1_DEVELOPMENT_FAIL_NO_CONFIRM"
    req(d["resolution"]==exp,"development resolution")
    req(bool(d["authorize_confirm"])==passed,"development authorization")
    return passed

def verify_conf(path,authorized):
    req(authorized,"confirmation only after authorized development")
    d=json.loads(Path(path).read_text())
    req(d["phase"]=="confirmatory","confirmatory phase")
    req(d["subjects"]==CONF,"confirmatory subjects")
    valid,D,C,R,joint,med=recount(d["records"])
    req((valid,D,C,R,joint)==(d["scientifically_valid_subjects"],d["D_pass_count"],d["C_pass_count"],d["R_pass_count"],d["joint_pass_count"]),"confirmatory counts")
    valid_ok=valid>=13; d_ok=D>=11; c_ok=C>=11; r_ok=R>=11; joint_ok=joint>=10
    specs=all(np.isfinite(v) and v>0 for v in med.values())
    full=valid_ok and d_ok and c_ok and r_ok and joint_ok and specs
    n=sum([d_ok,c_ok,r_ok])
    exp="E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_INVALID" if not valid_ok else "E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PASS" if full else "E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PARTIAL" if n==2 else "E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_FAIL"
    req(d["resolution"]==exp,"confirmatory resolution")
    req(d["boundaries"]["causal_neural_necessity"]=="NOT_ESTABLISHED","no causal overclaim")
    req(d["boundaries"]["E6b"]=="OPEN","E6b open")
    req(d["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True)
    p.add_argument("--confirmatory")
    a=p.parse_args()
    auth=verify_dev(a.development)
    if a.confirmatory: verify_conf(a.confirmatory,auth)
    print("E7-R1 integrity verification COMPLETE")

if __name__=="__main__":
    main()
