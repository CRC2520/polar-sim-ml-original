#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

DEV=[f"{x:03d}" for x in range(1,9)]
CONF=["009","010","011","012","014","015","016","017","018","019","020","021","022","023","024"]

def load_rows(paths):
    return sorted([json.loads(Path(p).read_text()) for p in paths],key=lambda r:r["subject"])

def aggregate(paths,phase,output):
    rows=load_rows(paths)
    expected=DEV if phase=="development" else CONF
    if [r["subject"] for r in rows] != expected:
        raise SystemExit("subject set mismatch")
    valid=sum(bool(r["scientifically_valid"]) for r in rows)
    D=sum(bool(r.get("D") and r["D"]["pass"]) for r in rows)
    C=sum(bool(r.get("C") and r["C"]["pass"]) for r in rows)
    R=sum(bool(r.get("R") and r["R"]["pass"]) for r in rows)
    joint=sum(bool(r.get("joint_pass",False)) for r in rows)
    vr=[r for r in rows if r["scientifically_valid"]]
    med={}
    for k in ("D","C","R"):
        vals=[float(r[k]["specificity"]) for r in vr if r.get(k)]
        med[f"{k}_median_specificity"]=float(np.median(vals)) if vals else float("nan")

    if phase=="development":
        invalid=valid<7
        passed=(not invalid and D>=6 and C>=6 and R>=6 and joint>=5 and all(np.isfinite(v) and v>0 for v in med.values()))
        resolution="E7_R1_DEVELOPMENT_INVALID" if invalid else "E7_R1_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "E7_R1_DEVELOPMENT_FAIL_NO_CONFIRM"
        out={
            "campaign":"E7-R1 prospective biological correspondence",
            "phase":"development","subjects":expected,
            "scientifically_valid_subjects":valid,
            "D_pass_count":D,"C_pass_count":C,"R_pass_count":R,"joint_pass_count":joint,
            "medians":med,"authorize_confirm":bool(passed),"resolution":resolution,
            "records":rows,
            "boundaries":{"observational_EEG":True,"causal_neural_necessity":"NOT_ESTABLISHED","biological_identity":"NOT_ESTABLISHED","E6b":"OPEN","consciousness":"NOT_ESTABLISHED"}
        }
    else:
        valid_ok=valid>=13; d_ok=D>=11; c_ok=C>=11; r_ok=R>=11; joint_ok=joint>=10
        specs=all(np.isfinite(v) and v>0 for v in med.values())
        full=bool(valid_ok and d_ok and c_ok and r_ok and joint_ok and specs)
        n=sum([d_ok,c_ok,r_ok])
        resolution="E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_INVALID" if not valid_ok else "E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PASS" if full else "E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PARTIAL" if n==2 else "E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_FAIL"
        out={
            "campaign":"E7-R1 prospective biological correspondence",
            "phase":"confirmatory","subjects":expected,
            "scientifically_valid_subjects":valid,
            "D_pass_count":D,"C_pass_count":C,"R_pass_count":R,"joint_pass_count":joint,
            "component_thresholds_reached":{"D":d_ok,"C":c_ok,"R":r_ok},
            "medians":med,"pass":full,"resolution":resolution,"records":rows,
            "boundaries":{"claim_scope":"prospective observational EEG correspondence in ds004117","causal_neural_necessity":"NOT_ESTABLISHED","biological_identity":"NOT_ESTABLISHED","global_minimality":"NOT_ESTABLISHED","E6b":"OPEN","AGI_ASI":"NOT_ESTABLISHED","consciousness":"NOT_ESTABLISHED"}
        }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--phase",choices=["development","confirmatory"],required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    aggregate(a.inputs,a.phase,a.output)

if __name__=="__main__":
    main()
