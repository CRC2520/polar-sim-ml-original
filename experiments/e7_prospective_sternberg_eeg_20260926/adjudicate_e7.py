#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

DEV=[f"{x:02d}" for x in range(4,20)]
CONF=[f"{x:02d}" for x in range(21,53)]

def load_rows(paths):
    rows=[json.loads(Path(p).read_text()) for p in paths]
    return sorted(rows,key=lambda r:r["subject"])

def aggregate(paths,phase,output):
    rows=load_rows(paths)
    expected=DEV if phase=="development" else CONF
    if [r["subject"] for r in rows] != expected:
        raise SystemExit(f"subject set mismatch: {[r['subject'] for r in rows]} != {expected}")

    valid=sum(bool(r["scientifically_valid"]) for r in rows)
    D=sum(bool(r.get("D") and r["D"]["pass"]) for r in rows)
    C=sum(bool(r.get("C") and r["C"]["pass"]) for r in rows)
    R=sum(bool(r.get("R") and r["R"]["pass"]) for r in rows)
    joint=sum(bool(r.get("joint_pass",False)) for r in rows)

    valid_rows=[r for r in rows if r["scientifically_valid"]]
    med={}
    for key in ("D","C","R"):
        vals=[float(r[key]["specificity"]) for r in valid_rows if r.get(key)]
        med[f"{key}_median_specificity"]=float(np.median(vals)) if vals else float("nan")

    if phase=="development":
        is_invalid=valid<14
        passed=(not is_invalid and joint>=10 and D>=11 and C>=11 and R>=10)
        if is_invalid:
            resolution="E7_DEVELOPMENT_INVALID"
        elif passed:
            resolution="E7_DEVELOPMENT_AUTHORIZE_CONFIRM"
        else:
            resolution="E7_DEVELOPMENT_FAIL_NO_CONFIRM"
        out={
            "campaign":"E7 prospective biological correspondence",
            "phase":"development",
            "subjects":expected,
            "scientifically_valid_subjects":valid,
            "D_pass_count":D,
            "C_pass_count":C,
            "R_pass_count":R,
            "joint_pass_count":joint,
            "medians":med,
            "authorize_confirm":bool(passed),
            "resolution":resolution,
            "records":rows,
            "boundaries":{
                "observational_EEG":True,
                "causal_neural_necessity":"NOT_ESTABLISHED",
                "biological_identity":"NOT_ESTABLISHED",
                "E6b":"OPEN",
                "consciousness":"NOT_ESTABLISHED"
            }
        }
    else:
        valid_ok=valid>=28
        d_ok=D>=22
        c_ok=C>=22
        r_ok=R>=21
        joint_ok=joint>=21
        specs_positive=all(np.isfinite(v) and v>0 for v in med.values())
        full=bool(valid_ok and d_ok and c_ok and r_ok and joint_ok and specs_positive)
        n_components=sum([d_ok,c_ok,r_ok])
        if not valid_ok:
            resolution="E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_INVALID"
        elif full:
            resolution="E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PASS"
        elif n_components==2:
            resolution="E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PARTIAL"
        else:
            resolution="E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_FAIL"
        out={
            "campaign":"E7 prospective biological correspondence",
            "phase":"confirmatory",
            "subjects":expected,
            "scientifically_valid_subjects":valid,
            "D_pass_count":D,
            "C_pass_count":C,
            "R_pass_count":R,
            "joint_pass_count":joint,
            "component_thresholds_reached":{"D":d_ok,"C":c_ok,"R":r_ok},
            "medians":med,
            "resolution":resolution,
            "pass":full,
            "records":rows,
            "boundaries":{
                "claim_scope":"prospective observational EEG correspondence in ds005095",
                "causal_neural_necessity":"NOT_ESTABLISHED",
                "biological_identity":"NOT_ESTABLISHED",
                "global_minimality":"NOT_ESTABLISHED",
                "E6b":"OPEN",
                "AGI_ASI":"NOT_ESTABLISHED",
                "consciousness":"NOT_ESTABLISHED"
            }
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
