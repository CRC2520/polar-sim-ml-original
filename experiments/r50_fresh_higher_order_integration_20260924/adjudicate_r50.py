#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

THRESHOLDS = {
    "D_prediction_damage": 0.0040,
    "C_recurrent_context_gain": 0.0005,
    "R_relation_lesion_damage": 0.0005,
    "memory_reentry_gain": 0.00010,
    "own_history_transplant_damage": 0.0008,
    "source_balanced_accuracy": 0.85,
    "time_order_accuracy": 0.98,
    "metacog_calibration_gap": 0.0,
    "metacog_gate_gain": 0.0,
    "planning_gain": 0.00020,
    "planning_invocation_rate_min": 0.02,
    "planning_invocation_rate_max": 0.50,
    "stable_fraction": 0.995,
    "same_agent_steps": 1800,
    "models_learned": 5,
}

COMPONENTS = [
    "D","C","R","memory","own_history","source","time","metacog_calibration",
    "metacog_gate","planning","planning_invocation","stability","same_lifetime","models"
]

def seed_flags(r):
    f = {
        "D": r["D_prediction_damage"] >= THRESHOLDS["D_prediction_damage"],
        "C": r["C_recurrent_context_gain"] >= THRESHOLDS["C_recurrent_context_gain"],
        "R": r["R_relation_lesion_damage"] >= THRESHOLDS["R_relation_lesion_damage"],
        "memory": r["memory_reentry_gain"] >= THRESHOLDS["memory_reentry_gain"],
        "own_history": r["own_history_transplant_damage"] >= THRESHOLDS["own_history_transplant_damage"],
        "source": r["source_balanced_accuracy"] >= THRESHOLDS["source_balanced_accuracy"],
        "time": r["time_order_accuracy"] >= THRESHOLDS["time_order_accuracy"],
        "metacog_calibration": r["metacog_calibration_gap"] > THRESHOLDS["metacog_calibration_gap"],
        "metacog_gate": r["metacog_gate_gain"] > THRESHOLDS["metacog_gate_gain"],
        "planning": r["planning_gain"] >= THRESHOLDS["planning_gain"],
        "planning_invocation": (
            r["planning_invocation_rate"] >= THRESHOLDS["planning_invocation_rate_min"]
            and r["planning_invocation_rate"] <= THRESHOLDS["planning_invocation_rate_max"]
        ),
        "stability": r["stable_fraction"] >= THRESHOLDS["stable_fraction"],
        "same_lifetime": int(r["same_agent_steps"]) == THRESHOLDS["same_agent_steps"],
        "models": int(r["models_learned"]) >= THRESHOLDS["models_learned"],
    }
    f["joint"] = all(f[k] for k in COMPONENTS)
    return f

def adjudicate(input_path, phase, output_path):
    d=json.loads(Path(input_path).read_text())
    rows=d["records"]
    if phase=="development":
        expected=list(range(2206001,2206007)); required=4
    else:
        expected=list(range(2207001,2207017)); required=12
    seeds=[int(r["seed"]) for r in rows]
    if seeds != expected:
        raise SystemExit(f"seed mismatch {seeds} != {expected}")

    enriched=[]
    counts={k:0 for k in COMPONENTS+["joint"]}
    for r in rows:
        rr=dict(r)
        rr["passes"]=seed_flags(r)
        for k,v in rr["passes"].items():
            counts[k]+=int(bool(v))
        enriched.append(rr)

    medians={}
    for k in rows[0]:
        if k=="seed": continue
        vals=[r[k] for r in rows]
        if isinstance(vals[0], (int,float)):
            medians[k]=float(np.median(vals))

    components_ok=all(counts[k]>=required for k in COMPONENTS)
    joint_ok=counts["joint"]>=required
    guards_ok=(
        medians.get("stable_fraction",0)>=THRESHOLDS["stable_fraction"]
        and medians.get("models_learned",0)>=THRESHOLDS["models_learned"]
    )
    passed=bool(components_ok and joint_ok and guards_ok)

    if phase=="development":
        resolution="R50_DEVELOPMENT_AUTHORIZE_CONFIRM" if passed else "R50_DEVELOPMENT_FAIL_NO_CONFIRM"
    else:
        resolution="R50_FRESH_HIGHER_ORDER_INTEGRATION_PASS" if passed else "R50_FRESH_HIGHER_ORDER_INTEGRATION_FAIL"

    out={
        "campaign":"R50 fresh higher-order integration in persistent agent",
        "phase":phase,
        "seeds":expected,
        "required_seed_count":required,
        "thresholds":THRESHOLDS,
        "component_counts":counts,
        "medians":medians,
        "pass":passed,
        "authorize_confirm": passed if phase=="development" else None,
        "resolution":resolution,
        "records":enriched,
        "boundaries":{
            "core_version":"POLAR Core v1.1 unchanged",
            "same_program":True,
            "global_minimality":"OPEN",
            "E6b":"OPEN",
            "E7":"OPEN",
            "identity":"OPERATIONAL_HISTORY_SPECIFICITY_ONLY",
            "consciousness":"NOT_ESTABLISHED",
            "open_ended_autonomy":"NOT_ESTABLISHED"
        }
    }
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "resolution":resolution,
        "pass":passed,
        "required_seed_count":required,
        "joint_count":counts["joint"],
        "component_counts":counts,
        "medians":medians
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
