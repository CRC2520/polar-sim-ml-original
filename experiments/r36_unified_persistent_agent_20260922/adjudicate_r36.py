#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def git_blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()

    r=json.loads(Path(a.results).read_text())
    f=json.loads(Path(a.freeze).read_text())
    blob=git_blob(a.source)

    assert f["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert blob==f["source_git_blob_sha"],(blob,f["source_git_blob_sha"])
    assert r["mode"]=="confirm"
    assert r["seeds"]==f["confirmatory_seeds"]
    assert r["mandatory_core"]=="D+C+R (POLAR Core v1.1)"
    assert len(r["records"])==12

    t=f["criteria"]
    def checks(x):
        return {
          "same_agent_lifetime": x["same_agent_steps"]==t["same_agent_steps"] and x["memory_entries"]==t["memory_entries"],
          "D": x["D_prediction_damage"]>=t["D_prediction_damage_min"],
          "C": x["C_recurrent_context_gain"]>=t["C_recurrent_context_gain_min"],
          "R": x["R_relation_lesion_damage"]>=t["R_relation_lesion_damage_min"],
          "memory_reentry": x["memory_reentry_gain"]>=t["memory_reentry_gain_min"],
          "own_history": x["own_history_transplant_damage"]>=t["own_history_transplant_damage_min"],
          "source": x["source_balanced_accuracy"]>=t["source_balanced_accuracy_min"],
          "time": x["time_order_accuracy"]>=t["time_order_accuracy_min"],
          "metacog_calibration": x["metacog_calibration_gap"]>=t["metacog_calibration_gap_min"],
          "metacog_function": x["metacog_gate_gain"]>=t["metacog_gate_gain_min"],
          "planning": x["planning_gain"]>=t["planning_gain_min"],
          "stability": x["stable_fraction"]>=t["stable_fraction_min"],
          "models": x["models_learned"]>=t["models_learned_min"],
        }

    rows=[]
    for x in r["records"]:
        c=checks(x)
        rows.append({"seed":x["seed"],"checks":c,"pass_all":all(c.values()),"metrics":x})

    names=list(rows[0]["checks"])
    counts={k:sum(int(z["checks"][k]) for z in rows) for k in names}
    joint=sum(int(z["pass_all"]) for z in rows)

    medians={}
    for k in f["median_criteria"]:
        medians[k]=float(np.median([float(z["metrics"][k]) for z in rows]))
    median_checks={}
    for k,v in f["median_criteria"].items():
        if "min" in v:
            median_checks[k]=medians[k]>=v["min"]
        elif "eq" in v:
            median_checks[k]=abs(medians[k]-v["eq"])<1e-12
        else:
            raise ValueError(k)

    component_pass=all(v>=f["component_required"] for v in counts.values())
    joint_pass=joint>=f["joint_required"]
    median_pass=all(median_checks.values())
    strong=bool(component_pass and joint_pass and median_pass)

    if strong:
        resolution="R36_UNIFIED_PERSISTENT_AGENT_PASS_INTERNAL"
    elif component_pass and median_pass:
        resolution="R36_CAPACITIES_PRESENT_BUT_JOINT_COOCCURRENCE_FAIL"
    else:
        resolution="R36_FAIL"

    out={
      "campaign":f["campaign"],
      "source_git_blob_sha":blob,
      "confirmatory_seeds":f["confirmatory_seeds"],
      "component_counts":counts,
      "joint_pass_count":joint,
      "component_required":f["component_required"],
      "joint_required":f["joint_required"],
      "medians":medians,
      "median_checks":median_checks,
      "component_pass":component_pass,
      "joint_pass":joint_pass,
      "median_pass":median_pass,
      "strong_pass":strong,
      "resolution":resolution,
      "records":rows,
      "boundaries":f["fixed_boundaries"]
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
      "resolution":resolution,
      "component_counts":counts,
      "joint_pass_count":joint,
      "medians":medians,
      "median_checks":median_checks
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
