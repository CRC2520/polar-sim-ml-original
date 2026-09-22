#!/usr/bin/env python3
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/"r36_r1_metacog_stabilization_20260922"/"r36_r1_metacog.py"
FREEZE_PATH=ROOT/"r36_r1_metacog_stabilization_20260922"/"R36_R1_CONFIRM_FREEZE.json"

spec=importlib.util.spec_from_file_location("r36r1base", BASE_PATH)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OriginalAgent=base.PersistentAgent
PARENT_FREEZE=json.loads(FREEZE_PATH.read_text())

VARIANTS=[
    ("BASE_R1",0.78,140,0.00,0),
    ("FAST_CONTEXT",0.70,140,0.00,1),
    ("LONG_MEMORY",0.78,220,0.00,2),
    ("ROBUST_PLAN",0.78,140,0.05,3),
    ("COMBINED_A",0.70,220,0.05,5),
    ("COMBINED_B",0.72,220,0.03,4),
]

class RobustAgent(OriginalAgent):
    PLAN_MARGIN_SCALE=0.0
    def act(self, obs, target):
        x=np.asarray(obs[:base.N],float)
        cue=obs[base.N:base.N+2]
        key=self.update_cue(cue)
        self.key_visits[key]=self.key_visits.get(key,0)+1
        visit=self.key_visits[key]
        m=self.model(key)
        if visit<=42 or not m.fitted:
            action=(visit-1)%len(base.ACTIONS)
            conf=self.confidence(key,x,action)
            use_plan=False
            predicted_plan_advantage=0.0
        else:
            a_plan=self.plan_action(x,target,key)
            a_myop=self.myopic_action(x,target,key)
            conf_plan=self.confidence(key,x,a_plan)
            plan_cost=self.model_two_step_cost(x,target,key,a_plan)
            myop_cost=self.model_two_step_cost(x,target,key,a_myop)
            predicted_plan_advantage=float(myop_cost-plan_cost)
            local_err=float(m.expected_error(x,a_plan))
            margin=max(base.META_PLAN_ADV_MIN,self.PLAN_MARGIN_SCALE*local_err)
            use_plan=bool(conf_plan>=base.META_PLAN_CONF_MIN and predicted_plan_advantage>=margin)
            action=a_plan if use_plan else a_myop
            conf=self.confidence(key,x,action)
        pred=m.predict(x,action)
        self.last_pred=pred
        self.last_key=key
        self.last_conf=conf
        return action,key,conf,pred,use_plan,predicted_plan_advantage

def checks(x,t):
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
      "planning_invocation": x["planning_invocation_rate"]>=t["planning_invocation_rate_min"],
      "stability": x["stable_fraction"]>=t["stable_fraction_min"],
      "models": x["models_learned"]>=t["models_learned_min"],
    }

def run_variant(name,cue_ema,window,margin,seeds):
    base.CUE_EMA=float(cue_ema)
    base.MODEL_WINDOW=int(window)
    RobustAgent.PLAN_MARGIN_SCALE=float(margin)
    base.PersistentAgent=RobustAgent
    recs=[base.run_seed(s) for s in seeds]
    t=PARENT_FREEZE["criteria"]
    rows=[]
    for x in recs:
        c=checks(x,t)
        rows.append({"seed":x["seed"],"checks":c,"pass_all":all(c.values()),"metrics":x})
    names=list(rows[0]["checks"])
    counts={k:sum(int(r["checks"][k]) for r in rows) for k in names}
    joint=sum(int(r["pass_all"]) for r in rows)
    medians={}
    for k in PARENT_FREEZE["median_criteria"]:
        medians[k]=float(np.median([float(r["metrics"][k]) for r in rows]))
    return {
        "name":name,
        "cue_ema":cue_ema,
        "model_window":window,
        "plan_margin_scale":margin,
        "records":rows,
        "component_counts":counts,
        "joint_pass_count":joint,
        "medians":medians,
    }

def eligible(v,n):
    counts=v["component_counts"]
    if any(counts[k]<6 for k in counts):
        return False
    return counts["same_agent_lifetime"]==n and counts["stability"]==n and counts["models"]==n

def select_variant(results):
    simplicity={name:rank for name,_,_,_,rank in VARIANTS}
    elig=[v for v in results if eligible(v,len(v["records"]))]
    if not elig:
        return None
    def key(v):
        c=v["component_counts"]
        return (
            v["joint_pass_count"],
            c["memory_reentry"]+c["planning"],
            c["metacog_calibration"]+c["metacog_function"],
            -simplicity[v["name"]],
        )
    return max(elig,key=key)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",default="2026001:2026008")
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    lo,hi=map(int,a.seeds.split(":"))
    seeds=list(range(lo,hi+1))
    assert seeds==list(range(2026001,2026009))
    results=[]
    for name,cue,window,margin,_ in VARIANTS:
        results.append(run_variant(name,cue,window,margin,seeds))
    chosen=select_variant(results)
    out={
      "campaign":"R36-R2 joint robustness development",
      "development_seeds":seeds,
      "parent":"R36_R1_COMPONENTS_PASS_JOINT_FAIL",
      "parent_thresholds":"R36_R1_CONFIRM_FREEZE.json unchanged",
      "variants":results,
      "selected_variant":None if chosen is None else {
          "name":chosen["name"],
          "cue_ema":chosen["cue_ema"],
          "model_window":chosen["model_window"],
          "plan_margin_scale":chosen["plan_margin_scale"],
          "joint_pass_count":chosen["joint_pass_count"],
          "component_counts":chosen["component_counts"],
          "medians":chosen["medians"],
      },
      "confirmatory_authorized":chosen is not None,
      "confirmatory_seeds_status":"UNOPENED_2027001_2027012",
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    compact={
      v["name"]:{
        "joint":v["joint_pass_count"],
        "memory":v["component_counts"]["memory_reentry"],
        "planning":v["component_counts"]["planning"],
        "meta_cal":v["component_counts"]["metacog_calibration"],
        "meta_gate":v["component_counts"]["metacog_function"],
        "eligible":eligible(v,len(seeds)),
      } for v in results
    }
    print(json.dumps({"variants":compact,"selected":out["selected_variant"]},indent=2,sort_keys=True))

if __name__=="__main__":
    main()
