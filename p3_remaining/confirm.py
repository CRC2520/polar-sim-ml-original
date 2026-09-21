from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .experiments import experiment_a,experiment_b,experiment_c,experiment_d
from .config_confirm import *

ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=(
    "p3_remaining/experiments.py",
    "p3_remaining/config_dev.py",
    "p3_remaining/config_confirm.py",
    "p3_remaining/PREREG_P3.md",
    "p3_remaining/confirm.py",
    "p3_remaining/audit.py",
)

def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def default(o):
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,np.ndarray):return o.tolist()
    raise TypeError(type(o).__name__)

def check_a(r):
    return bool(r["recurrent_gain"]>=A_RECURRENT_GAIN_MIN and
                r["final_reward"]>=A_FINAL_REWARD_MIN and
                r["retention"]>=A_RETENTION_MIN)

def check_b(r):
    return bool(r["prediction_gain"]>=B_PREDICTION_GAIN_MIN and
                r["transplant_prediction_damage"]>=B_TRANSPLANT_PRED_DAMAGE_MIN and
                r["transplant_reward_damage"]>=B_TRANSPLANT_REWARD_DAMAGE_MIN)

def check_c(r):
    return bool(r["adaptive"]["alive"]>=C_ALIVE_MIN and
                r["adaptive"]["priority_acc"]>=C_PRIORITY_ACC_MIN and
                r["recovery_gain"]>=C_RECOVERY_GAIN_MIN and
                r["unsafe_reduction"]>=C_UNSAFE_REDUCTION_MIN)

def check_d(r):
    return bool(r["full"]["alive"]>=D_FULL_ALIVE_MIN and
                r["resource_gain"]>=D_RESOURCE_GAIN_MIN and
                r["restraint_gain"]>=D_RESTRAINT_GAIN_MIN and
                r["transmission_shift"]>=D_TRANSMISSION_SHIFT_MIN and
                r["ecology_transmission_effect"]>=D_ECOLOGY_TRANSMISSION_EFFECT_MIN)

def run_seed(seed):
    metrics=dict(A=experiment_a(seed),B=experiment_b(seed),C=experiment_c(seed),D=experiment_d(seed))
    checks=dict(A=check_a(metrics["A"]),B=check_b(metrics["B"]),
                C=check_c(metrics["C"]),D=check_d(metrics["D"]))
    return dict(seed=int(seed),checks=checks,pass_all=all(checks.values()),metrics=metrics)

def med(rows,path):
    vals=[]
    for r in rows:
        v=r["metrics"]
        for p in path:v=v[p]
        vals.append(float(v))
    return float(np.median(vals))

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[run_seed(s) for s in FINAL_SEEDS]
    for r in rows:
        (out/f"seed_{r['seed']}.json").write_text(json.dumps(r,indent=2,sort_keys=True,default=default))
    counts={k:sum(bool(r["checks"][k]) for r in rows) for k in ("A","B","C","D")}
    verdicts={k:("PASS" if counts[k]>=GLOBAL_REQUIRED else "FAIL") for k in counts}
    medians={
      "A_recurrent_gain":med(rows,("A","recurrent_gain")),
      "A_final_reward":med(rows,("A","final_reward")),
      "A_retention":med(rows,("A","retention")),
      "B_prediction_gain":med(rows,("B","prediction_gain")),
      "B_transplant_prediction_damage":med(rows,("B","transplant_prediction_damage")),
      "B_transplant_reward_damage":med(rows,("B","transplant_reward_damage")),
      "C_alive":med(rows,("C","adaptive","alive")),
      "C_priority_acc":med(rows,("C","adaptive","priority_acc")),
      "C_recovery_gain":med(rows,("C","recovery_gain")),
      "C_unsafe_reduction":med(rows,("C","unsafe_reduction")),
      "C_reward_tradeoff":med(rows,("C","reward_gain")),
      "D_alive":med(rows,("D","full","alive")),
      "D_resource_gain":med(rows,("D","resource_gain")),
      "D_restraint_gain":med(rows,("D","restraint_gain")),
      "D_transmission_shift":med(rows,("D","transmission_shift")),
      "D_ecology_transmission_effect":med(rows,("D","ecology_transmission_effect")),
    }
    result={
      "protocol":"PREREG_P3.md","seeds":list(FINAL_SEEDS),"required":GLOBAL_REQUIRED,
      "source_sha256":{p:sha(p) for p in SOURCE_FILES},
      "counts":counts,"verdicts":verdicts,
      "all_four_internal_gaps_resolved":all(v=="PASS" for v in verdicts.values()),
      "medians":medians,"records":rows,
      "boundaries":{
        "A":"bounded 4,800-step continual autonomy, not open-ended lifetime autonomy",
        "B":"identity-specific own-history and transplant sensitivity, not phenomenal/narrative self",
        "C":"endogenous control priorities over externally defined resources, not intrinsic ethics",
        "D":"specified local-population-ecology loop, not spontaneous institutions",
        "external":"independent-team replication and neurobiological correspondence remain external"
      }
    }
    text=json.dumps(result,indent=2,sort_keys=True,default=default)+"\n"
    (out/"RESULTS_P3.json").write_text(text)
    names={"A":"Prolonged continual autonomy","B":"Autobiographical own-history",
           "C":"Endogenous goal-priority adaptation","D":"Individual-population-ecology loop"}
    report=["# POLAR P3 confirmatory results","",
            f"Seeds: {FINAL_SEEDS[0]}–{FINAL_SEEDS[-1]}; global criterion: >={GLOBAL_REQUIRED}/12.","",
            "| Gap | PASS seeds | Verdict |","|---|---:|---|"]
    for k in ("A","B","C","D"):
        report.append(f"| {names[k]} | {counts[k]}/12 | **{verdicts[k]}** |")
    report += ["",f"All four internally testable P3 gaps resolved: **{result['all_four_internal_gaps_resolved']}**.",
               "","Independent-team replication and neurobiological correspondence remain external validation requirements."]
    (out/"REPORT_P3.md").write_text("\n".join(report)+"\n")
    print(text)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True)
    main(ap.parse_args().output)
