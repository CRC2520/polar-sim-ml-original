from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config_dev import DEV_SEEDS,STRESS_SEEDS
from .experiments import run_seed

def default(o):
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)

def summarize(rows):
    counts={k:sum(bool(r[k]["pass_seed"]) for r in rows) for k in ("A","B","C","D")}
    def med(path):
        vals=[]
        for r in rows:
            v=r
            for p in path:v=v[p]
            vals.append(float(v))
        return float(np.median(vals))
    return dict(n=len(rows),counts=counts,
      A=dict(recurrent_gain=med(("A","recurrent_gain")),final_reward=med(("A","final_reward")),retention=med(("A","retention"))),
      B=dict(prediction_gain=med(("B","prediction_gain")),
             transplant_prediction_damage=med(("B","transplant_prediction_damage")),
             transplant_reward_damage=med(("B","transplant_reward_damage"))),
      C=dict(alive=med(("C","adaptive","alive")),priority_acc=med(("C","adaptive","priority_acc")),
             reward_gain=med(("C","reward_gain")),recovery_gain=med(("C","recovery_gain"))),
      D=dict(full_alive=med(("D","full","alive")),resource_gain=med(("D","resource_gain")),
             restraint_gain=med(("D","restraint_gain")),transmission_shift=med(("D","transmission_shift")),
             ecology_transmission_effect=med(("D","ecology_transmission_effect"))))

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dev=[run_seed(s) for s in DEV_SEEDS]
    stress=[run_seed(s) for s in STRESS_SEEDS]
    for label,rows in (("dev",dev),("stress",stress)):
        d=out/label;d.mkdir(exist_ok=True)
        for r in rows:
            (d/f"seed_{r['seed']}.json").write_text(json.dumps(r,indent=2,default=default))
    summary=dict(development=summarize(dev),stress=summarize(stress))
    (out/"PILOT_RESULTS.json").write_text(json.dumps(dict(summary=summary,development=dev,stress=stress),indent=2,default=default))
    (out/"PILOT_REPORT.md").write_text("# P3 development + stress pilot\n\n"+json.dumps(summary,indent=2,default=default)+"\n")
    print(json.dumps(summary,indent=2,default=default))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True)
    main(ap.parse_args().output)
