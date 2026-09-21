from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config_dev import DEV_SEEDS
from .experiments import run_seed

def default(o):
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[run_seed(s) for s in DEV_SEEDS]
    for r in rows:
        (out/f"seed_{r['seed']}.json").write_text(json.dumps(r,indent=2,default=default))
    counts={k:sum(bool(r[k]["pass_seed"]) for r in rows) for k in ("A","B","C","D")}
    def med(path):
        vals=[]
        for r in rows:
            v=r
            for p in path:v=v[p]
            vals.append(float(v))
        return float(np.median(vals))
    summary=dict(n=len(rows),counts=counts,
      A=dict(recurrent_gain=med(("A","recurrent_gain")),final_reward=med(("A","final_reward")),retention=med(("A","retention"))),
      B=dict(prediction_gain=med(("B","prediction_gain")),reward_gain=med(("B","reward_gain")),transplant_damage=med(("B","transplant_damage"))),
      C=dict(alive_gain=med(("C","alive_gain")),priority_acc=med(("C","adaptive","priority_acc")),recovery_gain=med(("C","recovery_gain"))),
      D=dict(resource_gain=med(("D","resource_gain")),alive_gain=med(("D","alive_gain")),transmission_shift=med(("D","transmission_shift"))))
    (out/"PILOT_RESULTS.json").write_text(json.dumps(dict(summary=summary,records=rows),indent=2,default=default))
    (out/"PILOT_REPORT.md").write_text("# P3 development pilot\n\n"+json.dumps(summary,indent=2,default=default)+"\n")
    print(json.dumps(summary,indent=2,default=default))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True)
    main(ap.parse_args().output)
