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
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[run_seed(s) for s in DEV_SEEDS]
    for r in rows:
        (out/f"seed_{r['seed']}.json").write_text(json.dumps(r,indent=2,default=default))
    counts={k:sum(bool(r[k]["pass_seed"]) for r in rows) for k in ("E1","E2","E3","E4")}
    def med(path):
        vals=[]
        for r in rows:
            v=r
            for p in path: v=v[p]
            vals.append(float(v))
        return float(np.median(vals))
    summary=dict(n=len(rows),counts=counts,
      E1=dict(polar_minus_generic=med(("E1","polar_minus_generic")),
              polar_overlap=med(("E1","polar","true_overlap")),
              generic_overlap=med(("E1","generic","true_overlap")),
              random_adv_polar=med(("E1","random_advantage_polar"))),
      E2=dict(pre_overlap=med(("E2","adaptive","pre_overlap")),
              post_overlap=med(("E2","adaptive","post_overlap")),
              advantage_vs_frozen=med(("E2","advantage_vs_frozen")),
              lesion_damage=med(("E2","lesion_damage"))),
      E3=dict(primary=med(("E3","full","primary_accuracy")),
              workspace=med(("E3","full","workspace_recall")),
              source=med(("E3","full","source_ba")),
              polar_workspace_damage=med(("E3","polar_lesion","workspace_damage")),
              polar_meta_damage=med(("E3","polar_lesion","meta_damage")),
              polar_source_damage=med(("E3","polar_lesion","source_damage")),
              gwt_primary_damage=med(("E3","gwt_lesion","primary_damage")),
              hot_meta_damage=med(("E3","hot_lesion","meta_damage"))),
      E4={})
    for task in ("logistic_harvest","thermal_rc","queue_service"):
        summary["E4"][task]=dict(
          ratio=med(("E4","tasks",task,"return_ratio")),
          alive=med(("E4","tasks",task,"agent","alive_fraction")),
          oracle_alive=med(("E4","tasks",task,"oracle","alive_fraction")))
    (out/"PILOT_RESULTS.json").write_text(json.dumps(dict(summary=summary,records=rows),indent=2,default=default))
    (out/"PILOT_REPORT.md").write_text("# P2-Critical development pilot\n\n"+json.dumps(summary,indent=2,default=default)+"\n")
    print(json.dumps(summary,indent=2,default=default))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True)
    main(ap.parse_args().output)
