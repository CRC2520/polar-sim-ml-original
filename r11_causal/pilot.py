from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config_dev import DEV_SEEDS
from .benchmarks import relational_benchmark,gate_benchmark,transfer_benchmark,integrated_benchmark

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in DEV_SEEDS:
        r=dict(seed=seed,
               specificity=relational_benchmark(seed),
               gate=gate_benchmark(seed),
               transfer=transfer_benchmark(seed),
               integrated=integrated_benchmark(seed))
        rows.append(r)
        (out/f"seed_{seed}.json").write_text(json.dumps(r,indent=2))
    def med(path):
        vals=[]
        for r in rows:
            v=r
            for p in path:v=v[p]
            vals.append(float(v))
        return float(np.median(vals))
    summary=dict(n=len(rows),
      specificity=dict(recovered=med(("specificity","recovered")),gain=med(("specificity","mse_gain"))),
      gate=dict(accuracy=med(("gate","accuracy")),gain=med(("gate","gain")),perm_damage=med(("gate","permutation_damage"))),
      transfer=dict(cyclic_return=med(("transfer","cyclic_buffer","adaptive","return_mean")),
                    cyclic_alive=med(("transfer","cyclic_buffer","adaptive","alive_fraction")),
                    cyclic_gain=med(("transfer","cyclic_buffer","reward_gain")),
                    repair_return=med(("transfer","repair_queue","adaptive","return_mean")),
                    repair_alive=med(("transfer","repair_queue","adaptive","alive_fraction")),
                    repair_gain=med(("transfer","repair_queue","reward_gain"))),
      integrated=dict(alive=med(("integrated","full_alive")),reward=med(("integrated","full_return")),
                      self_world=med(("integrated","self_world")),source=med(("integrated","source")),
                      time=med(("integrated","time")),auc=med(("integrated","metacog_auc")),
                      cf=med(("integrated","counterfactual")),memory_damage=med(("integrated","memory_damage")),
                      content_drop=med(("integrated","content_drop")),cross_drop=med(("integrated","cross_drop"))))
    (out/"PILOT_SUMMARY.json").write_text(json.dumps(summary,indent=2))
    (out/"PILOT_REPORT.md").write_text("# R11 development pilot\n\n"+json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True)
    main(ap.parse_args().output)
