#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

SEEDS=[2196001,2196002,2196003]
ISO=1e-12

def eligible(r):
    return (
        r["intact_accuracy"]>=0.90 and
        r["d_effect"]>=0.25 and
        r["c_effect"]>=0.25 and
        r["r_effect"]>=0.20 and
        r["iso_gap"]<=ISO and
        r["iso_action_agreement"]>=1.0-ISO
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a=p.parse_args()
    rows=[json.loads(Path(x).read_text()) for x in a.inputs]
    rows=sorted(rows,key=lambda r:int(r["train_seed"]))
    if [int(r["train_seed"]) for r in rows] != SEEDS:
        raise SystemExit("canonical R48 dev seed set mismatch")
    for r in rows:
        r["eligible"]=eligible(r)
    n=sum(bool(r["eligible"]) for r in rows)
    out={
      "campaign":"R48 neural relation discovery from reward",
      "phase":"development",
      "canonical_scientific_run":36089447230,
      "scientific_head_sha":"33ef845eca64ad80ea956b8ecc5a758f6a883843",
      "training_seeds":SEEDS,
      "eligible_checkpoints":n,
      "required_eligible_checkpoints":2,
      "authorize_confirm":n>=2,
      "resolution":"R48_DEVELOPMENT_AUTHORIZE_CONFIRM" if n>=2 else "R48_DEVELOPMENT_FAIL_NO_CONFIRM",
      "checkpoints":rows,
      "boundaries":{
        "confirmatory_seeds_opened":False,
        "cross_domain_transport":"NOT_TESTED",
        "global_minimality":"OPEN",
        "E6b":"OPEN",
        "E7":"OPEN",
        "POLAR_superiority":"NOT_ESTABLISHED",
        "consciousness":"NOT_ESTABLISHED"
      }
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="checkpoints"},indent=2))
    for r in rows:
        print(json.dumps({k:r[k] for k in ["train_seed","intact_accuracy","d_effect","c_effect","r_effect","iso_gap","iso_action_agreement","eligible"]},sort_keys=True))

if __name__=="__main__":
    main()
