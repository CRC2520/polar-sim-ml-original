"""Aggregate independent ON/OFF initialization probes. QA only."""
from __future__ import annotations
import argparse, json
from pathlib import Path

BOUNDARY=dict(scientific_training_performed=False,scientific_evaluation_performed=False,B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",B1E_executed=False,final_seeds_generated=False,H_CAT="NOT_EVALUABLE",H_TRANSFER="NOT_EVALUATED")

def read_one(root:Path,condition:str):
    hits=[]
    for p in root.rglob("PROBE.json"):
        try:d=json.loads(p.read_text(encoding="utf-8"))
        except Exception:continue
        if d.get("condition")==condition:hits.append(d)
    if len(hits)!=1:raise RuntimeError(f"Expected one probe for {condition}, got {len(hits)}")
    return hits[0]

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    on=read_one(a.root,"S6-ON");off=read_one(a.root,"S6-OFF-TRAIN")
    if on["torch"]!="2.2.2+cpu" or off["torch"]!="2.2.2+cpu":raise RuntimeError("Wrong torch runtime")
    if on["numpy"]!="1.26.4" or off["numpy"]!="1.26.4":raise RuntimeError("Wrong numpy runtime")
    by=[]
    for block in range(8):
        o=next(x for x in on["blocks"] if x["block"]==block);f=next(x for x in off["blocks"] if x["block"]==block)
        keys=("initial_state_sha256","actor_digest","critic_digest","source_seed","file","forward_digest")
        mismatches=[k for k in keys if o[k]!=f[k]]
        if mismatches:raise RuntimeError(f"Frozen init mismatch block {block}: {mismatches}")
        by.append(dict(block=block,initial_state_sha256=o["initial_state_sha256"],actor_digest=o["actor_digest"],critic_digest=o["critic_digest"],forward_digest=o["forward_digest"],exact_cross_runner_match=True))
    result=dict(status="D2_R1_FROZEN_INITIALIZATION_QUALIFIED_NOT_EXECUTED",independent_condition_runners=2,blocks_verified=8,all_cross_runner_exact=True,by_block=by,next_step_requires_explicit_execution_pipeline_authorization=True,**BOUNDARY)
    (a.out/"INIT_QA_STATUS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
