"""Independent verifier for repository-versioned D3 trainable snapshots."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from b1s.execution.core import Value, matrix
from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0.init_format import load_trainable_into,sha256_file,trainable_digest

ROOT=Path(__file__).resolve().parent
ALLOWED_FREEZE_STATUS={"D3_FROZEN_INITIALIZATION_PERSISTED_PENDING_INDEPENDENT_VERIFICATION","D3_FROZEN_INITIALIZATION_QUALIFIED_NOT_EXECUTED"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((ROOT/"INIT_FREEZE.json").read_text());reg=json.loads((ROOT/"FIT_REGISTRY.json").read_text())
    if freeze.get("status") not in ALLOWED_FREEZE_STATUS:raise RuntimeError("D3 init freeze status is not verifiable")
    rows=[]
    for block in range(d3.BLOCKS):
        fr=next(r for r in freeze["blocks"] if r["block"]==block);path=ROOT/"frozen_init"/f"block-{block}.bin"
        if sha256_file(path)!=fr["sha256"]:raise RuntimeError(f"Persisted D3 snapshot hash mismatch block {block}")
        condition_digests=[]
        for condition in d3.CONDITIONS:
            cfg=next(x for x in reg if x["block"]==block and x["condition"]==condition)
            actor=d3.D3RoutingActor(cfg["mask"],cfg["information_mode"]);critic=Value()
            header,_=load_trainable_into(path,actor,critic,fr["sha256"])
            if int(header["block"])!=block or int(header["source_seed"])!=int(fr["source_seed"]):raise RuntimeError("D3 snapshot header mismatch")
            ad=trainable_digest(actor);cd=trainable_digest(critic)
            if ad!=fr["actor_trainable_digest"] or cd!=fr["critic_trainable_digest"]:raise RuntimeError(f"D3 trainable digest mismatch {condition} b{block}")
            expected_support=torch.from_numpy(matrix(cfg["mask"]))
            if not torch.equal(actor.support.cpu(),expected_support):raise RuntimeError(f"D3 support buffer overwritten {condition} b{block}")
            if actor.information_mode!=cfg["information_mode"]:raise RuntimeError(f"D3 information mode overwritten {condition} b{block}")
            condition_digests.append(dict(condition=condition,actor_trainable_digest=ad,critic_trainable_digest=cd))
        if len({x["actor_trainable_digest"] for x in condition_digests})!=1 or len({x["critic_trainable_digest"] for x in condition_digests})!=1:raise RuntimeError(f"Four-condition trainable initialization mismatch block {block}")
        rows.append(dict(block=block,sha256=fr["sha256"],source_seed=fr["source_seed"],conditions_verified=4,actor_trainable_digest=fr["actor_trainable_digest"],critic_trainable_digest=fr["critic_trainable_digest"]))
    report=dict(status="D3_REPOSITORY_FROZEN_INIT_VERIFICATION_PASS",verifier_label=a.label,freeze_status=freeze["status"],blocks=rows,block_count=8,conditions_per_block=4,scientific_training_performed=False,scientific_evaluation_performed=False,B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",B1E_executed=False,final_seeds_generated=False,H_CAT="NOT_EVALUABLE",H_TRANSFER="NOT_EVALUATED")
    (a.out/f"VERIFY_{a.label}.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
