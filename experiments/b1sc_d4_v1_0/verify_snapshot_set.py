"""Verifier A for a persisted/candidate D4 snapshot set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from b1s.execution.core import Value
from experiments.b1sc_d4_v1_0 import implementation as d4
from experiments.b1sc_d4_v1_0.init_format import decode_snapshot, load_trainable_into, sha256_file, trainable_digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    manifest_path=a.manifest or (a.root/"CANDIDATE_MANIFEST.json")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["blocks"])!=d4.BLOCKS: raise RuntimeError("D4 verifier A expected 12 blocks")
    rows=[]
    for expected in sorted(manifest["blocks"],key=lambda x:int(x["block"])):
        block=int(expected["block"]); path=a.root/f"block-{block}.bin"
        if not path.is_file(): raise RuntimeError(f"Missing D4 snapshot {block}")
        actual=sha256_file(path)
        if actual!=expected["sha256"]: raise RuntimeError(f"D4 snapshot SHA mismatch block {block}")
        header,_=decode_snapshot(path.read_bytes())
        if int(header["block"])!=block or int(header["source_seed"])!=d4.snapshot_seed(block):
            raise RuntimeError(f"D4 header mismatch block {block}")
        torch.manual_seed(0)
        actor=d4.D4RoutingActor();critic=Value()
        load_trainable_into(path,actor,critic,expected["sha256"])
        ad=trainable_digest(actor);cd=trainable_digest(critic)
        if ad!=expected["actor_trainable_digest"] or cd!=expected["critic_trainable_digest"]:
            raise RuntimeError(f"D4 decoded trainable digest mismatch block {block}")
        rows.append(dict(block=block,sha256=actual,bytes=path.stat().st_size,source_seed=header["source_seed"],
                         actor_trainable_digest=ad,critic_trainable_digest=cd))
    report=dict(status="D4_SNAPSHOT_VERIFIER_A_PASS",blocks=rows,block_count=len(rows),
                scientific_training_performed=False,scientific_evaluation_performed=False,
                scientific_results_exist=False,B1E_executed=False,final_seeds_generated=False)
    (a.out/"VERIFY_A.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
