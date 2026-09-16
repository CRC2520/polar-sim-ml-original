"""Generate one canonical trainable initialization candidate per D3 block.

QA/init-freeze only: no environment, optimizer, training, scientific evaluation
or B1-E action is executed here.
"""
from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path
import numpy as np
import torch
from b1s.execution.core import Value
from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0.init_format import write_snapshot,sha256_file,trainable_digest

BOUNDARY=dict(scientific_training_performed=False,scientific_evaluation_performed=False,scientific_results_exist=False,B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",B1E_executed=False,final_seeds_generated=False,H_CAT="NOT_EVALUABLE",H_TRANSFER="NOT_EVALUATED")


def snapshot_seed(block:int)->int:
    return d3.seed("training",{"block":int(block),"purpose":"initial-trainable-snapshot"})

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if torch.__version__!="2.2.2+cpu":raise RuntimeError("D3 snapshot generation requires torch 2.2.2+cpu")
    if np.__version__!="1.26.4":raise RuntimeError("D3 snapshot generation requires numpy 1.26.4")
    d3.validate_design();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[]
    for block in range(d3.BLOCKS):
        seed=snapshot_seed(block);torch.manual_seed(seed)
        actor=d3.D3RoutingActor(d3.G0,d3.GLOBAL);critic=Value()
        path=a.out/f"block-{block}.bin";file_sha=write_snapshot(path,block,seed,actor,critic)
        rows.append(dict(block=block,source_seed=seed,file=path.name,sha256=file_sha,actor_trainable_digest=trainable_digest(actor),critic_trainable_digest=trainable_digest(critic),bytes=path.stat().st_size,shared_conditions=list(d3.CONDITIONS),support_buffer_included=False))
    manifest=dict(status="D3_INIT_CANDIDATES_GENERATED_NOT_SCIENTIFIC",schema="B1-SC-D3-INIT-FREEZE-1.0.0-20260916",design_commit="53bd59db47c6f0eac123e8777dd99b47e34f4051",source_branch="research/b1sc-d3-v1.0-byte-frozen-init-20260916",runtime=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),torch_threads=torch.get_num_threads()),snapshot_contract=dict(trainable_only=True,one_per_block=True,shared_across_four_conditions=True,support_buffer_excluded=True,seed_identity_excludes_condition=True),blocks=rows,**BOUNDARY)
    (a.out/"CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for r in rows:
        if sha256_file(a.out/r["file"])!=r["sha256"]:raise RuntimeError("D3 candidate snapshot changed before publication")
    print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__":main()
