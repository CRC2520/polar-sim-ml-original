"""Generate candidate initial-state snapshots under the preserved D2 runtime. QA only."""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import torch, numpy as np
from b1s.execution.core import RoutingActor, Value, model_digest
from experiments.b1sc_d2_v1_0 import implementation as d2
from experiments.b1sc_d2_r1_v1_0.init_format import write_snapshot, sha256_file

BOUNDARY=dict(scientific_training_authorized=False,scientific_evaluation_authorized=False,B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",B1E_executed=False,final_seeds_generated=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if torch.__version__!="2.2.2+cpu":raise RuntimeError("Snapshot generation requires torch 2.2.2+cpu")
    if np.__version__!="1.26.4":raise RuntimeError("Snapshot generation requires numpy 1.26.4")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rows=[]
    for block in range(d2.BLOCKS):
        on,_=d2.block_pair(block); seed=int(on["initial_seed"])
        torch.manual_seed(seed)
        actor=RoutingActor(d2.S6_MASK); critic=Value()
        path=a.out/f"block-{block}.bin"
        file_sha=write_snapshot(path,block,seed,actor,critic)
        rows.append(dict(block=block,source_seed=seed,file=path.name,sha256=file_sha,actor_digest=model_digest(actor),critic_digest=model_digest(critic),bytes=path.stat().st_size))
    manifest=dict(status="R1_INIT_CANDIDATES_GENERATED_NOT_SCIENTIFIC",schema="B1-SC-D2-R1-INIT-FREEZE-1.0.0-20260915",source_branch="research/b1sc-d2-r1-v1.0-byte-frozen-init-20260915",base_commit="3b65403c309d4fe3ef01be4c1fd2693c48f02702",runtime=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),torch_threads=torch.get_num_threads()),blocks=rows,**BOUNDARY)
    (a.out/"CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    # Re-read every file and verify manifest hash before upload.
    for r in rows:
        if sha256_file(a.out/r["file"])!=r["sha256"]:raise RuntimeError("Candidate snapshot changed before publication")
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=="__main__":main()
