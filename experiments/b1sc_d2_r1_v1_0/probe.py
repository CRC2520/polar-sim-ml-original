"""Independent-runner probe for frozen D2-R1 initialization bytes. QA only."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from experiments.b1sc_d2_r1_v1_0.initialization import build_initial_modules

BOUNDARY=dict(scientific_training_performed=False,scientific_evaluation_performed=False,B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",B1E_executed=False,final_seeds_generated=False,H_CAT="NOT_EVALUABLE",H_TRANSFER="NOT_EVALUATED")

def main():
    p=argparse.ArgumentParser();p.add_argument("--condition",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); rows=[]
    z=np.linspace(-1.0,1.0,4*93,dtype=np.float32).reshape(4,93)
    for block in range(8):
        actor,critic,meta=build_initial_modules(a.condition,block)
        with torch.no_grad():
            mu,_=actor(torch.from_numpy(z)); val=critic(torch.from_numpy(z))
        h=hashlib.sha256();h.update(mu.detach().cpu().numpy().tobytes());h.update(val.detach().cpu().numpy().tobytes())
        rows.append(dict(**meta,forward_digest=h.hexdigest()))
    result=dict(status="R1_FROZEN_INIT_PROBE_PASS_NOT_SCIENTIFIC",condition=a.condition,blocks=rows,torch=torch.__version__,numpy=np.__version__,**BOUNDARY)
    (a.out/"PROBE.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
