"""CLI for the guarded B1-SC-D2 v1.0 execution pipeline."""
from __future__ import annotations
import argparse,traceback
from pathlib import Path
from experiments.b1sc_d2_v1_0 import execution as ex

def dispatch(a):
    if a.stage=="authorize":
        req=ex.execution_guard(); ex.write_json(a.out/"AUTHORIZATION_RECEIPT.json",dict(status="D2_AUTHORIZED_FOR_SINGLE_RUN_ATTEMPT_1",request=req,**ex.d2.BOUNDARY)); return
    if a.stage=="fit":
        ex.require(a.condition is not None and a.block is not None,"fit requires --condition and --block"); ex.fit(a.condition,a.block,a.out); return
    if a.stage=="training-aggregate":
        ex.require(a.fits is not None,"training-aggregate requires --fits"); ex.aggregate_training(a.fits,a.out); return
    if a.stage=="online":
        ex.require(a.block is not None and a.fits is not None,"online requires --block and --fits"); ex.online_block(a.block,a.fits,a.out); return
    if a.stage=="online-aggregate":
        ex.require(a.online_root is not None,"online-aggregate requires --online-root"); ex.aggregate_online(a.online_root,a.out); return
    if a.stage=="final":
        ex.require(a.training is not None and a.online_file is not None,"final requires --training and --online"); ex.final_aggregate(a.training,a.online_file,a.out); return
    raise RuntimeError("Unsupported stage")

def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["authorize","fit","training-aggregate","online","online-aggregate","final"]); p.add_argument("--condition",choices=["S6-ON","S6-OFF-TRAIN"]); p.add_argument("--block",type=int); p.add_argument("--out",type=Path,required=True); p.add_argument("--fits",type=Path); p.add_argument("--online-root",type=Path); p.add_argument("--training",type=Path); p.add_argument("--online",dest="online_file",type=Path); a=p.parse_args()
    try: dispatch(a)
    except Exception:
        a.out.mkdir(parents=True,exist_ok=True); ex.write_json(a.out/"FAILED.json",dict(status="FAILED_RETAINED",stage=a.stage,error=traceback.format_exc(),replacement_seed=False,**ex.d2.BOUNDARY)); raise

if __name__=="__main__": main()
