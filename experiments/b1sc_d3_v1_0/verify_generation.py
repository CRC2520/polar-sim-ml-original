"""Compare two independent D3 initialization generations byte-for-byte."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from experiments.b1sc_d3_v1_0.init_format import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument("--a",type=Path,required=True);p.add_argument("--b",type=Path,required=True);p.add_argument("--out",type=Path,required=True);x=p.parse_args();x.out.mkdir(parents=True,exist_ok=False)
    ma=json.loads((x.a/"CANDIDATE_MANIFEST.json").read_text());mb=json.loads((x.b/"CANDIDATE_MANIFEST.json").read_text())
    if len(ma["blocks"])!=8 or len(mb["blocks"])!=8:raise RuntimeError("D3 independent generation did not produce eight blocks")
    rows=[]
    for block in range(8):
        ra=next(r for r in ma["blocks"] if r["block"]==block);rb=next(r for r in mb["blocks"] if r["block"]==block)
        pa=x.a/ra["file"];pb=x.b/rb["file"]
        ba=pa.read_bytes();bb=pb.read_bytes()
        if ba!=bb:raise RuntimeError(f"Independent D3 initialization mismatch at block {block}")
        if ra["sha256"]!=rb["sha256"] or sha256_file(pa)!=ra["sha256"]:raise RuntimeError(f"D3 hash mismatch at block {block}")
        if ra["actor_trainable_digest"]!=rb["actor_trainable_digest"] or ra["critic_trainable_digest"]!=rb["critic_trainable_digest"]:raise RuntimeError(f"D3 trainable digest mismatch at block {block}")
        shutil.copyfile(pa,x.out/pa.name)
        rows.append(dict(block=block,sha256=ra["sha256"],bytes=len(ba),source_seed=ra["source_seed"],actor_trainable_digest=ra["actor_trainable_digest"],critic_trainable_digest=ra["critic_trainable_digest"],independent_generations_byte_identical=True,shared_conditions=ra["shared_conditions"]))
    shutil.copyfile(x.a/"CANDIDATE_MANIFEST.json",x.out/"CANDIDATE_MANIFEST.json")
    report=dict(status="D3_DUAL_INDEPENDENT_INITIALIZATION_GENERATION_EXACT",blocks=rows,block_count=8,all_byte_identical=True,scientific_training_performed=False,scientific_evaluation_performed=False,B1E_executed=False,final_seeds_generated=False)
    (x.out/"DUAL_GENERATION_VERIFICATION.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
