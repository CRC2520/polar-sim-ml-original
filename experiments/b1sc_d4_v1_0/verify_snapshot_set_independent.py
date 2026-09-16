"""Independent verifier B for D4 snapshot bytes.

Intentionally parses the byte format directly instead of calling init_format.decode_snapshot.
"""
from __future__ import annotations
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from experiments.b1sc_d4_v1_0 import implementation as d4

MAGIC=b"B1SC-D4-TRAINABLE-INIT-v1\n"
SCHEMA="B1-SC-D4-INIT-1.0.0-20260916"


def digest_state(entries,payload,prefix):
    h=hashlib.sha256()
    subset=[e for e in entries if e["name"].startswith(prefix+".")]
    for e in sorted(subset,key=lambda x:x["name"].split(".",1)[1]):
        name=e["name"].split(".",1)[1]
        start=int(e["offset"]);end=start+int(e["nbytes"])
        arr=np.frombuffer(payload[start:end],dtype=np.dtype(e["dtype"])).reshape(e["shape"])
        h.update(name.encode("utf-8"));h.update(arr.dtype.str.encode("ascii"));h.update(str(tuple(arr.shape)).encode("ascii"));h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--manifest",type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.manifest or (a.root/"CANDIDATE_MANIFEST.json")).read_text(encoding="utf-8"))
    if len(manifest["blocks"])!=12: raise RuntimeError("D4 verifier B expected 12 blocks")
    rows=[]
    for expected in sorted(manifest["blocks"],key=lambda x:int(x["block"])):
        block=int(expected["block"]);data=(a.root/f"block-{block}.bin").read_bytes()
        sha=hashlib.sha256(data).hexdigest()
        if sha!=expected["sha256"]: raise RuntimeError(f"Verifier B SHA mismatch block {block}")
        if not data.startswith(MAGIC): raise RuntimeError(f"Verifier B magic mismatch block {block}")
        pos=len(MAGIC);n=struct.unpack(">Q",data[pos:pos+8])[0];pos+=8
        header=json.loads(data[pos:pos+n].decode("utf-8"));pos+=n
        if header.get("schema")!=SCHEMA or header.get("trainable_only") is not True:
            raise RuntimeError(f"Verifier B contract mismatch block {block}")
        if header.get("support_buffer_included") is not False or header.get("temporal_gate_state_included") is not False or header.get("condition_included") is not False:
            raise RuntimeError(f"Verifier B excluded-state mismatch block {block}")
        if int(header["block"])!=block or int(header["source_seed"])!=d4.snapshot_seed(block):
            raise RuntimeError(f"Verifier B header lineage mismatch block {block}")
        payload=memoryview(data)[pos:]
        entries=header["tensors"]
        ad=digest_state(entries,payload,"actor");cd=digest_state(entries,payload,"critic")
        if ad!=expected["actor_trainable_digest"] or cd!=expected["critic_trainable_digest"]:
            raise RuntimeError(f"Verifier B digest mismatch block {block}")
        rows.append(dict(block=block,sha256=sha,bytes=len(data),source_seed=header["source_seed"],actor_trainable_digest=ad,critic_trainable_digest=cd))
    report=dict(status="D4_SNAPSHOT_VERIFIER_B_INDEPENDENT_PASS",blocks=rows,block_count=len(rows),
                independent_parser=True,scientific_training_performed=False,scientific_evaluation_performed=False,
                scientific_results_exist=False,B1E_executed=False,final_seeds_generated=False)
    (a.out/"VERIFY_B.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
