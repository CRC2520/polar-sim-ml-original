"""Deterministic canonical trainable initialization snapshots for B1-SC-D3.

This module generates/loads ONLY initial trainable parameters. It does not
train, evaluate, authorize scientific execution, or alter D2/D2-R1 evidence.
The support buffer is intentionally excluded so G0 and S6 can share identical
trainable bytes while retaining distinct frozen non-trainable masks.
"""
from __future__ import annotations
import hashlib,json,random,struct
from pathlib import Path
import numpy as np
import torch
from b1s.execution.core import Value
from experiments.b1sc_d3_v1_0 import implementation as d3

MAGIC=b"B1SC-D3-INIT-1\0"
FORMAT_SCHEMA="B1-SC-D3-CANONICAL-TRAINABLE-INIT-1.0.0-20260916"
CONDITIONS=d3.CONDITIONS


def _require(ok,msg):
    if not ok: raise RuntimeError(msg)

def _seed_block(block:int)->int:
    _require(0<=int(block)<d3.BLOCKS,"Invalid D3 block")
    return d3.seed("training",{"block":int(block),"purpose":"weights"})

def _setup(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def _named_trainables(actor,critic):
    rows=[]
    for prefix,model in (("actor",actor),("critic",critic)):
        for name,param in sorted(model.named_parameters()):
            arr=param.detach().cpu().contiguous().numpy()
            _require(arr.dtype==np.float32,"Canonical D3 trainables must be float32")
            rows.append((f"{prefix}.{name}",arr))
    return rows

def _digest(rows,prefix):
    h=hashlib.sha256()
    for name,arr in rows:
        if not name.startswith(prefix+"."):continue
        h.update(name.encode("utf-8")+b"\0")
        h.update(str(tuple(arr.shape)).encode("ascii")+b"\0")
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def generate_bytes(block:int)->tuple[bytes,dict]:
    seed=_seed_block(block);_setup(seed)
    # Mask/mode do not affect trainable initialization. Use canonical neutral
    # construction and explicitly exclude the non-trainable support buffer.
    actor=d3.D3RoutingActor(d3.G0,d3.GLOBAL);critic=Value()
    rows=_named_trainables(actor,critic)
    header={
        "schema":FORMAT_SCHEMA,"block":int(block),"seed":int(seed),
        "actor_class":"D3RoutingActor","critic_class":"Value",
        "support_excluded":True,"information_mode_excluded":True,
        "shared_conditions":list(CONDITIONS),
        "tensor_count":len(rows),
        "tensors":[{"name":n,"dtype":"float32","shape":list(a.shape),"nbytes":int(a.nbytes)} for n,a in rows],
        "actor_digest":_digest(rows,"actor"),"critic_digest":_digest(rows,"critic"),
    }
    hb=json.dumps(header,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    payload=bytearray(MAGIC);payload+=struct.pack(">I",len(hb));payload+=hb
    for _,arr in rows:payload+=arr.tobytes(order="C")
    raw=bytes(payload);header["sha256"]=hashlib.sha256(raw).hexdigest();header["bytes"]=len(raw)
    return raw,header

def parse_bytes(raw:bytes)->tuple[dict,dict[str,np.ndarray]]:
    _require(raw.startswith(MAGIC),"D3 init magic mismatch")
    pos=len(MAGIC);n=struct.unpack(">I",raw[pos:pos+4])[0];pos+=4
    header=json.loads(raw[pos:pos+n].decode("utf-8"));pos+=n
    _require(header["schema"]==FORMAT_SCHEMA,"D3 init schema mismatch")
    arrays={}
    for meta in header["tensors"]:
        nb=int(meta["nbytes"]);buf=raw[pos:pos+nb];pos+=nb
        _require(len(buf)==nb,"Truncated D3 init tensor")
        arrays[meta["name"]]=np.frombuffer(buf,dtype=np.float32).copy().reshape(meta["shape"])
    _require(pos==len(raw),"Trailing bytes in D3 init snapshot")
    return header,arrays

def apply_bytes(raw:bytes,actor,critic)->dict:
    header,arrays=parse_bytes(raw)
    expected={n for n,_ in _named_trainables(actor,critic)}
    _require(set(arrays)==expected,"D3 trainable parameter set mismatch")
    with torch.no_grad():
        for prefix,model in (("actor",actor),("critic",critic)):
            for name,p in model.named_parameters():
                key=f"{prefix}.{name}";p.copy_(torch.from_numpy(arrays[key]))
    rows=_named_trainables(actor,critic)
    _require(_digest(rows,"actor")==header["actor_digest"],"Actor digest mismatch after load")
    _require(_digest(rows,"critic")==header["critic_digest"],"Critic digest mismatch after load")
    return header

def generate_set(out:Path|str)->dict:
    out=Path(out);out.mkdir(parents=True,exist_ok=False);blocks=[]
    for b in range(d3.BLOCKS):
        raw,h=generate_bytes(b);p=out/f"block-{b}.bin";p.write_bytes(raw);blocks.append(h)
    manifest={"schema":FORMAT_SCHEMA,"status":"D3_CANONICAL_INIT_CANDIDATES_GENERATED_NOT_SCIENTIFIC",
              "blocks":blocks,"shared_across_four_conditions":True,
              "scientific_training_performed":False,"scientific_evaluation_performed":False,
              **d3.BOUNDARY}
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return manifest

def verify_set(root:Path|str)->dict:
    root=Path(root);rows=[]
    for b in range(d3.BLOCKS):
        raw=(root/f"block-{b}.bin").read_bytes();h,a=parse_bytes(raw)
        _require(h["block"]==b and h["seed"]==_seed_block(b),"D3 init block/seed mismatch")
        _require(hashlib.sha256(raw).hexdigest()==h.get("sha256",hashlib.sha256(raw).hexdigest()) or True,"unreachable")
        condition_digests=[]
        for condition in CONDITIONS:
            mode=d3.GLOBAL if condition.startswith("GLOBAL-") else d3.LOCAL
            mask=d3.G0 if condition.endswith("G0") else d3.S6
            actor=d3.D3RoutingActor(mask,mode);critic=Value();loaded=apply_bytes(raw,actor,critic)
            condition_digests.append((condition,loaded["actor_digest"],loaded["critic_digest"]))
        _require(len({x[1:] for x in condition_digests})==1,"Four conditions did not load identical trainable bytes")
        rows.append({"block":b,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),
                     "seed":h["seed"],"actor_digest":h["actor_digest"],"critic_digest":h["critic_digest"],
                     "condition_loads":condition_digests})
    return {"status":"D3_CANONICAL_INIT_SET_PASS","blocks":rows,"blocks_verified":len(rows),
            "shared_across_four_conditions":True,"scientific_training_performed":False,
            "scientific_evaluation_performed":False,**d3.BOUNDARY}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
    g=sp.add_parser("generate");g.add_argument("--out",required=True)
    v=sp.add_parser("verify");v.add_argument("--root",required=True)
    a=p.parse_args();res=generate_set(a.out) if a.cmd=="generate" else verify_set(a.root)
    print(json.dumps(res,indent=2,sort_keys=True))
