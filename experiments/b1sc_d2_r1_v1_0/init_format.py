"""Canonical byte format for B1-SC-D2-R1 initial actor/critic snapshots."""
from __future__ import annotations
import hashlib, json, struct
from pathlib import Path
import numpy as np

MAGIC=b"B1SC-D2-R1-INIT-v1\n"
SCHEMA="B1-SC-D2-R1-INIT-1.0.0-20260915"

def canonical_json(obj)->bytes:
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

def sha256_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def sha256_file(path)->str:return sha256_bytes(Path(path).read_bytes())

def tensor_entries(actor,critic):
    rows=[]
    for prefix,module in (("actor",actor),("critic",critic)):
        for name,t in sorted(module.state_dict().items()):
            a=t.detach().cpu().contiguous().numpy()
            rows.append((f"{prefix}.{name}",a))
    return rows

def encode_snapshot(block:int,source_seed:int,actor,critic)->bytes:
    arrays=tensor_entries(actor,critic); payload=bytearray(); meta=[]
    for name,a in arrays:
        raw=a.tobytes(order="C"); off=len(payload); payload.extend(raw)
        meta.append(dict(name=name,dtype=a.dtype.str,shape=list(a.shape),offset=off,nbytes=len(raw)))
    header=dict(schema=SCHEMA,block=int(block),source_seed=int(source_seed),tensors=meta)
    hb=canonical_json(header)
    return MAGIC+struct.pack(">Q",len(hb))+hb+bytes(payload)

def decode_snapshot(data:bytes):
    if not data.startswith(MAGIC):raise RuntimeError("Wrong R1 snapshot magic")
    pos=len(MAGIC); n=struct.unpack(">Q",data[pos:pos+8])[0]; pos+=8
    header=json.loads(data[pos:pos+n].decode("utf-8")); pos+=n
    if header.get("schema")!=SCHEMA:raise RuntimeError("Wrong R1 snapshot schema")
    payload=memoryview(data)[pos:]; states={"actor":{},"critic":{}}
    for e in header["tensors"]:
        start=int(e["offset"]); end=start+int(e["nbytes"])
        arr=np.frombuffer(payload[start:end],dtype=np.dtype(e["dtype"])).reshape(e["shape"]).copy()
        prefix,name=e["name"].split(".",1); states[prefix][name]=arr
    return header,states

def write_snapshot(path,block,source_seed,actor,critic):
    data=encode_snapshot(block,source_seed,actor,critic); Path(path).write_bytes(data); return sha256_bytes(data)

def load_into(path,actor,critic,expected_sha256:str|None=None):
    data=Path(path).read_bytes(); actual=sha256_bytes(data)
    if expected_sha256 is not None and actual!=expected_sha256:raise RuntimeError("Frozen init SHA-256 mismatch")
    header,states=decode_snapshot(data)
    import torch
    actor.load_state_dict({k:torch.from_numpy(v) for k,v in states["actor"].items()},strict=True)
    critic.load_state_dict({k:torch.from_numpy(v) for k,v in states["critic"].items()},strict=True)
    return header,actual
