"""Canonical trainable-only byte format for B1-SC-D3 initial snapshots.

The support buffer and information mode are deliberately excluded so the exact
same snapshot bytes can initialize GLOBAL-G0, GLOBAL-S6, LOCAL-G0 and LOCAL-S6
within one block.
"""
from __future__ import annotations
import hashlib, json, struct
from pathlib import Path
import numpy as np

MAGIC=b"B1SC-D3-TRAINABLE-INIT-v1\n"
SCHEMA="B1-SC-D3-INIT-1.0.0-20260916"


def canonical_json(obj)->bytes:
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

def sha256_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def sha256_file(path)->str:return sha256_bytes(Path(path).read_bytes())

def trainable_digest(module)->str:
    h=hashlib.sha256()
    for name,p in sorted(module.named_parameters()):
        a=p.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"));h.update(a.dtype.str.encode("ascii"));h.update(str(tuple(a.shape)).encode("ascii"));h.update(a.tobytes(order="C"))
    return h.hexdigest()

def parameter_entries(actor,critic):
    rows=[]
    for prefix,module in (("actor",actor),("critic",critic)):
        for name,p in sorted(module.named_parameters()):
            rows.append((f"{prefix}.{name}",p.detach().cpu().contiguous().numpy()))
    return rows

def encode_snapshot(block:int,source_seed:int,actor,critic)->bytes:
    payload=bytearray();meta=[]
    for name,a in parameter_entries(actor,critic):
        raw=a.tobytes(order="C");off=len(payload);payload.extend(raw)
        meta.append(dict(name=name,dtype=a.dtype.str,shape=list(a.shape),offset=off,nbytes=len(raw)))
    header=dict(schema=SCHEMA,block=int(block),source_seed=int(source_seed),trainable_only=True,support_buffer_included=False,tensors=meta)
    hb=canonical_json(header)
    return MAGIC+struct.pack(">Q",len(hb))+hb+bytes(payload)

def decode_snapshot(data:bytes):
    if not data.startswith(MAGIC):raise RuntimeError("Wrong D3 snapshot magic")
    pos=len(MAGIC);n=struct.unpack(">Q",data[pos:pos+8])[0];pos+=8
    header=json.loads(data[pos:pos+n].decode("utf-8"));pos+=n
    if header.get("schema")!=SCHEMA or header.get("trainable_only") is not True or header.get("support_buffer_included") is not False:
        raise RuntimeError("Wrong D3 snapshot contract")
    payload=memoryview(data)[pos:];states={"actor":{},"critic":{}}
    for e in header["tensors"]:
        start=int(e["offset"]);end=start+int(e["nbytes"])
        arr=np.frombuffer(payload[start:end],dtype=np.dtype(e["dtype"])).reshape(e["shape"]).copy()
        prefix,name=e["name"].split(".",1);states[prefix][name]=arr
    return header,states

def write_snapshot(path,block,source_seed,actor,critic):
    data=encode_snapshot(block,source_seed,actor,critic);Path(path).write_bytes(data);return sha256_bytes(data)

def load_trainable_into(path,actor,critic,expected_sha256:str|None=None):
    import torch
    data=Path(path).read_bytes();actual=sha256_bytes(data)
    if expected_sha256 is not None and actual!=expected_sha256:raise RuntimeError("Frozen D3 init SHA-256 mismatch")
    header,states=decode_snapshot(data)
    for key,module in (("actor",actor),("critic",critic)):
        params=dict(module.named_parameters())
        if set(params)!=set(states[key]):raise RuntimeError(f"D3 trainable parameter set mismatch: {key}")
        with torch.no_grad():
            for name,p in params.items():
                src=torch.from_numpy(states[key][name]).to(dtype=p.dtype,device=p.device)
                if tuple(src.shape)!=tuple(p.shape):raise RuntimeError(f"D3 trainable shape mismatch: {key}.{name}")
                p.copy_(src)
    return header,actual
