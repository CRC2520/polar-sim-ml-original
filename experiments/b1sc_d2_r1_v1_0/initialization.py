"""Load D2-R1 initial modules exclusively from repository-versioned canonical bytes."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from b1s.execution.core import RoutingActor, Value, model_digest
from experiments.b1sc_d2_v1_0 import implementation as d2
from experiments.b1sc_d2_r1_v1_0.init_format import load_into, sha256_file
ROOT=Path(__file__).resolve().parent
MANIFEST_PATH=ROOT/"INIT_FREEZE.json"
PERSISTED_ROOT=ROOT/"frozen_init"
PERSISTED_MANIFEST=PERSISTED_ROOT/"PERSISTED_MANIFEST.json"
CONDITIONS=("S6-ON","S6-OFF-TRAIN")
def manifest(): return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
def persisted_manifest(): return json.loads(PERSISTED_MANIFEST.read_text(encoding="utf-8"))
def entry(block:int):
    b=int(block)
    if not 0<=b<8: raise RuntimeError("Invalid D2-R1 block")
    rows=[x for x in manifest()["blocks"] if x["block"]==b]; prows=[x for x in persisted_manifest()["blocks"] if x["block"]==b]
    if len(rows)!=1 or len(prows)!=1: raise RuntimeError("Missing/duplicate frozen block")
    e,p=rows[0],prows[0]
    if p["sha256"]!=e["sha256"]: raise RuntimeError("Persisted/frozen SHA contract mismatch")
    return e,p
def artifact_dir()->Path:
    if not PERSISTED_ROOT.is_dir(): raise RuntimeError("Repository-persisted frozen init directory is missing")
    return PERSISTED_ROOT
def build_initial_modules(condition:str,block:int):
    if condition not in CONDITIONS: raise RuntimeError("Unknown D2-R1 condition")
    e,pe=entry(block); p=artifact_dir()/f"block-{int(block)}.bin"
    if not p.is_file() or p.stat().st_size!=int(pe["size"]) or sha256_file(p)!=e["sha256"]: raise RuntimeError("Frozen block byte hash mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0); actor=RoutingActor(d2.S6_MASK); critic=Value()
    header,actual=load_into(p,actor,critic,e["sha256"])
    if int(header["block"])!=int(block) or int(header["source_seed"])!=int(e["source_seed"]): raise RuntimeError("Frozen snapshot header mismatch")
    ad=model_digest(actor); cd=model_digest(critic)
    if ad!=e["actor_digest"] or cd!=e["critic_digest"]: raise RuntimeError("Loaded parameter digest mismatch")
    pm=persisted_manifest()
    return actor,critic,dict(block=int(block),condition=condition,initial_state_sha256=actual,actor_digest=ad,critic_digest=cd,source_seed=int(e["source_seed"]),file=f"block-{int(block)}.bin",artifact_id=pm["canonical_artifact_id"],storage="repository-versioned",persisted_manifest_sha256=sha256_file(PERSISTED_MANIFEST))
