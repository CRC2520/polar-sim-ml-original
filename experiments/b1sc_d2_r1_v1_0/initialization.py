"""Load byte-frozen D2-R1 initial modules from the canonical GitHub artifact."""
from __future__ import annotations
import json, os
from pathlib import Path
import torch
from b1s.execution.core import RoutingActor, Value, model_digest
from experiments.b1sc_d2_v1_0 import implementation as d2
from experiments.b1sc_d2_r1_v1_0.init_format import load_into, sha256_file

ROOT=Path(__file__).resolve().parent
MANIFEST_PATH=ROOT/"INIT_FREEZE.json"
CONDITIONS=("S6-ON","S6-OFF-TRAIN")

def manifest():return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
def entry(block:int):
    b=int(block)
    if not 0<=b<8:raise RuntimeError("Invalid D2-R1 block")
    rows=[x for x in manifest()["blocks"] if x["block"]==b]
    if len(rows)!=1:raise RuntimeError("Missing/duplicate frozen block")
    return rows[0]
def artifact_dir()->Path:
    raw=os.environ.get("D2R1_INIT_DIR")
    if not raw:raise RuntimeError("D2R1_INIT_DIR must point to the extracted canonical artifact")
    p=Path(raw)
    if not p.is_dir():raise RuntimeError("Frozen init artifact directory is missing")
    return p
def build_initial_modules(condition:str,block:int):
    if condition not in CONDITIONS:raise RuntimeError("Unknown D2-R1 condition")
    e=entry(block);p=artifact_dir()/e["file"]
    if not p.is_file() or sha256_file(p)!=e["sha256"]:raise RuntimeError("Frozen block byte hash mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);actor=RoutingActor(d2.S6_MASK);critic=Value()
    header,actual=load_into(p,actor,critic,e["sha256"])
    if int(header["block"])!=int(block) or int(header["source_seed"])!=int(e["source_seed"]):raise RuntimeError("Frozen snapshot header mismatch")
    ad=model_digest(actor);cd=model_digest(critic)
    if ad!=e["actor_digest"] or cd!=e["critic_digest"]:raise RuntimeError("Loaded parameter digest mismatch")
    return actor,critic,dict(block=int(block),condition=condition,initial_state_sha256=actual,actor_digest=ad,critic_digest=cd,source_seed=int(e["source_seed"]),file=e["file"],artifact_id=manifest()["canonical_artifact"]["artifact_id"])
