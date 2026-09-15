"""B1-SC v1.0 executable design contract.

This module implements only deterministic design, gates, seed commitments and
intervention plans. It does not train or evaluate scientific policies.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DESIGN_PATH = ROOT / "DESIGN_FREEZE.json"
PROTOCOL_PATH = ROOT / "PROTOCOL_ES.md"

SCHEMA = "B1-SC-1.0.0-20260914"
BRANCH = "research/b1sc-v1.0-structural-causal-design-20260914"
REPOSITORY = "CRC2520/polar-sim-ml-original"
BASE_COMMIT = "7c263798531675a5542c62533f17a23858167737"
DESIGN_COMMIT = "d960907330c17ac6c2967f199eeefa8ef30e5114"
EDGE_ORDER = ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
PANEL = (
    ("G0","000000","primary_utility_reference"),
    ("GD","111111","dense_reference"),
    ("S1","000001","candidate"),
    ("S2","000100","candidate_direction_control"),
    ("S3","000101","candidate_reciprocity"),
    ("S4","100100","candidate_chain"),
    ("S5","011001","candidate_cycle"),
    ("S6","100110","candidate_inverse_cycle_control"),
    ("PPO",None,"generic_competent_reference"),
)
GRAPH_MASK = {role:mask for role,mask,_ in PANEL if mask is not None}
GRAPH_ROLES = tuple(GRAPH_MASK)
ALL_ROLES = tuple(x[0] for x in PANEL)
BLOCKS = 8
NATIVE_STEPS = 1_048_576
EPISODES_PER_BLOCK = 64
MIN_BLOCKS = 6
LOSS_MARGIN = 130/30
DISPLACEMENT_MARGIN = 1.0
CAUSAL_WINDOW = (9,40)
BOOTSTRAP_RESAMPLES = 10_000
ALLOWED_SPLITS = ("training","admission","causal","sham","bootstrap","qa")
SEED_NAMESPACE = SCHEMA + ":development"
BOUNDARY = dict(
    B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",
    ready_for_b1e_protocol_design=False,
    ready_for_b1e_freeze=False,
    ready_for_b1e_confirmatory_run=False,
    B1E_executed=False,
    final_seeds_generated=False,
    H_CAT="NOT_EVALUABLE",
    H_TRANSFER="NOT_EVALUATED",
)

def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)

def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False) + "\n").encode()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def design():
    return read_json(DESIGN_PATH)

def validate_design_contract() -> dict:
    d=design()
    require(d["schema"]==SCHEMA, "Unexpected design schema")
    require(d["status"]=="DESIGN_FROZEN_NOT_EXECUTED", "Design status changed")
    require(d["repository"]==REPOSITORY and d["branch"]==BRANCH, "Repository/branch contract changed")
    require(d["base_commit"]==BASE_COMMIT, "Base commit changed")
    require(tuple(tuple(x) for x in d["edge_order"])==EDGE_ORDER, "Edge order changed")
    require(tuple(tuple(x) for x in d["panel"])==PANEL, "Panel changed")
    require(d["training"]=={
        "blocks":8,"early_stopping":False,"fits":72,"hyperparameter_search":False,
        "input_mode":"raw","native_steps":NATIVE_STEPS}, "Training design changed")
    require(d["admission"]["total_blocks"]==BLOCKS and d["admission"]["minimum_blocks_passing"]==MIN_BLOCKS, "Admission block gate changed")
    require(d["admission"]["episodes_per_block"]==EPISODES_PER_BLOCK, "Admission episode count changed")
    require(abs(d["admission"]["loss_improvement_gt"]-LOSS_MARGIN)<1e-15, "Loss margin changed")
    require(d["admission"]["additional_displacement_gte"]==DISPLACEMENT_MARGIN, "Displacement margin changed")
    require(tuple(d["causal"]["window_zero_based"])==CAUSAL_WINDOW, "Causal window changed")
    require(d["causal"]["paired_episodes_per_block"]==EPISODES_PER_BLOCK, "Causal episode count changed")
    require(d["causal"]["minimum_blocks_with_practical_harm"]==MIN_BLOCKS, "Causal 6/8 gate changed")
    require(d["utility"]["minimum_blocks_with_practical_advantage"]==MIN_BLOCKS, "Utility 6/8 gate changed")
    require(d["utility"]["primary_reference"]=="G0", "G0 is no longer primary utility reference")
    require(d["analysis"]["bootstrap_resamples"]==BOOTSTRAP_RESAMPLES, "Bootstrap count changed")
    require(d["seed_commitment"]["B1E_final_seed_namespace_absent"] is True, "Final seed namespace boundary changed")
    require(tuple(d["seed_commitment"]["splits"])==ALLOWED_SPLITS, "Seed splits changed")
    for k,v in BOUNDARY.items():
        require(d["B1E_boundary"][k]==v, f"B1-E boundary changed: {k}")
    require(sha256_file(PROTOCOL_PATH)==d["protocol_sha256"], "Protocol bytes do not match design freeze")
    return {
        "status":"PASS","schema":SCHEMA,"panel_roles":len(PANEL),
        "graph_roles":len(GRAPH_ROLES),"fits":len(PANEL)*BLOCKS,
        "protocol_sha256":d["protocol_sha256"],
        **BOUNDARY,
    }

def mask_active_edges(mask: str) -> tuple[int,...]:
    require(len(mask)==6 and not (set(mask)-{"0","1"}), "Mask must have six binary digits")
    return tuple(i for i,b in enumerate(mask) if b=="1")

def role_active_edges(role: str) -> tuple[int,...]:
    require(role in ALL_ROLES, f"Unknown role: {role}")
    require(role!="PPO", "PPO has no graph-edge intervention")
    return mask_active_edges(GRAPH_MASK[role])

def role_active_pairs(role: str) -> tuple[tuple[int,int],...]:
    return tuple(EDGE_ORDER[i] for i in role_active_edges(role))

def inactive_edge_control(role: str) -> int | None:
    require(role in GRAPH_ROLES, "Only graph roles have edge controls")
    inactive=[i for i in range(6) if i not in role_active_edges(role)]
    return inactive[0] if inactive else None

def primary_lesion(role: str) -> tuple[int,...]:
    """All active routes for a graph role; G0 therefore returns empty."""
    return role_active_edges(role)

def g0_negative_control_lesion() -> tuple[int,...]:
    """All six unavailable routes; must be exactly null for G0."""
    return tuple(range(6))

def secondary_lesions(role: str) -> tuple[tuple[int,...],...]:
    """Only active routes, one at a time; caller must enforce causal gate."""
    return tuple((i,) for i in role_active_edges(role))

def in_causal_window(zero_based_decision: int) -> bool:
    return CAUSAL_WINDOW[0] <= int(zero_based_decision) <= CAUSAL_WINDOW[1]

def seed(split: str, *identity: object, bits: int=32) -> int:
    require(split in ALLOWED_SPLITS, "Only frozen B1-SC development splits are allowed")
    require(bits in (32,63), "Unsupported seed width")
    raw=json.dumps([SEED_NAMESPACE,split,*identity],separators=(",",":"),ensure_ascii=False).encode()
    value=int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")
    return value % (2**bits)

def registry() -> list[dict]:
    out=[]
    for role,mask,_ in PANEL:
        for block in range(BLOCKS):
            family="RoutingActor" if mask is not None else "PPO"
            out.append({
                "id":f"{role}-b{block}",
                "role":role,"mask":mask,"block":block,
                "input_mode":"raw","native_steps":NATIVE_STEPS,
                "initial_seed":seed("training","weights",family,block),
                "admission_panel_seed":seed("admission","panel",block),
                "causal_panel_seed":seed("causal","panel",block),
            })
    require(len(out)==72 and len({x["id"] for x in out})==72, "Registry must contain exactly 72 unique fits")
    return out

def block_competence(model_loss: float, model_disp: float, witness_loss: float, witness_disp: float) -> dict:
    li=float(witness_loss)-float(model_loss)
    dd=float(model_disp)-float(witness_disp)
    lp=li>LOSS_MARGIN
    dp=dd>=DISPLACEMENT_MARGIN
    return {"loss_improvement":li,"additional_displacement":dd,
            "loss_component_pass":lp,"displacement_component_pass":dp,
            "pass":bool(lp and dp)}

def admission_gate(blocks: Sequence[dict], aggregate: dict) -> dict:
    require(len(blocks)==BLOCKS, "Admission requires exactly 8 independent blocks")
    block_passes=sum(bool(x["pass"]) for x in blocks)
    agg=bool(aggregate["pass"])
    return {"aggregate_pass":agg,"blocks_passing":block_passes,
            "minimum_blocks":MIN_BLOCKS,"admitted":bool(agg and block_passes>=MIN_BLOCKS)}

def practical_harm(loss_harm: float, displacement_harm: float) -> bool:
    l=float(loss_harm); d=float(displacement_harm)
    return bool((l>LOSS_MARGIN and d>=-DISPLACEMENT_MARGIN) or
                (d>DISPLACEMENT_MARGIN and l>=-LOSS_MARGIN))

def causal_gate(block_harms: Sequence[bool], aggregate_harm: bool) -> dict:
    require(len(block_harms)==BLOCKS, "Causal gate requires exactly 8 independent blocks")
    n=sum(bool(x) for x in block_harms)
    return {"aggregate_practical_harm":bool(aggregate_harm),"blocks_with_practical_harm":n,
            "minimum_blocks":MIN_BLOCKS,"causal_gate_pass":bool(aggregate_harm and n>=MIN_BLOCKS)}

def practical_advantage(loss_advantage: float, displacement_advantage: float) -> bool:
    l=float(loss_advantage); d=float(displacement_advantage)
    return bool((l>LOSS_MARGIN and d>=-DISPLACEMENT_MARGIN) or
                (d>DISPLACEMENT_MARGIN and l>=-LOSS_MARGIN))

def utility_gate(block_advantages: Sequence[bool], aggregate_advantage: bool) -> dict:
    require(len(block_advantages)==BLOCKS, "Utility gate requires exactly 8 independent blocks")
    n=sum(bool(x) for x in block_advantages)
    return {"aggregate_practical_advantage":bool(aggregate_advantage),"blocks_with_practical_advantage":n,
            "minimum_blocks":MIN_BLOCKS,"utility_gate_pass":bool(aggregate_advantage and n>=MIN_BLOCKS)}

def implementation_summary() -> dict:
    d=validate_design_contract()
    reg=registry()
    return {
        "status":"IMPLEMENTATION_CONTRACT_READY_FOR_QA",
        "design":d,
        "edge_order":[list(x) for x in EDGE_ORDER],
        "panel":[list(x) for x in PANEL],
        "registry_entries":len(reg),
        "registry_sha256":sha256_bytes(json_bytes(reg)),
        "s6_edges":[list(x) for x in role_active_pairs("S6")],
        "causal_window_zero_based":list(CAUSAL_WINDOW),
        "training_callable_present":False,
        "start_request_required_for_any_future_execution":True,
        **BOUNDARY,
    }
