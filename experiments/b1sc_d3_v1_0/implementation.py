"""B1-SC-D3 v1.0 design implementation contract.

Design/QA only.  This module contains no scientific trainer, evaluator, workflow
authorization, or B1-E progression entry point.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch

from b1s.execution.core import RoutingActor, Value, EDGES

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "DESIGN_FREEZE.json"
PROTOCOL_PATH = ROOT / "PROTOCOL_ES.md"
REGISTRY_PATH = ROOT / "FIT_REGISTRY.json"

SCHEMA = "B1-SC-D3-1.0.0-20260916"
BRANCH = "research/b1sc-d3-v1.0-information-partitioning-design-20260916"
BASE_COMMIT = "e9d752683e5b1b52165d4a660668a714e33c3d0f"
GLOBAL = "GLOBAL_SHARED"
LOCAL = "LOCAL_PARTITIONED"
MODES = (GLOBAL, LOCAL)
G0 = "000000"
S6 = "100110"
ACTIVE_EDGES = (0, 3, 4)
INACTIVE_CONTROL_EDGE = 1
BLOCKS = 8
CONDITIONS = ("GLOBAL-G0", "GLOBAL-S6", "LOCAL-G0", "LOCAL-S6")
FITS = 32
NATIVE_STEPS = 1_048_576
CHECKPOINTS = (262_144, 524_288, 786_432, 1_048_576)
DIAGNOSTIC_EPISODES = 32
ENDPOINT_EPISODES = 64
CAUSAL_EPISODES = 64
MIN_BLOCKS = 6
LOSS_MARGIN = 130 / 30
DISPLACEMENT_MARGIN = 1.0
BOOTSTRAP_RESAMPLES = 10_000
NUMERICAL_TOLERANCE = 1e-8
SEED_ROOT = "a01253416f7e869e98b94dd321607b3826eebadb395048a580c652f1a22b62bf"
ALLOWED_SPLITS = ("training","diagnostic","endpoint","causal","sham","bootstrap","qa")
LOCAL_SLICES = ((0,31),(31,62),(62,93))

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

def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False)

def sha256_file(path: Path | str) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def seed(split: str, identity: object) -> int:
    require(split in ALLOWED_SPLITS, "Forbidden D3 split")
    raw=(SEED_ROOT+"|"+split+"|"+canonical_json(identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")%(2**32)

def node_views(z: torch.Tensor, mode: str) -> torch.Tensor:
    if z.ndim != 2 or z.shape[1] != 93:
        raise ValueError("D3 observation must be [batch,93]")
    if mode == GLOBAL:
        return z[:,None,:].expand(-1,3,-1)
    if mode != LOCAL:
        raise ValueError("Unknown D3 information mode")
    out=torch.zeros((z.shape[0],3,93),dtype=z.dtype,device=z.device)
    for i,(lo,hi) in enumerate(LOCAL_SLICES):
        out[:,i,lo:hi]=z[:,lo:hi]
    return out

class D3RoutingActor(RoutingActor):
    """RoutingActor with frozen D3 node-information view; trainable shapes unchanged."""
    def __init__(self, mask: str, information_mode: str):
        super().__init__(mask)
        if information_mode not in MODES:
            raise ValueError("Unknown D3 information mode")
        self.information_mode=information_mode

    def forward(self,z,lesion: tuple[int,...]=(),trace: bool=False):
        views=node_views(z,self.information_mode)
        onehot=torch.eye(3,device=z.device,dtype=z.dtype).expand(z.shape[0],-1,-1)
        h=torch.tanh(self.E(torch.cat((views,onehot),-1)))
        msgs=torch.stack([torch.tanh(self.A(torch.cat((h[:,i],h[:,j]),-1))) for i,j in EDGES],1)
        gate=z.new_tensor([self.support[i,j].item() for i,j in EDGES])
        lam=torch.ones_like(gate)
        if len(set(lesion))!=len(lesion) or any(e not in range(6) for e in lesion):
            raise ValueError("Invalid edge lesion")
        for e in lesion:
            lam[e]=0
        used=msgs*(gate*lam)[None,:,None]
        r=torch.stack([used[:,[e for e,(_,j) in enumerate(EDGES) if j==node]].sum(1)/2 for node in range(3)],1)
        g=torch.tanh(self.B(torch.cat((h,r),-1)))
        mu=self.D(g).flatten(1)
        ls=self.log_std.repeat(3).expand_as(mu)
        if trace:
            return mu,ls,{"node_views":views,"h":h,"m":msgs,"used":used,"r":r,
                          "lambda":lam,"support":self.support.clone(),"mu":mu,"action":mu.tanh()}
        return mu,ls

def practical_advantage(loss_advantage: float, displacement_advantage: float) -> bool:
    l=float(loss_advantage);d=float(displacement_advantage)
    return bool((l>LOSS_MARGIN and d>=-DISPLACEMENT_MARGIN) or
                (d>DISPLACEMENT_MARGIN and l>=-LOSS_MARGIN))

def practical_harm(loss_harm: float, displacement_harm: float) -> bool:
    return practical_advantage(loss_harm,displacement_harm)

def competence(loss_improvement: float, additional_displacement: float) -> bool:
    return bool(float(loss_improvement)>LOSS_MARGIN and float(additional_displacement)>=DISPLACEMENT_MARGIN)

def utility_gate(block_advantages: Sequence[bool], aggregate_advantage: bool,
                 local_s6_competence: Sequence[bool]) -> dict:
    require(len(block_advantages)==BLOCKS and len(local_s6_competence)==BLOCKS,
            "D3 utility gate requires exactly eight blocks")
    n=sum(bool(x) for x in block_advantages)
    nc=sum(bool(x) for x in local_s6_competence)
    passed=bool(aggregate_advantage and n>=MIN_BLOCKS and nc>=MIN_BLOCKS)
    return dict(aggregate_practical_advantage=bool(aggregate_advantage),
                blocks_with_practical_advantage=n,
                local_s6_competent_blocks=nc,minimum_blocks=MIN_BLOCKS,
                H_INFO_UTILITY=passed)

def causal_gate(block_harms: Sequence[bool], aggregate_harm: bool, controls_pass: bool) -> dict:
    require(len(block_harms)==BLOCKS,"D3 causal gate requires exactly eight blocks")
    n=sum(bool(x) for x in block_harms)
    passed=bool(aggregate_harm and n>=MIN_BLOCKS and controls_pass)
    return dict(aggregate_practical_harm=bool(aggregate_harm),
                blocks_with_practical_harm=n,minimum_blocks=MIN_BLOCKS,
                controls_pass=bool(controls_pass),H_INFO_CAUSAL=passed)

def adjudicate(h_utility: bool, h_causal: bool, integrity_pass: bool=True) -> str:
    if not integrity_pass:
        return "INVALID_OR_INCONCLUSIVE_D3_ADJUDICATION"
    return {
        (True,True):"PARTITIONED_INFORMATION_STRUCTURAL_DEPENDENCE_SUPPORTED",
        (True,False):"PARTITIONED_TRAINING_UTILITY_WITHOUT_ONLINE_DEPENDENCE",
        (False,True):"PARTITIONED_ONLINE_DEPENDENCE_WITHOUT_TRAINING_UTILITY",
        (False,False):"NO_REPRODUCIBLE_PARTITION_INDUCED_S6_NECESSITY",
    }[(bool(h_utility),bool(h_causal))]

def validate_design() -> dict:
    d=read_json(DESIGN_PATH);reg=read_json(REGISTRY_PATH)
    require(d["schema"]==SCHEMA and d["status"]=="D3_DESIGN_FROZEN_NOT_EXECUTED","D3 freeze changed")
    require(d["branch"]==BRANCH and d["base_commit"]==BASE_COMMIT,"D3 lineage changed")
    require(sha256_file(PROTOCOL_PATH)==d["protocol_sha256"],"D3 protocol hash mismatch")
    require(d["observation_contract"]["native_flat_dim"]==93,"Native observation dimension changed")
    require(d["observation_contract"]["native_dim_per_node"]==31,"Local observation dimension changed")
    require(d["observation_contract"]["critic_input"]=="global raw 93-D in all conditions","Critic boundary changed")
    require(d["observation_contract"]["actor_global_side_channel"] is False,"Actor side channel enabled")
    require(d["supports"]["G0"]["mask"]==G0 and d["supports"]["S6"]["mask"]==S6,"Support changed")
    require(tuple(d["supports"]["S6"]["active_edge_indices"])==ACTIVE_EDGES,"S6 edges changed")
    require(d["training"]["blocks"]==BLOCKS and d["training"]["fits"]==FITS,"D3 dimensions changed")
    require(d["training"]["native_steps_per_fit"]==NATIVE_STEPS,"D3 budget changed")
    require(d["H_INFO_UTILITY"]["minimum_blocks"]==MIN_BLOCKS,"Utility gate changed")
    require(d["H_INFO_CAUSAL"]["minimum_blocks"]==MIN_BLOCKS,"Causal gate changed")
    require(d["seed_commitment"]["seed_root_sha256"]==SEED_ROOT,"Seed root changed")
    require(tuple(d["seed_commitment"]["allowed_splits"])==ALLOWED_SPLITS,"Seed splits changed")
    require(len(reg)==FITS and len({x["id"] for x in reg})==FITS,"Registry must contain 32 unique fits")
    paired=("policy_sampling_seed","training_environment_seed_root","diagnostic_panel_seed",
            "endpoint_panel_seed","causal_panel_seed","initial_snapshot_block")
    for b in range(BLOCKS):
        rows=[x for x in reg if x["block"]==b]
        require(len(rows)==4,"Each block must contain four conditions")
        for key in paired:
            require(len({x[key] for x in rows})==1,f"Block pairing mismatch: {b} {key}")
    eb=d["execution_boundary"]
    require(all(eb[k] is False for k in ("scientific_runner_present","scientific_workflow_present",
        "start_request_exists","execution_authorized","canonical_snapshots_generated",
        "training_started","evaluation_started","results_exist")),"Execution boundary changed")
    for k,v in BOUNDARY.items():
        require(d["B1E_boundary"][k]==v,f"B1-E boundary changed: {k}")
    return dict(status="PASS",schema=SCHEMA,registry_entries=len(reg),protocol_sha256=d["protocol_sha256"],**BOUNDARY)

def qa_tests() -> dict:
    torch.set_num_threads(1);torch.manual_seed(20260916)
    contract=validate_design()
    z=torch.randn(7,93)
    gv=node_views(z,GLOBAL);lv=node_views(z,LOCAL)
    require(torch.equal(gv[:,0],z) and torch.equal(gv[:,1],z) and torch.equal(gv[:,2],z),
            "GLOBAL_SHARED does not reproduce shared observation")
    for i,(lo,hi) in enumerate(LOCAL_SLICES):
        require(torch.equal(lv[:,i,lo:hi],z[:,lo:hi]),f"Local slice lost for node {i}")
        outside=torch.cat((lv[:,i,:lo],lv[:,i,hi:]),1)
        require(torch.count_nonzero(outside).item()==0,f"Cross-node information leak for node {i}")

    base=RoutingActor(S6);global_actor=D3RoutingActor(S6,GLOBAL)
    global_actor.load_state_dict(base.state_dict())
    with torch.no_grad():
        bm,_=base(z);gm,_=global_actor(z)
    require(torch.equal(bm,gm),"D3 GLOBAL_SHARED is not bitwise-equivalent to historical RoutingActor")

    local_g0=D3RoutingActor(G0,LOCAL)
    global_g0=D3RoutingActor(G0,GLOBAL)
    require(sum(p.numel() for p in local_g0.parameters())==sum(p.numel() for p in global_g0.parameters()),
            "Parameter count differs across information modes")
    for node,(lo,hi) in enumerate(LOCAL_SLICES):
        z2=z.clone();mask=torch.ones(93,dtype=torch.bool);mask[lo:hi]=False
        z2[:,mask]=torch.randn_like(z2[:,mask])*5
        with torch.no_grad():
            m1,_,t1=local_g0(z,trace=True);m2,_,t2=local_g0(z2,trace=True)
        require(torch.equal(t1["h"][:,node],t2["h"][:,node]),f"Pre-message leakage into node {node}")
        require(torch.equal(m1[:,4*node:4*node+4],m2[:,4*node:4*node+4]),
                f"LOCAL-G0 remote observation changed node {node} action")
    with torch.no_grad():
        _,_,g0trace=local_g0(z,trace=True)
    require(torch.count_nonzero(g0trace["used"]).item()==0,"G0 used messages")

    local_s6=D3RoutingActor(S6,LOCAL)
    with torch.no_grad():
        _,_,tr=local_s6(z,trace=True)
    require(tuple(map(tuple,torch.nonzero(tr["support"],as_tuple=False).tolist()))==((0,1),(1,2),(2,0)),
            "S6 support changed")
    critic=Value()
    require(critic.net[0].in_features==93,"Critic is no longer centralized 93-D")

    require(utility_gate([1,1,1,1,1,1,0,0],True,[1]*8)["H_INFO_UTILITY"] is True,
            "Utility 6/8 gate failed")
    require(utility_gate([1,1,1,1,1,0,0,0],True,[1]*8)["H_INFO_UTILITY"] is False,
            "Utility 5/8 gate passed")
    require(causal_gate([1,1,1,1,1,1,0,0],True,True)["H_INFO_CAUSAL"] is True,
            "Causal 6/8 gate failed")
    require(causal_gate([1]*8,True,False)["H_INFO_CAUSAL"] is False,
            "Causal control requirement failed")
    require(adjudicate(False,False)=="NO_REPRODUCIBLE_PARTITION_INDUCED_S6_NECESSITY",
            "Adjudication matrix changed")
    return dict(status="D3_DESIGN_QA_PASS_NOT_EXECUTED",design=contract,
                global_equivalence="BITWISE_PASS",local_partition="PASS",
                local_g0_anti_leakage="PASS",support_contract="PASS",
                gate_contracts="PASS",scientific_training_performed=False,
                scientific_evaluation_performed=False,scientific_results_exist=False,
                **BOUNDARY)

if __name__=="__main__":
    print(json.dumps(qa_tests(),indent=2,sort_keys=True))
