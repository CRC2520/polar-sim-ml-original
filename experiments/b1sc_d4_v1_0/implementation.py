"""B1-SC-D4 v1.0 executable design contract for temporal S6 gating.

This module implements the prospectively frozen temporal lambda manipulation only.
It contains no scientific trainer, workflow authorization, or B1-E progression.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from b1s.execution.core import EDGES
from experiments.b1sc_d3_v1_0 import implementation as d3

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "DESIGN_FREEZE.json"
PROTOCOL_PATH = ROOT / "PROTOCOL_ES.md"
REGISTRY_PATH = ROOT / "FIT_REGISTRY.json"

SCHEMA = "B1-SC-D4-1.0.0-20260916"
REPOSITORY = "CRC2520/polar-sim-ml-original"
DESIGN_BRANCH = "research/b1sc-d4-v1.0-late-route-transition-design-20260916"
DESIGN_COMMIT = "9e70e8917cc0c20c9410f0318f48b95fbf0fdb9e"
PARENT_CLOSURE_COMMIT = "7842ef2933edf4800d59f06af79f0368cf03cd9d"
INIT_BRANCH = "research/b1sc-d4-v1.0-byte-frozen-init-20260916"

S6 = "100110"
ACTIVE_EDGES = (0, 3, 4)
INACTIVE_CONTROL_EDGE = 1
LOCAL = d3.LOCAL
BLOCKS = 12
CONDITIONS = ("LOCAL-S6-OFF", "LOCAL-S6-LATE", "LOCAL-S6-ALWAYS")
FITS = 36
NATIVE_STEPS = 1_048_576
SWITCH_STEPS = 786_432
CHECKPOINTS = (262_144, 524_288, 786_432, 851_968, 917_504, 983_040, 1_048_576)
DIAGNOSTIC_EPISODES = 32
ENDPOINT_EPISODES = 64
CAUSAL_EPISODES = 64
MIN_BLOCKS = 9
LOSS_MARGIN = 130 / 30
DISPLACEMENT_MARGIN = 1.0
NUMERICAL_TOLERANCE = 1e-8
BOOTSTRAP_RESAMPLES = 10_000
SEED_ROOT = "b8359617997d4b0403cd588908d4dac99fd50170ae43b63cc6323c297ca27f89"
ALLOWED_SPLITS = ("training", "diagnostic", "endpoint", "causal", "sham", "bootstrap", "qa")

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
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed(split: str, identity: object) -> int:
    require(split in ALLOWED_SPLITS, "Forbidden D4 split")
    require("condition" not in identity if isinstance(identity, dict) else True,
            "Paired D4 seed identity must exclude condition")
    raw = (SEED_ROOT + "|" + split + "|" + canonical_json(identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def paired_seed_identity(block: int, purpose: str, **extra) -> dict:
    require(0 <= int(block) < BLOCKS, "Invalid D4 block")
    require("condition" not in extra, "D4 paired identity cannot include condition")
    return {"block": int(block), "purpose": str(purpose), **extra}


def snapshot_seed(block: int) -> int:
    return seed("training", paired_seed_identity(block, "initial-trainable-snapshot"))


def training_lambda(condition: str, completed_native_steps: int) -> float:
    """Route lambda used for the next rollout after exactly this many completed steps."""
    require(condition in CONDITIONS, "Unknown D4 condition")
    s = int(completed_native_steps)
    require(0 <= s <= NATIVE_STEPS, "D4 completed step count outside frozen budget")
    if condition == "LOCAL-S6-OFF":
        return 0.0
    if condition == "LOCAL-S6-ALWAYS":
        return 1.0
    return 0.0 if s < SWITCH_STEPS else 1.0


def diagnostic_lambda(condition: str, checkpoint_native_steps: int) -> float:
    """Configured route state at a frozen diagnostic checkpoint.

    The 786432 checkpoint is explicitly pre-switch for LOCAL-S6-LATE.
    """
    require(condition in CONDITIONS, "Unknown D4 condition")
    s = int(checkpoint_native_steps)
    require(s in CHECKPOINTS, "Unknown D4 checkpoint")
    if condition == "LOCAL-S6-OFF":
        return 0.0
    if condition == "LOCAL-S6-ALWAYS":
        return 1.0
    return 0.0 if s <= SWITCH_STEPS else 1.0


def endpoint_lambda(condition: str) -> float:
    require(condition in CONDITIONS, "Unknown D4 condition")
    return 0.0 if condition == "LOCAL-S6-OFF" else 1.0


class D4RoutingActor(d3.D3RoutingActor):
    """D3 LOCAL_PARTITIONED actor with an explicit frozen scalar route gate λ."""

    def __init__(self):
        super().__init__(S6, LOCAL)

    def forward(self, z, route_lambda: float = 1.0, lesion: tuple[int, ...] = (), trace: bool = False):
        lam_value = float(route_lambda)
        if lam_value not in (0.0, 1.0):
            raise ValueError("D4 route_lambda is frozen to binary values 0 or 1")
        views = d3.node_views(z, LOCAL)
        onehot = torch.eye(3, device=z.device, dtype=z.dtype).expand(z.shape[0], -1, -1)
        h = torch.tanh(self.E(torch.cat((views, onehot), -1)))
        msgs = torch.stack([torch.tanh(self.A(torch.cat((h[:, i], h[:, j]), -1))) for i, j in EDGES], 1)
        gate = z.new_tensor([self.support[i, j].item() for i, j in EDGES])
        edge_lambda = torch.ones_like(gate) * lam_value
        if len(set(lesion)) != len(lesion) or any(e not in range(6) for e in lesion):
            raise ValueError("Invalid D4 edge lesion")
        for e in lesion:
            edge_lambda[e] = 0
        used = msgs * (gate * edge_lambda)[None, :, None]
        r = torch.stack(
            [used[:, [e for e, (_, j) in enumerate(EDGES) if j == node]].sum(1) / 2 for node in range(3)], 1
        )
        g = torch.tanh(self.B(torch.cat((h, r), -1)))
        mu = self.D(g).flatten(1)
        ls = self.log_std.repeat(3).expand_as(mu)
        if trace:
            return mu, ls, {
                "node_views": views,
                "h": h,
                "m": msgs,
                "used": used,
                "r": r,
                "route_lambda": lam_value,
                "edge_lambda": edge_lambda,
                "support": self.support.clone(),
                "mu": mu,
                "action": mu.tanh(),
            }
        return mu, ls


def registry() -> list[dict]:
    rows = read_json(REGISTRY_PATH)
    require(len(rows) == FITS and len({x["id"] for x in rows}) == FITS, "D4 registry must contain 36 unique fits")
    return rows


def practical_advantage(loss_advantage: float, displacement_advantage: float) -> bool:
    l = float(loss_advantage)
    d = float(displacement_advantage)
    return bool(
        (l > LOSS_MARGIN and d >= -DISPLACEMENT_MARGIN)
        or (d > DISPLACEMENT_MARGIN and l >= -LOSS_MARGIN)
    )


def practical_harm(loss_harm: float, displacement_harm: float) -> bool:
    return practical_advantage(loss_harm, displacement_harm)


def replicated_gate(block_flags: Sequence[bool], aggregate_flag: bool, competence_flags: Sequence[bool] | None = None) -> dict:
    require(len(block_flags) == BLOCKS, "D4 gate requires exactly 12 blocks")
    n = sum(bool(x) for x in block_flags)
    if competence_flags is None:
        nc = None
        passed = bool(aggregate_flag and n >= MIN_BLOCKS)
    else:
        require(len(competence_flags) == BLOCKS, "D4 competence gate requires exactly 12 blocks")
        nc = sum(bool(x) for x in competence_flags)
        passed = bool(aggregate_flag and n >= MIN_BLOCKS and nc >= MIN_BLOCKS)
    return dict(
        aggregate=bool(aggregate_flag),
        blocks_passing=n,
        competent_blocks=nc,
        minimum_blocks=MIN_BLOCKS,
        pass_gate=passed,
    )


def validate_design() -> dict:
    d = read_json(DESIGN_PATH)
    rows = registry()
    require(d["schema"] == SCHEMA and d["status"] == "D4_DESIGN_FROZEN_NOT_EXECUTED", "D4 freeze changed")
    require(d["repository"] == REPOSITORY and d["branch"] == DESIGN_BRANCH, "D4 repository/branch changed")
    require(d["parent_closure_commit"] == PARENT_CLOSURE_COMMIT, "D4 parent closure changed")
    require(d["architecture"]["support_mask"] == S6, "D4 support changed")
    require(tuple(d["architecture"]["active_edge_indices"]) == ACTIVE_EDGES, "D4 active edges changed")
    require(d["architecture"]["information_mode"] == LOCAL, "D4 information mode changed")
    require(d["timing"]["switch_after_completed_native_steps"] == SWITCH_STEPS, "D4 switch changed")
    require(d["timing"]["checkpoint_786432_is_pre_switch"] is True, "D4 pre-switch checkpoint changed")
    require(tuple(d["diagnostics"]["checkpoints_native_steps"]) == CHECKPOINTS, "D4 checkpoints changed")
    require(d["training"]["blocks"] == BLOCKS and d["training"]["fits"] == FITS, "D4 dimensions changed")
    require(d["training"]["native_steps_per_fit"] == NATIVE_STEPS, "D4 budget changed")
    require(d["training"]["fresh_byte_frozen_snapshots_required"] == BLOCKS, "D4 snapshot count changed")
    require(d["training"]["reuse_d3_snapshot_forbidden"] is True, "D3 snapshot reuse enabled")
    require(d["training"]["reuse_d3_model_or_checkpoint_forbidden"] is True, "D3 model reuse enabled")
    require(d["seed_commitment"]["seed_root_sha256"] == SEED_ROOT, "D4 seed root changed")
    require(tuple(d["seed_commitment"]["allowed_splits"]) == ALLOWED_SPLITS, "D4 seed splits changed")
    require(d["seed_commitment"]["d3_block_identity_forbidden"] is True, "D3 block identity allowed")
    require(d["anti_selection"]["d3_block_reuse"] is False, "D3 blocks reused")
    require(d["anti_selection"]["d3_blocks_6_7_special_handling"] is False, "D3 blocks 6/7 special handling enabled")
    require(d["anti_selection"]["switch_tuning_after_d4"] is False, "D4 switch tuning enabled")
    require(d["gates"]["minimum_blocks"] == MIN_BLOCKS, "D4 replication threshold changed")
    require(abs(d["gates"]["loss_margin_strict_gt"] - LOSS_MARGIN) < 1e-15, "D4 loss margin changed")
    require(d["gates"]["displacement_margin"] == DISPLACEMENT_MARGIN, "D4 displacement margin changed")
    eb = d["execution_boundary"]
    require(all(eb[k] is False for k in (
        "implementation_present", "qa_performed", "canonical_snapshots_generated", "runner_present",
        "workflow_present", "start_request_exists", "execution_authorized", "training_started",
        "evaluation_started", "results_exist", "D4_executed",
    )), "D4 execution boundary changed")
    for k, v in BOUNDARY.items():
        require(d["B1E_boundary"][k] == v, f"B1-E boundary changed: {k}")

    paired = (
        "initial_snapshot_block", "initial_snapshot_seed", "policy_sampling_seed",
        "training_environment_seed_root", "diagnostic_panel_seed", "endpoint_panel_seed", "causal_panel_seed",
    )
    for b in range(BLOCKS):
        rs = [x for x in rows if int(x["block"]) == b]
        require(len(rs) == 3 and {x["condition"] for x in rs} == set(CONDITIONS), f"D4 block {b} incomplete")
        require(all("d3_block" not in x for x in rs), f"D3 block identity leaked into D4 registry block {b}")
        for key in paired:
            require(len({x[key] for x in rs}) == 1, f"D4 pairing mismatch block {b}: {key}")
        require(next(iter({x["initial_snapshot_seed"] for x in rs})) == snapshot_seed(b),
                f"D4 snapshot seed mismatch block {b}")
    return dict(
        status="D4_IMPLEMENTATION_CONTRACT_READY_FOR_QA",
        blocks=BLOCKS,
        fits=FITS,
        conditions=list(CONDITIONS),
        switch_steps=SWITCH_STEPS,
        registry_sha256=sha256_file(REGISTRY_PATH),
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        scientific_results_exist=False,
        **BOUNDARY,
    )
