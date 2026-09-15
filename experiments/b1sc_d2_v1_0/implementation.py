"""B1-SC-D2 v1.0 executable design contract.

Implements only the frozen D2 mechanism contract, paired seed registry,
route operators and adjudication gates. It contains no scientific trainer or
evaluation campaign entry point.
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

SCHEMA = "B1-SC-D2-1.0.0-20260915"
REPOSITORY = "CRC2520/polar-sim-ml-original"
BRANCH = "research/b1sc-d2-v1.0-training-scaffold-20260915"
BASE_COMMIT = "2550c75336fb0af094bc02d2e99ddb680bd54933"
DESIGN_COMMIT = "c5d286743d613c09ef032317bebb10ae1871db00"
PARENT_RUN_ID = 34934384067
PARENT_AUTHORIZATION_COMMIT = "5ae02be4a8d80485e41898a2c4e8b9c466959b76"

EDGE_ORDER = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
S6_MASK = "100110"
ACTIVE_EDGES = (0, 3, 4)
INACTIVE_CONTROL_EDGE = 1
CONDITIONS = ("S6-ON", "S6-OFF-TRAIN")
BLOCKS = 8
FITS = 16
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
SEED_ROOT = "ae302b8be2005f914c9e1998303f9508c64c600cd140dff324c5d600be034f4f"
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


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def design() -> dict:
    return read_json(DESIGN_PATH)


def seed(split: str, identity: object) -> int:
    """Frozen D2 seed32 derivation; paired identities omit condition by design."""
    require(split in ALLOWED_SPLITS, "Only frozen D2 development split namespaces are allowed")
    raw = (SEED_ROOT + "|" + split + "|" + canonical_json(identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def active_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(EDGE_ORDER[i] for i in ACTIVE_EDGES)


def training_lesion(condition: str) -> tuple[int, ...]:
    require(condition in CONDITIONS, "Unknown D2 condition")
    return () if condition == "S6-ON" else ACTIVE_EDGES


def endpoint_lesion(condition: str) -> tuple[int, ...]:
    return training_lesion(condition)


def permanent_lesion() -> tuple[int, ...]:
    return ACTIVE_EDGES


def inactive_control_lesion() -> tuple[int, ...]:
    return (INACTIVE_CONTROL_EDGE,)


def window_9_40_lesion(zero_based_decision: int) -> tuple[int, ...]:
    return ACTIVE_EDGES if 9 <= int(zero_based_decision) <= 40 else ()


def paired_seed_identity(block: int, purpose: str, **extra) -> dict:
    require(0 <= int(block) < BLOCKS, "Invalid paired block")
    identity = {"block": int(block), "purpose": str(purpose), **extra}
    require("condition" not in identity, "Paired seed identity must exclude condition")
    return identity


def registry() -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        for block in range(BLOCKS):
            rows.append(dict(
                id=f"{condition}-b{block}",
                condition=condition,
                block=block,
                role="S6",
                mask=S6_MASK,
                lambda_train_active_routes=1.0 if condition == "S6-ON" else 0.0,
                endpoint_lambda_active_routes=1.0 if condition == "S6-ON" else 0.0,
                input_mode="raw",
                native_steps=NATIVE_STEPS,
                initial_seed=seed("training", paired_seed_identity(block, "weights")),
                policy_sampling_seed=seed("training", paired_seed_identity(block, "policy-sampling")),
                training_environment_seed_root=seed("training", paired_seed_identity(block, "environment-stream")),
                diagnostic_panel_seed=seed("diagnostic", paired_seed_identity(block, "panel")),
                endpoint_panel_seed=seed("endpoint", paired_seed_identity(block, "panel")),
                causal_panel_seed=seed("causal", paired_seed_identity(block, "panel")),
            ))
    require(len(rows) == FITS and len({x["id"] for x in rows}) == FITS, "D2 registry must contain 16 unique fits")
    return rows


def block_pair(block: int) -> tuple[dict, dict]:
    rows = [x for x in registry() if x["block"] == int(block)]
    require(len(rows) == 2, "Each D2 block must have exactly two paired conditions")
    on = next(x for x in rows if x["condition"] == "S6-ON")
    off = next(x for x in rows if x["condition"] == "S6-OFF-TRAIN")
    return on, off


def paired_seed_contract() -> dict:
    keys = (
        "initial_seed", "policy_sampling_seed", "training_environment_seed_root",
        "diagnostic_panel_seed", "endpoint_panel_seed", "causal_panel_seed",
    )
    for block in range(BLOCKS):
        on, off = block_pair(block)
        for key in keys:
            require(on[key] == off[key], f"ON/OFF paired seed mismatch at block {block}: {key}")
    return {"status": "PASS", "paired_blocks": BLOCKS, "paired_seed_fields": list(keys)}


def practical_advantage(loss_advantage: float, displacement_advantage: float) -> bool:
    l = float(loss_advantage)
    d = float(displacement_advantage)
    return bool(
        (l > LOSS_MARGIN and d >= -DISPLACEMENT_MARGIN)
        or (d > DISPLACEMENT_MARGIN and l >= -LOSS_MARGIN)
    )


def training_gate(block_advantages: Sequence[bool], aggregate_advantage: bool) -> dict:
    require(len(block_advantages) == BLOCKS, "H_TRAIN-D2 requires exactly 8 paired blocks")
    n = sum(bool(x) for x in block_advantages)
    return dict(
        aggregate_practical_advantage=bool(aggregate_advantage),
        blocks_with_practical_advantage=n,
        minimum_blocks=MIN_BLOCKS,
        H_TRAIN_D2=bool(aggregate_advantage and n >= MIN_BLOCKS),
    )


def practical_harm(loss_harm: float, displacement_harm: float) -> bool:
    l = float(loss_harm)
    d = float(displacement_harm)
    return bool(
        (l > LOSS_MARGIN and d >= -DISPLACEMENT_MARGIN)
        or (d > DISPLACEMENT_MARGIN and l >= -LOSS_MARGIN)
    )


def online_gate(block_harms: Sequence[bool], aggregate_harm: bool, controls_pass: bool) -> dict:
    require(len(block_harms) == BLOCKS, "H_ONLINE-D2 requires exactly 8 paired blocks")
    n = sum(bool(x) for x in block_harms)
    return dict(
        aggregate_practical_harm=bool(aggregate_harm),
        blocks_with_practical_harm=n,
        minimum_blocks=MIN_BLOCKS,
        controls_pass=bool(controls_pass),
        H_ONLINE_D2=bool(aggregate_harm and n >= MIN_BLOCKS and controls_pass),
    )


def adjudicate(h_train: bool, h_online: bool, integrity_pass: bool = True) -> str:
    if not integrity_pass:
        return "INVALID_OR_INCONCLUSIVE_MECHANISTIC_ADJUDICATION"
    table = {
        (True, False): "TRAINING_SCAFFOLD_SUPPORTED",
        (True, True): "TRAINING_AND_ONLINE_DEPENDENCE_SUPPORTED",
        (False, True): "ONLINE_DEPENDENCE_ONLY_SUPPORTED",
        (False, False): "NO_REPRODUCIBLE_S6_MECHANISM_UNDER_D2",
    }
    return table[(bool(h_train), bool(h_online))]


def validate_design_contract() -> dict:
    d = design()
    require(d["schema"] == SCHEMA, "Unexpected D2 schema")
    require(d["status"] == "DESIGN_FROZEN_NOT_EXECUTED", "D2 design status changed")
    require(d["repository"] == REPOSITORY and d["branch"] == BRANCH, "Repository/branch contract changed")
    require(d["base_commit"] == BASE_COMMIT, "D2 base commit changed")
    require(d["support"]["mask"] == S6_MASK, "S6 mask changed")
    require(tuple(d["support"]["active_edge_indices"]) == ACTIVE_EDGES, "S6 active edges changed")
    require(tuple(tuple(x) for x in d["support"]["edge_order"]) == EDGE_ORDER, "Edge order changed")
    require(d["support"]["inactive_control_edge_index"] == INACTIVE_CONTROL_EDGE, "Inactive control changed")
    require(d["training"]["blocks"] == BLOCKS and d["training"]["fits"] == FITS, "Training dimensions changed")
    require(d["training"]["native_steps_per_fit"] == NATIVE_STEPS, "Training budget changed")
    require(d["training"]["total_native_training_steps"] == FITS * NATIVE_STEPS, "Total training budget changed")
    require(d["training"]["input_mode"] == "raw", "Input mode changed")
    require(d["training"]["paired_blocks"] is True and d["training"]["train_from_scratch"] is True, "Pairing/from-scratch changed")
    require(d["training"]["hyperparameter_search"] is False and d["training"]["early_stopping"] is False, "Tuning boundary changed")
    require(tuple(d["checkpoint_diagnostics"]["checkpoints_native_steps"]) == CHECKPOINTS, "Checkpoints changed")
    require(d["checkpoint_diagnostics"]["episodes_per_checkpoint_per_block_per_condition"] == DIAGNOSTIC_EPISODES, "Diagnostic panel changed")
    require(d["training_effect_primary"]["episodes_per_block_per_condition"] == ENDPOINT_EPISODES, "Endpoint panel changed")
    require(d["training_effect_primary"]["minimum_blocks_with_practical_advantage"] == MIN_BLOCKS, "Training 6/8 gate changed")
    require(abs(d["training_effect_primary"]["loss_margin_strict_gt"] - LOSS_MARGIN) < 1e-15, "Training loss margin changed")
    require(d["online_dependence_primary"]["episodes_per_block"] == CAUSAL_EPISODES, "Causal panel changed")
    require(d["online_dependence_primary"]["minimum_blocks_with_practical_harm"] == MIN_BLOCKS, "Online 6/8 gate changed")
    require(tuple(d["online_dependence_primary"]["primary_intervention"]["active_edge_indices"]) == ACTIVE_EDGES, "Permanent lesion changed")
    require(d["online_dependence_primary"]["primary_intervention"]["window_zero_based"] == [0, "terminal"], "Permanent window changed")
    require(d["online_dependence_primary"]["secondary_replication"]["window_zero_based"] == [9, 40], "Replication window changed")
    require(d["analysis"]["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES, "Bootstrap count changed")
    require(d["qa_contract"]["numerical_tolerance"] == NUMERICAL_TOLERANCE, "QA tolerance changed")
    require(d["seed_commitment"]["seed_root_sha256"] == SEED_ROOT, "Seed root changed")
    require(tuple(d["seed_commitment"]["allowed_splits"]) == ALLOWED_SPLITS, "Seed splits changed")
    require(d["seed_commitment"]["paired_seed_identity_excludes_condition"] is True, "Paired seed identity changed")
    require(d["lineage"]["parent_run_id"] == PARENT_RUN_ID, "Parent run changed")
    require(d["lineage"]["parent_authorization_commit"] == PARENT_AUTHORIZATION_COMMIT, "Parent authorization changed")
    require(d["lineage"]["motivating_observations"]["S6_competence_blocks"] == 8, "Motivating S6 observation changed")
    require(d["lineage"]["motivating_observations"]["primary_causal_passed_roles"] == [], "Motivating causal observation changed")
    require(sha256_file(PROTOCOL_PATH) == d["protocol_sha256"], "Protocol bytes do not match D2 freeze")
    for k, v in BOUNDARY.items():
        require(d["B1E_boundary"][k] == v, f"B1-E boundary changed: {k}")
    eb = d["execution_boundary"]
    require(all(eb[k] is False for k in ("execution_authorized", "workflow_installed", "start_request_exists", "training_started", "evaluation_started", "results_exist")), "Execution boundary changed")
    return dict(
        status="PASS", schema=SCHEMA, fits=FITS, blocks=BLOCKS,
        conditions=list(CONDITIONS), protocol_sha256=d["protocol_sha256"],
        **BOUNDARY,
    )


def implementation_summary() -> dict:
    contract = validate_design_contract()
    reg = registry()
    pairing = paired_seed_contract()
    return dict(
        status="D2_IMPLEMENTATION_CONTRACT_READY_FOR_QA",
        design=contract,
        pairing=pairing,
        registry_entries=len(reg),
        registry_sha256=sha256_bytes(json_bytes(reg)),
        active_edges=list(ACTIVE_EDGES),
        active_pairs=[list(x) for x in active_pairs()],
        inactive_control_edge=INACTIVE_CONTROL_EDGE,
        checkpoints=list(CHECKPOINTS),
        scientific_training_callable_present=False,
        scientific_evaluation_callable_present=False,
        start_request_required_for_future_execution=True,
        **BOUNDARY,
    )
