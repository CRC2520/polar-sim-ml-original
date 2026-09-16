"""Fail-closed execution preflight for B1-SC-D3 v1.0.

Security/scientific invariant
-----------------------------
No optimizer may be constructed and no training step may be attempted until the
exact immutable snapshot bytes for the requested block have passed length,
SHA-256, header and trainable-state validation.

The validated bytes are retained inside the permit and are the bytes later
loaded into the model.  The runtime never reconstructs scientific
initialization from a seed and never re-reads a snapshot after validation,
which closes the validate-then-swap (TOCTOU) window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import torch

from b1s.execution.core import Value
from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0.init_format import decode_snapshot, trainable_digest

ROOT = Path(__file__).resolve().parent
INIT_FREEZE_PATH = ROOT / "INIT_FREEZE.json"
FROZEN_INIT_DIR = ROOT / "frozen_init"

# Independent code-level pin of the eight repository bytes.  INIT_FREEZE.json
# is checked against this table; editing the manifest and the snapshots together
# therefore cannot silently authorize a different initialization.
IMMUTABLE_BLOCKS = (
    # block, bytes, sha256, source_seed, actor_digest, critic_digest
    (0, 51519, "30ad184c683b471eb198aeeaeb1a24b835cefd1f36f0090d7da074cb7b7a6756", 500584769,
     "8705ed7f0deec9eada0e458b0f20df0c854c6a63414681cafc8ec82e7210cbde",
     "da820b7a23c4bb276c6300b76af450bcc5a6a24e0e272e66f7be9371ebc423ce"),
    (1, 51520, "07d69728675ec90a2aa024d85f4153939bccf5a62fc4414b8209874f03deb737", 1942996382,
     "bc38c042578d4318cbbbac8ee21ce372b6bfac84e1483eb4fca4de8a04ef763f",
     "b99e5abd6fbad6db8b20157b3191095f6cfd87b75688e19ee1c9f8d8f92b83cb"),
    (2, 51520, "096dc088d21180f972e48e6ed1870463244b5ee4d47392880b7f700084e85b7b", 3051758160,
     "4e93119d4641b69a536d6bc89cdb64a8cf796964be919b46af034ce467e9bc9e",
     "7cdc0d79e66e5f8e0b559bf81961e2fffb8c354a06570a63add4536e101a963e"),
    (3, 51520, "2ae02b44def6e41e12f18e0672cd2afa2406c2a409fe3be3e33a99e8690dbf9d", 2038421252,
     "d82a788a4acee7d916a62449266eeb4cbd48a04a89238c301df5b3205cd49d47",
     "8ad0895d89efa1b45e081e605342e6eece9d564f4292ca1f3c295fb032da9fd7"),
    (4, 51520, "24eced5465c7ad649d5c9d47fe4407765c0e700e4a46b304e7b67b548112d849", 4289014941,
     "f3ac11cb68b8dbff9b6c24fa899409c9f4a2062d1f79bd687ae05d6d78fd6420",
     "b03fc39607e38324bbe9e66bf0113a6968a4ac91bfbadd30cd2d91f313a96c6d"),
    (5, 51520, "7c4b337ae420d6c20d7abc6ccd9ec04677cc07389c0414a12174cad9feb9a199", 3098980698,
     "e40ab6eeba6a1dfa2a937e3b88d687abcebec68ff23683084c0758c336b9e028",
     "5247edc886b7fc9c15b85401b15f654be305e56bd7b80d4cfc2c5aaab89a9dcb"),
    (6, 51520, "7f0be21fe5f192a226a055bec8b734cef9ae8a5d59487d452ad5b9a481d01c91", 3340279315,
     "595b572896f8cbaa3e20d6a196609360ba7aafe5b141039e8590dea297f15150",
     "a3a762f25e5986da2d17204cdae13c61f53160cf40818087a68e768721752d32"),
    (7, 51520, "6cc4ed0b22c35988bde5a94fc5f6e571951d37a79f6ffb3fc7b8c268b40fadc1", 3982676336,
     "b8b9a89a858051ab503d56751e2d45b0cf4c5bbe5a94a357ea9d7cae51dd5a10",
     "9d155cf25a8319a4b5bc9d09fa5d30ac0a415b5ce61df0fd9a4c75acd252805f"),
)

_EXPECTED = {
    block: {
        "block": block,
        "bytes": nbytes,
        "sha256": sha256,
        "source_seed": seed,
        "actor_trainable_digest": actor_digest,
        "critic_trainable_digest": critic_digest,
    }
    for block, nbytes, sha256, seed, actor_digest, critic_digest in IMMUTABLE_BLOCKS
}
_EXPECTED_CONDITIONS = tuple(d3.CONDITIONS)
_PERMIT_GUARD = object()


class PreflightError(RuntimeError):
    """Raised before optimizer construction when D3 execution integrity fails."""


@dataclass(frozen=True)
class SnapshotPermit:
    """Opaque capability produced only after validation of one immutable block."""

    block: int
    sha256: str
    byte_count: int
    source_seed: int
    actor_trainable_digest: str
    critic_trainable_digest: str
    snapshot_bytes: bytes = field(repr=False)
    _guard: object = field(repr=False, compare=False, default=None)

    def assert_valid(self, block: int | None = None) -> None:
        if self._guard is not _PERMIT_GUARD:
            raise PreflightError("Untrusted D3 snapshot permit")
        if block is not None and int(block) != self.block:
            raise PreflightError(
                f"D3 snapshot permit is for block {self.block}, not block {block}"
            )
        expected = _EXPECTED.get(self.block)
        if expected is None or self.sha256 != expected["sha256"]:
            raise PreflightError("D3 snapshot permit no longer matches immutable authority")

    def public_receipt(self) -> dict:
        self.assert_valid()
        return {
            "block": self.block,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "source_seed": self.source_seed,
            "actor_trainable_digest": self.actor_trainable_digest,
            "critic_trainable_digest": self.critic_trainable_digest,
            "status": "VALIDATED_IMMUTABLE_D3_SNAPSHOT",
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_freeze() -> dict:
    try:
        freeze = json.loads(INIT_FREEZE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"Cannot read D3 INIT_FREEZE.json: {exc}") from exc
    if freeze.get("schema") != "B1-SC-D3-INIT-FREEZE-1.0.0-20260916":
        raise PreflightError("Unexpected D3 initialization freeze schema")
    if freeze.get("status") != "D3_FROZEN_INITIALIZATION_QUALIFIED_NOT_EXECUTED":
        raise PreflightError("D3 initialization is not in the qualified frozen state")
    contract = freeze.get("snapshot_contract", {})
    required_contract = {
        "one_snapshot_per_block": True,
        "seed_reconstruction_at_scientific_runtime_forbidden": True,
        "shared_byte_for_byte_across_four_conditions": True,
        "support_buffer_included": False,
        "trainable_only": True,
    }
    if any(contract.get(k) is not v for k, v in required_contract.items()):
        raise PreflightError("D3 snapshot contract changed")
    qualification = freeze.get("qualification", {})
    if qualification.get("blocks_verified") != 8 or qualification.get("two_independent_persisted_verifiers_pass") is not True:
        raise PreflightError("D3 persisted snapshots are not doubly qualified")
    rows = freeze.get("blocks")
    if not isinstance(rows, list) or len(rows) != 8:
        raise PreflightError("D3 INIT_FREEZE must contain exactly eight blocks")
    for row in rows:
        block = row.get("block")
        if block not in _EXPECTED:
            raise PreflightError(f"Unexpected block in D3 INIT_FREEZE: {block!r}")
        expected = _EXPECTED[block]
        for key in ("bytes", "sha256", "source_seed", "actor_trainable_digest", "critic_trainable_digest"):
            if row.get(key) != expected[key]:
                raise PreflightError(f"D3 INIT_FREEZE mismatch for block {block}: {key}")
        if tuple(row.get("shared_conditions", ())) != _EXPECTED_CONDITIONS:
            raise PreflightError(f"D3 condition sharing contract changed for block {block}")
    return freeze


def validate_snapshot(block: int, path: Path | str | None = None) -> SnapshotPermit:
    """Validate raw bytes first and return the only optimizer-enabling capability."""
    block = int(block)
    if block not in _EXPECTED:
        raise PreflightError(f"D3 block must be 0..7, got {block}")
    _load_freeze()
    expected = _EXPECTED[block]
    snapshot_path = Path(path) if path is not None else FROZEN_INIT_DIR / f"block-{block}.bin"
    try:
        data = snapshot_path.read_bytes()
    except Exception as exc:
        raise PreflightError(f"Cannot read frozen D3 block {block}: {exc}") from exc

    # Raw-byte checks are deliberately before decode/deserialization/model/optimizer.
    if len(data) != expected["bytes"]:
        raise PreflightError(
            f"Frozen D3 block {block} byte length mismatch: {len(data)} != {expected['bytes']}"
        )
    actual_sha = _sha256(data)
    if actual_sha != expected["sha256"]:
        raise PreflightError(
            f"Frozen D3 block {block} SHA-256 mismatch: {actual_sha} != {expected['sha256']}"
        )

    try:
        header, states = decode_snapshot(data)
    except Exception as exc:
        raise PreflightError(f"Frozen D3 block {block} decode failed after byte validation: {exc}") from exc
    if header.get("block") != block:
        raise PreflightError(f"Frozen D3 block header mismatch: {header.get('block')} != {block}")
    if header.get("source_seed") != expected["source_seed"]:
        raise PreflightError(f"Frozen D3 block {block} source-seed metadata mismatch")
    if set(states) != {"actor", "critic"} or not states["actor"] or not states["critic"]:
        raise PreflightError(f"Frozen D3 block {block} has incomplete trainable state")

    return SnapshotPermit(
        block=block,
        sha256=actual_sha,
        byte_count=len(data),
        source_seed=expected["source_seed"],
        actor_trainable_digest=expected["actor_trainable_digest"],
        critic_trainable_digest=expected["critic_trainable_digest"],
        snapshot_bytes=data,
        _guard=_PERMIT_GUARD,
    )


def _copy_state(module: torch.nn.Module, state: dict, label: str) -> None:
    params = dict(module.named_parameters())
    if set(params) != set(state):
        missing = sorted(set(params) - set(state))
        extra = sorted(set(state) - set(params))
        raise PreflightError(f"D3 {label} trainable parameter-set mismatch; missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, parameter in params.items():
            source = torch.from_numpy(state[name]).to(dtype=parameter.dtype, device=parameter.device)
            if tuple(source.shape) != tuple(parameter.shape):
                raise PreflightError(f"D3 {label}.{name} trainable shape mismatch")
            parameter.copy_(source)


def load_validated_models(
    permit: SnapshotPermit,
    mask: str,
    information_mode: str,
) -> tuple[d3.D3RoutingActor, Value]:
    """Instantiate and load models exclusively from already-validated in-memory bytes."""
    permit.assert_valid()
    actor = d3.D3RoutingActor(mask, information_mode)
    critic = Value()
    header, states = decode_snapshot(permit.snapshot_bytes)
    if header.get("block") != permit.block or header.get("source_seed") != permit.source_seed:
        raise PreflightError("Validated D3 snapshot metadata changed in memory")
    _copy_state(actor, states["actor"], "actor")
    _copy_state(critic, states["critic"], "critic")
    actor_digest = trainable_digest(actor)
    critic_digest = trainable_digest(critic)
    if actor_digest != permit.actor_trainable_digest:
        raise PreflightError("Loaded D3 actor digest differs from frozen trainable digest")
    if critic_digest != permit.critic_trainable_digest:
        raise PreflightError("Loaded D3 critic digest differs from frozen trainable digest")
    return actor, critic


def create_optimizer_after_preflight(
    permit: SnapshotPermit,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    *,
    lr: float,
    eps: float = 1e-5,
    optimizer_factory: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
) -> torch.optim.Optimizer:
    """Single optimizer construction gate shared by QA and scientific runner."""
    permit.assert_valid()
    if trainable_digest(actor) != permit.actor_trainable_digest:
        raise PreflightError("Refuse optimizer: actor is not at validated frozen initialization")
    if trainable_digest(critic) != permit.critic_trainable_digest:
        raise PreflightError("Refuse optimizer: critic is not at validated frozen initialization")
    params = list(actor.parameters()) + list(critic.parameters())
    if not params:
        raise PreflightError("Refuse optimizer: empty D3 trainable parameter set")
    return optimizer_factory(params, lr=float(lr), eps=float(eps))


def full_preflight(blocks: Iterable[int] = range(8)) -> tuple[dict, dict[int, SnapshotPermit]]:
    """Run design/anti-leakage checks and validate all requested immutable blocks."""
    design = d3.qa_tests()
    if design.get("status") != "D3_DESIGN_QA_PASS_NOT_EXECUTED":
        raise PreflightError("D3 design QA did not pass")
    _load_freeze()
    requested = tuple(int(b) for b in blocks)
    if len(requested) != len(set(requested)):
        raise PreflightError("D3 full preflight received duplicate blocks")
    permits = {block: validate_snapshot(block) for block in requested}
    report = {
        "status": "D3_EXECUTION_PREFLIGHT_PASS_NOT_AUTHORIZATION",
        "blocks_validated": len(permits),
        "receipts": [permits[b].public_receipt() for b in sorted(permits)],
        "optimizer_created": False,
        "training_step_performed": False,
        "scientific_training_performed": False,
        "scientific_evaluation_performed": False,
        "scientific_results_exist": False,
        "start_request_required_for_scientific_execution": True,
        "seed_reconstruction_used": False,
        "B1E_executed": False,
    }
    return report, permits


if __name__ == "__main__":
    report, _ = full_preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
