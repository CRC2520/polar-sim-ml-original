"""Deterministic D3 microfit QA.

This module is explicitly non-scientific.  It exercises the exact execution
ordering required by the frozen protocol:

    immutable raw bytes -> validated permit -> load exact bytes -> optimizer
    -> first backward/step

It also contains a corruption probe proving that optimizer construction remains
at zero when a single snapshot byte is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Callable

import torch
from torch import nn

from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0.execution_preflight import (
    FROZEN_INIT_DIR,
    PreflightError,
    create_optimizer_after_preflight,
    load_validated_models,
    validate_snapshot,
)
from experiments.b1sc_d3_v1_0.init_format import trainable_digest


CONDITION_MAP = {
    "GLOBAL-G0": (d3.G0, d3.GLOBAL),
    "GLOBAL-S6": (d3.S6, d3.GLOBAL),
    "LOCAL-G0": (d3.G0, d3.LOCAL),
    "LOCAL-S6": (d3.S6, d3.LOCAL),
}


def _joint_digest(actor: nn.Module, critic: nn.Module) -> str:
    return f"{trainable_digest(actor)}:{trainable_digest(critic)}"


def _fixed_microfit_batch(block: int, condition: str, n: int = 64):
    # QA namespace only; this batch is not a native environment trajectory and
    # is not retained or interpreted as scientific evidence.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(d3.seed("qa", {"purpose": "microfit", "block": block, "condition": condition}))
    z = torch.randn((n, 93), generator=generator)
    target_action = torch.tanh(0.35 * z[:, :12] + 0.15 * z[:, 31:43])
    target_value = 0.25 * z[:, :8].sum(dim=1) - 0.10 * z[:, 62:70].sum(dim=1)
    return z, target_action, target_value


def run_microfit(
    block: int = 0,
    condition: str = "LOCAL-S6",
    *,
    steps: int = 16,
    lr: float = 1e-3,
    snapshot_path: Path | str | None = None,
    optimizer_factory: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
) -> dict:
    """Run a tiny deterministic QA fit, never a scientific D3 fit."""
    if condition not in CONDITION_MAP:
        raise ValueError(f"Unknown D3 condition: {condition}")
    if steps < 1:
        raise ValueError("microfit steps must be >= 1")

    # Critical ordering: there is no model optimizer and no backward before this.
    permit = validate_snapshot(block, snapshot_path)
    mask, information_mode = CONDITION_MAP[condition]
    actor, critic = load_validated_models(permit, mask, information_mode)

    digest_after_load = _joint_digest(actor, critic)
    optimizer = create_optimizer_after_preflight(
        permit,
        actor,
        critic,
        lr=lr,
        eps=1e-5,
        optimizer_factory=optimizer_factory,
    )
    digest_after_optimizer_creation = _joint_digest(actor, critic)
    if digest_after_optimizer_creation != digest_after_load:
        raise RuntimeError("Optimizer construction mutated D3 parameters before first step")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    z, target_action, target_value = _fixed_microfit_batch(block, condition)

    losses: list[float] = []
    grad_norms: list[float] = []
    first_step_parameter_delta = False
    digest_before_first_step = _joint_digest(actor, critic)

    for step in range(steps):
        mu, log_std = actor(z)
        pred_action = mu.tanh()
        value = critic(z)
        actor_loss = (pred_action - target_action).square().mean()
        # Include log_std so every trainable actor component is exercised.
        scale_loss = 0.01 * (log_std + 0.5).square().mean()
        critic_loss = (value - target_value).square().mean()
        loss = actor_loss + scale_loss + 0.5 * critic_loss
        if not torch.isfinite(loss):
            raise RuntimeError("D3 microfit produced non-finite loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), max_norm=10.0
        )
        if not torch.isfinite(grad) or float(grad) <= 0.0:
            raise RuntimeError("D3 microfit produced non-finite or zero gradient")
        optimizer.step()

        losses.append(float(loss.detach()))
        grad_norms.append(float(grad.detach()))
        if step == 0:
            first_step_parameter_delta = _joint_digest(actor, critic) != digest_before_first_step
            if not first_step_parameter_delta:
                raise RuntimeError("D3 microfit first authorized optimizer step changed no parameter")

    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("D3 microfit loss series contains non-finite values")
    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"D3 microfit fixed-batch loss did not decrease: {losses[0]} -> {losses[-1]}"
        )

    return {
        "status": "D3_MICROFIT_QA_PASS_NOT_SCIENTIFIC",
        "block": int(block),
        "condition": condition,
        "snapshot_sha256": permit.sha256,
        "snapshot_bytes": permit.byte_count,
        "snapshot_validated_before_optimizer": True,
        "optimizer_created_after_preflight": True,
        "optimizer_created_before_preflight": False,
        "first_step_after_preflight": True,
        "first_step_parameter_delta": first_step_parameter_delta,
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "minimum_grad_norm": min(grad_norms),
        "maximum_grad_norm": max(grad_norms),
        "seed_reconstruction_used": False,
        "native_environment_used": False,
        "scientific_training_performed": False,
        "scientific_evaluation_performed": False,
        "scientific_results_exist": False,
        "B1E_executed": False,
    }


def corruption_gate_qa() -> dict:
    """Prove a one-byte mutation fails before optimizer factory invocation."""
    source = FROZEN_INIT_DIR / "block-0.bin"
    data = bytearray(source.read_bytes())
    if not data:
        raise RuntimeError("Frozen D3 block-0 is unexpectedly empty")
    data[-1] ^= 0x01

    calls = {"optimizer_factory": 0}

    def counted_factory(*args, **kwargs):
        calls["optimizer_factory"] += 1
        return torch.optim.Adam(*args, **kwargs)

    with tempfile.TemporaryDirectory(prefix="d3-microfit-corrupt-") as td:
        bad = Path(td) / "block-0-corrupt.bin"
        bad.write_bytes(data)
        try:
            run_microfit(
                block=0,
                condition="LOCAL-S6",
                steps=1,
                snapshot_path=bad,
                optimizer_factory=counted_factory,
            )
        except PreflightError as exc:
            failure = str(exc)
        else:
            raise RuntimeError("Corrupted D3 snapshot unexpectedly passed preflight")

    if calls["optimizer_factory"] != 0:
        raise RuntimeError("Optimizer factory was invoked before corrupted snapshot rejection")
    return {
        "status": "D3_CORRUPTION_GATE_QA_PASS",
        "mutation": "single_last_byte_xor_0x01",
        "preflight_rejected": True,
        "optimizer_factory_calls": 0,
        "training_steps": 0,
        "failure": failure,
        "scientific_training_performed": False,
    }


def qa_suite(block: int = 0, condition: str = "LOCAL-S6") -> dict:
    corrupt = corruption_gate_qa()
    microfit = run_microfit(block=block, condition=condition)
    return {
        "status": "D3_MICROFIT_EXECUTION_GATE_QA_PASS",
        "corruption_gate": corrupt,
        "microfit": microfit,
        "scientific_training_performed": False,
        "scientific_evaluation_performed": False,
        "scientific_results_exist": False,
        "B1E_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--condition", choices=sorted(CONDITION_MAP), default="LOCAL-S6")
    args = parser.parse_args()
    print(json.dumps(qa_suite(args.block, args.condition), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
