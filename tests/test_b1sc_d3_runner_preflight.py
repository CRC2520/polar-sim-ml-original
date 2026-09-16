from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch

from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0.execution_preflight import (
    FROZEN_INIT_DIR,
    IMMUTABLE_BLOCKS,
    PreflightError,
    SnapshotPermit,
    create_optimizer_after_preflight,
    full_preflight,
    load_validated_models,
    validate_snapshot,
)
from experiments.b1sc_d3_v1_0.microfit_qa import corruption_gate_qa, run_microfit


def test_all_eight_immutable_snapshots_pass_full_preflight_without_optimizer():
    report, permits = full_preflight(range(8))
    assert report["status"] == "D3_EXECUTION_PREFLIGHT_PASS_NOT_AUTHORIZATION"
    assert report["blocks_validated"] == 8
    assert report["optimizer_created"] is False
    assert report["training_step_performed"] is False
    assert set(permits) == set(range(8))
    expected = {block: (nbytes, sha) for block, nbytes, sha, *_ in IMMUTABLE_BLOCKS}
    for block, permit in permits.items():
        assert permit.byte_count == expected[block][0]
        assert permit.sha256 == expected[block][1]
        permit.assert_valid(block)


def test_one_byte_corruption_is_rejected_before_optimizer_factory():
    result = corruption_gate_qa()
    assert result["preflight_rejected"] is True
    assert result["optimizer_factory_calls"] == 0
    assert result["training_steps"] == 0


def test_wrong_length_is_rejected_before_decode_or_optimizer():
    source = (FROZEN_INIT_DIR / "block-0.bin").read_bytes()
    with tempfile.TemporaryDirectory(prefix="d3-preflight-short-") as td:
        bad = Path(td) / "block-0.bin"
        bad.write_bytes(source[:-1])
        with pytest.raises(PreflightError, match="byte length mismatch"):
            validate_snapshot(0, bad)


def test_forged_permit_cannot_open_optimizer_gate():
    genuine = validate_snapshot(0)
    forged = SnapshotPermit(
        block=genuine.block,
        sha256=genuine.sha256,
        byte_count=genuine.byte_count,
        source_seed=genuine.source_seed,
        actor_trainable_digest=genuine.actor_trainable_digest,
        critic_trainable_digest=genuine.critic_trainable_digest,
        snapshot_bytes=genuine.snapshot_bytes,
    )
    actor = d3.D3RoutingActor(d3.S6, d3.LOCAL)
    from b1s.execution.core import Value
    critic = Value()
    with pytest.raises(PreflightError, match="Untrusted"):
        create_optimizer_after_preflight(forged, actor, critic, lr=1e-3)


def test_same_validated_block_bytes_load_all_four_conditions():
    permit = validate_snapshot(0)
    for condition, (mask, mode) in {
        "GLOBAL-G0": (d3.G0, d3.GLOBAL),
        "GLOBAL-S6": (d3.S6, d3.GLOBAL),
        "LOCAL-G0": (d3.G0, d3.LOCAL),
        "LOCAL-S6": (d3.S6, d3.LOCAL),
    }.items():
        actor, critic = load_validated_models(permit, mask, mode)
        assert actor.mask == mask, condition
        assert actor.information_mode == mode, condition
        # Optimizer construction is intentionally omitted in this pairing test.
        assert sum(p.numel() for p in actor.parameters()) > 0
        assert sum(p.numel() for p in critic.parameters()) > 0


def test_valid_microfit_only_steps_after_snapshot_validation():
    calls = {"optimizer": 0}

    def counted_factory(*args, **kwargs):
        calls["optimizer"] += 1
        return torch.optim.Adam(*args, **kwargs)

    result = run_microfit(
        block=0,
        condition="LOCAL-S6",
        steps=8,
        optimizer_factory=counted_factory,
    )
    assert calls["optimizer"] == 1
    assert result["snapshot_validated_before_optimizer"] is True
    assert result["optimizer_created_before_preflight"] is False
    assert result["first_step_after_preflight"] is True
    assert result["first_step_parameter_delta"] is True
    assert result["loss_decreased"] is True
    assert result["scientific_training_performed"] is False


def test_scientific_runner_has_no_skip_preflight_switch_and_no_direct_optimizer_constructor():
    source = (Path(__file__).resolve().parents[1] / "experiments" / "b1sc_d3_v1_0" / "runner.py").read_text(encoding="utf-8")
    assert "skip-preflight" not in source.lower()
    assert "--skip" not in source.lower()
    assert "torch.optim." not in source
    assert "create_optimizer_after_preflight(" in source
    assert "full_preflight(range(d3.BLOCKS))" in source


def test_start_request_is_not_present_in_runner_build():
    # This task installs a runner/preflight; it must not self-authorize science.
    path = Path(__file__).resolve().parents[1] / "experiments" / "b1sc_d3_v1_0" / "START_REQUEST.json"
    assert not path.exists()
