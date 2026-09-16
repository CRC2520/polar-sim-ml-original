"""Qualification QA for the B1-SC-D4 scientific runner candidate.

No scientific namespace, endpoint campaign, causal campaign, or B1-E action is
executed here. Runtime QA uses three 1024-step microfits straddling the frozen
786432-step switch solely to verify runner mechanics.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from experiments.b1sc_d4_v1_0 import implementation as d4
from experiments.b1sc_d4_v1_0 import scientific_runner as runner

ROOT = Path(__file__).resolve().parent


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def static_rules() -> dict:
    require(not runner.START_REQUEST.exists(), "D4 START_REQUEST must be absent during runner qualification")
    require(runner.HP == runner.EXPECTED_HP and runner.N_ENVS == 8, "D4 runner changed historical PPO profile")
    require(d4.validate_design()["status"] == "D4_IMPLEMENTATION_CONTRACT_READY_FOR_QA", "D4 design contract failed")

    source = Path(runner.__file__).read_text(encoding="utf-8")
    for literal in ("generate_snapshots", "write_snapshot(", "encode_snapshot(", "D3 blocks 6", "block-6 special"):
        require(literal not in source, f"Forbidden runner source literal: {literal}")
    require("init_format.load_trainable_into" in source, "Frozen-byte loader absent from D4 runner")
    require("guarded_optimizer_step" in inspect.getsource(runner._train), "Guarded optimizer step absent")
    require("execution_guard()" in inspect.getsource(runner._train), "Scientific execution guard absent")

    snapshots = runner.verify_all_snapshots()
    require(len(snapshots) == 12, "Exactly twelve D4 frozen snapshots are required")

    initialization_matrix = {}
    for block in range(d4.BLOCKS):
        rows = []
        for condition in d4.CONDITIONS:
            prepared = runner.validate_and_load_frozen(condition, block)
            att = prepared.attestation
            require([x["event"] for x in prepared.audit] == ["validation_started", "snapshot_validated"],
                    "D4 validation created optimizer state")
            rows.append(dict(condition=condition, snapshot_sha256=att.snapshot_sha256,
                             actor_trainable_digest=att.actor_trainable_digest,
                             critic_trainable_digest=att.critic_trainable_digest))
        require(len({x["snapshot_sha256"] for x in rows}) == 1, f"D4 block {block} bytes differ by condition")
        require(len({x["actor_trainable_digest"] for x in rows}) == 1, f"D4 block {block} actor init differs")
        require(len({x["critic_trainable_digest"] for x in rows}) == 1, f"D4 block {block} critic init differs")
        initialization_matrix[str(block)] = rows

    reg = runner._registry()
    paired_keys = ("initial_snapshot_block", "initial_snapshot_seed", "policy_sampling_seed",
                   "training_environment_seed_root", "diagnostic_panel_seed",
                   "endpoint_panel_seed", "causal_panel_seed")
    for block in range(d4.BLOCKS):
        rows = [x for x in reg if int(x["block"]) == block]
        require(len(rows) == 3, f"D4 block {block} does not have three treatments")
        for key in paired_keys:
            require(len({x[key] for x in rows}) == 1, f"D4 seed pairing failed: block {block}, {key}")
        require(all("d3_block" not in x for x in rows), f"D3 block identity leaked into D4 block {block}")

    require(d4.training_lambda("LOCAL-S6-LATE", d4.SWITCH_STEPS-512) == 0.0, "D4 LATE pre-switch changed")
    require(d4.training_lambda("LOCAL-S6-LATE", d4.SWITCH_STEPS) == 1.0, "D4 LATE post-switch changed")
    require(d4.diagnostic_lambda("LOCAL-S6-LATE", d4.SWITCH_STEPS) == 0.0, "786432 diagnostic is no longer pre-switch")

    return dict(status="D4_RUNNER_STATIC_CONTRACT_PASS", frozen_base_head=runner.FROZEN_BASE_HEAD,
                snapshots=snapshots, initialization_matrix=initialization_matrix,
                registry_entries=len(reg), ppo_profile=runner.HP,
                switch_after_completed_native_steps=d4.SWITCH_STEPS,
                scientific_training_performed=False, scientific_evaluation_performed=False,
                scientific_results_exist=False, execution_authorized=False,
                B1E_executed=False, final_seeds_generated=False)


def runtime_microfits(out: Path) -> dict:
    require(not runner.START_REQUEST.exists(), "D4 START_REQUEST must be absent during QA")
    runtime_lock = runner.runtime()
    contract = json.loads(runner.RUNNER_CONTRACT.read_text(encoding="utf-8"))
    expected = contract["qa_contract"]["expected_route_schedules"]
    results = []
    block = int(contract["qa_contract"]["microfit_block"])
    budget = int(contract["qa_contract"]["microfit_native_steps_per_condition"])
    start = int(contract["qa_contract"]["microfit_start_completed_steps"])
    for condition in d4.CONDITIONS:
        fit_out = out / condition
        result = runner.qa_microfit(condition, block, fit_out, budget=budget, start_completed_steps=start)
        require(result["status"] == "D4_QA_MICROFIT_COMPLETE", f"D4 QA microfit failed: {condition}")
        require(result["split"] == "qa" and result["scientific_evidence"] is False, "D4 QA mislabeled as science")
        require(result["scientific_training_performed"] is False, "D4 QA marked scientific training")
        schedule = [float(x["route_lambda"]) for x in result["route_schedule"]]
        require(schedule == [float(x) for x in expected[condition]], f"D4 QA switch schedule mismatch: {condition}")
        events = [x["event"] for x in result["initialization_audit"]]
        require(events[:3] == ["validation_started", "snapshot_validated", "optimizer_created"],
                "D4 initialization audit order changed")
        require("optimizer_step" in events and events.index("snapshot_validated") < events.index("optimizer_created") < events.index("optimizer_step"),
                "D4 first optimizer step ordering violated")
        results.append(result)

    require(len({x["initialization"]["snapshot_sha256"] for x in results}) == 1,
            "D4 block-0 QA treatments did not consume identical bytes")
    for key in ("environment_seed_root", "policy_sampling_seed", "minibatch_seed"):
        require(len({x["seed_plan"][key] for x in results}) == 1, f"D4 QA paired seed mismatch: {key}")

    payload = dict(status="D4_RUNNER_MICROFIT_QA_PASS_NOT_SCIENCE", runtime=runtime_lock,
                   block=block, start_completed_steps=start, budget=budget,
                   conditions=list(d4.CONDITIONS), results=results,
                   scientific_training_performed=False, scientific_evaluation_performed=False,
                   scientific_results_exist=False, execution_authorized=False,
                   B1E_executed=False, final_seeds_generated=False)
    write_json(out / "MICROFIT_QA.json", payload)
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("phase", choices=("rules", "runtime"))
    p.add_argument("--out", type=Path, default=Path("/tmp/d4-runner-qa")); args = p.parse_args(argv)
    if args.phase == "rules":
        payload = static_rules(); write_json(args.out / "STATIC_CONTRACT.json", payload)
    else:
        payload = runtime_microfits(args.out)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
