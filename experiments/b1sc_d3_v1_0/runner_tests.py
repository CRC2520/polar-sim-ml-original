"""Qualification tests for the B1-SC-D3 scientific runner.

The rules phase performs no native-environment training. The runtime phase performs
bounded QA-only microfits and never produces scientific evidence.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0 import scientific_runner as runner

ROOT = Path(__file__).resolve().parent

BANNED_RUNNER_LITERALS = (
    "np.arange(",
    "np.zeros(",
    "_synthetic",
    "Synthetic data",
    "generate_snapshots",
    "encode_snapshot(",
    "write_snapshot(",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def static_rules() -> dict:
    require(not runner.START_REQUEST.exists(), "D3 START_REQUEST must be absent during runner qualification")
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for literal in BANNED_RUNNER_LITERALS:
        require(literal not in source, f"Forbidden runner literal: {literal}")

    require(runner.HP == runner.EXPECTED_HP, "PPO hyperparameters differ from the preserved D2 profile")
    require(runner.N_ENVS == 8, "PPO parallelism changed")
    require("load_trainable_into" in inspect.getsource(runner.validate_and_load_frozen), "Frozen-byte loader absent")
    require("torch.optim.Adam" in inspect.getsource(runner.create_optimizer), "Expected Adam optimizer absent")
    require("guarded_optimizer_step" in inspect.getsource(runner._train), "Guarded optimizer step absent")
    require("START_REQUEST.exists()" in inspect.getsource(runner.execution_guard), "Scientific authorization guard absent")

    snapshots = runner.verify_all_snapshots()
    require(len(snapshots) == 8, "Exactly eight frozen snapshots are required")

    by_block: dict[int, list[dict]] = {}
    for block in range(d3.BLOCKS):
        attestations = []
        for condition in d3.CONDITIONS:
            prepared = runner.validate_and_load_frozen(condition, block)
            att = prepared.attestation
            require(
                [x["event"] for x in prepared.audit] == ["validation_started", "snapshot_validated"],
                "Loading unexpectedly created optimizer state",
            )
            attestations.append(
                dict(
                    condition=condition,
                    snapshot_sha256=att.snapshot_sha256,
                    actor_trainable_digest=att.actor_trainable_digest,
                    critic_trainable_digest=att.critic_trainable_digest,
                    mask=att.mask,
                    information_mode=att.information_mode,
                )
            )
        require(len({x["snapshot_sha256"] for x in attestations}) == 1, f"Block {block} snapshot bytes differ by condition")
        require(len({x["actor_trainable_digest"] for x in attestations}) == 1, f"Block {block} actor trainables differ by condition")
        require(len({x["critic_trainable_digest"] for x in attestations}) == 1, f"Block {block} critic trainables differ by condition")
        by_block[block] = attestations

    rows = runner._registry()
    for block in range(d3.BLOCKS):
        paired = [x for x in rows if int(x["block"]) == block]
        for key in (
            "policy_sampling_seed",
            "training_environment_seed_root",
            "diagnostic_panel_seed",
            "endpoint_panel_seed",
            "causal_panel_seed",
            "initial_snapshot_block",
        ):
            require(len({x[key] for x in paired}) == 1, f"Registry pairing failed: block {block}, {key}")

    return dict(
        status="D3_RUNNER_STATIC_CONTRACT_PASS",
        frozen_base_head=runner.FROZEN_BASE_HEAD,
        snapshots=snapshots,
        initialization_matrix=by_block,
        ppo_profile=runner.HP,
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        scientific_results_exist=False,
        execution_authorized=False,
        B1E_executed=False,
        final_seeds_generated=False,
    )


def runtime_microfits(out: Path) -> dict:
    require(not runner.START_REQUEST.exists(), "D3 START_REQUEST must be absent during QA")
    runtime_lock = runner.runtime()
    runner.verify_all_snapshots()
    results = []
    block = 0
    for condition in d3.CONDITIONS:
        fit_out = out / condition
        result = runner.qa_microfit(condition, block, fit_out, purpose="qa_runner_microfit", budget=512)
        require(result["status"] == "D3_QA_MICROFIT_COMPLETE", f"QA microfit failed: {condition}")
        require(result["split"] == "qa" and result["scientific_evidence"] is False, "QA mislabeled as science")
        require(result["scientific_training_performed"] is False, "QA marked scientific training")
        events = [x["event"] for x in result["initialization_audit"]]
        require(events[:3] == ["validation_started", "snapshot_validated", "optimizer_created"], "Initialization order changed")
        require("optimizer_step" in events, "QA microfit never executed a guarded optimizer step")
        require(events.index("snapshot_validated") < events.index("optimizer_created") < events.index("optimizer_step"), "Frozen-init ordering violated")
        results.append(result)

    require(len({x["initialization"]["snapshot_sha256"] for x in results}) == 1, "Four block-0 QA conditions did not consume identical bytes")
    require(len({x["seed_plan"]["environment_seed_root"] for x in results}) == 1, "QA environment seeds are not paired")
    require(len({x["seed_plan"]["policy_sampling_seed"] for x in results}) == 1, "QA policy seeds are not paired")
    require(len({x["seed_plan"]["minibatch_seed"] for x in results}) == 1, "QA minibatch seeds are not paired")

    payload = dict(
        status="D3_RUNNER_MICROFIT_QA_PASS_NOT_SCIENCE",
        runtime=runtime_lock,
        block=block,
        conditions=list(d3.CONDITIONS),
        results=results,
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        scientific_results_exist=False,
        execution_authorized=False,
        B1E_executed=False,
        final_seeds_generated=False,
    )
    write_json(out / "MICROFIT_QA.json", payload)
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("rules", "runtime"))
    p.add_argument("--out", type=Path, default=Path("/tmp/d3-runner-qa"))
    args = p.parse_args(argv)
    if args.phase == "rules":
        payload = static_rules()
        write_json(args.out / "STATIC_CONTRACT.json", payload)
    else:
        payload = runtime_microfits(args.out)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
