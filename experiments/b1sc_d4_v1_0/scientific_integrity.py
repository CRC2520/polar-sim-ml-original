"""Integrity-only aggregation for a completed B1-SC-D4 36-fit execution.

Validates completeness, frozen initialization lineage and the prospectively
frozen temporal-route schedule. It does not adjudicate any D4 hypothesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ("LOCAL-S6-OFF", "LOCAL-S6-LATE", "LOCAL-S6-ALWAYS")
BLOCKS = 12
NATIVE_STEPS = 1_048_576
ROLLOUT_STEPS = 512
SWITCH_STEPS = 786_432
CHECKPOINTS = (262_144, 524_288, 786_432, 851_968, 917_504, 983_040, 1_048_576)
QUALIFIED_RUNNER_COMMIT = "b475a41b583b50a516bebce16d030518c8dd02f2"
EXPECTED_SNAPSHOTS = {
    0: "cb0431aff2d54aad63acced23524ea0092dde789b1511cef0c78aca1e47dc6dc",
    1: "82be59782d9d817a18959dd6950eed6c16a8981fd362eaa7e1c11956c3349af4",
    2: "b59beb5e1e5f241d93e4a5cc4fcdd26c036989df11238eb67a2f7040e4a18871",
    3: "89693e76e375b708d15505146b1350b4a9fd664aa2999afc0835f11f49eccc9c",
    4: "858499a679cc143632fab388a23ec036479fc8af7fdf9df636aeec56ce2433ad",
    5: "c059bf7ec2169d7ceb1f834808298776a4176ed5374fceef611b428f578a3d59",
    6: "2491bb4bd1059d37a0d3e155e5d11f9caebd0f03fcded878b5d14ee1cad3d130",
    7: "15b493900d93dcef844bffe622b50c961a67762770698983bd109c798f394f88",
    8: "8771fa86fde3edf7a8a4c1fac565fc563ce74f01ca2cd54f1f6f6fd979010514",
    9: "c193a05da6cfa22fb496e230662e5b29cc55c78f25ace68c881b0f130a62b829",
    10: "934905e39eacf74f4e4d3504558d0167afe97f18183ab3b617427a6c46be893a",
    11: "7b5c72dbd38fa7e6e7c6e3168d9799db62dae11678887c7464ca6d7b8563c42a",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_lambda(condition: str, completed_steps: int) -> float:
    if condition == "LOCAL-S6-OFF":
        return 0.0
    if condition == "LOCAL-S6-ALWAYS":
        return 1.0
    return 0.0 if int(completed_steps) < SWITCH_STEPS else 1.0


def expected_diagnostic_lambda(condition: str, checkpoint: int) -> float:
    if condition == "LOCAL-S6-OFF":
        return 0.0
    if condition == "LOCAL-S6-ALWAYS":
        return 1.0
    return 0.0 if int(checkpoint) <= SWITCH_STEPS else 1.0


def validate_schedule(condition: str, schedule: list[dict]) -> None:
    require(len(schedule) == NATIVE_STEPS // ROLLOUT_STEPS,
            f"Wrong D4 route schedule length: {condition}")
    for i, row in enumerate(schedule):
        completed = i * ROLLOUT_STEPS
        require(int(row["completed_native_steps_before_rollout"]) == completed,
                f"D4 route schedule step mismatch: {condition}, rollout {i}")
        require(float(row["route_lambda"]) == expected_lambda(condition, completed),
                f"D4 route schedule lambda mismatch: {condition}, step {completed}")


def validate(root: Path) -> dict:
    result_paths = list(root.rglob("RUN_RESULT.json"))
    require(len(result_paths) == 36, f"Expected 36 RUN_RESULT.json files, found {len(result_paths)}")
    seen = set()
    rows = []
    by_block = {b: [] for b in range(BLOCKS)}
    execution_commits = set()

    for path in result_paths:
        r = read_json(path)
        key = (r["condition"], int(r["block"]))
        require(key not in seen, f"Duplicate D4 fit: {key}")
        seen.add(key)
        require(key[0] in CONDITIONS and 0 <= key[1] < BLOCKS, f"Unexpected D4 fit: {key}")
        require(r["status"] == "D4_SCIENTIFIC_FIT_COMPLETE", f"Incomplete D4 fit: {key}")
        require(r["split"] == "training" and r["scientific_evidence"] is True,
                f"Non-scientific D4 fit artifact: {key}")
        require(r["scientific_training_performed"] is True, f"Scientific training flag missing: {key}")
        require(int(r["environment_steps"]) == NATIVE_STEPS, f"Wrong D4 budget: {key}")
        require(int(r["start_completed_steps"]) == 0 and
                int(r["final_completed_steps"]) == NATIVE_STEPS, f"Wrong D4 training interval: {key}")
        require(r["B1E_executed"] is False and r["final_seeds_generated"] is False,
                f"B1-E boundary violation: {key}")

        snap = r["initialization"]["snapshot_sha256"]
        require(snap == EXPECTED_SNAPSHOTS[key[1]], f"Frozen D4 init mismatch: {key}")
        by_block[key[1]].append(snap)

        auth = r["authorization"]
        require(auth["qualified_runner_commit"] == QUALIFIED_RUNNER_COMMIT,
                f"D4 runner lineage mismatch: {key}")
        require(int(auth["scientific_fits"]) == 36 and int(auth["blocks"]) == BLOCKS,
                f"D4 authorization scope mismatch: {key}")
        require(auth["conditions"] == list(CONDITIONS), f"D4 authorization conditions mismatch: {key}")
        require(int(auth["switch_after_completed_native_steps"]) == SWITCH_STEPS,
                f"D4 authorization switch mismatch: {key}")
        execution_commits.add(auth["execution_commit"])

        validate_schedule(key[0], r["route_schedule"])

        fit_dir = path.parent
        require((fit_dir / "ENDPOINT_CONFIGURED.json").is_file(), f"Missing configured endpoint: {key}")
        endpoint = read_json(fit_dir / "ENDPOINT_CONFIGURED.json")
        expected_endpoint = 0.0 if key[0] == "LOCAL-S6-OFF" else 1.0
        require(endpoint["status"] == "D4_ENDPOINT_CONFIGURED_COMPLETE" and
                float(endpoint["route_lambda"]) == expected_endpoint,
                f"D4 endpoint route state mismatch: {key}")
        require(int(endpoint["summary"]["episodes"]) == 64, f"D4 endpoint episode count mismatch: {key}")

        require((fit_dir / "NO_ACTION_WITNESS.json").is_file(), f"Missing D4 witness: {key}")
        witness = read_json(fit_dir / "NO_ACTION_WITNESS.json")
        require(witness["status"] == "D4_NO_ACTION_WITNESS_COMPLETE" and
                int(witness["summary"]["episodes"]) == 64, f"D4 witness mismatch: {key}")
        require((fit_dir / "TRAINING_RESOURCES.json").is_file(), f"Missing D4 training resources: {key}")
        require((fit_dir / "model-final.pt").is_file(), f"Missing D4 final model: {key}")

        checkpoints = sorted(fit_dir.glob("checkpoint-*.pt"))
        require(len(checkpoints) == len(CHECKPOINTS), f"Missing D4 checkpoint models: {key}")
        diagnostics = sorted(fit_dir.glob("DIAGNOSTIC_*.json"))
        require(len(diagnostics) == len(CHECKPOINTS), f"Missing D4 diagnostics: {key}")
        for step in CHECKPOINTS:
            dp = fit_dir / f"DIAGNOSTIC_{step}.json"
            require(dp.is_file(), f"Missing D4 diagnostic {step}: {key}")
            d = read_json(dp)
            require(d["status"] == "D4_DIAGNOSTIC_COMPLETE" and int(d["native_steps"]) == step,
                    f"Malformed D4 diagnostic {step}: {key}")
            require(float(d["route_lambda"]) == expected_diagnostic_lambda(key[0], step),
                    f"D4 diagnostic lambda mismatch {step}: {key}")
            require(int(d["summary"]["episodes"]) == 32, f"D4 diagnostic episode count mismatch {step}: {key}")

        if key[0] in ("LOCAL-S6-LATE", "LOCAL-S6-ALWAYS"):
            require((fit_dir / "ENDPOINT_PERMANENT_LESION.json").is_file(),
                    f"Missing D4 permanent-lesion endpoint: {key}")
            lesion = read_json(fit_dir / "ENDPOINT_PERMANENT_LESION.json")
            require(lesion["status"] == "D4_ENDPOINT_PERMANENT_LESION_COMPLETE" and
                    int(lesion["summary"]["episodes"]) == 64,
                    f"D4 lesion endpoint mismatch: {key}")
            require((fit_dir / "CAUSAL.json").is_file(), f"Missing D4 causal panel: {key}")
            causal = read_json(fit_dir / "CAUSAL.json")
            require(causal["status"] == "D4_CAUSAL_PANEL_COMPLETE", f"D4 causal status mismatch: {key}")
            require(set(causal["regimes"]) == {"intact", "sham", "inactive_edge_control", "permanent_all_active_lesion"},
                    f"D4 causal regimes mismatch: {key}")
            for mode, payload in causal["regimes"].items():
                require(int(payload["summary"]["episodes"]) == 64,
                        f"D4 causal episode count mismatch {mode}: {key}")
        else:
            require(not (fit_dir / "ENDPOINT_PERMANENT_LESION.json").exists(),
                    f"Unexpected D4 lesion endpoint for OFF: {key}")
            require(not (fit_dir / "CAUSAL.json").exists(), f"Unexpected D4 causal panel for OFF: {key}")

        rows.append({"condition": key[0], "block": key[1], "snapshot_sha256": snap})

    require(seen == {(c, b) for c in CONDITIONS for b in range(BLOCKS)},
            "D4 condition/block matrix incomplete")
    for block, snaps in by_block.items():
        require(len(snaps) == 3 and len(set(snaps)) == 1,
                f"D4 block {block} did not share identical initialization bytes")
    require(len(execution_commits) == 1, "D4 fits do not share one immutable execution commit")

    return {
        "schema": "B1-SC-D4-SCIENTIFIC-INTEGRITY-1.0.0-20260916",
        "status": "D4_SCIENTIFIC_EXECUTION_COMPLETE_INTEGRITY_PASS_NOT_ADJUDICATED",
        "fits_verified": 36,
        "blocks_verified": 12,
        "conditions_verified": list(CONDITIONS),
        "native_steps_per_fit": NATIVE_STEPS,
        "total_native_steps": 36 * NATIVE_STEPS,
        "switch_after_completed_native_steps": SWITCH_STEPS,
        "checkpoint_786432_is_pre_switch": True,
        "checkpoints_verified": list(CHECKPOINTS),
        "late_causal_panels_verified": 12,
        "always_causal_panels_verified": 12,
        "initialization_pairing": "PASS",
        "route_schedule_integrity": "PASS",
        "qualified_runner_commit": QUALIFIED_RUNNER_COMMIT,
        "execution_commit": next(iter(execution_commits)),
        "B1E_executed": False,
        "final_seeds_generated": False,
        "H_D4_LATE_TOTAL": "NOT_EVALUATED",
        "H_D4_EARLY_PREPARATION": "NOT_EVALUATED",
        "H_D4_LATE_SCAFFOLD": "NOT_EVALUATED",
        "H_D4_ONLINE_LATE": "NOT_EVALUATED",
        "H_D4_ONLINE_ALWAYS": "NOT_EVALUATED",
        "adjudication_performed": False,
        "fits": sorted(rows, key=lambda x: (x["condition"], x["block"])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    payload = validate(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
