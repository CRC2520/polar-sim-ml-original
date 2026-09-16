"""Execution preflight for B1-SC-D3 runner readiness.

This module qualifies the runner after independent static-contract and microfit jobs.
It never creates START_REQUEST.json and never executes scientific training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from experiments.b1sc_d3_v1_0 import scientific_runner as runner

ROOT = Path(__file__).resolve().parent
ALLOWED_DIFF = {
    ".github/workflows/b1sc-d3-runner-qa.yml",
    "experiments/b1sc_d3_v1_0/RUNNER_CONTRACT.json",
    "experiments/b1sc_d3_v1_0/scientific_runner.py",
    "experiments/b1sc_d3_v1_0/runner_tests.py",
    "experiments/b1sc_d3_v1_0/runner_preflight.py",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def remote_head() -> str:
    output = git("ls-remote", "origin", f"refs/heads/{runner.RUNNER_BRANCH}")
    require(bool(output), "Runner branch is absent on origin")
    parts = output.split()
    require(len(parts) >= 2, "Malformed ls-remote response")
    return parts[0]


def verify_frozen_git_objects(contract: dict) -> dict:
    verified = {}
    for rel, expected in contract["frozen_metadata_git_blobs"].items():
        base_blob = git("rev-parse", f"{runner.FROZEN_BASE_HEAD}:{rel}")
        head_blob = git("rev-parse", f"HEAD:{rel}")
        require(base_blob == expected, f"Frozen base metadata blob mismatch: {rel}")
        require(head_blob == expected, f"Frozen metadata mutated on runner branch: {rel}")
        verified[rel] = expected

    for row in contract["immutable_snapshot_blobs"]:
        rel = f"experiments/b1sc_d3_v1_0/frozen_init/block-{int(row['block'])}.bin"
        base_blob = git("rev-parse", f"{runner.FROZEN_BASE_HEAD}:{rel}")
        head_blob = git("rev-parse", f"HEAD:{rel}")
        require(base_blob == row["git_blob"], f"Frozen base snapshot git blob mismatch: block {row['block']}")
        require(head_blob == row["git_blob"], f"Frozen snapshot git blob mutated: block {row['block']}")
    return verified


def qualify(out: Path, source_commit: str, qa_run_id: str, static_evidence: Path, microfit_evidence: Path) -> dict:
    contract = read_json(runner.RUNNER_CONTRACT)
    require(contract["status"] == "D3_RUNNER_CONTRACT_FROZEN_NOT_AUTHORIZED", "Wrong D3 runner contract status")
    require(contract["frozen_base_head"] == runner.FROZEN_BASE_HEAD, "D3 runner base changed")
    require(contract["runner_branch"] == runner.RUNNER_BRANCH, "D3 runner branch changed")
    require(contract["execution_boundary"]["execution_authorized"] is False, "Runner contract unexpectedly authorizes science")
    require(contract["execution_boundary"]["start_request_exists"] is False, "Runner contract unexpectedly includes START_REQUEST")

    require(not runner.START_REQUEST.exists(), "D3 START_REQUEST must remain absent during preflight")
    require(os.environ.get("GITHUB_REPOSITORY") == contract["repository"], "Wrong preflight repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + runner.RUNNER_BRANCH, "Wrong preflight branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Only qualification run attempt 1 is admissible")
    require(os.environ.get("GITHUB_SHA") == source_commit, "Workflow source commit mismatch")
    require(git("rev-parse", "HEAD") == source_commit, "Checkout commit mismatch")
    require(git("status", "--porcelain") == "", "Dirty checkout during D3 preflight")

    subprocess.check_call(["git", "merge-base", "--is-ancestor", runner.FROZEN_BASE_HEAD, "HEAD"])
    diff = set(filter(None, git("diff", "--name-only", runner.FROZEN_BASE_HEAD, "HEAD").splitlines()))
    require(diff == ALLOWED_DIFF, f"Unexpected runner-readiness diff: {sorted(diff)}")
    require(remote_head() == source_commit, "Remote runner branch HEAD no longer equals the qualified commit")

    frozen_metadata = verify_frozen_git_objects(contract)
    snapshots = runner.verify_all_snapshots()
    for actual, expected in zip(snapshots, contract["immutable_snapshot_blobs"], strict=True):
        require(actual["block"] == expected["block"], "Snapshot block order changed")
        require(actual["bytes"] == expected["bytes"], f"Snapshot byte length mismatch: block {actual['block']}")
        require(actual["sha256"] == expected["sha256"], f"Snapshot SHA mismatch: block {actual['block']}")

    static = read_json(static_evidence)
    microfit = read_json(microfit_evidence)
    require(static["status"] == "D3_RUNNER_STATIC_CONTRACT_PASS", "Static runner contract did not pass")
    require(microfit["status"] == "D3_RUNNER_MICROFIT_QA_PASS_NOT_SCIENCE", "Microfit QA did not pass")
    for evidence in (static, microfit):
        require(evidence["scientific_training_performed"] is False, "QA evidence claims scientific training")
        require(evidence["scientific_evaluation_performed"] is False, "QA evidence claims scientific evaluation")
        require(evidence["scientific_results_exist"] is False, "QA evidence claims scientific results")
        require(evidence["execution_authorized"] is False, "QA evidence authorizes execution")
        require(evidence["B1E_executed"] is False and evidence["final_seeds_generated"] is False, "B1-E boundary violated")

    require(len(microfit["results"]) == 4, "Expected four D3 QA microfits")
    require({x["condition"] for x in microfit["results"]} == set(contract["qa_contract"]["microfit_conditions"]), "QA condition coverage changed")
    require(all(int(x["environment_steps"]) == 512 for x in microfit["results"]), "QA microfit budget changed")
    require(len({x["initialization"]["snapshot_sha256"] for x in microfit["results"]}) == 1, "Block-0 microfits did not share identical bytes")

    payload = dict(
        schema="B1-SC-D3-RUNNER-PREFLIGHT-1.0.0-20260916",
        status="D3_RUNNER_PREFLIGHT_QUALIFIED_NOT_AUTHORIZED",
        repository=contract["repository"],
        branch=runner.RUNNER_BRANCH,
        source_commit=source_commit,
        frozen_base_head=runner.FROZEN_BASE_HEAD,
        qa_run_id=str(qa_run_id),
        qa_run_attempt=1,
        static_contract_status=static["status"],
        microfit_status=microfit["status"],
        frozen_metadata_git_blobs=frozen_metadata,
        immutable_snapshots=snapshots,
        start_request_exists=False,
        execution_authorized=False,
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        scientific_results_exist=False,
        B1E_executed=False,
        final_seeds_generated=False,
        next_required_action="SEPARATE_IMMUTABLE_START_REQUEST_AUTHORIZATION",
    )
    write_json(out / "EXECUTION_PREFLIGHT.json", payload)
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--qa-run-id", required=True)
    p.add_argument("--static-evidence", type=Path, required=True)
    p.add_argument("--microfit-evidence", type=Path, required=True)
    args = p.parse_args(argv)
    payload = qualify(
        args.out,
        args.source_commit,
        args.qa_run_id,
        args.static_evidence,
        args.microfit_evidence,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
