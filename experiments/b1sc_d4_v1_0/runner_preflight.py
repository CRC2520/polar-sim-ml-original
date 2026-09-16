"""Qualification preflight for the B1-SC-D4 scientific runner candidate.

This preflight consumes static and microfit QA evidence, verifies the entire D4
frozen design/initialization lineage, and never creates START_REQUEST or runs
scientific training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from experiments.b1sc_d4_v1_0 import scientific_runner as runner

ALLOWED_DIFF = {
    ".github/workflows/b1sc-d4-runner-qa.yml",
    "experiments/b1sc_d4_v1_0/RUNNER_CONTRACT.json",
    "experiments/b1sc_d4_v1_0/scientific_runner.py",
    "experiments/b1sc_d4_v1_0/runner_tests.py",
    "experiments/b1sc_d4_v1_0/runner_preflight.py",
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
    out = git("ls-remote", "origin", f"refs/heads/{runner.RUNNER_BRANCH}")
    require(bool(out), "D4 runner branch absent on origin")
    return out.split()[0]


def verify_frozen_lineage(contract: dict) -> dict:
    verified = {}
    for rel in contract["frozen_paths"]:
        base_blob = git("rev-parse", f"{runner.FROZEN_BASE_HEAD}:{rel}")
        head_blob = git("rev-parse", f"HEAD:{rel}")
        require(base_blob == head_blob, f"Frozen D4 path mutated on runner branch: {rel}")
        verified[rel] = base_blob
    for row in contract["immutable_snapshots"]:
        block = int(row["block"])
        rel = f"experiments/b1sc_d4_v1_0/frozen_init/block-{block}.bin"
        base_blob = git("rev-parse", f"{runner.FROZEN_BASE_HEAD}:{rel}")
        head_blob = git("rev-parse", f"HEAD:{rel}")
        require(base_blob == head_blob, f"Frozen D4 snapshot Git blob mutated: block {block}")
    return verified


def qualify(out: Path, source_commit: str, qa_run_id: str,
            static_evidence: Path, microfit_evidence: Path) -> dict:
    contract = read_json(runner.RUNNER_CONTRACT)
    require(contract["status"] == "D4_RUNNER_CONTRACT_FROZEN_NOT_AUTHORIZED", "Wrong D4 runner contract status")
    require(contract["frozen_base_head"] == runner.FROZEN_BASE_HEAD, "D4 runner frozen base changed")
    require(contract["runner_branch"] == runner.RUNNER_BRANCH, "D4 runner branch changed")
    require(contract["execution_boundary"]["execution_authorized"] is False, "D4 runner contract authorizes science")
    require(contract["execution_boundary"]["start_request_exists"] is False, "D4 runner contract contains START")
    require(not runner.START_REQUEST.exists(), "D4 START_REQUEST must remain absent during qualification")

    require(os.environ.get("GITHUB_REPOSITORY") == contract["repository"], "Wrong D4 qualification repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + runner.RUNNER_BRANCH, "Wrong D4 qualification branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Only D4 qualification attempt 1 is admissible")
    require(os.environ.get("GITHUB_SHA") == source_commit and git("rev-parse", "HEAD") == source_commit,
            "D4 qualification checkout mismatch")
    require(git("status", "--porcelain") == "", "Dirty D4 runner qualification checkout")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", runner.FROZEN_BASE_HEAD, "HEAD"])
    diff = set(filter(None, git("diff", "--name-only", runner.FROZEN_BASE_HEAD, "HEAD").splitlines()))
    require(diff == ALLOWED_DIFF, f"Unexpected D4 runner-readiness diff: {sorted(diff)}")
    require(remote_head() == source_commit, "Remote D4 runner branch HEAD changed during qualification")

    frozen_paths = verify_frozen_lineage(contract)
    snapshots = runner.verify_all_snapshots()
    require(len(snapshots) == 12, "D4 qualification did not verify twelve snapshots")
    for actual, expected in zip(snapshots, contract["immutable_snapshots"], strict=True):
        require(actual["block"] == expected["block"], "D4 snapshot ordering changed")
        require(actual["bytes"] == expected["bytes"], f"D4 snapshot bytes changed: block {actual['block']}")
        require(actual["sha256"] == expected["sha256"], f"D4 snapshot SHA changed: block {actual['block']}")

    static = read_json(static_evidence); microfit = read_json(microfit_evidence)
    require(static["status"] == "D4_RUNNER_STATIC_CONTRACT_PASS", "D4 static contract did not pass")
    require(microfit["status"] == "D4_RUNNER_MICROFIT_QA_PASS_NOT_SCIENCE", "D4 microfit QA did not pass")
    for evidence in (static, microfit):
        require(evidence["scientific_training_performed"] is False, "D4 QA evidence claims scientific training")
        require(evidence["scientific_evaluation_performed"] is False, "D4 QA evidence claims scientific evaluation")
        require(evidence["scientific_results_exist"] is False, "D4 QA evidence claims scientific results")
        require(evidence["execution_authorized"] is False, "D4 QA evidence authorizes execution")
        require(evidence["B1E_executed"] is False and evidence["final_seeds_generated"] is False,
                "D4 QA violates B1-E/final seed boundary")

    qc = contract["qa_contract"]
    require(len(microfit["results"]) == 3, "D4 qualification expected three QA microfits")
    require({x["condition"] for x in microfit["results"]} == set(qc["microfit_conditions"]),
            "D4 QA condition coverage changed")
    require(all(int(x["environment_steps"]) == int(qc["microfit_native_steps_per_condition"])
                for x in microfit["results"]), "D4 QA microfit budget changed")
    require(len({x["initialization"]["snapshot_sha256"] for x in microfit["results"]}) == 1,
            "D4 block-0 QA microfits did not share identical bytes")
    for row in microfit["results"]:
        schedule = [float(x["route_lambda"]) for x in row["route_schedule"]]
        require(schedule == [float(x) for x in qc["expected_route_schedules"][row["condition"]]],
                f"D4 qualified schedule mismatch: {row['condition']}")

    payload = dict(
        schema="B1-SC-D4-RUNNER-PREFLIGHT-1.0.0-20260916",
        status="D4_RUNNER_PREFLIGHT_QUALIFIED_NOT_AUTHORIZED",
        repository=contract["repository"], branch=runner.RUNNER_BRANCH,
        source_commit=source_commit, frozen_base_head=runner.FROZEN_BASE_HEAD,
        qa_run_id=str(qa_run_id), qa_run_attempt=1,
        static_contract_status=static["status"], microfit_status=microfit["status"],
        frozen_path_git_blobs=frozen_paths, immutable_snapshots=snapshots,
        initialization_authority=contract["initialization_authority"],
        start_request_exists=False, execution_authorized=False,
        scientific_training_performed=False, scientific_evaluation_performed=False,
        scientific_results_exist=False, B1E_executed=False, final_seeds_generated=False,
        next_required_action="SEPARATE_IMMUTABLE_START_REQUEST_AND_SCIENTIFIC_EXECUTION_WORKFLOW",
    )
    write_json(out / "EXECUTION_PREFLIGHT.json", payload)
    return payload


def main(argv=None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-commit",required=True); p.add_argument("--qa-run-id",required=True)
    p.add_argument("--static-evidence",type=Path,required=True); p.add_argument("--microfit-evidence",type=Path,required=True)
    a=p.parse_args(argv)
    payload=qualify(a.out,a.source_commit,a.qa_run_id,a.static_evidence,a.microfit_evidence)
    print(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
