"""B1-SC-D2 v1.0 implementation QA/preflight closure. No scientific execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.b1sc_d2_v1_0 import implementation as d2
from b1s.execution.instrument import runtime_lock

WORKFLOW = ".github/workflows/b1sc-d2-v1-0-qa.yml"
ALLOWED_ADDITIONS = {
    "experiments/b1sc_d2_v1_0/implementation.py",
    "experiments/b1sc_d2_v1_0/tests.py",
    "experiments/b1sc_d2_v1_0/preflight.py",
    WORKFLOW,
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def preservation() -> dict:
    diff = git("diff", "--name-status", d2.DESIGN_COMMIT + "..HEAD").splitlines()
    observed = {}
    for line in diff:
        if not line:
            continue
        status, path = line.split("\t", 1)
        observed[path] = status
    require(set(observed) == ALLOWED_ADDITIONS, "Unexpected changed paths after D2 design freeze: " + repr(observed))
    require(all(v == "A" for v in observed.values()), "Only new D2 implementation/QA files are allowed")
    require(git("status", "--porcelain") == "", "Working tree is not clean")
    return {
        "status": "PASS",
        "design_commit": d2.DESIGN_COMMIT,
        "added_paths": sorted(observed),
        "design_files_unchanged": True,
    }


def workflow_contract() -> dict:
    path = d2.REPO_ROOT / WORKFLOW
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    require("workflow_dispatch" not in lower, "Manual dispatch is forbidden in QA-only D2 workflow")
    require("matrix:" not in lower, "Scientific fit matrix is forbidden in QA-only D2 workflow")
    forbidden_invocations = (
        "python -m experiments.b1sc_d2_v1_0.runner",
        "python experiments/b1sc_d2_v1_0/runner.py",
    )
    require(all(x not in lower for x in forbidden_invocations), "Scientific runner must not be invoked by QA-only D2 workflow")
    require("start_request.json" in lower, "QA workflow must explicitly prove START_REQUEST absence")
    require("permissions:\n  contents: read" in text, "QA workflow must have read-only repository permission")
    require("python -m experiments.b1sc_d2_v1_0.tests" in text, "D2 QA suite is not invoked")
    require("python -m experiments.b1sc_d2_v1_0.preflight" in text, "D2 preflight is not invoked")
    require("b1sc-d2-v1-0-execution" not in lower, "QA workflow must not invoke a scientific execution workflow")
    return {
        "status": "PASS",
        "workflow_sha256": d2.sha256_file(path),
        "training_job_present": False,
        "scientific_evaluation_job_present": False,
        "repository_write_permission": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    require(os.environ.get("GITHUB_REPOSITORY") == d2.REPOSITORY, "Wrong repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + d2.BRANCH, "Wrong D2 branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "QA/preflight reruns are not accepted as qualification")
    require(git("rev-parse", "HEAD") == os.environ.get("GITHUB_SHA"), "Unexpected checkout")
    require(not (d2.ROOT / "START_REQUEST.json").exists(), "START_REQUEST exists: D2 execution boundary violated")
    require(not (d2.ROOT / "runner.py").exists(), "Scientific runner exists during implementation-only qualification")
    require(not (d2.REPO_ROOT / ".github/workflows/b1sc-d2-v1-0-execution.yml").exists(), "Scientific D2 workflow exists during QA-only qualification")

    validation = json.loads((a.qa / "VALIDATION.json").read_text(encoding="utf-8"))
    require(validation["status"] == "D2_QA_PASS_NOT_SCIENTIFIC_EXECUTION", "D2 QA did not pass")
    require(validation["scientific_training_performed"] is False, "QA unexpectedly performed scientific training")
    require(validation["scientific_evaluation_performed"] is False, "QA unexpectedly performed scientific evaluation")
    require(validation["qa_microfit_performed"] is False, "D2 QA was not supposed to run a microfit")
    require(validation["start_request_present"] is False, "QA observed a start request")

    design = d2.validate_design_contract()
    pairing = d2.paired_seed_contract()
    preserve = preservation()
    wf = workflow_contract()
    lock = runtime_lock()
    registry = d2.registry()

    sources = [
        "experiments/b1sc_d2_v1_0/PROTOCOL_ES.md",
        "experiments/b1sc_d2_v1_0/DESIGN_FREEZE.json",
        "experiments/b1sc_d2_v1_0/implementation.py",
        "experiments/b1sc_d2_v1_0/tests.py",
        "experiments/b1sc_d2_v1_0/preflight.py",
        WORKFLOW,
        "b1s/execution/core.py",
        "b1s/execution/instrument.py",
        "b1s/execution/requirements.txt",
    ]
    hashes = {rel: d2.sha256_file(d2.REPO_ROOT / rel) for rel in sources}
    qa_hashes = {p.name: d2.sha256_file(p) for p in sorted(a.qa.iterdir()) if p.is_file()}

    freeze = {
        "status": "D2_IMPLEMENTATION_QUALIFIED_NOT_EXECUTED",
        "schema": d2.SCHEMA,
        "repository": d2.REPOSITORY,
        "branch": d2.BRANCH,
        "design_commit": d2.DESIGN_COMMIT,
        "design_freeze_sha256": d2.sha256_file(d2.DESIGN_PATH),
        "protocol_sha256": d2.sha256_file(d2.PROTOCOL_PATH),
        "implementation_source_commit": os.environ["GITHUB_SHA"],
        "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "source_hashes": hashes,
        "qa_hashes": qa_hashes,
        "registry_entries": len(registry),
        "registry_sha256": d2.sha256_bytes(d2.json_bytes(registry)),
        "design_contract": design,
        "paired_seed_contract": pairing,
        "runtime": lock,
        "historical_preservation": preserve,
        "workflow_contract": wf,
        "execution_boundary": {
            "execution_authorized": False,
            "start_request_exists": False,
            "training_started": False,
            "evaluation_started": False,
            "results_exist": False,
            "scientific_runner_present": False,
            "scientific_workflow_present": False,
            "training_job_present_in_qa_workflow": False,
        },
        "qa_scope": {
            "functional_equivalence_S6_OFF_vs_G0": "TESTED",
            "E_B_D_gradient_equivalence": "TESTED",
            "message_A_zero_functional_gradient_when_off": "TESTED",
            "paired_ON_OFF_initialization": "TESTED",
            "permanent_lesion_operator": "TESTED",
            "inactive_edge_control": "TESTED",
            "scientific_training": "NOT_PERFORMED",
            "scientific_evaluation": "NOT_PERFORMED",
        },
        "next_step_requires_explicit_execution_readiness_authorization": True,
        **d2.BOUNDARY,
    }
    write(a.out / "IMPLEMENTATION_FREEZE.json", freeze)
    write(a.out / "PREFLIGHT_STATUS.json", {
        "status": "D2_IMPLEMENTATION_QUALIFIED_NOT_EXECUTED",
        "implementation_freeze_sha256": d2.sha256_file(a.out / "IMPLEMENTATION_FREEZE.json"),
        "scientific_training_performed": False,
        "scientific_evaluation_performed": False,
        "start_request_present": False,
        "scientific_runner_present": False,
        "scientific_workflow_present": False,
        "next_step": "AUTHOR_REVIEW_BEFORE_EXECUTION_PIPELINE_IMPLEMENTATION",
        **d2.BOUNDARY,
    })
    print(json.dumps(json.loads((a.out / "PREFLIGHT_STATUS.json").read_text(encoding="utf-8")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
