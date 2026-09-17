"""Immutable authorization envelope for B1-SC-D4 scientific execution.

The qualified scientific runner at b475a41b remains byte-for-byte unchanged.
Scientific execution requires a later EXECUTE_REQUEST-only commit whose parent is
the immutable START_REQUEST authorization commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from experiments.b1sc_d4_v1_0 import implementation as d4
from experiments.b1sc_d4_v1_0 import scientific_runner as qualified

ROOT = Path(__file__).resolve().parent
BRANCH = "research/b1sc-d4-v1.0-runner-readiness-20260916"
QUALIFIED_RUNNER_COMMIT = "b475a41b583b50a516bebce16d030518c8dd02f2"
QUALIFIED_RUNNER_BLOB = "8fcdad1a0bb26cf37ef3596b9bc20a1ff7f56481"
QUALIFIED_CONTRACT_BLOB = "da9cdf3c647960d7bd6ef1a244177e60a1fa1f52"
START_REQUEST = ROOT / "START_REQUEST.json"
EXECUTE_REQUEST = ROOT / "EXECUTE_REQUEST.json"
AUTHORITY = ROOT / "EXECUTION_AUTHORITY.json"
PREFLIGHT_RECEIPT = ROOT / "RUNNER_PREFLIGHT_RECEIPT.json"
PREFLIGHT_RECEIPT_SHA256 = "833b47fff8ad0d3088efdaad1902bc164dabc7c7d56a19d92a90b046c31d580d"
PREFLIGHT_RUN_ID = 35105193510
PREFLIGHT_ARTIFACT_ID = 10450457252
PREFLIGHT_ARTIFACT_DIGEST = "sha256:cec902fa93740e510922f6932756f39f4950ff12ee021fe3ade0da7327ede095"
WORKFLOW = ".github/workflows/b1sc-d4-v1-0-scientific.yml"
START_REL = "experiments/b1sc_d4_v1_0/START_REQUEST.json"
EXECUTE_REL = "experiments/b1sc_d4_v1_0/EXECUTE_REQUEST.json"
AUTHORITY_REL = "experiments/b1sc_d4_v1_0/EXECUTION_AUTHORITY.json"
WRAPPER_REL = "experiments/b1sc_d4_v1_0/authorized_runner.py"
INTEGRITY_REL = "experiments/b1sc_d4_v1_0/scientific_integrity.py"
RECEIPT_REL = "experiments/b1sc_d4_v1_0/RUNNER_PREFLIGHT_RECEIPT.json"
RUNNER_REL = "experiments/b1sc_d4_v1_0/scientific_runner.py"
CONTRACT_REL = "experiments/b1sc_d4_v1_0/RUNNER_CONTRACT.json"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _blob_at(ref: str, rel: str) -> str:
    return git("rev-parse", f"{ref}:{rel}")


def authorization_guard() -> dict:
    require(START_REQUEST.is_file(), "D4 START_REQUEST.json is absent")
    require(EXECUTE_REQUEST.is_file(), "D4 EXECUTE_REQUEST.json is absent")
    require(AUTHORITY.is_file() and PREFLIGHT_RECEIPT.is_file(), "D4 authority evidence is incomplete")
    require(os.environ.get("GITHUB_REPOSITORY") == d4.REPOSITORY, "Wrong D4 repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + BRANCH, "Wrong D4 branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Scientific reruns are not authorized")

    current = git("rev-parse", "HEAD")
    require(current == os.environ.get("GITHUB_SHA"), "Unexpected D4 checkout")
    authorization_commit = git("rev-parse", "HEAD^")
    authorization_base = git("rev-parse", "HEAD^^")
    require(git("diff", "--name-only", authorization_commit, current).splitlines() == [EXECUTE_REL],
            "Execution activation commit must contain only EXECUTE_REQUEST.json")
    require(git("diff", "--name-only", authorization_base, authorization_commit).splitlines() == [START_REL],
            "Authorization commit must contain only START_REQUEST.json")
    require(_blob_at(authorization_commit, START_REL) == _blob_at("HEAD", START_REL),
            "START_REQUEST changed after authorization")

    req = read_json(START_REQUEST)
    exe = read_json(EXECUTE_REQUEST)
    authority = read_json(AUTHORITY)
    receipt = read_json(PREFLIGHT_RECEIPT)

    require(req.get("schema") == "B1-SC-D4-START-1.0.0-20260916" and
            req.get("authorize_b1sc_d4_v1_0") is True, "Invalid D4 START_REQUEST")
    require(req.get("authorized_run_attempt") == 1, "Only D4 run attempt 1 is authorized")
    require(req.get("authorization_base_commit") == authorization_base, "Authorization base commit mismatch")
    require(req.get("qualified_runner_commit") == QUALIFIED_RUNNER_COMMIT, "Qualified D4 runner commit mismatch")
    require(req.get("preflight_run_id") == PREFLIGHT_RUN_ID and
            req.get("preflight_artifact_id") == PREFLIGHT_ARTIFACT_ID, "Preflight identity mismatch")
    require(req.get("preflight_artifact_digest") == PREFLIGHT_ARTIFACT_DIGEST,
            "Preflight artifact digest mismatch")
    require(req.get("preflight_receipt_sha256") == PREFLIGHT_RECEIPT_SHA256,
            "Preflight receipt hash mismatch")
    require(req.get("B1E_executed") is False and req.get("final_seeds_generated") is False,
            "B1-E boundary violated")
    require(req.get("scientific_fits") == d4.FITS and req.get("native_steps_per_fit") == d4.NATIVE_STEPS,
            "Authorized D4 scope changed")
    require(req.get("conditions") == list(d4.CONDITIONS) and req.get("blocks") == d4.BLOCKS,
            "Authorized D4 matrix changed")
    require(req.get("switch_after_completed_native_steps") == d4.SWITCH_STEPS,
            "Authorized D4 switch changed")
    require(req.get("execution_workflow_git_blob") == _blob_at("HEAD", WORKFLOW),
            "Scientific workflow blob mismatch")
    require(req.get("authority_manifest_git_blob") == _blob_at("HEAD", AUTHORITY_REL),
            "Authority manifest blob mismatch")
    require(req.get("authorization_wrapper_git_blob") == _blob_at("HEAD", WRAPPER_REL),
            "Authorization wrapper blob mismatch")
    require(req.get("scientific_integrity_git_blob") == _blob_at("HEAD", INTEGRITY_REL),
            "Scientific integrity blob mismatch")
    require(req.get("preflight_receipt_git_blob") == _blob_at("HEAD", RECEIPT_REL),
            "Preflight receipt blob mismatch")

    require(exe.get("schema") == "B1-SC-D4-EXECUTE-1.0.0-20260916", "Wrong EXECUTE_REQUEST schema")
    require(exe.get("execute_b1sc_d4_v1_0") is True, "D4 execution activation absent")
    require(exe.get("authorization_commit") == authorization_commit,
            "EXECUTE_REQUEST authorization commit mismatch")
    require(exe.get("qualified_runner_commit") == QUALIFIED_RUNNER_COMMIT, "EXECUTE_REQUEST runner mismatch")
    require(exe.get("authorized_run_attempt") == 1, "EXECUTE_REQUEST attempt mismatch")
    require(exe.get("confirmation") == "EXECUTE_D4_36_FITS", "Wrong execution confirmation token")
    require(exe.get("conditions") == list(d4.CONDITIONS) and exe.get("blocks") == d4.BLOCKS,
            "EXECUTE_REQUEST matrix mismatch")
    require(exe.get("scientific_fits") == d4.FITS and
            exe.get("native_steps_per_fit") == d4.NATIVE_STEPS,
            "EXECUTE_REQUEST scope mismatch")
    require(exe.get("B1E_executed") is False and exe.get("final_seeds_generated") is False,
            "EXECUTE_REQUEST B1-E boundary violated")

    require(authority["qualified_runner_commit"] == QUALIFIED_RUNNER_COMMIT, "Authority runner lineage changed")
    require(authority["qualified_scientific_runner_git_blob"] == QUALIFIED_RUNNER_BLOB,
            "Authority runner blob changed")
    require(authority["qualified_runner_contract_git_blob"] == QUALIFIED_CONTRACT_BLOB,
            "Authority contract blob changed")
    require(authority["preflight"]["receipt_sha256"] == PREFLIGHT_RECEIPT_SHA256,
            "Authority receipt hash changed")
    require(sha256_file(PREFLIGHT_RECEIPT) == PREFLIGHT_RECEIPT_SHA256,
            "Persisted preflight receipt SHA-256 mismatch")
    require(receipt["status"] == "D4_RUNNER_PREFLIGHT_QUALIFIED_NOT_AUTHORIZED", "Wrong preflight status")
    require(receipt["source_commit"] == QUALIFIED_RUNNER_COMMIT and
            str(receipt["qa_run_id"]) == str(PREFLIGHT_RUN_ID), "Preflight receipt lineage mismatch")
    require(receipt["scientific_training_performed"] is False and
            receipt["scientific_evaluation_performed"] is False and
            receipt["scientific_results_exist"] is False, "Preflight receipt contains science")

    for rel, expected in receipt["frozen_path_git_blobs"].items():
        require(_blob_at(QUALIFIED_RUNNER_COMMIT, rel) == expected, f"Qualified frozen blob mismatch: {rel}")
        require(_blob_at("HEAD", rel) == expected, f"Frozen D4 authority mutated after qualification: {rel}")

    require(_blob_at(QUALIFIED_RUNNER_COMMIT, RUNNER_REL) == QUALIFIED_RUNNER_BLOB and
            _blob_at("HEAD", RUNNER_REL) == QUALIFIED_RUNNER_BLOB,
            "Qualified scientific runner changed")
    require(_blob_at(QUALIFIED_RUNNER_COMMIT, CONTRACT_REL) == QUALIFIED_CONTRACT_BLOB and
            _blob_at("HEAD", CONTRACT_REL) == QUALIFIED_CONTRACT_BLOB,
            "Qualified runner contract changed")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", QUALIFIED_RUNNER_COMMIT, "HEAD"])

    actual_snapshots = qualified.verify_all_snapshots()
    expected_snapshots = receipt["immutable_snapshots"]
    require(len(actual_snapshots) == len(expected_snapshots) == d4.BLOCKS,
            "D4 snapshot cardinality changed")
    for actual, expected in zip(actual_snapshots, expected_snapshots, strict=True):
        require(int(actual["block"]) == int(expected["block"]), "D4 snapshot block order changed")
        require(int(actual["bytes"]) == int(expected["bytes"]), "D4 snapshot byte length changed")
        require(actual["sha256"] == expected["sha256"], "D4 snapshot SHA-256 changed")
        require(int(actual["source_seed"]) == int(expected["source_seed"]), "D4 snapshot lineage changed")

    return {
        "status": "D4_SCIENTIFIC_EXECUTION_ACTIVATION_VALID",
        "execution_commit": current,
        "authorization_commit": authorization_commit,
        "authorization_base_commit": authorization_base,
        "qualified_runner_commit": QUALIFIED_RUNNER_COMMIT,
        "preflight_run_id": PREFLIGHT_RUN_ID,
        "preflight_artifact_id": PREFLIGHT_ARTIFACT_ID,
        "preflight_artifact_digest": PREFLIGHT_ARTIFACT_DIGEST,
        "preflight_receipt_sha256": PREFLIGHT_RECEIPT_SHA256,
        "scientific_fits": d4.FITS,
        "blocks": d4.BLOCKS,
        "conditions": list(d4.CONDITIONS),
        "native_steps_per_fit": d4.NATIVE_STEPS,
        "total_native_steps": d4.FITS * d4.NATIVE_STEPS,
        "switch_after_completed_native_steps": d4.SWITCH_STEPS,
        "B1E_executed": False,
        "final_seeds_generated": False,
    }


def scientific_fit(condition: str, block: int, out: Path) -> dict:
    auth = authorization_guard()
    qualified.runtime()
    result = qualified._train(condition, block, out, d4.NATIVE_STEPS, "training", 0)
    result["authorization"] = auth
    qualified.write_json(out / "RUN_RESULT.json", result)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--condition", choices=d4.CONDITIONS)
    p.add_argument("--block", type=int, choices=range(d4.BLOCKS))
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.preflight_only:
        payload = authorization_guard()
    else:
        require(args.condition is not None and args.block is not None and args.out is not None,
                "Scientific fit arguments are required")
        payload = scientific_fit(args.condition, args.block, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
