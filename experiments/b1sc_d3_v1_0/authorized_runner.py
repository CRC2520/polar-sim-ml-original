"""Immutable authorization envelope for B1-SC-D3 scientific execution.

The qualified scientific runner at e05223a remains byte-for-byte unchanged.
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

from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0 import scientific_runner as qualified

ROOT = Path(__file__).resolve().parent
BRANCH = "research/b1sc-d3-v1.0-runner-readiness-20260916"
QUALIFIED_RUNNER_COMMIT = "e05223aad09b75e075d6c4fac87565d95531779c"
QUALIFIED_RUNNER_BLOB = "ca9f6c991b563a7fceb0d2e05933e931a1195a75"
QUALIFIED_CONTRACT_BLOB = "c64058ce8f89d48fd82d3a060c6d83a6213ac24b"
START_REQUEST = ROOT / "START_REQUEST.json"
EXECUTE_REQUEST = ROOT / "EXECUTE_REQUEST.json"
AUTHORITY = ROOT / "EXECUTION_AUTHORITY.json"
PREFLIGHT_RECEIPT = ROOT / "RUNNER_PREFLIGHT_RECEIPT.json"
PREFLIGHT_RECEIPT_SHA256 = "b657ce0b189e54f3a725f9877103afc05ddaee0b2f00f32fd991bea80489f4a4"
PREFLIGHT_RUN_ID = 35057029187
PREFLIGHT_ARTIFACT_ID = 10431735009
PREFLIGHT_ARTIFACT_DIGEST = "sha256:ca2eee6c9a1007709757c76ff2622758dc62bf88bf6388dae809d19288236009"
WORKFLOW = ".github/workflows/b1sc-d3-v1-0-scientific.yml"
START_REL = "experiments/b1sc_d3_v1_0/START_REQUEST.json"
EXECUTE_REL = "experiments/b1sc_d3_v1_0/EXECUTE_REQUEST.json"


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
    require(START_REQUEST.is_file(), "D3 START_REQUEST.json is absent")
    require(EXECUTE_REQUEST.is_file(), "D3 EXECUTE_REQUEST.json is absent")
    require(AUTHORITY.is_file() and PREFLIGHT_RECEIPT.is_file(), "D3 authority evidence is incomplete")
    require(os.environ.get("GITHUB_REPOSITORY") == "CRC2520/polar-sim-ml-original", "Wrong D3 repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + BRANCH, "Wrong D3 branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Scientific reruns are not authorized")

    current = git("rev-parse", "HEAD")
    require(current == os.environ.get("GITHUB_SHA"), "Unexpected D3 checkout")
    authorization_commit = git("rev-parse", "HEAD^")
    authorization_base = git("rev-parse", "HEAD^^")
    require(git("diff", "--name-only", authorization_commit, current).splitlines() == [EXECUTE_REL], "Execution activation commit must contain only EXECUTE_REQUEST.json")
    require(git("diff", "--name-only", authorization_base, authorization_commit).splitlines() == [START_REL], "Authorization commit must contain only START_REQUEST.json")
    require(_blob_at(authorization_commit, START_REL) == _blob_at("HEAD", START_REL), "START_REQUEST changed after authorization")

    req = read_json(START_REQUEST)
    exe = read_json(EXECUTE_REQUEST)
    authority = read_json(AUTHORITY)
    receipt = read_json(PREFLIGHT_RECEIPT)

    require(req.get("schema") == "B1-SC-D3-START-1.0.0-20260916" and req.get("authorize_b1sc_d3_v1_0") is True, "Invalid D3 START_REQUEST")
    require(req.get("authorized_run_attempt") == 1, "Only D3 run attempt 1 is authorized")
    require(req.get("authorization_base_commit") == authorization_base, "Authorization base commit mismatch")
    require(req.get("qualified_runner_commit") == QUALIFIED_RUNNER_COMMIT, "Qualified D3 runner commit mismatch")
    require(req.get("preflight_run_id") == PREFLIGHT_RUN_ID and req.get("preflight_artifact_id") == PREFLIGHT_ARTIFACT_ID, "Preflight identity mismatch")
    require(req.get("preflight_artifact_digest") == PREFLIGHT_ARTIFACT_DIGEST, "Preflight artifact digest mismatch")
    require(req.get("preflight_receipt_sha256") == PREFLIGHT_RECEIPT_SHA256, "Preflight receipt hash mismatch")
    require(req.get("B1E_executed") is False and req.get("final_seeds_generated") is False, "B1-E boundary violated")
    require(req.get("scientific_fits") == 32 and req.get("native_steps_per_fit") == d3.NATIVE_STEPS, "Authorized D3 scope changed")
    require(req.get("conditions") == list(d3.CONDITIONS) and req.get("blocks") == d3.BLOCKS, "Authorized D3 matrix changed")
    require(req.get("execution_workflow_git_blob") == _blob_at("HEAD", WORKFLOW), "Scientific workflow blob mismatch")
    require(req.get("authority_manifest_git_blob") == _blob_at("HEAD", "experiments/b1sc_d3_v1_0/EXECUTION_AUTHORITY.json"), "Authority manifest blob mismatch")
    require(req.get("authorization_wrapper_git_blob") == _blob_at("HEAD", "experiments/b1sc_d3_v1_0/authorized_runner.py"), "Authorization wrapper blob mismatch")

    require(exe.get("schema") == "B1-SC-D3-EXECUTE-1.0.0-20260916", "Wrong EXECUTE_REQUEST schema")
    require(exe.get("execute_b1sc_d3_v1_0") is True, "D3 execution activation absent")
    require(exe.get("authorization_commit") == authorization_commit, "EXECUTE_REQUEST authorization commit mismatch")
    require(exe.get("qualified_runner_commit") == QUALIFIED_RUNNER_COMMIT, "EXECUTE_REQUEST runner mismatch")
    require(exe.get("authorized_run_attempt") == 1, "EXECUTE_REQUEST attempt mismatch")
    require(exe.get("confirmation") == "EXECUTE_D3_32_FITS", "Wrong execution confirmation token")
    require(exe.get("B1E_executed") is False and exe.get("final_seeds_generated") is False, "EXECUTE_REQUEST B1-E boundary violated")

    require(authority["qualified_runner_commit"] == QUALIFIED_RUNNER_COMMIT, "Authority runner lineage changed")
    require(authority["qualified_scientific_runner_git_blob"] == QUALIFIED_RUNNER_BLOB, "Authority runner blob changed")
    require(authority["qualified_runner_contract_git_blob"] == QUALIFIED_CONTRACT_BLOB, "Authority contract blob changed")
    require(authority["preflight"]["receipt_sha256"] == PREFLIGHT_RECEIPT_SHA256, "Authority receipt hash changed")
    require(sha256_file(PREFLIGHT_RECEIPT) == PREFLIGHT_RECEIPT_SHA256, "Persisted preflight receipt SHA-256 mismatch")
    require(receipt["status"] == "D3_RUNNER_PREFLIGHT_QUALIFIED_NOT_AUTHORIZED", "Wrong preflight status")
    require(receipt["source_commit"] == QUALIFIED_RUNNER_COMMIT and str(receipt["qa_run_id"]) == str(PREFLIGHT_RUN_ID), "Preflight receipt lineage mismatch")
    require(receipt["scientific_training_performed"] is False and receipt["scientific_evaluation_performed"] is False and receipt["scientific_results_exist"] is False, "Preflight receipt contains science")

    runner_rel = "experiments/b1sc_d3_v1_0/scientific_runner.py"
    contract_rel = "experiments/b1sc_d3_v1_0/RUNNER_CONTRACT.json"
    require(_blob_at(QUALIFIED_RUNNER_COMMIT, runner_rel) == QUALIFIED_RUNNER_BLOB and _blob_at("HEAD", runner_rel) == QUALIFIED_RUNNER_BLOB, "Qualified scientific runner changed")
    require(_blob_at(QUALIFIED_RUNNER_COMMIT, contract_rel) == QUALIFIED_CONTRACT_BLOB and _blob_at("HEAD", contract_rel) == QUALIFIED_CONTRACT_BLOB, "Qualified runner contract changed")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", QUALIFIED_RUNNER_COMMIT, "HEAD"])
    qualified.verify_all_snapshots()

    return {
        "status": "D3_SCIENTIFIC_EXECUTION_ACTIVATION_VALID",
        "execution_commit": current,
        "authorization_commit": authorization_commit,
        "authorization_base_commit": authorization_base,
        "qualified_runner_commit": QUALIFIED_RUNNER_COMMIT,
        "preflight_run_id": PREFLIGHT_RUN_ID,
        "preflight_artifact_id": PREFLIGHT_ARTIFACT_ID,
        "preflight_artifact_digest": PREFLIGHT_ARTIFACT_DIGEST,
        "preflight_receipt_sha256": PREFLIGHT_RECEIPT_SHA256,
        "scientific_fits": 32,
        "native_steps_per_fit": d3.NATIVE_STEPS,
        "B1E_executed": False,
        "final_seeds_generated": False,
    }


def scientific_fit(condition: str, block: int, out: Path) -> dict:
    auth = authorization_guard()
    qualified.runtime()
    result = qualified._train(condition, block, out, d3.NATIVE_STEPS, "training")
    result["authorization"] = auth
    qualified.write_json(out / "RUN_RESULT.json", result)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--condition", choices=d3.CONDITIONS)
    p.add_argument("--block", type=int, choices=range(d3.BLOCKS))
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.preflight_only:
        payload = authorization_guard()
    else:
        require(args.condition is not None and args.block is not None and args.out is not None, "Scientific fit arguments are required")
        payload = scientific_fit(args.condition, args.block, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
