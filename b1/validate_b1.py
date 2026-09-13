"""Read-only B1 provenance and documentary validation; never runs a study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


CODE_SOURCE = "c958049743f989b5c1f8866018ea48b60bae9ed7"
HISTORICAL_REALIZATION = "05caf7abef6fdcb769604539a22cdcd1cfd6b397"
B0_FREEZE_SHA256 = "7d9abe376401ca53c8e8f9f519b75c6bb227dfc49749059bb72eb24050978cc1"
WORKFLOW = ".github/workflows/b1-development-validation.yml"
REQUIRED = ["README.md", "B1D_IMPLEMENTATION_SPEC.md", "B1D_CONFIG.json",
            "B1D_CONTRACT_IMPORT.json", "B1D_SEED_POLICY.json",
            "B1D_COMPARATOR_MATRIX.csv", "B1D_INVARIANT_MATRIX.csv",
            "B1D_RESULTS.json", "B1D_RESULTS.md", "B1D_AUDIT.json", "B1D_MANIFEST.json"]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(path, mode="100644"):
    path = Path(path)
    data = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def allowed_path(path):
    return path.startswith("b1/") or path == WORKFLOW


def safe_artifact(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in name:
        raise ValueError(f"unsafe artifact path: {name}")
    result = Path(root).joinpath(*p.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"artifact escapes repository: {name}")
    return result


def git(root, *args):
    command = ["git", "--literal-pathspecs", "-C", str(root), *args]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def tree_blobs(root, commit):
    result = {}
    for record in git(root, "ls-tree", "-rz", commit).split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, digest = meta.decode().split()
        if kind != "blob":
            raise ValueError(f"unsupported source object {kind}: {name!r}")
        result[name.decode()] = {"mode": mode, "sha": digest}
    return result


def verify_preservation(root, baseline, *, sparse=False, enforce_pins=True):
    """Compare immutable source tree, HEAD, and working bytes; sparse mode is partial only."""
    errors = []
    if enforce_pins and (baseline.get("code_source_commit") != CODE_SOURCE or
                         baseline.get("historical_realization_commit") != HISTORICAL_REALIZATION):
        errors.append("historical baseline immutable commit mismatch")
    records = baseline.get("blobs", [])
    expected = {row["path"]: {"sha": row["sha"], "mode": row["mode"]} for row in records}
    if len(records) != len(expected) or baseline.get("blob_count") != len(expected):
        errors.append("historical baseline duplicate/count mismatch")
    if enforce_pins and len(expected) != 773:
        errors.append("expected all 773 original source blobs")
    checked = 0
    if not sparse:
        try:
            source = baseline["code_source_commit"]
            git(root, "merge-base", "--is-ancestor", baseline["historical_realization_commit"], source)
            git(root, "merge-base", "--is-ancestor", source, "HEAD")
            actual_source = tree_blobs(root, source)
            if actual_source != expected:
                errors.append("baseline inventory does not equal the complete immutable source tree")
            current = tree_blobs(root, "HEAD")
            for name, record in expected.items():
                if current.get(name) != record:
                    errors.append(f"historical HEAD blob changed or missing: {name}")
            for name in current.keys() - expected.keys():
                if not allowed_path(name):
                    errors.append(f"addition outside B1 allowlist: {name}")
            changed = git(root, "diff", "--name-only", "-z", "HEAD").split(b"\0")
            for name in changed:
                if name and not allowed_path(name.decode()):
                    errors.append(f"working change outside B1 allowlist: {name.decode()}")
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))
    for name, record in expected.items():
        path = safe_artifact(root, name)
        if not path.exists() and not path.is_symlink():
            if not sparse:
                errors.append(f"missing historical working file: {name}")
            continue
        checked += 1
        if git_blob(path, record["mode"]) != record["sha"]:
            errors.append(f"historical working bytes changed: {name}")
        if not sparse and os.name != "nt" and record["mode"] in {"100644", "100755"}:
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (record["mode"] == "100755"):
                errors.append(f"historical executable mode changed: {name}")
    return {"errors": errors, "historical_working_blobs_checked": checked,
            "remote_integrity_verified": False if sparse else not errors,
            "validation_scope": "sparse_local_only" if sparse else "git_ancestry_HEAD_and_working_tree"}


def validate_artifact_hashes(root, manifest, own_path):
    errors = []
    hashes = manifest.get("artifact_hashes", manifest.get("artifact_sha256"))
    if not isinstance(hashes, dict) or not hashes:
        return ["missing nonempty artifact_hashes mapping"]
    if own_path in hashes:
        errors.append("manifest cannot include its own SHA256")
    for name, expected in hashes.items():
        try:
            path = safe_artifact(root, name)
            if not re.fullmatch(r"[a-f0-9]{64}", str(expected)):
                errors.append(f"invalid SHA256 value: {name}")
            elif not path.is_file() or sha256(path) != expected:
                errors.append(f"artifact SHA256 mismatch or missing: {name}")
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    return errors


def manifest_coverage(root, hashes, own_path):
    expected = {str(p.relative_to(root)) for p in (Path(root) / "b1").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                and str(p.relative_to(root)) != own_path}
    return [f"B1 artifact omitted from freeze hashes: {name}" for name in sorted(expected - set(hashes))]


def validate_structures(root):
    errors = []
    count = 0
    for path in sorted((Path(root) / "b1").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            if path.suffix == ".json":
                read_json(path)
                count += 1
            elif path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.reader(stream))
                if not rows or len(rows[0]) != len(set(rows[0])):
                    raise ValueError("empty CSV or duplicate header")
                if any(len(row) != len(rows[0]) for row in rows[1:]):
                    raise ValueError("CSV field count mismatch")
                count += 1
        except (ValueError, OSError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    return count, errors


def validate(root, *, pre_freeze=False, sparse=False):
    root = Path(root).resolve()
    errors = []
    warnings = []
    if sparse:
        warnings.append("Sparse workspace: absent historical files and Git ancestry are NOT verified; this is not a remote-integrity PASS.")
    for name in REQUIRED:
        if not (root / "b1" / name).is_file():
            if pre_freeze:
                warnings.append(f"not yet present before freeze: b1/{name}")
            else:
                errors.append(f"missing required artifact: b1/{name}")
    baseline_path = root / "b1/manifests/HISTORICAL_BASELINE.json"
    if baseline_path.is_file():
        preservation = verify_preservation(root, read_json(baseline_path), sparse=sparse)
        errors.extend(preservation["errors"])
    else:
        preservation = {"remote_integrity_verified": False}
        errors.append("missing historical baseline inventory")
    frozen = root / "b1/contracts/frozen_b0"
    if not (frozen / "B0_FREEZE.json").is_file() or sha256(frozen / "B0_FREEZE.json") != B0_FREEZE_SHA256:
        errors.append("imported B0 freeze SHA256 mismatch")
    else:
        b0 = read_json(frozen / "B0_FREEZE.json")
        for name in ("PILOT_CONTRACTS_v1.json", "CONSTRUCT_DICTIONARY_v1.yaml", "B1_PROTOCOL_DRAFT_v1.md", "B1_DECISION_RULES_v1.yaml", "PD_OPERATIONAL_FOUNDATIONS_v3.0.md"):
            expected = b0["artifact_sha256"].get("b0/" + name)
            if not (frozen / name).is_file() or sha256(frozen / name) != expected:
                errors.append(f"imported frozen canonical file changed: {name}")
    structure_count, structure_errors = validate_structures(root)
    errors.extend(structure_errors)
    results_path = root / "b1/B1D_RESULTS.json"
    if results_path.is_file():
        results = read_json(results_path)
        for field, value in (("development_only", True), ("confirmatory", False), ("reusable_as_final", False)):
            if results.get(field) is not value:
                errors.append(f"B1D_RESULTS must set {field}={value}")
    manifest_path = root / "b1/B1D_MANIFEST.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        errors.extend(validate_artifact_hashes(root, manifest, "b1/B1D_MANIFEST.json"))
        hashes = manifest.get("artifact_hashes", manifest.get("artifact_sha256", {}))
        if isinstance(hashes, dict):
            errors.extend(manifest_coverage(root, hashes, "b1/B1D_MANIFEST.json"))
        if manifest.get("status") not in {"B1D_COMPLETE", "B1D_BLOCKED"}:
            errors.append("invalid B1D manifest status")
        if manifest.get("status") == "B1D_BLOCKED" and not manifest.get("unresolved_blockers"):
            errors.append("B1D_BLOCKED requires explicit unresolved_blockers")
        for field in ("final_seeds_generated", "ready_for_b1e_confirmatory_run"):
            if manifest.get(field) is not False:
                errors.append(f"{field} must be explicitly false in this assignment")
        for field in ("historical_code_changed", "historical_results_changed"):
            if manifest.get(field) is not False:
                errors.append(f"{field} must be explicitly false")
    return {"status": "FAIL" if errors else ("PASS_SPARSE_LOCAL" if sparse else "PASS"),
            "pre_freeze": pre_freeze, "structures_checked": structure_count,
            "preservation": preservation, "errors": errors, "warnings": warnings,
            "experiments_executed_by_validator": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pre-freeze", action="store_true")
    parser.add_argument("--sparse-workspace", action="store_true", help="Local incomplete staging only; never a remote integrity claim")
    args = parser.parse_args()
    try:
        result = validate(args.repo_root, pre_freeze=args.pre_freeze, sparse=args.sparse_workspace)
    except (OSError, ValueError, KeyError) as exc:
        result = {"status": "FAIL", "errors": [str(exc)]}
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
