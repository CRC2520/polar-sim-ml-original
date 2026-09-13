"""Recover a missing reporting sidecar from immutable, closed trace archives.

This is reporting version 1, added AFTER calibration closure. It is outside the
original pre-tuning inventory. It cannot resume calibration or satisfy the
original runner's inventory gate. Frozen experimental code/data stay unchanged.
No tasks, controllers, seed policy, runner, or diagnostic execution functions
are imported. Only the unchanged decision and precision evaluators are reused.
The descriptive reporting body is an explicit adaptation of frozen run.export;
its sole data correction is the separately identified derived diagnostics input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from b1.contracts import load_contracts
from b1.contracts.loader import strict_json
from b1.evaluation.decision import evaluate_development
from b1.evaluation.precision import precision_plan

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = load_contracts()
FLAGS = {"development_only": True, "confirmatory": False, "reusable_as_final": False}
RESULTS_DIR = "development_results"
VERSION = "closed-trace-reporting-recovery-v1"
SIDECAR = "postprocessing/bundle_diagnostics.recovered.jsonl"
INCIDENT = "postprocessing/EXPORT_RECOVERY_INCIDENT.json"
SCOPE = "Fresh deterministic imposed-law clone diagnostics within this bundle; no independent empirical confirmation or route-specific inference"


class RecoveryError(RuntimeError):
    """Incomplete, changed or ambiguous evidence blocks reporting recovery."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def write_bytes_new(path, data):
    """Never overwrite an existing artifact, except accepting identical bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RecoveryError(f"Refusing to overwrite retained artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def write_json(path, value):
    write_bytes_new(path, (json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def verify_snapshot(root, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise RecoveryError("A nonempty frozen content snapshot is required")
    root = Path(root).resolve()
    for name, expected in hashes.items():
        target = (root / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(root):
            raise RecoveryError("Unsafe snapshot path")
        if not target.is_file() or digest(target) != expected:
            raise RecoveryError(f"Frozen artifact changed or missing: {name}")


def verify_reporting_inputs(root):
    """Verify frozen bytes; explicitly list the separate reporting inventory.

    This verifies past closure inputs, not authorization to open an experiment.
    Unlike run.verify_pre_tuning it does not claim that new reporting sources
    belonged to the experimental freeze. Any other Python addition is rejected.
    """
    root = Path(root)
    files = {name: read_json(root / name) for name in (
        "PRE_TUNING_FREEZE.json", "INSTRUMENT_GATE.json", "TUNING_FREEZE.json",
        "CALIBRATION_TECHNICAL.json", "CALIBRATION_STARTED.json")}
    expected = {"PRE_TUNING_FREEZE.json": "IMPLEMENTATION_CONFIG_FROZEN_BEFORE_SCORES",
                "INSTRUMENT_GATE.json": "PASS", "TUNING_FREEZE.json": "TUNING_CLOSED",
                "CALIBRATION_TECHNICAL.json": "TECHNICAL_CALIBRATION_CLOSED"}
    checks = []
    for name, status in expected.items():
        if files[name].get("status") != status:
            raise RecoveryError(f"Invalid frozen status: {name}")
        verify_snapshot(root, files[name]["artifact_sha256"])
        checks.append({"manifest": name, "sha256": digest(root / name),
                       "artifacts_verified": len(files[name]["artifact_sha256"])})
    pre, gate, tuning, technical, started = (files[name] for name in files)
    if pre["source_commit"] != CONTRACTS.source_commit or tuning["source_commit"] != CONTRACTS.source_commit:
        raise RecoveryError("Canonical source commit mismatch")
    if gate["pre_tuning_freeze_sha256"] != digest(root / "PRE_TUNING_FREEZE.json"):
        raise RecoveryError("Instrument gate is not linked to original freeze")
    if (tuning.get("tuning_bundle_indices") != list(range(32)) or
            tuning.get("configuration_frozen") is not True or
            tuning.get("calibration_comparator_identities_masked") is not True or
            tuning.get("calibration_consumed") is not False):
        raise RecoveryError("Tuning closure declarations invalid")
    required = tuning["implementation_files"] + [tuning["configuration_file"], tuning["tuning_ledger_file"]]
    if not all(name in tuning["artifact_sha256"] for name in required):
        raise RecoveryError("Tuning closure evidence missing")
    ledger = read_json(root / tuning["tuning_ledger_file"])
    if ledger["bundle_indices"] != list(range(32)) or ledger.get("development_only") is not True:
        raise RecoveryError("Invalid tuning ledger")
    frozen_hash = digest(root / "TUNING_FREEZE.json")
    if (technical["tuning_freeze_sha256"] != frozen_hash or
            started["tuning_freeze_sha256"] != frozen_hash or
            started.get("calibration_consumed") is not True or
            technical["bundle_indices"] != list(range(32, 64))):
        raise RecoveryError("Calibration closure linkage or consumed-block range invalid")
    original_inventory = set(pre["source_files"])
    current_py = {p.relative_to(root).as_posix() for p in root.rglob("*.py")
                  if RESULTS_DIR not in p.parts and "__pycache__" not in p.parts}
    additions = sorted(current_py - original_inventory)
    if not all(name.startswith("postprocessing/") for name in additions):
        raise RecoveryError("Unexpected addition to experimental Python inventory")
    if set(pre["artifact_sha256"]) != original_inventory:
        raise RecoveryError("Pre-tuning inventory and hashes differ")
    identity = root / RESULTS_DIR / "calibration/IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json"
    if read_json(identity).get("technical_close_sha256") != digest(root / "CALIBRATION_TECHNICAL.json"):
        raise RecoveryError("Disclosure not linked to immutable technical closure")
    return {"frozen_snapshots": checks,
            "postclosure_reporting_python_additions": {name: digest(root / name) for name in additions},
            "identity_disclosure_sha256": digest(identity),
            "original_inventory_gate_satisfied_by_reporting_additions": False,
            "experimental_freeze_modified": False}


def collect_archived_diagnostics(directory, technical):
    """Hash-check every shard and verify archive totals before using records."""
    directory = Path(directory)
    index = read_json(directory / "traces/INDEX.json")
    if index != technical["trace_index"]:
        raise RecoveryError("Archive index differs from technical closure")
    seen_paths = set()
    records, totals = [], Counter()
    for shard in index["shards"]:
        name = shard["path"]
        if Path(name).name != name or name in seen_paths:
            raise RecoveryError("Duplicate or unsafe trace shard")
        seen_paths.add(name)
        data = (directory / "traces" / name).read_bytes()
        if len(data) != shard["bytes"] or sha256(data).hexdigest() != shard["sha256"]:
            raise RecoveryError(f"Trace shard checksum/size mismatch: {name}")
        try:
            raw = gzip.decompress(data)
        except (OSError, EOFError) as exc:
            raise RecoveryError(f"Invalid gzip shard: {name}") from exc
        lines = raw.splitlines(keepends=True)
        if len(lines) != shard["records"] or any(not line.endswith(b"\n") for line in lines):
            raise RecoveryError(f"Trace record count/framing mismatch: {name}")
        totals.update(records=len(lines), raw_bytes=len(raw), compressed_bytes=len(data), shards=1)
        # All bytes were authenticated above. Parse only the designated record
        # type; the other closed episodes are not rerun or reinterpreted here.
        for line in lines:
            if b'"record_type":"BUNDLE_SOURCE_DIAGNOSTIC"' in line:
                record = strict_json(line.decode("utf-8"))
                if record.get("record_type") != "BUNDLE_SOURCE_DIAGNOSTIC":
                    raise RecoveryError("Unexpected diagnostic record type")
                records.append(record)
    for name in ("records", "raw_bytes", "compressed_bytes"):
        if totals[name] != index[name]:
            raise RecoveryError(f"Archive total mismatch: {name}")
    if {p.name for p in (directory / "traces").glob("*.jsonl.gz")} != seen_paths:
        raise RecoveryError("Unindexed trace shard present")
    return records, dict(totals)


def reconstruct_records(archived, fixtures, original_lines):
    """Use fixture metadata only after exact event equality for every world.

    An identical fixture supplies metadata of an already executed deterministic
    trace. It never supplies a missing trace or an additional observation.
    """
    pilots, indices = ("P5", "P6", "P7"), range(32, 64)
    replicas = CONTRACTS.pilots["shared_contract"]["clock"]["replicated_cells"]
    expected, actual = {}, {}
    for pilot in pilots:
        for index in indices:
            for replica in range(replicas):
                for context in ("kappa1", "kappa2"):
                    for arm in ("source", "sham"):
                        for event in fixtures[pilot]["contexts"][context][arm]["trace"]:
                            key = (pilot, index, replica, context, arm, event["epoch"])
                            if key in expected:
                                raise RecoveryError("Duplicate epoch in retained fixture")
                            expected[key] = canonical_bytes(event)
    expected_keys = {"record_type", "pilot", "bundle_index", "role", "replicate_id",
                     "context", "diagnostic_arm", "event", *FLAGS}
    for record in archived:
        if (set(record) != expected_keys or record["role"] != "calibration" or
                record["record_type"] != "BUNDLE_SOURCE_DIAGNOSTIC" or
                any(record[k] is not value for k, value in FLAGS.items())):
            raise RecoveryError("Diagnostic envelope metadata mismatch")
        key = (record["pilot"], record["bundle_index"], record["replicate_id"],
               record["context"], record["diagnostic_arm"], record["event"]["epoch"])
        if key in actual:
            raise RecoveryError("Duplicated archived diagnostic event")
        if key not in expected or canonical_bytes(record["event"]) != expected[key]:
            raise RecoveryError("Archived event does not exactly equal retained deterministic fixture")
        actual[key] = record["event"]
    if set(actual) != set(expected):
        raise RecoveryError("Incomplete archived diagnostic events")
    recovered = []
    for index in indices:
        for pilot in pilots:
            reps = []
            for replica in range(replicas):
                result = deepcopy(fixtures[pilot])
                for context in ("kappa1", "kappa2"):
                    for arm in ("source", "sham"):
                        result["contexts"][context][arm]["trace"] = [
                            actual[(pilot, index, replica, context, arm, event["epoch"])]
                            for event in result["contexts"][context][arm]["trace"]]
                reps.append({"replicate_id": replica, "result": result})
            outcomes = {context: sum(r["result"]["contexts"][context]["effect"] for r in reps) / len(reps)
                        for context in ("kappa1", "kappa2")}
            recovered.append({"pilot": pilot, "bundle_index": index, "role": "calibration",
                "replicates": reps, "executed_new_clone_worlds": True,
                "mechanism.kappa1": outcomes["kappa1"], "mechanism.kappa2": outcomes["kappa2"],
                "context.interaction": outcomes["kappa1"] - outcomes["kappa2"],
                "task_law_pass": all(r["result"]["task_law_pass"] for r in reps),
                "mediator_contract_verified": all(r["result"]["mediator_contract_verified"] for r in reps),
                "scope": SCOPE, **FLAGS})
    by_key = {(r["pilot"], r["bundle_index"]): canonical_bytes(r) + b"\n" for r in recovered}
    seen = set()
    for original in original_lines:
        row = strict_json(original.decode("utf-8"))
        key = (row["pilot"], row["bundle_index"])
        if key in seen or key not in by_key:
            raise RecoveryError("Duplicated or unexpected original diagnostic row")
        seen.add(key)
        if original != by_key[key]:
            raise RecoveryError("Recovered row differs byte-for-byte from retained original row")
    return recovered, {"archive_diagnostic_events_verified": len(actual),
        "total_diagnostic_bundle_rows": len(recovered), "original_rows_byte_identical": len(seen),
        "recovered_derived_rows": len(recovered) - len(seen),
        "derived_row_keys": [[r["pilot"], r["bundle_index"]] for r in recovered
                             if (r["pilot"], r["bundle_index"]) not in seen],
        "new_diagnostic_executions": 0, "new_independent_observations": 0,
        "fixture_metadata_use_condition": "Exact canonical equality of every archived event in every replica/context/arm before metadata reuse",
        "executed_new_clone_worlds_semantics": "Original execution metadata preserved; this postprocessor executed zero worlds"}


def recover_diagnostics(root, technical, input_verification):
    directory = root / RESULTS_DIR / "calibration"
    fixtures = read_json(directory / "mechanism_diagnostics.json")
    original = directory / "bundle_diagnostics.jsonl"
    archived, archive_totals = collect_archived_diagnostics(directory, technical)
    recovered, validation = reconstruct_records(archived, fixtures, original.read_bytes().splitlines(keepends=True))
    if (validation["original_rows_byte_identical"] != 42 or
            validation["recovered_derived_rows"] != 54 or
            validation["archive_diagnostic_events_verified"] != 2304):
        raise RecoveryError("Inputs do not match the documented 42/54-row incident")
    data = b"".join(canonical_bytes(row) + b"\n" for row in recovered)
    incident = {"version": VERSION, "status": "DERIVED_REPORTING_SIDECAR_RECOVERED",
        "recorded_utc": utc_now(), "original_export_status": "FAILED_RETAINED_UNCHANGED",
        "original_export_invocation": "python -B -m b1.run export && python -B -m b1.evaluation.reporting && python -B -m b1.validate_b1 --help",
        "original_export_exit_code": 2,
        "original_export_tool_output": {"status": "BLOCKED", "stage": "export",
            "error": "StageError: Missing or duplicated actual calibration bundle diagnostics", **FLAGS},
        "original_export_transcript_provenance": "Root agent supplied exact exec_command tool stdout and exit code; no on-disk transcript existed; following shell && commands did not run",
        "cause": "Original summary JSONL has 42 rows through bundle 45; immutable gzip archive contains all 96 bundles through 63. Why the original stream retained only 42 rows is not established.",
        "source_input": {"path": original.relative_to(root).as_posix(), "sha256": digest(original)},
        "fixture": {"path": "development_results/calibration/mechanism_diagnostics.json",
                    "sha256": digest(directory / "mechanism_diagnostics.json")},
        "derived_sidecar": {"path": SIDECAR, "sha256": sha256(data).hexdigest()},
        "technical_close_sha256": digest(root / "CALIBRATION_TECHNICAL.json"),
        "archive_totals_verified": archive_totals, "validation": validation,
        "input_verification": input_verification,
        "original_export_repaired_or_declared_successful": False,
        "scientific_code_changed_by_postprocessor": False, "frozen_artifacts_changed": False,
        "new_experiments": 0, "new_seeds": 0, "retuning": False,
        "evaluator": {name: digest(root / name) for name in (
            "evaluation/decision.py", "evaluation/precision.py", "run.py")},
        "reporting_changes": "Standalone adaptation of frozen run.export reporting formulas; corrected diagnostics input path, explicit incident provenance, no change to evaluator or thresholds",
        "limitations": ["Reporting inventory was added after calibration closure and was never part of the experimental freeze.",
                        "Legacy run.export remains failed and its original inventory gate is not satisfied by the additions.",
                        "Derived records recover a redundant representation of archived observations, not a calibration rerun or new evidence.",
                        "Deterministic imposed-law diagnostic repeats are not independent empirical confirmation.",
                        "P5 architecture comparison and resource certificates remain blocked; no B1-E authorization."],
        **FLAGS}
    write_bytes_new(root / SIDECAR, data)
    write_json(root / INCIDENT, incident)
    write_bytes_new(root / "postprocessing/EXPORT_RECOVERY_INCIDENT.md", (
        "# Closed calibration export incident\n\n"
        "The original `python -B -m b1.run export` failed with exit code 2: "
        "`StageError: Missing or duplicated actual calibration bundle diagnostics`. "
        "Its failure is preserved; the original exporter was not repaired or rerun.\n\n"
        f"The immutable summary contains {validation['original_rows_byte_identical']} rows through bundle 45. "
        f"All {archive_totals['shards']} compressed shards were hash checked and their record/byte counts verified. "
        f"They contain {validation['archive_diagnostic_events_verified']} diagnostic events for all 96 pilot/bundle records. "
        "The cause of the incomplete summary stream is unknown.\n\n"
        "A separate postclosure reporting module verifies every archived diagnostic event against the retained "
        "deterministic fixture before using its metadata. It rebuilds the redundant JSONL representation, "
        "proves byte-for-byte equality of every one of the original 42 rows, and derives the other 54 records. "
        "The sidecar retains original execution metadata: `executed_new_clone_worlds=true` refers to the "
        "already archived calibration executions. This postprocessor runs zero worlds and opens zero seeds.\n\n"
        "All original source, raw data, snapshots, and technical-closure bytes remain unchanged. "
        "New reporting Python files are outside the experimental freeze; consequently the original "
        "runner inventory gate is not satisfied by this additive tree. The independent reporting verifier "
        "checks the frozen inventory's bytes and separately identifies the reporting additions. "
        "It supplies no permission to resume or rerun any experimental stage.\n\n"
        "The standalone export uses the frozen decision evaluator, precision calculation, endpoints, thresholds "
        "and descriptive formulas. Only the identified diagnostics input sidecar and reporting provenance differ. "
        "The result remains B1D_BLOCKED: P5 architecture comparison and resource certification are unresolved. "
        "Recovered diagnostics add no independent empirical evidence and allow no O3/O4 promotion or B1-E run.\n\n"
        "See `EXPORT_RECOVERY_INCIDENT.json` for hashes, original tool-output provenance, derived-row keys, "
        "and the exact checks.\n").encode("utf-8"))
    return recovered, incident


def actual_resources(root):
    disk = shutil.disk_usage(root)
    cpu_quota = None
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.is_file():
        fields = path.read_text().split()
        if len(fields) == 2 and fields[0] != "max":
            cpu_quota = int(fields[0]) / int(fields[1])
    return {"measured_utc": utc_now(), "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free, "logical_cpu_count": os.cpu_count(),
            "cgroup_cpu_quota": cpu_quota, "future_cpu_time_budget": None,
            "future_budget_authorized": False}


def validate_trials(rows, technical):
    expected = set()
    for pilot in ("P6", "P7"):
        for index in range(32, 64):
            for comparator in ("C0", "C1", "C2", "C3", "C4", "C5", "C6"):
                for cycle in ((1, 2) if comparator == "C2" else (1,)):
                    expected.add((pilot, index, comparator, cycle, None, False))
            for context in ("kappa1", "kappa2"):
                for lesion in (False, True):
                    expected.add((pilot, index, "C1", 1, context, lesion))
    keys = [(r["pilot"], r["bundle_index"], r["comparator"], r["cycle"],
             r["acute_context"], r["acute_lesion"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RecoveryError("Incomplete or duplicated closed calibration trial schedule")
    if len(rows) != technical["trial_count"] or len(rows) != technical["expected_trial_count"]:
        raise RecoveryError("Closed trial count differs from technical closure")
    counts = Counter()
    for row in rows:
        if any(row[k] is not value for k, value in FLAGS.items()):
            raise RecoveryError("Closed trial evidence flags changed")
        counts.update(row["guardrails"])
    if dict(counts) != technical["guardrails"]:
        raise RecoveryError("Trial guardrails differ from immutable technical closure")
    if sum(r["controller_failure"] for r in rows) != technical["failed_attempts"]:
        raise RecoveryError("Retained trial failures differ from technical closure")
    return len(rows)


def export_recovered(root=ROOT):
    root = Path(root)
    input_verification = verify_reporting_inputs(root)
    path = root / "CALIBRATION_TECHNICAL.json"
    if not path.is_file():
        raise RecoveryError("Technical calibration must close before descriptive export or unblinding")
    technical = read_json(path)
    CONFLICT_ID = technical["P5_conflict_id"]
    if technical.get("status") != "TECHNICAL_CALIBRATION_CLOSED":
        raise RecoveryError("Calibration did not close")
    verify_snapshot(root, technical["artifact_sha256"])
    directory = root / RESULTS_DIR / "calibration"
    identity_record = read_json(directory / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json")
    if identity_record.get("technical_close_sha256") != digest(path):
        raise RecoveryError("Identity disclosure is not linked to the immutable technical close")
    identity_map = identity_record["identities"]
    rows = []
    with (directory / "masked_trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = strict_json(line)
            row.update(identity_map[row["arm_id"]])
            rows.append(row)
    validate_trials(rows, technical)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pilot"], row["comparator"], row["cycle"], row["acute_context"], row["acute_lesion"])].append(row)
    summaries = [{"pilot": key[0], "comparator": key[1], "cycle": key[2], "acute_context": key[3], "acute_lesion": key[4],
                  "N": len(group), "mean_development_loss": sum(r["loss"] for r in group) / len(group),
                  "failed_attempts": sum(r["controller_failure"] for r in group), **FLAGS}
                 for key, group in sorted(grouped.items(), key=lambda pair: str(pair[0]))]
    paired = {}
    for pilot in ("P6", "P7"):
        ordinary = [r for r in rows if r["pilot"] == pilot and r["acute_context"] is None]
        values = {(r["bundle_index"], r["comparator"], r["cycle"]): r["loss"] for r in ordinary}
        paired[pilot] = {}
        for comparator in ("C0", "C2", "C3", "C4"):
            differences = [values[(i, "C1", 1)] - (sum(values[(i, "C2", cycle)] for cycle in (1, 2)) / 2
                           if comparator == "C2" else values[(i, comparator, 1)]) for i in range(32, 64)]
            paired[pilot][f"C1_minus_{comparator}"] = {"mean": sum(differences) / len(differences),
                                                     "bundle_values": differences, "favorable_direction": "negative", **FLAGS}
        for context in ("kappa1", "kappa2"):
            acute = {(r["bundle_index"], r["acute_lesion"]): r["loss"] for r in rows
                     if r["pilot"] == pilot and r["acute_context"] == context}
            differences = [acute[(i, False)] - acute[(i, True)] for i in range(32, 64)]
            paired[pilot][f"acute_route.{context}"] = {"mean": sum(differences) / len(differences),
                "formula": "L_intact_minus_L_bypass", "bundle_values": differences, **FLAGS}
    diagnostic_fixtures = read_json(directory / "mechanism_diagnostics.json")
    bundle_diagnostics, incident = recover_diagnostics(root, technical, input_verification)
    expected_diagnostics = {(pilot, index) for pilot in ("P5", "P6", "P7") for index in range(32, 64)}
    observed_diagnostics = [(r["pilot"], r["bundle_index"]) for r in bundle_diagnostics]
    if len(observed_diagnostics) != len(expected_diagnostics) or set(observed_diagnostics) != expected_diagnostics:
        raise RecoveryError("Missing or duplicated actual calibration bundle diagnostics")
    registry_evaluation = {}
    for pilot in ("P5", "P6", "P7"):
        actual_diagnostics = [r for r in bundle_diagnostics if r["pilot"] == pilot]
        endpoints = {endpoint: [{"bundle_index": r["bundle_index"], "value": r[endpoint]}
                                 for r in actual_diagnostics]
                     for endpoint in ("mechanism.kappa1", "mechanism.kappa2", "context.interaction")}
        if pilot in paired:
            for comparator in ("C0", "C2", "C3", "C4"):
                endpoints[f"utility.{comparator}"] = [{"bundle_index": i, "value": value}
                    for i, value in zip(range(32, 64), paired[pilot][f"C1_minus_{comparator}"]["bundle_values"])]
            for context in ("kappa1", "kappa2"):
                endpoints[f"acute_route.{context}"] = [{"bundle_index": i, "value": value}
                    for i, value in zip(range(32, 64), paired[pilot][f"acute_route.{context}"]["bundle_values"])]
        pilot_counts = Counter()
        for row in rows:
            if row["pilot"] == pilot:
                pilot_counts.update(row["guardrails"])
        for record in actual_diagnostics:
            for replicate in record["replicates"]:
                for context in replicate["result"]["contexts"].values():
                    for arm in ("source", "sham"):
                        for event in context[arm]["trace"]:
                            pilot_counts.update(event["guardrails"])
        diag = diagnostic_fixtures[pilot]
        c6_rows = [r for r in technical["coordinate_invariance_checks"] if r["pilot"] == pilot]
        instrumentation = {
            "guardrail_counts": dict(pilot_counts),
            "invariants": {},  # No substitute of declared allocations for a measured certificate.
            "manipulation_changed_axes": [],
            "sensitivity_checks": {context: {"observed": diag["contexts"][context]["effect"],
                                              "expected": diag["contexts"][context]["expected_imposed_effect"]}
                                   for context in ("kappa1", "kappa2")},
            "c6_checks": {str(row["bundle_index"]): {
                "observed": row["loss_equal_exactly"] and row["event_trace_equal_exactly"] and row["controller_failures_absent"],
                "expected": True} for row in c6_rows},
            "mediator_checks": {"delivery": {"observed": diag["mediator_contract_verified"], "expected": True}},
            "blocked_reasons": ([CONFLICT_ID] if pilot == "P5" else []) +
                ["six_axis_resource_certificate_pending"],
        }
        registry_evaluation[pilot] = evaluate_development(pilot, endpoints, instrumentation, contracts=CONTRACTS)
        registry_evaluation[pilot]["instrument_diagnostics"] = {
            "data_class": "deterministic_fixture", "independent_bundle_sample": False,
            "fixture_file": f"{RESULTS_DIR}/calibration/mechanism_diagnostics.json",
            "fixture_file_sha256": digest(directory / "mechanism_diagnostics.json"),
            "fixture": diag,
            "registry_bundle_diagnostics_file": SIDECAR,
            "registry_bundle_diagnostics_sha256": digest(root / SIDECAR),
            "postprocessing_incident": INCIDENT,
            "original_summary_row_count": 42, "derived_summary_row_count": 54,
            "registry_observations": "Closed archived events cover every one of the 32 calibration bundles per pilot and three local worlds per bundle. The separate sidecar reconstructs 54 missing reporting rows after exact event equality and byte-identical validation of the 42 retained rows. No diagnostic executions or seeds were opened by postprocessing. Repeated imposed deterministic laws are instrument observations, not independent empirical confirmation.", **FLAGS}
    write_json(root / "B1D_REGISTRY_EVALUATION.json", {"registered_endpoint_count": 27,
                "pilot_evaluations": registry_evaluation, **FLAGS})
    precision = precision_plan(CONTRACTS)
    resources = actual_resources(root)
    n = precision["B1E_required_N"]
    trace_bytes = technical["trace_index"]["compressed_bytes"]
    projected_seconds = technical["wall_seconds"] / 32 * n
    projected_bytes = trace_bytes / 32 * n
    projection = {"precision_plan": precision, "available_resources": resources,
                  "observed_block_bundles": 32, "projected_N": n,
                  "serial_wall_seconds_partial_schedule": projected_seconds,
                  "compressed_trace_bytes_partial_schedule": projected_bytes,
                  "observed_disk_insufficient_for_partial_schedule": projected_bytes > resources["disk_free_bytes"],
                  "projection_basis": "Observed P6/P7 12-arm episode schedule plus P5 two task-law witnesses and fresh source diagnostics for every pilot/bundle; does not supply missing P5 architecture runs",
                  "limits": "Linear extrapolation of this host's measured wall time and gzip bytes; not a hardware-independent bound, CPU reservation or complete protocol certification",
                  "budget_compatible": False, "full_protocol_resources_certified": False,
                  "infeasible_action": "Block B1-E; never reduce N and proceed", **FLAGS}
    write_json(root / "B1E_RESOURCE_PROJECTION.json", projection)
    blockers = [{"id": CONFLICT_ID, "reason": "Canonical P5 C4 refill actions conflict; neither labeled witness is silently selected"},
                {"id": "B1-RESOURCE-CERTIFICATE-PENDING", "reason": "Smallest common memory/operation cap, worst-input latency and complete protocol budget remain uncertified"}]
    blockers.append({"id": "B1D-ORIGINAL-EXPORT-PROVENANCE-INCIDENT", "reason": "Original diagnostic summary remains incomplete and original export failed; derived reporting recovery is documented separately without repairing the frozen execution"})
    if projection["observed_disk_insufficient_for_partial_schedule"]:
        blockers.append({"id": "B1E-RESOURCE-INFEASIBLE-CURRENT-DISK", "reason": "Projected partial-schedule compressed traces exceed currently available disk"})
    if not technical["pipeline_technical_pass"]:
        blockers.append({"id": "B1D-TECHNICAL-CALIBRATION-FAILED", "reason": "See exact retained technical failures; no failed attempt was filtered"})
    results = {"postprocessing": {"version": VERSION, "incident_file": INCIDENT, "incident_sha256": digest(root / INCIDENT),
                                "derived_sidecar_file": SIDECAR, "derived_sidecar_sha256": digest(root / SIDECAR),
                                "original_export_status": "FAILED_RETAINED_UNCHANGED", "new_experiments": 0, "new_seeds": 0},
               "status": "B1D_BLOCKED", "created_utc": utc_now(), "unresolved_blockers": blockers,
               "tuning": read_json(root / "TUNING_LEDGER.json"),
               "calibration_technical": technical, "calibration_summaries": summaries,
               "paired_development_differences": paired,
               "registry_evaluation": registry_evaluation,
               "mechanism_diagnostics": diagnostic_fixtures,
               "P5": {"architecture_comparison": "BLOCKED", "conflict_id": CONFLICT_ID,
                      "witness_results": f"{RESULTS_DIR}/calibration/P5_ceiling_witnesses.json",
                      "no_C4_selected_from_scores": True},
               "resource_projection_file": "B1E_RESOURCE_PROJECTION.json",
               "interpretation": "Development descriptions only. Imposed source laws do not validate pairing; absence of a route-lesion effect does not establish necessity. B0 O2 levels are not automatically promoted.",
               "unblinded_after_technical_close": True,
               "technical_close_sha256": digest(path), "retuning_after_unblinding_permitted": False,
               "ready_for_b1e_freeze": False, "ready_for_b1e_confirmatory_run": False,
               "final_seeds_generated": False, "B1E_executed": False, **FLAGS}
    write_json(root / "B1D_RESULTS.json", results)
    lines = ["# B1-D development results — derived reporting recovery", "",
             "The original exporter failed because its immutable summary retained 42 of 96 diagnostic rows. This separate postclosure exporter reconstructed the redundant sidecar from 2,304 hash-verified archived events: 42 rows match the original byte-for-byte and 54 are derived reporting records. No new worlds, trials or seeds were opened. See `postprocessing/EXPORT_RECOVERY_INCIDENT.md`. The original failure and frozen artifacts remain unchanged.", "", "Status: **B1D_BLOCKED**. All observations are development-only, nonconfirmatory and forbidden for reuse as final observations.", "",
             "The implementation and configuration grid were content-frozen before tuning. All registered attempts, including failures, were retained. The technical calibration block opened only after a verified tuning/configuration freeze; descriptive identities were disclosed only after its technical summary closed.", "",
             "| Pilot | Comparator | Cycle | Context | Route bypass | Bundles | Mean development loss | Failed attempts |",
             "|---|---|---:|---|---|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['pilot']} | {row['comparator']} | {row['cycle']} | {row['acute_context'] or 'ordinary'} | {row['acute_lesion']} | {row['N']} | {row['mean_development_loss']:.8f} | {row['failed_attempts']} |")
    lines += ["", "C2 cycles remain separate in this table; paired comparisons average both equally. C5 is descriptive. C6 is an exact re-encoding check, not an independent trained competitor.", "",
              f"Canonical precision planning requires **N = {n:,}** independent bundles. The measured partial schedule projects approximately **{projected_seconds / 3600:,.2f} serial hours** and **{projected_bytes / 10**9:,.2f} GB** of compressed traces. These are host-specific linear extrapolations, not an approved complete-run budget. P5 architecture execution remains missing.", "",
              "Unresolved blockers:", ""]
    lines += [f"- `{b['id']}`: {b['reason']}." for b in blockers]
    lines += ["", "No final seeds were generated, no B1-E run was performed, and no historical results were modified by this runner. Source/sham diagnostics check imposed task laws; they do not supply independent empirical support or prove pairing utility.", ""]
    write_bytes_new(root / "B1D_RESULTS.md", "\n".join(lines).encode("utf-8"))
    verify_reporting_inputs(root)
    outputs = [SIDECAR, INCIDENT, "postprocessing/EXPORT_RECOVERY_INCIDENT.md",
               "B1D_REGISTRY_EVALUATION.json", "B1E_RESOURCE_PROJECTION.json", "B1D_RESULTS.json", "B1D_RESULTS.md"]
    write_json(root / "postprocessing/REPORT_OUTPUT_MANIFEST.json", {
        "version": VERSION, "output_sha256": {name: digest(root / name) for name in outputs},
        "source_sha256": {name: digest(root / name) for name in input_verification["postclosure_reporting_python_additions"]},
        "new_experiments": 0, "new_seeds": 0, **FLAGS})
    return {"stage": "export", "status": results["status"], "unresolved_blockers": blockers,
            "B1E_required_N": n, "ready_for_b1e_confirmatory_run": False, **FLAGS}


def verify_only(root=ROOT):
    """Read-only verification of frozen inputs, derivation, and saved exports."""
    root = Path(root)
    verification = verify_reporting_inputs(root)
    manifest = read_json(root / "postprocessing/REPORT_OUTPUT_MANIFEST.json")
    if manifest.get("version") != VERSION:
        raise RecoveryError("Unknown reporting manifest version")
    verify_snapshot(root, manifest["output_sha256"])
    verify_snapshot(root, manifest["source_sha256"])
    if manifest["source_sha256"] != verification["postclosure_reporting_python_additions"]:
        raise RecoveryError("Reporting source inventory differs from report manifest")
    incident = read_json(root / INCIDENT)
    technical = read_json(root / "CALIBRATION_TECHNICAL.json")
    directory = root / RESULTS_DIR / "calibration"
    archived, totals = collect_archived_diagnostics(directory, technical)
    recovered, validation = reconstruct_records(archived,
        read_json(directory / "mechanism_diagnostics.json"),
        (directory / "bundle_diagnostics.jsonl").read_bytes().splitlines(keepends=True))
    expected_data = b"".join(canonical_bytes(row) + b"\n" for row in recovered)
    if (root / SIDECAR).read_bytes() != expected_data:
        raise RecoveryError("Saved derived sidecar differs from closed trace reconstruction")
    if incident["validation"] != validation or incident["archive_totals_verified"] != totals:
        raise RecoveryError("Incident derivation counts or archive totals changed")
    if incident["technical_close_sha256"] != digest(root / "CALIBRATION_TECHNICAL.json"):
        raise RecoveryError("Incident not linked to original technical closure")
    if incident["input_verification"] != verification:
        raise RecoveryError("Incident frozen-source provenance differs")
    result = read_json(root / "B1D_RESULTS.json")
    if (result["status"] != "B1D_BLOCKED" or result["ready_for_b1e_confirmatory_run"] is not False or
            result["postprocessing"]["incident_sha256"] != digest(root / INCIDENT) or
            result["postprocessing"]["derived_sidecar_sha256"] != digest(root / SIDECAR)):
        raise RecoveryError("Saved results lost blocked status or recovery provenance")
    return {"status": "PASS", "verification": "read_only_closed_trace_reporting",
            "experimental_status": result["status"], "frozen_artifacts_unchanged": True,
            **validation, "new_experiments": 0, "new_seeds": 0,
            "files_written": 0, **FLAGS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true", help="Read and verify only; never regenerate reports or resource measurements")
    args = parser.parse_args()
    try:
        result = verify_only(args.root) if args.verify_only else export_recovered(args.root)
    except (RecoveryError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reporting_version": VERSION,
                          "error": f"{type(exc).__name__}: {exc}", **FLAGS}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
