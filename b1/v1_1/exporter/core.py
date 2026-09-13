"""Version 1.1 exporter: authenticate, validate complete IDs, then publish atomically.

The historical adapter reports retained observations only. The normalized adapter
accepts prospectively declared QA schedules. Neither path can start experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from b1.contracts.loader import strict_json
from b1.postprocessing.export_recovery import reconstruct_records

ROOT = Path(__file__).resolve().parents[2]
VERSION = "b1d-exporter-1.1"
FLAGS = {"development_only": True, "confirmatory": False,
         "reusable_as_final": False, "reusable_for_B1E": False}
DIMENSIONS = ("record_kind", "pilot", "bundle_index", "comparator", "context",
              "route_status", "cycle", "replicate_id", "epoch")
V1_MANIFEST_SHA256 = "e5de2f406cf34ea5ce122443890ac9344e4df68206edd267dd1ff7eaadb7dc2b"
V1_COMMIT = "c27235deb1688c105bfe96597dfd233108307895"


class ExportError(RuntimeError):
    """Incomplete, malformed or unauthenticated input prevents publication."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return strict_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ExportError(f"Invalid JSON input: {path}: {exc}") from exc


def document(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def verify_v1(root=ROOT):
    """Pin the old manifest, then verify its 480 paths; additions are separate."""
    root = Path(root)
    if digest(root / "B1D_MANIFEST.json") != V1_MANIFEST_SHA256:
        raise ExportError("Historical B1-D v1 manifest changed")
    manifest = read_json(root / "B1D_MANIFEST.json")
    for name, expected in manifest["artifact_hashes"].items():
        p = root.parent / name
        if not p.is_file() or digest(p) != expected:
            raise ExportError(f"Historical artifact changed: {name}")
    return {"code_commit": V1_COMMIT, "manifest_sha256": V1_MANIFEST_SHA256,
            "artifact_count": len(manifest["artifact_hashes"]), "all_unchanged": True}


def authenticated_records(directory, index_sha256, expected_index=None):
    """Streaming parser: one gzip shard at a time, including all record kinds."""
    directory = Path(directory)
    index_path = directory / "INDEX.json"
    if not index_path.is_file() or digest(index_path) != index_sha256:
        raise ExportError("Archive index SHA256 mismatch")
    index = read_json(index_path)
    if expected_index is not None and index != expected_index:
        raise ExportError("Archive index differs from frozen technical closure")
    names, total = set(), Counter()
    for shard in index["shards"]:
        name = shard["path"]
        if Path(name).name != name or name in names:
            raise ExportError("Duplicate or unsafe shard path")
        names.add(name)
        path = directory / name
        if not path.is_file():
            raise ExportError(f"Missing shard: {name}")
        data = path.read_bytes()
        if len(data) != shard["bytes"] or sha256(data).hexdigest() != shard["sha256"]:
            raise ExportError(f"Shard SHA256/size mismatch: {name}")
        count = 0
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        raise ExportError(f"Incomplete JSONL framing: {name}")
                    count += 1
                    total["raw_bytes"] += len(line)
                    try:
                        record = strict_json(line.decode("utf-8"))
                    except (ValueError, UnicodeError) as exc:
                        raise ExportError(f"Invalid record JSON: {name}") from exc
                    if not isinstance(record, dict):
                        raise ExportError("Record schema requires object")
                    yield record
        except (OSError, EOFError) as exc:
            raise ExportError(f"Invalid gzip: {name}") from exc
        if count != shard["records"]:
            raise ExportError(f"Shard record count mismatch: {name}")
        total.update(records=count, compressed_bytes=len(data))
    if names != {p.name for p in directory.glob("*.jsonl.gz")}:
        raise ExportError("Unindexed shard present")
    for field in ("records", "raw_bytes", "compressed_bytes"):
        if total[field] != index[field]:
            raise ExportError(f"Archive total mismatch: {field}")


def validate_event(event, pilot, epoch):
    required = {"pilot", "cell_id", "epoch", "proposal", "state_before", "state_after",
                "channels", "completed", "missed", "demand", "guardrails",
                "development_only", "confirmatory", "reusable_as_final"}
    if not isinstance(event, dict) or not required <= event.keys():
        raise ExportError("Event schema missing fields")
    if pilot not in ("P5", "P6", "P7") or event["pilot"] != pilot or event["epoch"] != epoch:
        raise ExportError("Event/envelope identity mismatch")
    if type(event["cell_id"]) is not int or event["cell_id"] not in range(3):
        raise ExportError("Event cell schema invalid")
    if type(epoch) is not int or epoch not in range(32):
        raise ExportError("Event epoch schema invalid")
    if not isinstance(event["proposal"], dict):
        raise ExportError("Event proposal schema invalid")
    common_state = {"pilot", "cell_id", "epoch", "demand", "horizon", "context", "permissions"}
    pilot_state = {"P5": {"reserve", "pending_deliveries", "replenishment_tokens"},
                   "P6": {"reports", "pending_assays", "current_routine", "selector_history", "adoption_permission"},
                   "P7": {"instances", "pending_deployments", "required_version", "approved_versions", "snapshot_history"}}
    for label in ("state_before", "state_after"):
        state = event[label]
        if not isinstance(state, dict) or not common_state | pilot_state[pilot] <= state.keys():
            raise ExportError("State/mediator schema incomplete")
        if (state["pilot"] != pilot or state["cell_id"] != event["cell_id"] or state["epoch"] != epoch
                or state["horizon"] != 32 or state["demand"] != 2 or not isinstance(state["context"], dict)):
            raise ExportError("State identity/domain schema invalid")
        permissions = state["permissions"]
        if not isinstance(permissions, dict) or set(permissions) != {"internal", "operation"}:
            raise ExportError("State permissions schema incomplete")
        if any(not isinstance(arm, dict) or set(arm) != {"A", "B"} or
               any(type(v) is not bool for v in arm.values()) for arm in permissions.values()):
            raise ExportError("State permission domain invalid")
        if pilot == "P5" and (type(state["reserve"]) is not int or not 0 <= state["reserve"] <= 8
                              or not isinstance(state["pending_deliveries"], list)):
            raise ExportError("P5 mediator schema invalid")
        if pilot == "P6" and (state["current_routine"] not in ("q0", "q1") or
                              type(state["adoption_permission"]) is not bool or
                              any(not isinstance(state[k], list) for k in ("reports", "pending_assays", "selector_history"))):
            raise ExportError("P6 mediator schema invalid")
        if pilot == "P7":
            if any(not isinstance(state[k], list) for k in ("instances", "pending_deployments", "approved_versions", "snapshot_history")):
                raise ExportError("P7 mediator schema invalid")
            if len(state["instances"]) != 2 or state["required_version"] not in (0, 1):
                raise ExportError("P7 instance domain invalid")
            for instance in state["instances"]:
                if not isinstance(instance, dict) or not {"instance_id", "drift", "target_version", "locked"} <= instance.keys():
                    raise ExportError("P7 instance schema incomplete")
                if (type(instance["instance_id"]) is not int or instance["instance_id"] not in (0, 1)
                        or type(instance["drift"]) is not int or instance["drift"] not in (0, 1)
                        or type(instance["locked"]) is not bool or instance["target_version"] not in (0, 1)):
                    raise ExportError("P7 instance value domain invalid")
    if any(event[k] is not v for k, v in FLAGS.items() if k != "reusable_for_B1E"):
        raise ExportError("Event development flags invalid")
    for name in ("completed", "missed", "demand"):
        if type(event[name]) is not int or event[name] < 0:
            raise ExportError("Event outcome schema invalid")
    if event["completed"] + event["missed"] != event["demand"]:
        raise ExportError("Event demand accounting invalid")
    required_channel = {"requested_activation", "external_requested_dose",
                        "external_admitted_dose", "executed_operation", "environmental_effect"}
    if not isinstance(event["channels"], dict) or set(event["channels"]) != {"A", "B"}:
        raise ExportError("Event channel schema invalid")
    if any(not isinstance(channel, dict) or not required_channel <= channel.keys() for channel in event["channels"].values()):
        raise ExportError("Event requested/admitted/executed/effect schema incomplete")
    effect_keys = {"P5": {"A": {"reserve_withdrawn", "service_completed"},
                          "B": {"delivered", "due_delivery", "enqueued_units", "overflow", "scheduled_delivery_epoch", "sham_tokens_dissipated", "tokens_consumed"}},
                   "P6": {"A": {"arriving_reports", "assay_tokens_consumed", "live_service_directly_completed"},
                          "B": {"routine_id", "correct_jobs", "incorrect_jobs"}},
                   "P7": {"A": {"repair_completion", "sham", "sham_locked_operations", "target_version_changed"},
                          "B": {"deployment_completions", "rollback_events", "snapshot_created"}}}
    for pole, channel in event["channels"].items():
        if not isinstance(channel["environmental_effect"], dict):
            raise ExportError("Environmental effect schema invalid")
        if not effect_keys[pilot][pole] <= channel["environmental_effect"].keys():
            raise ExportError("Environmental effect/mediator schema incomplete")
        for quantity in ("external_requested_dose", "external_admitted_dose", "executed_operation"):
            if type(channel[quantity]) is not int or channel[quantity] < 0:
                raise ExportError("Channel dose domain invalid")
        if type(channel.get("valid_request")) is not bool:
            raise ExportError("Channel validity flag missing")
        activation = channel["requested_activation"]
        if channel["valid_request"] and (type(activation) not in (int, float) or not math.isfinite(activation) or not 0 <= activation <= 1):
            raise ExportError("Channel activation domain invalid")
    guardrails = {"unauthorized_executions", "invalid_concurrent_writes",
                  "executed_infeasible_actions", "physical_state_violations",
                  "undeclared_budget_overruns"}
    if not isinstance(event["guardrails"], dict) or set(event["guardrails"]) != guardrails:
        raise ExportError("Guardrail schema incomplete")
    if any(type(n) is not int or n < 0 for n in event["guardrails"].values()):
        raise ExportError("Guardrail values invalid")


def normalized(record):
    required = {"schema_version", "namespace", "event", *DIMENSIONS, *FLAGS}
    if set(record) != required or record["schema_version"] != VERSION:
        raise ExportError("Normalized record schema/key dimension mismatch")
    if record["namespace"] not in ("PD-B1-D-v1.1-QA", "PD-B1-D-v1.1"):
        raise ExportError("This normalized adapter is restricted to development namespaces")
    if any(record[k] is not v for k, v in FLAGS.items()):
        raise ExportError("Normalized development flags invalid")
    key = tuple(record[k] for k in DIMENSIONS)
    if any(type(record[k]) is not int for k in ("bundle_index", "cycle", "replicate_id", "epoch")):
        raise ExportError("Integer key dimension invalid")
    if any(type(record[k]) is not str or not record[k] for k in
           ("record_kind", "pilot", "comparator", "context", "route_status")):
        raise ExportError("String key dimension invalid")
    validate_event(record["event"], record["pilot"], record["epoch"])
    if record["event"]["cell_id"] != record["replicate_id"]:
        raise ExportError("Event cell/envelope mismatch")
    return key


def historic_key(record, identities):
    """Explicit NA labels prevent collapsing nonarchitecture observations."""
    if record.get("record_type") == "BUNDLE_SOURCE_DIAGNOSTIC":
        required = {"record_type", "pilot", "bundle_index", "role", "replicate_id",
                    "context", "diagnostic_arm", "event", "development_only",
                    "confirmatory", "reusable_as_final"}
        if set(record) != required or record["role"] != "calibration":
            raise ExportError("Historical diagnostic schema/key dimension mismatch")
        key = ("diagnostic", record["pilot"], record["bundle_index"],
               record["diagnostic_arm"], record["context"], "joint_source_not_selective",
               0, record["replicate_id"], record["event"]["epoch"])
    elif "witness" in record:
        required = {"pilot", "bundle_index", "witness", "event", "development_only",
                    "confirmatory", "reusable_as_final"}
        if set(record) != required:
            raise ExportError("Historical witness schema/key dimension mismatch")
        key = ("witness", record["pilot"], record["bundle_index"], record["witness"],
               "ordinary", "not_applicable", 0, record["event"]["cell_id"], record["event"]["epoch"])
    else:
        required = {"pilot", "bundle_index", "arm_id", "config_id", "cycle", "acute_lesion",
                    "acute_context", "internal_integrated_content", "internal_memory",
                    "controller_report", "event"}
        if set(record) != required or record["arm_id"] not in identities:
            raise ExportError("Historical trial schema/key dimension mismatch")
        identity = identities[record["arm_id"]]
        if (record["config_id"] != record["arm_id"] or
                any(record[k] != identity[k] for k in ("pilot", "cycle", "acute_context", "acute_lesion"))):
            raise ExportError("Historical trial identity map mismatch")
        key = ("trial", record["pilot"], record["bundle_index"], identity["comparator"],
               record["acute_context"] or "ordinary", "bypassed" if record["acute_lesion"] else "intact",
               record["cycle"], record["event"]["cell_id"], record["event"]["epoch"])
    validate_event(record["event"], record["pilot"], key[-1])
    return key


def historic_expected(fixtures):
    expected = set()
    for bundle in range(32, 64):
        for pilot in ("P5", "P6", "P7"):
            for cell in range(3):
                for context in ("kappa1", "kappa2"):
                    for arm in ("source", "sham"):
                        for event in fixtures[pilot]["contexts"][context][arm]["trace"]:
                            expected.add(("diagnostic", pilot, bundle, arm, context,
                                          "joint_source_not_selective", 0, cell, event["epoch"]))
                if pilot == "P5":
                    for witness in ("json_minimal_refill_witness", "protocol_full_refill_witness"):
                        for epoch in range(32):
                            expected.add(("witness", pilot, bundle, witness, "ordinary", "not_applicable", 0, cell, epoch))
                else:
                    schedule = [(c, "ordinary", "intact", cycle) for c in
                                ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
                                for cycle in ((1, 2) if c == "C2" else (1,))]
                    schedule += [("C1", context, route, 1) for context in ("kappa1", "kappa2")
                                 for route in ("intact", "bypassed")]
                    for comparator, context, route, cycle in schedule:
                        for epoch in range(32):
                            expected.add(("trial", pilot, bundle, comparator, context, route, cycle, cell, epoch))
    if len(expected) != 82176:
        raise ExportError("Frozen historical schedule no longer has 82,176 events")
    return expected


def validate_coverage(records, expected, key_function):
    seen, diagnostics = set(), []
    coverage = {k: Counter() for k in DIMENSIONS}
    guardrails = Counter()
    for record in records:
        try:
            key = key_function(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise ExportError(f"Record schema/key dimension invalid: {exc}") from exc
        if key in seen:
            raise ExportError(f"Duplicate complete event ID: {key}")
        if key not in expected:
            raise ExportError(f"Unexpected complete event ID: {key}")
        seen.add(key)
        for dimension, value in zip(DIMENSIONS, key):
            coverage[dimension][str(value)] += 1
        guardrails.update(record["event"]["guardrails"])
        if key[0] == "diagnostic":
            diagnostics.append(record)
    if seen != expected:
        raise ExportError(f"Missing complete event IDs: {len(expected - seen)}")
    return diagnostics, {"expected_records": len(expected), "validated_records": len(seen),
                          "complete_id_dimensions": list(DIMENSIONS),
                          "coverage": {k: dict(v) for k, v in coverage.items()},
                          "guardrails": dict(guardrails), "coverage_pass": True}


def publish_atomic(output, files, manifest):
    """Absent destination required. One directory rename publishes complete set."""
    output = Path(output)
    if output.exists():
        raise ExportError("Refusing to overwrite an existing export destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".export-stage-", dir=output.parent))
    try:
        for name, data in files.items():
            if Path(name).name != name:
                raise ExportError("Unsafe output name")
            with (stage / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        manifest = {**manifest, "artifact_sha256": {n: sha256(d).hexdigest() for n, d in files.items()},
                    "publication": "all validations completed before atomic directory rename",
                    "manifest_self_hash": "MANIFEST.sha256; no self-reference", **FLAGS}
        manifest_data = document(manifest)
        (stage / "MANIFEST.json").write_bytes(manifest_data)
        (stage / "MANIFEST.sha256").write_text(sha256(manifest_data).hexdigest() + "  MANIFEST.json\n")
        os.rename(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def export_historical(output, root=ROOT):
    root = Path(root)
    provenance = verify_v1(root)
    directory = root / "development_results/calibration"
    technical = read_json(root / "CALIBRATION_TECHNICAL.json")
    fixtures = read_json(directory / "mechanism_diagnostics.json")
    identities = read_json(directory / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json")["identities"]
    expected = historic_expected(fixtures)
    index_sha = digest(directory / "traces/INDEX.json")
    records = authenticated_records(directory / "traces", index_sha, technical["trace_index"])
    diagnostic_events, audit = validate_coverage(records, expected, lambda r: historic_key(r, identities))
    original = (directory / "bundle_diagnostics.jsonl").read_bytes().splitlines(keepends=True)
    try:
        rows, reconstruction = reconstruct_records(diagnostic_events, fixtures, original)
    except RuntimeError as exc:
        raise ExportError(f"Historical reconstruction failed: {exc}") from exc
    data = b"".join(canonical(row) + b"\n" for row in rows)
    prior = (root / "postprocessing/bundle_diagnostics.recovered.jsonl").read_bytes()
    if (data != prior or reconstruction["original_rows_byte_identical"] != 42 or
            reconstruction["recovered_derived_rows"] != 54 or len(rows) != 96):
        raise ExportError("96/96 or 42-original/54-sidecar equality failed")
    audit.update({"version": VERSION, "status": "PASS", "historical_provenance": provenance,
                  "diagnostic_rows_expected": 96, "diagnostic_rows_exported": len(rows),
                  "original_rows_byte_identical": 42, "previously_derived_rows_byte_identical": 54,
                  "prior_sidecar_byte_identical": True, "archive_diagnostic_events": len(diagnostic_events),
                  "original_export_status": "FAILED_RETAINED_UNCHANGED", "new_scientific_runs": 0,
                  "new_seeds": 0, "new_independent_observations": 0,
                  "inference_scope": "archived v1 representation only; imposed-law diagnostics; joint source/sham",
                  "index_sha256": index_sha, **FLAGS})
    return publish_atomic(output, {"bundle_diagnostics.jsonl": data, "AUDIT.json": document(audit)},
                          {"version": VERSION, "source": "immutable B1-D v1 archive", "audit": audit})


def export_development(directory, plan_path, plan_sha256, output, *, index_sha256):
    if digest(plan_path) != plan_sha256:
        raise ExportError("QA expected-plan SHA256 mismatch")
    plan = read_json(plan_path)
    if plan.get("namespace") not in ("PD-B1-D-v1.1-QA", "PD-B1-D-v1.1") or plan.get("fixed_before_execution") is not True:
        raise ExportError("Development plan must be fixed before task execution")
    expected = {tuple(key) for key in plan["expected_ids"]}
    if not expected:
        raise ExportError("Empty expected development plan")
    if len(expected) != len(plan["expected_ids"]):
        raise ExportError("Duplicate expected IDs in plan")
    records = authenticated_records(directory, index_sha256)
    def adapter(record):
        if record.get("namespace") != plan["namespace"]:
            raise ExportError("Record namespace differs from declared development plan")
        return normalized(record)
    _, audit = validate_coverage(records, expected, adapter)
    audit.update({"version": VERSION, "namespace": plan["namespace"], "status": "PASS",
                  "qa_only": plan["namespace"] == "PD-B1-D-v1.1-QA",
                  "counts_as_confirmatory_evidence": False,
                  "new_independent_observations_from_export": 0,
                  "expected_plan_sha256": plan_sha256, "index_sha256": index_sha256, **FLAGS})
    return publish_atomic(output, {"AUDIT.json": document(audit)},
                          {"version": VERSION, "source": "development normalized archive", "audit": audit})


def export_qa(directory, plan_path, plan_sha256, output, *, index_sha256):
    if read_json(plan_path).get("namespace") != "PD-B1-D-v1.1-QA":
        raise ExportError("QA adapter requires the QA namespace")
    return export_development(directory, plan_path, plan_sha256, output, index_sha256=index_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("archive", "verify-v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify-v1":
        result = verify_v1()
    else:
        if args.output is None:
            parser.error("archive requires a new --output directory")
        result = export_historical(args.output)["audit"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
