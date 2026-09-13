"""A/B/C retention decisions fixed independently of scientific outcomes.

Level B retains every external task event, including both mediator states, and
the complete route payload/consumption report. Only verbose internal memories
and unrelated reporting metadata are omitted. It is not a replacement for the
full record in the mandatory Level C cases.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import shutil

AUDIT_MODULUS = 1024
GUARDRAILS = (
    "unauthorized_executions", "invalid_concurrent_writes",
    "executed_infeasible_actions", "physical_state_violations",
    "undeclared_budget_overruns",
)
ROUTE_FIELDS = (
    "route_slots", "route_consumed", "route_use_masked",
    "nominated_route_bypassed", "joint_manipulation",
    "raw_information_preserved", "planning_horizon", "acute_lesion",
    "useful_operations", "decision_cap", "memory_scalars", "memory_cap",
    "failed_decision", "failure_operations_available",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def causal_record(record):
    """Project a record without losing any external event or route evidence."""
    if not isinstance(record, dict) or "event" not in record:
        raise ValueError("A causal record needs the complete task event")
    event = record["event"]
    required = {"channels", "state_before", "state_after", "guardrails",
                "controller_failure", "cell_id", "epoch", "pilot"}
    if not required.issubset(event):
        raise ValueError("Incomplete causal event")
    for channel in ("A", "B"):
        if not {"requested_activation", "admitted_activation", "executed_operation",
                "environmental_effect"}.issubset(event["channels"][channel]):
            raise ValueError("Missing requested/admitted/executed/effect fields")
    # Preserve all top-level experiment identity/context fields as well as the
    # complete integrated-content record. Never infer an omitted dimension.
    out = {key: deepcopy(value) for key, value in record.items()
           if key not in ("internal_memory", "controller_report")}
    if "controller_report" in record:
        out["controller_report"] = {
            key: deepcopy(value) for key, value in record["controller_report"].items()
            if key in ROUTE_FIELDS
        }
    out["retention_level"] = "B"
    return out


def full_trace_required(bundle_index, *, failed=False, guardrails=None,
                        deterministic_diagnostic=False):
    if type(bundle_index) is not int or bundle_index < 0:
        raise ValueError("An audit index is required before observing outcomes")
    return (bundle_index % AUDIT_MODULUS == 0 or failed
            or any((guardrails or {}).values()) or deterministic_diagnostic)


def verify_causal_projection(original, projected):
    return canonical(causal_record(original)) == canonical(projected)


class RetentionBudgetExceeded(RuntimeError):
    """Collection must close as incomplete; never silently drop a trace."""


def storage_gate(directory, *, next_bundle_worst_case_bytes,
                 reserve_bytes, available_bytes=None):
    """Reserve full-trace capacity BEFORE starting the next bundle.

    A failure of this gate ends collection as incomplete. A later resource
    allocation cannot be used to extend sampling on the basis of outcomes.
    No partial fixed-N result can acquire confirmatory status.
    """
    if next_bundle_worst_case_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("A prospective positive bound and reserve are required")
    free = shutil.disk_usage(directory).free if available_bytes is None else available_bytes
    if free < reserve_bytes + next_bundle_worst_case_bytes:
        raise RetentionBudgetExceeded("Insufficient pre-bundle full-trace reserve")
    return True


def encode_records(records):
    raw = b"".join(canonical(row) + b"\n" for row in records)
    return gzip.compress(raw, compresslevel=6, mtime=0)


def decode_records(blob):
    return [json.loads(line) for line in gzip.decompress(blob).splitlines()]


def manifest_entry(name, blob, records):
    return {"path": name, "bytes": len(blob), "records": records,
            "sha256": hashlib.sha256(blob).hexdigest()}
