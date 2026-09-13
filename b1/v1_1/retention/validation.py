"""Complete retention IDs and causal schema; no task execution or seed draws.

Archived records use the exporter's frozen identity adapter. New v1.1 trial
records use their explicit comparator/context/route/cycle/cell/epoch fields.
An identity map is mandatory for blinded historical architecture trials.
"""
from b1.v1_1.exporter.core import DIMENSIONS, FLAGS, historic_key, validate_event

REQUIRED_ROUTE = {
    "route_slots", "route_consumed", "route_use_masked",
    "nominated_route_bypassed", "joint_manipulation",
    "raw_information_preserved", "planning_horizon", "acute_lesion",
    "useful_operations", "decision_cap", "memory_scalars", "memory_cap",
}


def validate_key(key):
    key = tuple(key)
    if len(key) != len(DIMENSIONS):
        raise ValueError("Incomplete retention event ID")
    for i, dimension in enumerate(DIMENSIONS):
        if dimension in ("bundle_index", "cycle", "replicate_id", "epoch"):
            if type(key[i]) is not int or key[i] < 0:
                raise ValueError("Invalid integer retention ID dimension")
        elif type(key[i]) is not str or not key[i]:
            raise ValueError("Invalid string retention ID dimension")
    if key[1] not in ("P5", "P6", "P7") or key[-2] not in range(3) or key[-1] not in range(32):
        raise ValueError("Retention ID outside frozen task domain")
    return key


def record_key(record, identities=None):
    if not isinstance(record, dict) or not isinstance(record.get("event"), dict):
        raise ValueError("Complete retention record required")
    if "namespace" in record:
        if record["namespace"] not in ("PD-B1-D-v1.1", "PD-B1-D-v1.1-QA"):
            raise ValueError("Unapproved record namespace")
        required = {"pilot", "bundle_index", "comparator", "context", "route_status",
                    "cycle", "cell_id", "epoch", "source_commit", "config_sha256"}
        if not required <= record.keys():
            raise ValueError("Incomplete v1.1 retention identity")
        validate_event(record["event"], record["pilot"], record["epoch"])
        if record["cell_id"] != record["event"]["cell_id"]:
            raise ValueError("Event cell/envelope mismatch")
        key = ("trial", record["pilot"], record["bundle_index"], record["comparator"],
               record["context"], record["route_status"], record["cycle"],
               record["cell_id"], record["epoch"])
    else:
        key = historic_key(record, identities or {})
    key = validate_key(key)
    for name, value in FLAGS.items():
        if name == "reusable_for_B1E":
            continue
        # Frozen blinded trial envelopes have no flags; the shared event schema
        # always verifies them. Diagnostic, witness and new envelopes have them.
        if "arm_id" not in record or name in record:
            if record.get(name) is not value:
                raise ValueError("Retention record development flags invalid")
    if key[0] == "trial":
        report = record.get("controller_report")
        # The separately frozen P5 fixed conventional adapter has no mask or
        # route to mask. Preserve its exact report without inventing that field.
        fixed_p5 = ("namespace" in record and record["pilot"] == "P5"
                    and record["comparator"] in ("C4", "C4_FULL_REFILL_CEILING_WITNESS"))
        required_route = REQUIRED_ROUTE - {"route_use_masked"} if fixed_p5 else REQUIRED_ROUTE
        if not isinstance(report, dict) or not required_route <= report.keys():
            raise ValueError("Incomplete causal route report")
        if fixed_p5 and (report["route_slots"] != [] or report["route_consumed"] != []
                         or report["nominated_route_bypassed"] is not True):
            raise ValueError("Fixed P5 conventional controller cannot contain a route")
        for name in ("route_slots", "route_consumed"):
            if not isinstance(report[name], list):
                raise ValueError("Invalid causal route payload")
        for name in ("route_use_masked", "nominated_route_bypassed", "joint_manipulation",
                     "raw_information_preserved", "acute_lesion"):
            if fixed_p5 and name == "route_use_masked" and name not in report:
                continue
            if type(report[name]) is not bool:
                raise ValueError("Invalid causal route flag")
        for name in ("planning_horizon", "useful_operations", "decision_cap", "memory_scalars", "memory_cap"):
            if type(report[name]) is not int or report[name] < 0:
                raise ValueError("Invalid causal resource counter")
    if type(record["event"].get("controller_failure")) is not bool:
        raise ValueError("Controller failure flag required")
    return key
