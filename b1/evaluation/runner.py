"""Development-only execution and auditable trace export.

No final seed generator or confirmatory collection path exists in this module.
Tuning choices are locked before effect-blinded technical calibration opens.
"""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random
import time

from b1.contracts import load_contracts
from b1.controllers.core import Controller, ControllerFailure
from b1.metrics import primary_loss
from b1.tasks import P5Task, P6Task, P7Task
from b1.tasks.core import audit_execution, InstrumentViolation

CONTRACTS = load_contracts()


def encode(value):
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else float(value)
    raise TypeError(type(value).__name__)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=encode).encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False, default=encode) + "\n")


class TraceArchive:
    """Deterministic independent gzip shards; all records are retained."""
    def __init__(self, directory, records_per_shard=1000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records_per_shard = records_per_shard
        self.pending, self.shards = [], []
        self.record_count = 0
        self.raw_bytes = 0

    def append(self, record):
        data = canonical_bytes(record) + b"\n"
        self.pending.append(data)
        self.record_count += 1
        self.raw_bytes += len(data)
        if len(self.pending) >= self.records_per_shard:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        raw = b"".join(self.pending)
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        name = f"trace-{len(self.shards):05d}.jsonl.gz"
        (self.directory / name).write_bytes(compressed)
        self.shards.append({"path": name, "records": len(self.pending),
                            "bytes": len(compressed), "sha256": sha256(compressed).hexdigest()})
        self.pending.clear()

    def close(self):
        self.flush()
        out = {"format": "independent_gzip_jsonl_shards", "records": self.record_count,
               "raw_bytes": self.raw_bytes,
               "compressed_bytes": sum(s["bytes"] for s in self.shards),
               "shards": self.shards, "development_only": True,
               "confirmatory": False, "reusable_as_final": False}
        write_json(self.directory / "INDEX.json", out)
        return out


def sample_bundle(pilot, index, seeds, role):
    """Precompute action-independent exogenous substreams; never expose to policy."""
    contracts = CONTRACTS
    clock = contracts.pilots["shared_contract"]["clock"]
    cells = []
    def draw(cell, event, choices, epoch=0):
        sid = f"cell/{cell}/episode/utility/epoch/{epoch}/event/{event}"
        seed = seeds.derive(pilot, role, index, sid)
        return choices[random.Random(seed).randrange(len(choices))]
    for cell in range(clock["replicated_cells"]):
        args = {"cell_id": cell}
        constants = contracts.pilot(pilot)["task_contract"]["constants"]
        if pilot == "P5":
            args["initial_reserve"] = draw(cell, "initial_reserve", (0, constants["reserve_capacity"]))
        elif pilot == "P6":
            args.update(initial_mapping=draw(cell, "initial_mapping", constants["routine_catalogue"]),
                        flip_epoch=draw(cell, "change_schedule", (None, clock["episode_horizon"] // 2)),
                        transfer_eligible=draw(cell, "transferability", (False, True)))
        elif pilot == "P7":
            args.update(initial_drifts=tuple(draw(cell, f"initial_drift/{i}", constants["drift_values"])
                                            for i in range(constants["instances_per_cell"])),
                        mode=draw(cell, "migration_mode", ("state_preserving", "replacement")),
                        version_switch_epoch=draw(cell, "version_schedule", (None, clock["episode_horizon"] // 2)),
                        drift_events={epoch: draw(cell, "drift_event", (None, 0, 1), epoch)
                                      for epoch in (clock["episode_horizon"] // 4,
                                                    3 * clock["episode_horizon"] // 4)})
        else:
            raise ValueError("Unregistered pilot")
        cells.append(args)
    return {"pilot": pilot, "bundle_index": index, "role": role, "cells": cells,
            "development_only": True, "confirmatory": False, "reusable_as_final": False}


def contextual_bundle(bundle, context):
    """Prospective acute-episode moderator clamp, retaining all other histories."""
    out = deepcopy(bundle)
    for args in out["cells"]:
        if out["pilot"] == "P5":
            args["initial_reserve"] = 0 if context == "kappa1" else 8
        elif out["pilot"] == "P6":
            args["transfer_eligible"] = context == "kappa1"
        else:
            args["mode"] = "state_preserving" if context == "kappa1" else "replacement"
    out["acute_context"] = context
    return out


def run_episode(bundle, comparator, config_id="cfg00", *, cycle=1, lesion=False,
                archive=None, opaque_id=None, limits=None):
    if (bundle.get("development_only") is not True or bundle.get("confirmatory") is not False
            or bundle.get("reusable_as_final") is not False):
        raise ValueError("Only explicitly marked development bundles can be executed")
    role = bundle.get("role", "")
    if "final" in str(role).lower() or "B1-E" in str(bundle.get("namespace", "")):
        raise ValueError("Final streams cannot be relabeled as development")
    fixture = bundle.get("synthetic_fixture") is True and role == "deterministic-integration-fixture"
    if not fixture:
        index = bundle.get("bundle_index")
        permitted_roles = ("training", "tuning", "calibration", "evaluation-development")
        if role not in permitted_roles or type(index) is not int:
            raise ValueError("Unregistered development role or bundle index")
        if not ((32 <= index <= 63) if role == "calibration" else (0 <= index <= 31)):
            raise ValueError("Development stages and bundle ranges must remain disjoint")
    pilot = bundle["pilot"]
    task_class = {"P5": P5Task, "P6": P6Task, "P7": P7Task}[pilot]
    tasks = [task_class(**args) for args in bundle["cells"]]
    if comparator == "C6":
        from b1.controllers.c6 import C6Controller
        controller = C6Controller(Controller(pilot, "C1", config_id, limits=limits))
    else:
        controller = Controller(pilot, comparator, config_id, cycle=cycle, lesion=lesion, limits=limits)
    trace_hash = sha256()
    guards = {k: 0 for k in ("unauthorized_executions", "invalid_concurrent_writes",
                            "executed_infeasible_actions", "physical_state_violations",
                            "undeclared_budget_overruns")}
    failures, max_seconds, total_operations, max_memory = [], 0.0, 0, 0
    for epoch in range(tasks[0].horizon):
        observations = [task.observe() for task in tasks]
        started = time.perf_counter()
        controller.last_decision_report = {}
        try:
            actions = controller.act(observations)
            if not isinstance(actions, (list, tuple)) or len(actions) != len(tasks):
                raise ControllerFailure("malformed_action_count")
            allowed = {"A", "B"} | ({"candidate_id", "routine_id", "selector_source"} if pilot == "P6"
                       else {"repair_target", "deploy_target", "target_version"} if pilot == "P7" else set())
            for action in actions:
                if not isinstance(action, dict) or not {"A", "B"}.issubset(action) or set(action) - allowed:
                    raise ControllerFailure("malformed_action_schema")
                if pilot == "P6" and any(k in action and not isinstance(action[k], str)
                                         for k in ("candidate_id", "routine_id", "selector_source")):
                    raise ControllerFailure("malformed_routine_identifier")
                if pilot == "P7" and any(k in action and type(action[k]) is not int
                                         for k in ("repair_target", "deploy_target", "target_version")):
                    raise ControllerFailure("malformed_instance_identifier")
            if limits is not None and limits.latency_seconds is not None:
                if time.perf_counter() - started > limits.latency_seconds:
                    raise ControllerFailure("decision_deadline_missed")
        except Exception as exc:
            # This boundary contains controller execution/output validation only.
            # Task-law assertion failures below are instrument failures and must
            # stop collection rather than be disguised as weak controller scores.
            failures.append({"epoch": epoch, "cause": str(exc), "exception_type": type(exc).__name__})
            actions = [{"A": 0, "B": 0} for _ in tasks]
            controller.last_decision_report = {"failed_decision": True,
                                               "failure_operations_available": False}
        elapsed = time.perf_counter() - started
        max_seconds = max(max_seconds, elapsed)
        report = deepcopy(controller.last_decision_report)
        total_operations += report.get("useful_operations", 0)
        max_memory = max(max_memory, report.get("memory_scalars", 0))
        for task, action in zip(tasks, actions):
            event = task.step(action)
            if "useful_operations" in report and "decision_cap" in report:
                event["decision_compute"] = {"used": report["useful_operations"],
                                              "allowance": report["decision_cap"],
                                              "scope": "shared_three_cell_decision"}
                event["guardrails"] = audit_execution(event)
                if any(event["guardrails"].values()):
                    raise InstrumentViolation(event)
            for key in guards:
                guards[key] += event["guardrails"][key]
            if event["controller_failure"]:
                failures.append({"epoch": epoch, "cell": task.cell_id,
                                 "cause": "invalid_or_unauthorized_controller_request"})
            # Model internal state and external observation are separate fields.
            if comparator == "C6":
                from b1.controllers.c6 import decode_tree
                memory_record = decode_tree(controller.state)["memory"]
            else:
                memory_record = controller.memory
            record = {"bundle_index": bundle["bundle_index"], "pilot": pilot,
                      "arm_id": opaque_id or comparator, "config_id": config_id,
                      "cycle": cycle, "acute_lesion": lesion,
                      "acute_context": bundle.get("acute_context"),
                      "internal_integrated_content": {"available_epoch": epoch,
                          "route": report.get("route_slots", []),
                          "planning_horizon": report.get("planning_horizon", 1)},
                      "internal_memory": deepcopy(memory_record),
                      "controller_report": report, "event": event}
            trace_hash.update(canonical_bytes(event))
            if archive is not None:
                archive.append(record)
    failed_jobs = sum(t.missed_jobs for t in tasks)
    expected_demand = CONTRACTS.pilots["shared_contract"]["clock"]["primary_loss_denominator"]
    actual_demand = sum(sum(event["demand"] for event in task.events) for task in tasks)
    if actual_demand != expected_demand:
        raise RuntimeError("Instrument demand accounting is incomplete; comparisons must stop")
    loss = primary_loss(pilot, failed_jobs, controller_failure=bool(failures), contracts=CONTRACTS)
    return {"pilot": pilot, "bundle_index": bundle["bundle_index"],
            "comparator": comparator, "config_id": config_id, "cycle": cycle,
            "acute_lesion": lesion, "acute_context": bundle.get("acute_context"),
            "loss": float(loss), "loss_exact": [loss.numerator, loss.denominator],
            "failed_jobs_observed": failed_jobs, "controller_failure": bool(failures),
            "demanded_jobs_accounted": actual_demand,
            "failures": failures, "guardrails": guards, "event_trace_sha256": trace_hash.hexdigest(),
            "useful_operations": total_operations, "max_memory_scalars": max_memory,
            "max_decision_seconds": max_seconds, "development_only": True,
            "confirmatory": False, "reusable_as_final": False}


def run_ceiling_witness(pilot, witness, cell_args):
    """Exact finite law fixtures, not a selected P5 C4 architecture benchmark."""
    if pilot != "P5" or witness not in ("json_minimal_refill_witness", "protocol_full_refill_witness"):
        raise ValueError("Only explicit unresolved P5 ceiling witnesses are allowed")
    task = P5Task(**cell_args)
    cap = CONTRACTS.pilot("P5")["task_contract"]["constants"]
    for epoch in range(task.horizon):
        obs = task.observe()
        service = min(cap["production_capacity_per_epoch"], obs["reserve"], obs["demand"])
        if witness == "protocol_full_refill_witness":
            refill = cap["replenishment_capacity_per_epoch"]
        else:
            refill = min(cap["replenishment_capacity_per_epoch"],
                         max(0, cap["production_capacity_per_epoch"] - (obs["reserve"] - service)))
            if epoch + 1 == task.horizon:
                refill = 0
        task.step({"A": service / cap["production_capacity_per_epoch"],
                   "B": refill / cap["replenishment_capacity_per_epoch"]})
    return {"pilot": pilot, "witness": witness, "completed_jobs": task.completed_jobs,
            "missed_jobs": task.missed_jobs, "events": task.events,
            "frozen_C4_selected": False, "development_only": True,
            "confirmatory": False, "reusable_as_final": False}
