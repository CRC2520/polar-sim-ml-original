"""Auditable B1-D stage runner. No final seed or B1-E execution command exists.

The content snapshots cover implementation, tests, registered configurations,
canonical inputs and historical seed exclusions. Technical calibration cannot
open before a complete tuning ledger and configuration freeze are verified.
Changing frozen code requires a new development version, not silent resumption.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from b1.contracts import load_contracts
from b1.contracts.loader import strict_json
from b1.controllers import get_configurations, implementation_choices
from b1.controllers.core import CONFLICT_ID
from b1.evaluation.decision import evaluate_development
from b1.evaluation.precision import precision_plan
from b1.evaluation.runner import (TraceArchive, canonical_bytes, contextual_bundle,
                                  run_ceiling_witness, run_episode, sample_bundle,
                                  write_json)
from b1.interventions.diagnostics import all_mechanism_diagnostics, mechanism_diagnostic
from b1.metrics import guardrails_from_counts
from b1.seeds import DevelopmentSeeds, verify_tuning_freeze


ROOT = Path(__file__).resolve().parent
CONTRACTS = load_contracts()
FLAGS = {"development_only": True, "confirmatory": False, "reusable_as_final": False}
TUNABLE = ("C0", "C1", "C2", "C3", "C5")
COMPARATORS = ("C0", "C1", "C2", "C3", "C4", "C5")
WITNESSES = ("json_minimal_refill_witness", "protocol_full_refill_witness")
RESULTS_DIR = "development_results"


class StageError(RuntimeError):
    """A missing or altered gate blocks execution before another seed is opened."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def source_files(root):
    """Runnable code plus explicit inputs, excluding results and prose reports."""
    root = Path(root)
    paths = {p.relative_to(root).as_posix() for p in root.rglob("*.py")
             if "__pycache__" not in p.parts and RESULTS_DIR not in p.parts}
    for folder in ("contracts/frozen_b0", "configs"):
        directory = root / folder
        if directory.exists():
            paths.update(p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file())
    for name in ("B1D_CONFIG.json", "seeds/HISTORICAL_SEED_EXCLUSIONS.json"):
        if (root / name).is_file():
            paths.add(name)
    return sorted(paths)


def snapshot(root, names):
    return {name: digest(Path(root) / name) for name in sorted(names)}


def verify_snapshot(root, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise StageError("A nonempty content snapshot is required")
    root = Path(root).resolve()
    for name, expected in hashes.items():
        target = (root / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(root):
            raise StageError("Unsafe snapshot path")
        if not target.is_file() or digest(target) != expected:
            raise StageError(f"Frozen artifact changed or missing: {name}")


def verify_pre_tuning(root=ROOT):
    root = Path(root)
    path = root / "PRE_TUNING_FREEZE.json"
    if not path.is_file():
        raise StageError("Run prepare before instrumentation or development")
    frozen = read_json(path)
    if frozen.get("status") != "IMPLEMENTATION_CONFIG_FROZEN_BEFORE_SCORES":
        raise StageError("Invalid pre-tuning freeze status")
    verify_snapshot(root, frozen["artifact_sha256"])
    if source_files(root) != frozen["source_files"]:
        raise StageError("Frozen implementation/input inventory changed")
    return frozen


def historical_exclusions(root):
    path = Path(root) / "seeds/HISTORICAL_SEED_EXCLUSIONS.json"
    if not path.is_file():
        raise StageError("Historical seed-exclusion registry must exist before prepare")
    registry = read_json(path)
    seeds = registry.get("seeds", registry.get("historical_seeds", registry.get("excluded_seeds")))
    if not isinstance(seeds, list) or not all(type(x) is int and 0 <= x < 2**64 for x in seeds):
        raise StageError("Historical seed-exclusion registry requires an explicit uint64 list")
    return seeds


def new_seeds(root):
    return DevelopmentSeeds(contracts=CONTRACTS, historical_seeds=historical_exclusions(root))


def create_stage(root, stage):
    directory = Path(root) / RESULTS_DIR / stage
    if directory.exists():
        raise StageError(f"Stage already has retained artifacts: {stage}; use a new version instead of overwriting")
    directory.mkdir(parents=True)
    write_json(directory / "STARTED.json", {"stage": stage, "started_utc": utc_now(), **FLAGS})
    return directory


def append_row(stream, row):
    stream.write(canonical_bytes(row).decode("utf-8") + "\n")
    stream.flush()


def prepare(root=ROOT, *, code_commit=None):
    root = Path(root)
    if (root / "PRE_TUNING_FREEZE.json").exists() or (root / RESULTS_DIR / "tuning").exists():
        raise StageError("Do not overwrite a pre-tuning freeze or previous development attempt")
    contracts = load_contracts()
    historical_exclusions(root)
    choices = implementation_choices()
    config = {"version": "B1-D-v1", "canonical_source_commit": contracts.source_commit,
              "implementation_choices": choices,
              "configurations": {c: get_configurations(c) for c in COMPARATORS},
              "selected_configuration_rule": ["mean_bundle_primary_loss", "mean_useful_operations", "config_id"],
              "C2_cycle_policy": "Both fixed cycles retained; selection averages both cycles equally",
              "C4_search": "One frozen conventional policy, not eight fabricated search trials",
              "C6_search": "None: exact image of selected C1 configuration",
              "tuning_bundles": list(range(32)), "calibration_bundles": list(range(32, 64)),
              "architecture_pilots": ["P6", "P7"],
              "P5": {"architecture_status": "BLOCKED", "conflict_id": CONFLICT_ID,
                     "permitted_partial_checks": ["task_law_diagnostics", *WITNESSES]},
              "calibration": {"identities_masked": True, "opaque_identifier": "SHA256(tuning_freeze_hash|arm_description)[:16]",
                              "technical_summary_contains_comparator_losses": False,
                              "export_unblinding": "Only after immutable technical-calibration close",
                              "acute_contexts": ["kappa1", "kappa2"],
                              "acute_intervention": "C1 intact versus nominated-route bypass, complete 32-epoch episode",
                              "diagnostics": "Fresh source/sham state clones for every pilot and every tuning/calibration bundle, separate from architecture episodes",
                              "diagnostic_replicates_per_bundle": contracts.pilots["shared_contract"]["clock"]["replicated_cells"],
                              "diagnostic_endpoint_policy": "Record newly executed clone outcomes for each actual bundle; deterministic imposed-law instrument checks are not independent empirical validation",
                              "c6_equivalence": "Exact event trace hash and rational loss, tolerance zero"},
              "resources": {"limits": choices["limits"], "status": "provisional_not_smallest_cap_certified",
                            "C5_multiplier": 2, "planning_horizon": 1,
                            "latency_formula": "2 * maximum observed tuning decision time; reporting only pending worst-input certificate"},
              "failure_policy": "Retain all attempts; genuine controller failures have primary loss 1. Escaping instrument exceptions, incomplete guardrails or executed physical violations stop the stage and cannot enter configuration selection.",
              **FLAGS}
    write_json(root / "B1D_CONFIG.json", config)
    paths = source_files(root)
    frozen = {"status": "IMPLEMENTATION_CONFIG_FROZEN_BEFORE_SCORES", "created_utc": utc_now(),
              "source_commit": contracts.source_commit,
              "code_commit": code_commit or "uncommitted_content_addressed_snapshot",
              "code_commit_semantics": "Optional immutable pre-execution commit; artifact hashes are the exact runnable content identity",
              "canonical_input_sha256": contracts.hashes, "source_files": paths,
              "artifact_sha256": snapshot(root, paths), "comparator_scores_seen": False,
              "calibration_opened": False, "final_seeds_generated": False, **FLAGS}
    write_json(root / "PRE_TUNING_FREEZE.json", frozen)
    seeds = new_seeds(root)
    write_json(root / "B1D_SEED_POLICY.json", {**seeds.policy(), "historical_exclusion_registry_sha256": digest(root / "seeds/HISTORICAL_SEED_EXCLUSIONS.json"), **FLAGS})
    return {"stage": "prepare", "status": "PASS", "frozen_files": len(paths),
            "pre_tuning_freeze_sha256": digest(root / "PRE_TUNING_FREEZE.json"), **FLAGS}


def run_unit_gate(root):
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "b1/tests", "-p", "test_*.py", "-v"],
                            cwd=Path(root).parent, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout + result.stderr


def instrument(root=ROOT):
    root = Path(root)
    verify_pre_tuning(root)
    directory = create_stage(root, "instrumentation")
    code, transcript = run_unit_gate(root)
    (directory / "unit-tests.txt").write_text(transcript, encoding="utf-8")
    diagnostics = all_mechanism_diagnostics()
    write_json(directory / "mechanism_diagnostics.json", diagnostics)
    verify_pre_tuning(root)
    passed = code == 0 and all(d["task_law_pass"] and d["mediator_contract_verified"] for d in diagnostics.values())
    evidence = {name: digest(directory / name) for name in ("unit-tests.txt", "mechanism_diagnostics.json")}
    gate = {"stage": "instrumentation", "status": "PASS" if passed else "FAIL", "closed_utc": utc_now(),
            "pre_tuning_freeze_sha256": digest(root / "PRE_TUNING_FREEZE.json"),
            "unit_tests_exit_code": code, "task_law_diagnostics_pass": passed and code == 0,
            "artifact_sha256": {f"{RESULTS_DIR}/instrumentation/{name}": value for name, value in evidence.items()},
            "scope": "Instrument and implementation checks; no pairing-utility inference", **FLAGS}
    write_json(root / "INSTRUMENT_GATE.json", gate)
    return gate


def verify_instrument(root):
    frozen = verify_pre_tuning(root)
    path = Path(root) / "INSTRUMENT_GATE.json"
    if not path.is_file():
        raise StageError("Instrument gate must pass before tuning")
    gate = read_json(path)
    if gate.get("status") != "PASS" or gate.get("pre_tuning_freeze_sha256") != digest(Path(root) / "PRE_TUNING_FREEZE.json"):
        raise StageError("Instrument gate failed or no longer matches frozen implementation")
    verify_snapshot(root, gate["artifact_sha256"])
    return frozen


def execute_trial(bundle, comparator, config_id, *, cycle=1, lesion=False, archive=None, opaque_id=None):
    started = time.perf_counter()
    try:
        row = run_episode(bundle, comparator, config_id, cycle=cycle, lesion=lesion,
                          archive=archive, opaque_id=opaque_id)
    except Exception as exc:
        # The episode helper already converts genuine controller failures to
        # loss 1. An escaping exception is an instrument failure: retain the
        # partial trace and failed-attempt metadata, then stop this version.
        row = {"pilot": bundle["pilot"], "bundle_index": bundle["bundle_index"],
               "comparator": comparator, "config_id": config_id, "cycle": cycle,
               "acute_lesion": lesion, "acute_context": bundle.get("acute_context"),
               "loss": None, "loss_exact": None, "controller_failure": False,
               "failures": [{"cause": type(exc).__name__ + ": " + str(exc)}],
               "execution_exception": True, "instrument_failure": True, "guardrails": {}, "useful_operations": 0,
               "max_memory_scalars": 0, "max_decision_seconds": 0.0,
               "event_trace_sha256": None, **FLAGS}
        row["wall_seconds"] = time.perf_counter() - started
        if archive is not None:
            archive.append({"record_type": "INSTRUMENT_FAILED_ATTEMPT", **row})
        blocked = StageError("Instrument exception stops the benchmark: " + type(exc).__name__ + ": " + str(exc))
        blocked.attempt = row
        raise blocked from exc
    guard = guardrails_from_counts(row.get("guardrails", {}), contracts=CONTRACTS)
    if not guard["guardrails_pass"]:
        if archive is not None:
            archive.append({"record_type": "GUARDRAIL_FAILED_ATTEMPT", **row})
        blocked = StageError("Incomplete or violated hard guardrails stop the benchmark")
        blocked.attempt = row
        raise blocked
    row["wall_seconds"] = time.perf_counter() - started
    return row


def choose_configurations(rows):
    """Every expected slot/cycle/bundle is required; failures are included."""
    for row in rows:
        if not guardrails_from_counts(row.get("guardrails", {}), contracts=CONTRACTS)["guardrails_pass"]:
            raise StageError("A tuning trial has missing or violated hard guardrails")
    choices, summary = {}, []
    for pilot in ("P6", "P7"):
        choices[pilot] = {}
        for comparator in COMPARATORS:
            candidates = []
            for cfg in get_configurations(comparator):
                cfg_id = cfg["config_id"]
                group = [r for r in rows if r["pilot"] == pilot and r["comparator"] == comparator and r["config_id"] == cfg_id]
                expected = {(index, cycle) for index in range(32) for cycle in ((1, 2) if comparator == "C2" else (1,))}
                actual = [(r["bundle_index"], r["cycle"]) for r in group]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise StageError(f"Incomplete/duplicated tuning trials: {pilot}/{comparator}/{cfg_id}")
                result = {"pilot": pilot, "comparator": comparator, "config_id": cfg_id,
                          "trials": len(group), "bundle_count": 32,
                          "mean_loss": sum(r["loss"] for r in group) / len(group),
                          "mean_useful_operations": sum(r["useful_operations"] for r in group) / len(group),
                          "failed_attempts": sum(r["controller_failure"] for r in group),
                          "cycles_retained": [1, 2] if comparator == "C2" else [1], **FLAGS}
                candidates.append(result)
                summary.append(result)
            selected = min(candidates, key=lambda r: (r["mean_loss"], r["mean_useful_operations"], r["config_id"]))
            choices[pilot][comparator] = selected["config_id"]
        choices[pilot]["C6"] = choices[pilot]["C1"]
    return choices, summary


def run_witness_bundle(bundle, archive):
    rows = []
    for witness in WITNESSES:
        for cell in bundle["cells"]:
            result = run_ceiling_witness("P5", witness, cell)
            events = result.pop("events")
            for event in events:
                archive.append({"pilot": "P5", "bundle_index": bundle["bundle_index"],
                                "witness": witness, "event": event, **FLAGS})
            rows.append({**result, "bundle_index": bundle["bundle_index"], "cell_id": cell["cell_id"],
                         "initial_reserve": cell["initial_reserve"],
                         "event_trace_sha256": sha256(canonical_bytes(events)).hexdigest()})
    return rows


def run_bundle_diagnostics(bundle, archive, stream):
    """New clone executions within the registered bundle, never copied fixtures.

    Each independent local clone world has its own cell 0. The surrounding
    replicate_id identifies its bundle cell; no cross-world identity is shared.
    These deterministic source-law manipulations are not route-only contrasts.
    """
    replicas = []
    for replicate_id in range(CONTRACTS.pilots["shared_contract"]["clock"]["replicated_cells"]):
        result = mechanism_diagnostic(bundle["pilot"])
        for context, contrast in result["contexts"].items():
            for arm in ("source", "sham"):
                for event in contrast[arm]["trace"]:
                    archive.append({"record_type": "BUNDLE_SOURCE_DIAGNOSTIC", "pilot": bundle["pilot"],
                                    "bundle_index": bundle["bundle_index"], "role": bundle["role"],
                                    "replicate_id": replicate_id, "context": context, "diagnostic_arm": arm,
                                    "event": event, **FLAGS})
        replicas.append({"replicate_id": replicate_id, "result": result})
    outcomes = {context: sum(r["result"]["contexts"][context]["effect"] for r in replicas) / len(replicas)
                for context in ("kappa1", "kappa2")}
    record = {"pilot": bundle["pilot"], "bundle_index": bundle["bundle_index"], "role": bundle["role"],
              "replicates": replicas, "executed_new_clone_worlds": True,
              "mechanism.kappa1": outcomes["kappa1"], "mechanism.kappa2": outcomes["kappa2"],
              "context.interaction": outcomes["kappa1"] - outcomes["kappa2"],
              "task_law_pass": all(r["result"]["task_law_pass"] for r in replicas),
              "mediator_contract_verified": all(r["result"]["mediator_contract_verified"] for r in replicas),
              "scope": "Fresh deterministic imposed-law clone diagnostics within this bundle; no independent empirical confirmation or route-specific inference", **FLAGS}
    append_row(stream, record)
    if not record["task_law_pass"] or not record["mediator_contract_verified"]:
        raise StageError("A bundle source diagnostic failed; stop the instrument before architecture trials")
    return record


def tuning(root=ROOT):
    root = Path(root)
    verify_instrument(root)
    directory = create_stage(root, "tuning")
    started = time.perf_counter()
    seeds, rows, witnesses, bundles = new_seeds(root), [], [], []
    archive = TraceArchive(directory / "traces")
    try:
        with (directory / "all_trials.jsonl").open("x", encoding="utf-8") as stream, \
                (directory / "bundle_diagnostics.jsonl").open("x", encoding="utf-8") as diagnostic_stream:
            for index in range(32):
                for pilot in ("P5", "P6", "P7"):
                    bundle = sample_bundle(pilot, index, seeds, "tuning")
                    bundles.append(bundle)
                    run_bundle_diagnostics(bundle, archive, diagnostic_stream)
                    if pilot == "P5":
                        witnesses.extend(run_witness_bundle(bundle, archive))
                        continue
                    for comparator in COMPARATORS:
                        for config in get_configurations(comparator):
                            for cycle in ((1, 2) if comparator == "C2" else (1,)):
                                try:
                                    row = execute_trial(bundle, comparator, config["config_id"], cycle=cycle, archive=archive)
                                except StageError as exc:
                                    if hasattr(exc, "attempt"):
                                        append_row(stream, exc.attempt)
                                    raise
                                rows.append(row)
                                append_row(stream, row)
                print(json.dumps({"stage": "tuning", "bundles_finished": index + 1, "architecture_trials": len(rows)}), flush=True)
    finally:
        trace_index = archive.close()
        write_json(directory / "seed_usage.json", seeds.usage_log)
        write_json(directory / "bundles.json", bundles)
        write_json(directory / "P5_ceiling_witnesses.json", {"rows": witnesses, "conflict_id": CONFLICT_ID, "architecture_comparison_blocked": True, **FLAGS})
    selected, means = choose_configurations(rows)
    verify_instrument(root)
    ledger = {"status": "TUNING_CLOSED", "bundle_indices": list(range(32)),
              "architecture_pilots": ["P6", "P7"], "P5_architecture": "BLOCKED",
              "P5_conflict_id": CONFLICT_ID, "attempt_count": len(rows),
              "expected_attempt_count": 32 * 2 * (8 + 8 + 16 + 8 + 1 + 8),
              "all_attempts_retained": True, "failed_attempts": sum(r["controller_failure"] for r in rows),
              "configuration_results": means, "trace_index": trace_index,
              "wall_seconds": time.perf_counter() - started,
              "max_observed_decision_seconds": max(r["max_decision_seconds"] for r in rows),
              "evidence_files": [f"{RESULTS_DIR}/tuning/all_trials.jsonl", f"{RESULTS_DIR}/tuning/seed_usage.json"],
              "closed_utc": utc_now(), **FLAGS}
    write_json(root / "TUNING_LEDGER.json", ledger)
    write_json(root / "SELECTED_CONFIG.json", {"configurations": selected, "selected_from": "all 32 tuning bundles and every registered configuration, including failed attempts",
                                              "C2": "Mean of both fixed cycles; neither cycle selected", "C6": "Exact image of selected C1", **FLAGS})
    evidence = source_files(root) + ["PRE_TUNING_FREEZE.json", "INSTRUMENT_GATE.json", "TUNING_LEDGER.json", "SELECTED_CONFIG.json"]
    evidence += [p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file()]
    freeze = {"status": "TUNING_CLOSED", "source_commit": load_contracts().source_commit,
              "tuning_bundle_indices": list(range(32)), "configuration_frozen": True,
              "calibration_comparator_identities_masked": True, "calibration_consumed": False,
              "implementation_files": [name for name in source_files(root) if name.endswith(".py")],
              "configuration_file": "SELECTED_CONFIG.json", "tuning_ledger_file": "TUNING_LEDGER.json",
              "artifact_sha256": snapshot(root, evidence), "closed_utc": utc_now(),
              "scope": "P6/P7 architecture tuning complete; P5 conflict retained, never resolved by witness scores", **FLAGS}
    write_json(root / "TUNING_FREEZE.json", freeze)
    verify_tuning_freeze(root / "TUNING_FREEZE.json")
    return {"stage": "tuning", "status": "TUNING_CLOSED", "attempts": len(rows),
            "failed_attempts": ledger["failed_attempts"], "selected_configurations": selected,
            "calibration_opened": False, "tuning_freeze_sha256": digest(root / "TUNING_FREEZE.json"), **FLAGS}


def opaque_arm(freeze_hash, pilot, comparator, cycle=1, context=None, lesion=False):
    description = {"pilot": pilot, "comparator": comparator, "cycle": cycle, "context": context, "lesion": lesion}
    return "arm-" + sha256(freeze_hash.encode() + canonical_bytes(description)).hexdigest()[:16]


def redact_identity(value, arm_id):
    if isinstance(value, dict):
        return {key: (arm_id if key in {"comparator", "config_id", "arm_id"} else redact_identity(item, arm_id))
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_identity(item, arm_id) for item in value]
    return value


class MaskedArchive:
    def __init__(self, archive, arm_id):
        self.archive, self.arm_id = archive, arm_id

    def append(self, value):
        self.archive.append(redact_identity(value, self.arm_id))


def calibration(root=ROOT):
    root = Path(root)
    verify_instrument(root)
    token = verify_tuning_freeze(root / "TUNING_FREEZE.json")
    if (root / "CALIBRATION_STARTED.json").exists():
        raise StageError("Calibration has already been opened; it cannot be presented as untouched or rerun in this version")
    selected = read_json(root / "SELECTED_CONFIG.json")["configurations"]
    directory = create_stage(root, "calibration")
    freeze_hash = digest(root / "TUNING_FREEZE.json")
    write_json(root / "CALIBRATION_STARTED.json", {"opened_utc": utc_now(), "tuning_freeze_sha256": freeze_hash,
                                                "bundle_indices": list(range(32, 64)), "calibration_consumed": True, **FLAGS})
    seeds = new_seeds(root)
    seeds.open_calibration(token)
    archive = TraceArchive(directory / "traces")
    started = time.perf_counter()
    rows, identity_map, equivalence, witnesses, bundles = [], {}, [], [], []
    try:
        with (directory / "masked_trials.jsonl").open("x", encoding="utf-8") as stream, \
                (directory / "bundle_diagnostics.jsonl").open("x", encoding="utf-8") as diagnostic_stream:
            def execute(bundle, comparator, cycle=1, lesion=False):
                cfg = selected[bundle["pilot"]][comparator]
                arm = opaque_arm(freeze_hash, bundle["pilot"], comparator, cycle, bundle.get("acute_context"), lesion)
                identity_map[arm] = {"pilot": bundle["pilot"], "comparator": comparator,
                                     "config_id": cfg, "cycle": cycle,
                                     "acute_context": bundle.get("acute_context"), "acute_lesion": lesion}
                try:
                    result = execute_trial(bundle, comparator, cfg, cycle=cycle, lesion=lesion,
                                           archive=MaskedArchive(archive, arm), opaque_id=arm)
                except StageError as exc:
                    if hasattr(exc, "attempt"):
                        append_row(stream, {**redact_identity(exc.attempt, arm), "arm_id": arm})
                    raise
                masked = redact_identity(result, arm)
                masked["arm_id"] = arm
                append_row(stream, masked)
                rows.append(masked)
                return result

            for index in range(32, 64):
                for pilot in ("P5", "P6", "P7"):
                    bundle = sample_bundle(pilot, index, seeds, "calibration")
                    bundles.append(bundle)
                    run_bundle_diagnostics(bundle, archive, diagnostic_stream)
                    if pilot == "P5":
                        witnesses.extend(run_witness_bundle(bundle, archive))
                        continue
                    ordinary = {}
                    for comparator in (*COMPARATORS, "C6"):
                        for cycle in ((1, 2) if comparator == "C2" else (1,)):
                            ordinary[(comparator, cycle)] = execute(bundle, comparator, cycle)
                    original, conjugate = ordinary[("C1", 1)], ordinary[("C6", 1)]
                    equivalence.append({"pilot": pilot, "bundle_index": index,
                                        "loss_equal_exactly": original["loss_exact"] == conjugate["loss_exact"],
                                        "event_trace_equal_exactly": original["event_trace_sha256"] is not None and original["event_trace_sha256"] == conjugate["event_trace_sha256"],
                                        "controller_failures_absent": not (original["controller_failure"] or conjugate["controller_failure"]),
                                        "tolerance": 0, **FLAGS})
                    for context in ("kappa1", "kappa2"):
                        clamped = contextual_bundle(bundle, context)
                        execute(clamped, "C1", lesion=False)
                        execute(clamped, "C1", lesion=True)
                print(json.dumps({"stage": "calibration", "bundles_finished": index - 31, "masked_trials": len(rows), "comparator_effects_disclosed": False}), flush=True)
    finally:
        trace_index = archive.close()
        write_json(directory / "seed_usage.json", seeds.usage_log)
        write_json(directory / "bundles.json", bundles)
        write_json(directory / "P5_ceiling_witnesses.json", {"rows": witnesses, "architecture_comparison_blocked": True, "conflict_id": CONFLICT_ID, **FLAGS})
    diagnostics = all_mechanism_diagnostics()
    write_json(directory / "mechanism_diagnostics.json", diagnostics)
    # Recheck exact implementation and all tuning evidence after collection.
    verify_instrument(root)
    verify_tuning_freeze(root / "TUNING_FREEZE.json")
    guardrails = Counter()
    for row in rows:
        guardrails.update(row["guardrails"])
    expected = 32 * 2 * (8 + 4)
    c6_pass = len(equivalence) == 64 and all(r["loss_equal_exactly"] and r["event_trace_equal_exactly"] and r["controller_failures_absent"] for r in equivalence)
    failures = sum(r["controller_failure"] for r in rows)
    elapsed = time.perf_counter() - started
    technical = {"status": "TECHNICAL_CALIBRATION_CLOSED", "closed_utc": utc_now(),
                 "tuning_freeze_sha256": freeze_hash, "bundle_indices": list(range(32, 64)),
                 "identities_masked_during_collection": True,
                 "blinding_limit": "Identity-label masking for technical reporting, not structural anonymity or double blinding",
                 "comparator_losses_in_summary": False, "trial_count": len(rows), "expected_trial_count": expected,
                 "all_expected_trials_retained": len(rows) == expected,
                 "failed_attempts": failures, "guardrails": dict(guardrails),
                 "source_diagnostics_task_law_pass": all(x["task_law_pass"] for x in diagnostics.values()),
                 "source_diagnostics_instrument_valid": all(x["mediator_contract_verified"] for x in diagnostics.values()),
                 "exact_coordinate_invariance_pass": c6_pass, "coordinate_invariance_checks": equivalence,
                 "wall_seconds": elapsed, "sum_episode_wall_seconds": sum(r["wall_seconds"] for r in rows),
                 "max_observed_decision_seconds": max(r["max_decision_seconds"] for r in rows),
                 "max_observed_memory_scalars": max(r["max_memory_scalars"] for r in rows),
                 "max_observed_episode_useful_operations": max(r["useful_operations"] for r in rows),
                 "trace_index": trace_index, "smallest_common_resource_cap_certified": False,
                 "P5_architecture_status": "BLOCKED", "P5_conflict_id": CONFLICT_ID,
                 "final_seeds_generated": False, **FLAGS}
    technical["pipeline_technical_pass"] = (len(rows) == expected and not failures and not any(guardrails.values())
                                             and c6_pass and technical["source_diagnostics_task_law_pass"]
                                             and technical["source_diagnostics_instrument_valid"])
    technical["artifact_sha256"] = snapshot(root, [p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file()])
    write_json(root / "CALIBRATION_TECHNICAL.json", technical)
    # Disclosure is a separate, ordered artifact after technical closure. A
    # failed or interrupted collection never publishes the identity map.
    write_json(directory / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json", {
        "identities": identity_map, "created_utc": utc_now(),
        "technical_close_sha256": digest(root / "CALIBRATION_TECHNICAL.json"),
        "use_policy": "Export only after technical closure; no retuning", **FLAGS})
    return {key: value for key, value in technical.items() if key not in {"artifact_sha256", "coordinate_invariance_checks", "trace_index"}}


def actual_resources(root):
    disk = shutil.disk_usage(root)
    cpu_quota = None
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.is_file():
        fields = path.read_text().split()
        if len(fields) == 2 and fields[0] != "max":
            cpu_quota = int(fields[0]) / int(fields[1])
    return {"measured_utc": utc_now(), "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
            "logical_cpu_count": os.cpu_count(), "cgroup_cpu_quota": cpu_quota,
            "future_cpu_time_budget": None, "future_budget_authorized": False}


def export(root=ROOT):
    root = Path(root)
    verify_instrument(root)
    verify_tuning_freeze(root / "TUNING_FREEZE.json")
    path = root / "CALIBRATION_TECHNICAL.json"
    if not path.is_file():
        raise StageError("Technical calibration must close before descriptive export or unblinding")
    technical = read_json(path)
    if technical.get("status") != "TECHNICAL_CALIBRATION_CLOSED":
        raise StageError("Calibration did not close")
    verify_snapshot(root, technical["artifact_sha256"])
    directory = root / RESULTS_DIR / "calibration"
    identity_record = read_json(directory / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json")
    if identity_record.get("technical_close_sha256") != digest(path):
        raise StageError("Identity disclosure is not linked to the immutable technical close")
    identity_map = identity_record["identities"]
    rows = []
    with (directory / "masked_trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = strict_json(line)
            row.update(identity_map[row["arm_id"]])
            rows.append(row)
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
    with (directory / "bundle_diagnostics.jsonl").open(encoding="utf-8") as stream:
        bundle_diagnostics = [strict_json(line) for line in stream]
    expected_diagnostics = {(pilot, index) for pilot in ("P5", "P6", "P7") for index in range(32, 64)}
    observed_diagnostics = [(r["pilot"], r["bundle_index"]) for r in bundle_diagnostics]
    if len(observed_diagnostics) != len(expected_diagnostics) or set(observed_diagnostics) != expected_diagnostics:
        raise StageError("Missing or duplicated actual calibration bundle diagnostics")
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
            "registry_bundle_diagnostics_file": f"{RESULTS_DIR}/calibration/bundle_diagnostics.jsonl",
            "registry_bundle_diagnostics_sha256": digest(directory / "bundle_diagnostics.jsonl"),
            "registry_observations": "Fresh clone diagnostics were actually executed for each of the 32 calibration bundles, with three new local worlds per bundle. Repeated imposed deterministic laws are instrument observations, not independent empirical confirmation.", **FLAGS}
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
    if projection["observed_disk_insufficient_for_partial_schedule"]:
        blockers.append({"id": "B1E-RESOURCE-INFEASIBLE-CURRENT-DISK", "reason": "Projected partial-schedule compressed traces exceed currently available disk"})
    if not technical["pipeline_technical_pass"]:
        blockers.append({"id": "B1D-TECHNICAL-CALIBRATION-FAILED", "reason": "See exact retained technical failures; no failed attempt was filtered"})
    results = {"status": "B1D_BLOCKED", "created_utc": utc_now(), "unresolved_blockers": blockers,
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
    lines = ["# B1-D development results", "", "Status: **B1D_BLOCKED**. All observations are development-only, nonconfirmatory and forbidden for reuse as final observations.", "",
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
    (root / "B1D_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return {"stage": "export", "status": results["status"], "unresolved_blockers": blockers,
            "B1E_required_N": n, "ready_for_b1e_confirmatory_run": False, **FLAGS}


def smoke(output):
    """CI-sized instrument check: fixed development bundle 0, never calibration."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise StageError("Smoke output must be absent or empty; previous traces are retained")
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = all_mechanism_diagnostics()
    if not all(x["task_law_pass"] and x["mediator_contract_verified"] for x in diagnostics.values()):
        write_json(output / "smoke.json", {"status": "FAIL", "reason": "deterministic instrument gate", **FLAGS})
        raise StageError("Deterministic instrument gate failed before smoke bundle")
    seeds = new_seeds(ROOT)
    archive = TraceArchive(output / "traces")
    checks = []
    try:
        for pilot in ("P6", "P7"):
            bundle = sample_bundle(pilot, 0, seeds, "evaluation-development")
            original = execute_trial(bundle, "C1", "cfg00", archive=archive)
            conjugate = execute_trial(bundle, "C6", "cfg00", archive=archive)
            checks.append({"pilot": pilot, "bundle_index": 0,
                           "pass": not (original["controller_failure"] or conjugate["controller_failure"])
                                   and original["event_trace_sha256"] == conjugate["event_trace_sha256"]
                                   and original["loss_exact"] == conjugate["loss_exact"],
                           "C1": original, "C6": conjugate, **FLAGS})
    finally:
        index = archive.close()
    result = {"stage": "smoke", "status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
              "scope": "CI smoke, not a tuning or calibration block", "development_bundle_indices": [0],
              "calibration_opened": False, "checks": checks, "trace_index": index,
              "seed_usage": seeds.usage_log, "final_seeds_generated": False, **FLAGS}
    write_json(output / "smoke.json", result)
    return {key: value for key, value in result.items() if key not in {"checks", "trace_index", "seed_usage"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "instrument", "tuning", "calibration", "smoke", "export"))
    parser.add_argument("--output", type=Path, help="Dedicated output directory for smoke only")
    parser.add_argument("--code-commit", help="Immutable pre-execution code commit, recorded by prepare")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.command != "smoke":
            raise StageError("--output is only available for smoke; development artifacts remain under b1/")
        if args.command == "prepare":
            result = prepare(code_commit=args.code_commit)
        elif args.command == "smoke":
            result = smoke(args.output or ROOT / RESULTS_DIR / "smoke")
        else:
            result = {"instrument": instrument, "tuning": tuning, "calibration": calibration, "export": export}[args.command]()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result.get("status") == "FAIL" else 0
    except (OSError, ValueError, StageError) as exc:
        print(json.dumps({"status": "BLOCKED", "stage": args.command,
                          "error": type(exc).__name__ + ": " + str(exc), **FLAGS}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
