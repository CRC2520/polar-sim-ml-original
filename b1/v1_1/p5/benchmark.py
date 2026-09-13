"""P5 v1.1 development benchmark with a required published source freeze.

The only collection namespace supported here is PD-B1-D-v1.1. QA is explicit
and cannot be presented as a new scientific realization or final evidence.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

from b1.controllers.core import Controller
from b1.controllers.c6 import C6Controller, ConjugatedTask, decode_tree
from b1.evaluation.runner import TraceArchive, canonical_bytes, write_json
from b1.metrics import primary_loss
from b1.tasks import P5Task
from b1.tasks.core import audit_execution, GUARDRAILS
from b1.v1_1.p5.controllers import P5ConventionalController
from b1.v1_1.resources.accounting import limits, decision_charge, assert_within_envelope

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FLAGS = {"development_only": True, "confirmatory": False, "reusable_as_final": False}
ARMS = [("C0", 1), ("C1", 1), ("C2", 1), ("C2", 2), ("C3", 1),
        ("C4", 1), ("C5", 1), ("C6", 1), ("C4_FULL_REFILL_CEILING_WITNESS", 1)]


def protocol():
    return json.loads((HERE.parent / "P5_PROTOCOL_V1_1.json").read_text())


def source_freeze_inventory():
    paths = ["b1/v1_1/P5_C4_DISPOSITION.json", "b1/v1_1/P5_C4_DISPOSITION.md",
             "b1/v1_1/P5_PROTOCOL_V1_1.json", "b1/v1_1/RESOURCE_MATCHING_SPEC.json",
             "b1/v1_1/resources/accounting.py", "b1/v1_1/__init__.py",
             "b1/v1_1/resources/__init__.py", "b1/seeds/HISTORICAL_SEED_EXCLUSIONS.json"]
    paths += [str(p.relative_to(ROOT)) for p in sorted(HERE.glob("*.py"))]
    paths += [str(p.relative_to(ROOT)) for folder in ("tasks", "controllers", "contracts", "metrics")
              for p in sorted((ROOT / "b1" / folder).rglob("*"))
              if p.is_file() and p.suffix in (".py", ".json", ".yaml", ".md")]
    paths += ["b1/evaluation/runner.py"]
    # Pin every immutable v1 Python module, including indirect package imports,
    # without depending on an incomplete manually enumerated import graph.
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "b1").rglob("*.py"))
              if not p.is_relative_to(ROOT / "b1/v1_1")]
    return {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(set(paths))}


def verify_freeze(path, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Published prospective source commit is required")
    frozen = json.loads(Path(path).read_text())
    if frozen.get("status") != "P5_V1_1_FROZEN_BEFORE_BENCHMARK":
        raise ValueError("Prospective freeze status is missing")
    if frozen.get("namespace") != "PD-B1-D-v1.1":
        raise ValueError("Invalid development namespace")
    actual = source_freeze_inventory()
    if frozen.get("artifact_sha256") != actual:
        raise ValueError("Prospective source/configuration/resource freeze changed")
    return frozen


def bundle(index, namespace="PD-B1-D-v1.1"):
    if namespace not in ("PD-B1-D-v1.1", "PD-B1-D-v1.1-QA") or type(index) is not int or not 0 <= index < 32:
        raise ValueError("Only enumerated development or QA streams are available")
    p = protocol()
    exclusions = set(json.loads((ROOT / "b1/seeds/HISTORICAL_SEED_EXCLUSIONS.json").read_text())["seeds"])
    cells, ledger = [], []
    for cell in range(3):
        substream = f"cell/{cell}/episode/utility/epoch/0/event/initial_reserve"
        fields = [namespace, p["b0_source_commit"], "P5", "evaluation-development", str(index), substream]
        digest = sha256("|".join(fields).encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        if seed in exclusions:
            raise ValueError("Development seed collides with excluded historical seed")
        cells.append({"cell_id": cell, "initial_reserve": (0, 8)[random.Random(seed).randrange(2)]})
        ledger.append({"namespace": namespace, "bundle_index": index, "pilot": "P5",
                       "role": "evaluation-development", "substream_id": substream,
                       "seed": seed, "derivation_sha256": digest.hex(), **FLAGS})
    return {"pilot": "P5", "bundle_index": index, "namespace": namespace,
            "cells": cells, "seed_ledger": ledger, **FLAGS}


def controller_for(comparator, cycle=1):
    if comparator in ("C4", "C4_FULL_REFILL_CEILING_WITNESS"):
        return P5ConventionalController(limits=limits(), full_refill_witness=comparator != "C4")
    if comparator == "C6":
        return C6Controller(Controller("P5", "C1", "cfg00", limits=limits()))
    return Controller("P5", comparator, "cfg00", cycle=cycle, limits=limits())


def episode(b, comparator, cycle=1, *, archive=None, source_commit=None, config_hash=None):
    if any(b.get(k) != v for k, v in FLAGS.items()) or b.get("namespace") not in ("PD-B1-D-v1.1", "PD-B1-D-v1.1-QA"):
        raise ValueError("Explicit development scope is required")
    ctr = controller_for(comparator, cycle)
    tasks = [P5Task(**a) for a in b["cells"]]
    if comparator == "C6":
        conjugated_tasks = [ConjugatedTask(P5Task(**a)) for a in b["cells"]]
        reference = Controller("P5", "C1", "cfg00", limits=limits())
    digest = sha256()
    guards = dict.fromkeys(GUARDRAILS, 0)
    profile_max, failures, events = {}, [], []
    max_latency, cpu, replenishment, waste = 0.0, 0.0, 0, 0
    c6_counts = dict(actions=0, states=0, observations=0, events=0)
    for epoch in range(32):
        obs = [t.observe() for t in tasks]
        start, cpu_start = time.perf_counter(), time.process_time()
        failed_decision = False
        ctr.last_decision_report = {}
        try:
            actions = ctr.act(obs)
            if len(actions) != 3 or any(not isinstance(a, dict) or set(a) != {"A", "B"} for a in actions):
                raise ValueError("Malformed P5 controller action")
        except Exception as exc:
            failures.append({"epoch": epoch, "type": type(exc).__name__, "message": str(exc)})
            actions = [{"A": 0, "B": 0} for _ in tasks]
            failed_decision = True
        elapsed, used_cpu = time.perf_counter() - start, time.process_time() - cpu_start
        max_latency, cpu = max(max_latency, elapsed), cpu + used_cpu
        measured_ctr = SimpleNamespace(**decode_tree(ctr.state)) if comparator == "C6" else ctr
        if failed_decision:
            measured_ctr.last_decision_report = {"useful_operations": getattr(measured_ctr, "_operations", 0),
                                                "route_slots": [], "failed_decision": True}
        charge = decision_charge(obs, measured_ctr)
        assert_within_envelope("P5", "C1" if comparator == "C6" else comparator, charge)
        for k, value in charge.items():
            profile_max[k] = max(profile_max.get(k, 0), value)
        if comparator == "C6" and not failures:
            if obs != [t.observe() for t in conjugated_tasks]:
                raise AssertionError("C6 observation mismatch")
            expected_actions = reference.act(obs)
            if actions != expected_actions or decode_tree(ctr.state) != reference.__dict__:
                raise AssertionError("C6 action/state mismatch")
            c6_counts["observations"] += 3
            c6_counts["actions"] += 3
            c6_counts["states"] += 1
        for cell, (task, action) in enumerate(zip(tasks, actions)):
            event = task.step(action)
            if comparator == "C6" and not failures:
                if event != conjugated_tasks[cell].step(action):
                    raise AssertionError("C6 conjugated task event mismatch")
                c6_counts["events"] += 1
            event["decision_compute"] = {"used": charge["legacy_operations"],
                                         "allowance": measured_ctr.limits.decision_operations,
                                         "scope": "shared_three_cell_decision"}
            event["guardrails"] = audit_execution(event)
            if any(event["guardrails"].values()):
                raise AssertionError("P5 physical or execution guardrail violation")
            if event["controller_failure"]:
                failures.append({"epoch": epoch, "cell_id": cell, "type": "invalid_controller_request"})
            for k in guards:
                guards[k] += event["guardrails"][k]
            replenishment += event["channels"]["B"]["external_admitted_dose"]
            waste += event["channels"]["B"]["environmental_effect"]["overflow"]
            digest.update(canonical_bytes(event))
            if archive is not None:
                archive.append({"pilot": "P5", "namespace": b["namespace"],
                    "bundle_index": b["bundle_index"], "comparator": comparator,
                    "cycle": cycle, "context": "ordinary_utility", "route_status": "intact",
                    "epoch": epoch, "cell_id": cell, "source_commit": source_commit,
                    "config_sha256": config_hash, "action": action, "event": event,
                    "controller_report": deepcopy(ctr.last_decision_report),
                    "resource_charge": charge, **FLAGS})
            events.append(event)
    failed = sum(t.missed_jobs for t in tasks)
    analytic_min_failed = 2 * sum(a["initial_reserve"] == 0 for a in b["cells"])
    if failed < analytic_min_failed:
        raise AssertionError("Controller apparently exceeds the physical optimum")
    loss = primary_loss("P5", failed, controller_failure=bool(failures))
    task_hash = sha256(canonical_bytes({"task_version": protocol()["task_version"], "cells": b["cells"]})).hexdigest()
    result = {"pilot": "P5", "namespace": b["namespace"], "bundle_index": b["bundle_index"],
              "comparator": comparator, "cycle": cycle, "loss": float(loss),
              "loss_exact": [loss.numerator, loss.denominator], "missed_jobs": failed,
              "completed_jobs": 192 - failed, "analytical_minimum_missed_jobs": analytic_min_failed,
              "physical_bound_pass": True, "replenishment_units": replenishment, "overflow_units": waste,
              "terminal_refill_units": sum(e["channels"]["B"]["external_admitted_dose"] for e in events if e["epoch"] == 31),
              "guardrails": guards, "controller_failures": failures,
              "event_trace_sha256": digest.hexdigest(), "event_count": len(events),
              "task_sha256": task_hash, "source_commit": source_commit, "config_sha256": config_hash,
              "resource_maxima": profile_max, "max_decision_seconds": max_latency,
              "controller_cpu_seconds": cpu, "resource_cap_binding": False,
              "C6_exact_checks": c6_counts if comparator == "C6" else None, **FLAGS}
    return result


def run(freeze_path, source_commit, output):
    freeze = verify_freeze(freeze_path, source_commit)
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; scientific results must not be overwritten")
    output.mkdir(parents=True)
    start = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    cfg_hash = sha256((HERE.parent / "P5_PROTOCOL_V1_1.json").read_bytes()).hexdigest()
    write_json(output / "STARTED.json", {"started_utc": stamp, "source_commit": source_commit,
        "freeze_sha256": sha256(Path(freeze_path).read_bytes()).hexdigest(),
        "namespace": "PD-B1-D-v1.1", **FLAGS})
    archive = TraceArchive(output / "traces", records_per_shard=960)
    results, ledger, bundles = [], [], []
    try:
        for index in protocol()["bundle_indices"]:
            b = bundle(index)
            bundles.append(b)
            ledger += b["seed_ledger"]
            for comparator, cycle in ARMS:
                row = episode(b, comparator, cycle, archive=archive,
                              source_commit=source_commit, config_hash=cfg_hash)
                results.append(row)
        if len({x["seed"] for x in ledger}) != len(ledger):
            raise AssertionError("Distinct development seeds collide")
        index = archive.close()
        for name, rows in (("bundles.jsonl", bundles), ("seed_ledger.jsonl", ledger), ("episodes.jsonl", results)):
            (output / name).write_bytes(b"".join(canonical_bytes(r)+b"\n" for r in rows))
        means, contrasts = {}, {}
        for comparator in sorted(set(a for a, _ in ARMS)):
            rows = [r for r in results if r["comparator"] == comparator]
            means[comparator] = sum(r["loss"] for r in rows) / len(rows)
        for comparator in ("C0", "C2", "C3", "C4"):
            paired = []
            for bi in range(32):
                l1 = next(Fraction(*r["loss_exact"]) for r in results if r["bundle_index"] == bi and r["comparator"] == "C1")
                lc = [Fraction(*r["loss_exact"]) for r in results if r["bundle_index"] == bi and r["comparator"] == comparator]
                paired.append(l1 - sum(lc)/len(lc))
            contrasts["C1-"+comparator] = {"mean": float(sum(paired)/len(paired)),
                "all_bundles_zero": all(x == 0 for x in paired), "n": 32,
                "differences_exact": [[x.numerator, x.denominator] for x in paired]}
        c6 = [r for r in results if r["comparator"] == "C6"]
        failures_count = sum(len(r["controller_failures"]) for r in results)
        result = {"version": "P5-B1D-v1.1", "status": "PASS" if not failures_count else "FAILED", "namespace": "PD-B1-D-v1.1",
            "source_commit": source_commit, "freeze_sha256": sha256(Path(freeze_path).read_bytes()).hexdigest(),
            "started_utc": stamp, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter()-start, "bundle_count": 32,
            "seed_uses": len(ledger), "episode_count": len(results), "event_count": index["records"],
            "mean_losses": means, "paired_contrasts": contrasts,
            "physical_bound_pass": all(r["physical_bound_pass"] for r in results),
            "all_episodes_attain_physical_ceiling": all(r["missed_jobs"] == r["analytical_minimum_missed_jobs"] for r in results),
            "C6_status": "PASS" if not any(r["controller_failures"] for r in c6) else "FAILED", "C6_exact_checks": {k: sum(r["C6_exact_checks"][k] for r in c6) for k in c6[0]["C6_exact_checks"]},
            "C4_policy": "minimal-next-demand-v1.1", "full_refill_witness_separate": True,
            "C4_terminal_refill_total": sum(r["terminal_refill_units"] for r in results if r["comparator"] == "C4"),
            "full_witness_terminal_refill_total": sum(r["terminal_refill_units"] for r in results if r["comparator"] == "C4_FULL_REFILL_CEILING_WITNESS"),
            "secondary_replenishment_totals": {c: sum(r["replenishment_units"] for r in results if r["comparator"] == c) for c in means},
            "controller_failures": failures_count, "guardrail_violations": 0, "resource_cap_binding": False,
            "trace_compressed_bytes": index["compressed_bytes"], "trace_raw_bytes": index["raw_bytes"],
            "positive_utility_eligible": False, "positive_pairing_specificity_eligible": False,
            "interpretation": "Ceiling negative control. Equality is descriptive and provides no positive pairing or utility evidence.",
            "v1_results_combined": False, "final_seeds_generated": False, "B1E_executed": False, **FLAGS}
        write_json(output / "RESULTS.json", result)
        hashes = {str(p.relative_to(output)): sha256(p.read_bytes()).hexdigest()
                  for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "MANIFEST.json", {"source_commit": source_commit, "artifact_sha256": hashes, **FLAGS})
        return result
    except BaseException as exc:
        archive.close()
        write_json(output / "FAILED.json", {"status": "FAILED", "type": type(exc).__name__,
            "message": str(exc), "completed_episodes": len(results), "source_commit": source_commit, **FLAGS})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.freeze, a.source_commit, a.output), sort_keys=True))


if __name__ == "__main__":
    main()
