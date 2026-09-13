"""End-to-end development QA. This module is separate from the read-only exporter.

The immutable expected ID plan is written before the first task is executed.
The only derivation namespace is PD-B1-D-v1.1-QA; no final stream exists.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from b1.evaluation.runner import TraceArchive, contextual_bundle, run_episode, sample_bundle
from .core import (ROOT, VERSION, FLAGS, V1_COMMIT, ExportError, canonical,
                   digest, document, export_qa, read_json)

NAMESPACE = "PD-B1-D-v1.1-QA"
SCHEDULE = [(c, "ordinary", "intact", cycle) for c in ("C1", "C2", "C6")
            for cycle in ((1, 2) if c == "C2" else (1,))]
SCHEDULE += [("C1", context, route, 1) for context in ("kappa1", "kappa2")
             for route in ("intact", "bypassed")]


class QASeeds:
    def __init__(self):
        self.usage = []

    def derive(self, pilot, role, index, substream):
        if pilot not in ("P6", "P7") or role != "evaluation-development" or index != 0:
            raise ExportError("Only the declared QA streams are available")
        key = [NAMESPACE, V1_COMMIT, pilot, role, str(index), substream]
        value = int.from_bytes(sha256("|".join(key).encode()).digest()[:8], "big")
        self.usage.append({"key": key, "seed": value, **FLAGS})
        return value


class NormalizedArchive:
    def __init__(self, archive, pilot, comparator, context, route, cycle):
        self.archive, self.pilot, self.comparator = archive, pilot, comparator
        self.context, self.route, self.cycle = context, route, cycle

    def append(self, record):
        event = record["event"]
        self.archive.append({"schema_version": VERSION, "namespace": NAMESPACE,
                             "record_kind": "qa_trial", "pilot": self.pilot,
                             "bundle_index": 0, "comparator": self.comparator,
                             "context": self.context, "route_status": self.route,
                             "cycle": self.cycle, "replicate_id": event["cell_id"],
                             "epoch": event["epoch"], "event": event, **FLAGS})


def run(output):
    output = Path(output)
    if output.exists():
        raise ExportError("QA output must be absent; retained QA is never overwritten")
    output.mkdir(parents=True)
    plan = {"version": VERSION, "namespace": NAMESPACE, "fixed_before_execution": True,
            "purpose": "pipeline QA; no experimental evidence and no policy selection",
            "source_commit": V1_COMMIT, "bundle_indices": [0],
            "expected_ids": [["qa_trial", p, 0, c, context, route, cycle, cell, epoch]
                             for p in ("P6", "P7") for c, context, route, cycle in SCHEDULE
                             for cell in range(3) for epoch in range(32)], **FLAGS}
    plan_path = output / "EXPECTED_PLAN.json"
    plan_path.write_bytes(document(plan))
    plan_sha = digest(plan_path)
    selected = read_json(ROOT / "SELECTED_CONFIG.json")["configurations"]
    seeds = QASeeds()
    archive = TraceArchive(output / "traces", records_per_shard=192)
    trials, replay = [], []
    for pilot in ("P6", "P7"):
        bundle = sample_bundle(pilot, 0, seeds, "evaluation-development")
        bundle["namespace"] = NAMESPACE
        by_arm = {}
        for comparator, context, route, cycle in SCHEDULE:
            supplied = contextual_bundle(bundle, context) if context != "ordinary" else bundle
            adapter = NormalizedArchive(archive, pilot, comparator, context, route, cycle)
            row = run_episode(supplied, comparator, selected[pilot][comparator], cycle=cycle,
                              lesion=route == "bypassed", archive=adapter)
            row.update({"namespace": NAMESPACE, "qa_only": True,
                        "counts_as_experimental_evidence": False, **FLAGS})
            trials.append(row)
            by_arm[(comparator, context, route, cycle)] = row
        original = by_arm[("C1", "ordinary", "intact", 1)]
        conjugate = by_arm[("C6", "ordinary", "intact", 1)]
        replay.append({"pilot": pilot, "C6_event_trace_equal_exactly":
                       original["event_trace_sha256"] == conjugate["event_trace_sha256"],
                       "C6_loss_equal_exactly": original["loss_exact"] == conjugate["loss_exact"]})
    index = archive.close()
    if digest(plan_path) != plan_sha:
        raise ExportError("QA plan changed during execution")
    if any(t["controller_failure"] or any(t["guardrails"].values()) for t in trials):
        raise ExportError("QA execution or guardrail failure")
    if not all(r["C6_event_trace_equal_exactly"] and r["C6_loss_equal_exactly"] for r in replay):
        raise ExportError("QA C6 invariance failed")
    (output / "trials.jsonl").write_bytes(b"".join(canonical(t) + b"\n" for t in trials))
    (output / "SEED_USAGE.json").write_bytes(document({"namespace": NAMESPACE, "usage": seeds.usage, **FLAGS}))
    exported = export_qa(output / "traces", plan_path, plan_sha, output / "export",
                         index_sha256=digest(output / "traces/INDEX.json"))
    audit = {"status": "PASS", "namespace": NAMESPACE, "pipeline": ["task", "trace", "shard", "export", "audit"],
             "expected_plan_sha256_before_execution": plan_sha,
             "expected_plan_sha256_after_execution": digest(plan_path),
             "trial_count": len(trials), "trace_records": index["records"], "trace_index": index,
             "guardrail_violations": 0, "controller_failures": 0, "C6_checks": replay,
             "counts_as_experimental_evidence": False, "new_scientific_runs": 0,
             "final_seeds_generated": False, "B1E_executed": False,
             "export_manifest_sha256": digest(output / "export/MANIFEST.json"),
             "export_pass": exported["audit"]["coverage_pass"], **FLAGS}
    (output / "QA_AUDIT.json").write_bytes(document(audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "trace_index"}, sort_keys=True))


if __name__ == "__main__":
    main()
