"""Measure archived-v1 retention and replay; never generate a random bundle.

Run from repository root: python -B -m b1.v1_1.retention.measure
Outputs are measurements of existing evidence, not new scientific realizations.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import time

from .policy import canonical, causal_record, decode_records, encode_records
from b1.evaluation.runner import run_episode, contextual_bundle

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "development_results/calibration"
OUTPUT = ROOT / "v1_1/retention/MEASUREMENTS.json"
BASE_COMMIT = "c27235deb1688c105bfe96597dfd233108307895"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(value):
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Counter:
    def __init__(self):
        self.output = io.BytesIO()
        self.stream = gzip.GzipFile(fileobj=self.output, mode="wb", compresslevel=6, mtime=0)
        self.records = self.raw = 0
        self.hash = hashlib.sha256()

    def append(self, record):
        data = canonical(record) + b"\n"
        self.stream.write(data)
        self.hash.update(data)
        self.records += 1
        self.raw += len(data)

    def close(self, verify=False):
        self.stream.close()
        blob = self.output.getvalue()
        if verify:
            raw = gzip.decompress(blob)
            assert hashlib.sha256(raw).hexdigest() == self.hash.hexdigest()
            assert len(raw.splitlines()) == self.records
        return {"records": self.records, "raw_bytes": self.raw,
                "compressed_bytes": len(blob), "raw_sha256": self.hash.hexdigest(),
                "compressed_sha256": hashlib.sha256(blob).hexdigest()}


def measure():
    started = time.perf_counter()
    process_started = time.process_time()
    index = json.loads((ARCHIVE / "traces/INDEX.json").read_text())
    counters = {}
    source_records = 0
    projected_equal = 0
    # Ninety-six pilot/bundle groups retain compressed buffers and codec state.
    # No complete raw archive is assembled in memory; RSS is measured below.
    groups = {}
    for shard in index["shards"]:
        path = ARCHIVE / "traces" / shard["path"]
        assert digest(path) == shard["sha256"]
        lines = gzip.decompress(path.read_bytes()).splitlines()
        assert len(lines) == shard["records"]
        for line in lines:
            row = json.loads(line)
            key = (row["pilot"], row["bundle_index"])
            if key not in counters:
                counters[key] = {level: Counter() for level in ("B", "C_all", "C_diagnostic")}
            counters[key]["C_all"].append(row)
            projected = causal_record(row)
            assert projected["event"] == row["event"]
            # Preserve the complete internal route structure, including empty
            # routes. A null route does not become a missing route.
            if "internal_integrated_content" in row:
                assert projected["internal_integrated_content"] == row["internal_integrated_content"]
            counters[key]["B"].append(projected)
            projected_equal += 1
            if row.get("record_type") == "BUNDLE_SOURCE_DIAGNOSTIC":
                counters[key]["C_diagnostic"].append(row)
            source_records += 1
    assert source_records == index["records"]
    for key, levels in counters.items():
        groups[f"{key[0]}/{key[1]}"] = {level: counter.close(verify=True) for level, counter in levels.items()}
    trials = [json.loads(line) for line in (ARCHIVE / "masked_trials.jsonl").read_text().splitlines()]
    identities = json.loads((ARCHIVE / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json").read_text())["identities"]
    seeds = json.loads((ARCHIVE / "seed_usage.json").read_text())
    bundles = json.loads((ARCHIVE / "bundles.json").read_text())
    by_bundle = {(row["pilot"], row["bundle_index"]): row for row in bundles}
    selected_path = ROOT / "SELECTED_CONFIG.json"
    task_path = ROOT / "contracts/frozen_b0/PILOT_CONTRACTS_v1.json"
    a_sizes = defaultdict(list)
    for (pilot, bundle_index), bundle in by_bundle.items():
        arms = []
        for trial in trials:
            if (trial["pilot"], trial["bundle_index"]) != (pilot, bundle_index):
                continue
            decoded = deepcopy(trial)
            decoded.update(identities[trial["arm_id"]])
            arms.append(decoded)
        row = {"pilot": pilot, "bundle_index": bundle_index, "bundle": bundle,
               "seed_usage": [r for r in seeds if (r["pilot_id"], r["bundle_index"]) == (pilot, bundle_index)],
               "source_commit": BASE_COMMIT, "config_sha256": digest(selected_path),
               "task_sha256": digest(task_path), "outcomes_and_invariant_audits": arms,
               "trace_identity": groups[f"{pilot}/{bundle_index}"]["B"],
               "development_only": True, "confirmatory": False, "reusable_as_final": False}
        blob = encode_records([row])
        assert decode_records(blob) == [row]
        a_sizes[pilot].append(len(blob))
    summary = {}
    for pilot in ("P5", "P6", "P7"):
        pilot_groups = [value for key, value in groups.items() if key.startswith(pilot + "/")]
        summary[pilot] = {"bundles": len(pilot_groups), "level_A_compressed_bytes": sum(a_sizes[pilot]),
                          "level_A_max_bundle_bytes": max(a_sizes[pilot])}
        for level in ("B", "C_all", "C_diagnostic"):
            values = [g[level]["compressed_bytes"] for g in pilot_groups]
            summary[pilot][level] = {"compressed_bytes": sum(values), "max_bundle_bytes": max(values),
                                    "mean_bundle_bytes": sum(values) / len(values),
                                    "records": sum(g[level]["records"] for g in pilot_groups)}
        arm_times = [r["wall_seconds"] for r in trials if r["pilot"] == pilot]
        summary[pilot]["original_episode_wall_seconds"] = sum(arm_times)
        summary[pilot]["original_max_episode_wall_seconds"] = max(arm_times, default=0)
    return {"measured_utc": datetime.now(timezone.utc).isoformat(), "source_commit": BASE_COMMIT,
            "source_archive_index_sha256": digest(ARCHIVE / "traces/INDEX.json"),
            "source_records": source_records, "verified_projection_records": projected_equal,
            "source_shards_verified": len(index["shards"]), "groups": groups, "summary": summary,
            "roundtrip_lossless_B": True, "roundtrip_lossless_C": True,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - process_started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "original_archives_modified": False, "development_only": True,
            "confirmatory": False, "reusable_as_final": False,
            "retention_basis": "Level B stores complete task events and complete route payloads directly; feasibility does not depend on regenerating omitted internal memories"}


def replay():
    bundles = json.loads((ARCHIVE / "bundles.json").read_text())
    identities = json.loads((ARCHIVE / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json").read_text())["identities"]
    trials = [json.loads(line) for line in (ARCHIVE / "masked_trials.jsonl").read_text().splitlines()]
    results = []
    started = time.perf_counter()
    for trial in trials:
        if trial["bundle_index"] != 32:
            continue
        identity = identities[trial["arm_id"]]
        bundle = deepcopy(next(b for b in bundles if b["pilot"] == trial["pilot"] and b["bundle_index"] == 32))
        for cell in bundle["cells"]:
            if "drift_events" in cell:
                cell["drift_events"] = {int(k): v for k, v in cell["drift_events"].items()}
        if identity["acute_context"]:
            bundle = contextual_bundle(bundle, identity["acute_context"])
        result = run_episode(bundle, identity["comparator"], identity["config_id"],
                             cycle=identity["cycle"], lesion=identity["acute_lesion"])
        fields = ("event_trace_sha256", "loss_exact", "failed_jobs_observed", "guardrails", "useful_operations", "max_memory_scalars")
        matches = {field: result[field] == trial[field] for field in fields}
        if not all(matches.values()):
            raise AssertionError({"identity": identity, "matches": matches})
        results.append({"pilot": trial["pilot"], "bundle_index": 32, "arm": identity,
                        "matches": matches, "event_trace_sha256": result["event_trace_sha256"]})
    assert len(results) == 24
    return {"kind": "archived_input_replay_only", "episodes": len(results),
            "bundles": [32], "pilots": ["P6", "P7"], "source_seed_namespace": "PD-B1-D-v1",
            "new_seed_generation": False, "new_scientific_realizations": 0,
            "scientific_event_bytes_match": True,
            "excluded_nondeterministic_fields": ["wall_seconds", "max_decision_seconds"],
            "claim_scope": "24 archived episodes at bundle 32, covering all 12 frozen arms for P6 andP7; not universal cross-platform bitwise certification",
            "wall_seconds": time.perf_counter() - started, "results": results}


if __name__ == "__main__":
    result = measure()
    result["replay"] = replay()
    write(result)
    print(json.dumps({"summary": result["summary"], "replay_episodes": result["replay"]["episodes"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))
