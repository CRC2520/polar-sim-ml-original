"""Read-only retention schema/ID and incremental-cost QA on retained events."""
import argparse
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import time

from b1.v1_1.exporter.core import authenticated_records, digest
from .validation import record_key

ROOT = Path(__file__).resolve().parents[3]


def run():
    base = ROOT / "b1/development_results/calibration"
    identities = json.loads((base / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json").read_text())["identities"]
    counts, timings, sources = {}, {}, {}
    for label, directory, identity_map in (
            ("historical", base / "traces", identities),
            ("P5_v1_1", ROOT / "b1/v1_1/p5/results/traces", {})):
        index_sha = digest(directory / "INDEX.json")
        sources[str((directory / "INDEX.json").relative_to(ROOT))] = index_sha
        records = authenticated_records(directory, index_sha)
        seen_global, total = set(), 0
        for (pilot, bundle), group in groupby(records, lambda row: (row["pilot"], row["bundle_index"])):
            rows = list(group)
            start, cpu_start = time.perf_counter(), time.process_time()
            # Charge key derivation twice, including a fresh expected-set build.
            # Actual writer plans are supplied before collection, independently
            # of observed records. Here both derive from retained QA fixtures.
            keys = [record_key(row, identity_map) for row in rows]
            expected, seen = frozenset(keys), set()
            for row in rows:
                key = record_key(row, identity_map)
                if key not in expected or key in seen:
                    raise ValueError("Retained ID coverage mismatch")
                seen.add(key)
            if seen != expected or seen_global.intersection(seen):
                raise ValueError("Missing or globally duplicated retained IDs")
            cpu, wall = time.process_time() - cpu_start, time.perf_counter() - start
            seen_global.update(seen)
            total += len(rows)
            timings.setdefault(label + ":" + pilot, []).append({"bundle_index": bundle,
                "records": len(rows), "wall_seconds": wall, "CPU_seconds": cpu})
        counts[label] = total
    if counts != {"historical": 82176, "P5_v1_1": 27648}:
        raise ValueError("Retained archive count changed")
    cert_path = ROOT / "b1/v1_1/B1E_RESOURCE_CERTIFICATE.json"
    cert = json.loads(cert_path.read_text())
    sources[str(cert_path.relative_to(ROOT))] = digest(cert_path)
    summary = {label: {"bundles": len(rows), "max_wall_seconds_per_bundle": max(r["wall_seconds"] for r in rows),
        "max_CPU_seconds_per_bundle": max(r["CPU_seconds"] for r in rows),
        "total_wall_seconds": sum(r["wall_seconds"] for r in rows)} for label, rows in timings.items()}
    incremental = cert["N"] * summary["historical:P7"]["max_wall_seconds_per_bundle"]
    incremental += 32 * (summary["historical:P6"]["max_wall_seconds_per_bundle"]
                         + summary["P5_v1_1:P5"]["max_wall_seconds_per_bundle"] * 13 / 9)
    # The frozen future P5 schedule has 13 episodes versus 9 in the retained
    # v1.1 benchmark; scale the measured QA cost explicitly for planning.
    contingency = cert["CPU_and_wall_time"]["CPU_seconds_planning_budget"] - cert["CPU_and_wall_time"]["serial_seconds_precontingency"]
    return {"status": "PASS", "source_sha256": sources,
        "validator_sha256": digest(Path(__file__).with_name("validation.py")),
        "measurement_script_sha256": digest(Path(__file__)),
        "archive_unique_ID_counts": counts, "timing_summary": summary,
        "incremental_validation_seconds_projection": incremental,
        "existing_CPU_contingency_seconds": contingency,
        "remaining_CPU_contingency_after_measured_increment_seconds": contingency - incremental,
        "fits_existing_CPU_contingency": incremental < contingency,
        "projection_limit": "Sampled same-host incremental schema/ID cost, key derivation charged twice; not a worst-case runtime guarantee. Existing 1.5 planning contingency absorbs this measured increment; fixed N and numerical resource budget are unchanged.",
        "scientific_model_executions": 0, "development_only": True,
        "confirmatory": False, "reusable_as_final": False,
        "final_seeds_generated": False, "B1E_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
