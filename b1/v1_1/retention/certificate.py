"""Rebuild prospective resource arithmetic from measured archived evidence.

This performs no simulation, model execution or seed generation.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
V11 = ROOT / "v1_1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def block_bytes(size):
    return math.ceil(size / 4096) * 4096


def build():
    measure = json.loads((V11 / "retention/MEASUREMENTS.json").read_text())
    stats = json.loads((V11 / "B1E_STATISTICAL_REDESIGN.json").read_text())
    resource = json.loads((V11 / "RESOURCE_MATCHING_CERTIFICATE.json").read_text())
    n = stats["N_v2"]
    if n != 138688 or stats["N_v1"] != 915501:
        raise ValueError("Reassess changed statistical schedule prospectively")
    policy = {
        "version": "B1-E-v2-prospective-retention-1", "status": "DEVELOPMENT_QA_IMPLEMENTED_FUTURE_PROTOCOL_DRAFT",
        "selection_before_outcomes": True,
        "level_A": {"coverage": "every bundle", "format": "independent_gzip_JSON_bundle",
                    "fields": ["seed_usage", "source_commit", "config_sha256", "task_sha256", "comparator_identity",
                               "scalar_outcomes", "guardrails", "invariant_audit", "trace_identity"],
                    "metadata_deduplication": "Shared fixed versions may be referenced by their committed SHA256; bundle-specific seeds and scalar outcomes are explicit"},
        "level_B": {"coverage": "every causal record", "format": "independent_gzip_JSONL_pilot_bundle",
                    "fields": "Complete original task event, complete identity/context and integrated-content fields, route slots and consumed payloads, route masks, planning horizon and resource counters",
                    "external_fields_preserved": ["requested", "admitted", "executed", "environmental_effect", "mediator_state_before", "mediator_state_after", "route", "context"],
                    "omitted_fields": ["verbose_internal_memory", "unrelated_controller_reporting_metadata"],
                    "when_C_full_exists": "B is directly available by deterministic field projection from C; do not store a duplicate B file"},
        "level_C": {"audit_subset_rule": "bundle_index % 1024 == 0, indices start at zero, fixed independently of outcomes",
                    "P7_subset_count": math.ceil(n / 1024),
                    "additional_full_bundle_triggers": ["any_controller_failure", "any_guardrail_violation", "any_failed_invariant_audit"],
                    "deterministic_diagnostics": "All source/sham/context diagnostic records retained fully in every bundle, even outside audit subset",
                    "late_failure": "Spool all full records until bundle audit; a final audit failure retains the entire bundle"},
        "replay": {"required_for_feasibility": False, "verification": "24 archived P6/P7 bundle 32 episodes, all 12 arms per pilot; exact scientific event SHA256, loss, guardrail, operation and memory agreement",
                   "nondeterministic_exclusions": ["wall_time", "decision_latency"],
                   "cross_platform_universal_bitwise_claim": False,
                   "source_namespace": "PD-B1-D-v1", "new_scientific_realizations": 0,
                   "fallback": "Required causal evidence is stored directly at B or C; no reconstruction of omitted internal memories is required for causal audit"},
        "failure_policy": {"expected_record_counts_fixed_before_collection": True,
                           "missing_or_malformed_audit": "Preserve staged evidence, reject completion; no MANIFEST.json",
                           "failed_audit": "Retain full C plus A and FAILED_MANIFEST.json with complete=false, raise error",
                           "hash_manifest": "Every finalized A/B/C artifact has SHA256 and size; completion manifest published last",
                           "storage_check": "Before each next bundle, reserve its declared full-trace staging allowance; technical exhaustion closes collection as incomplete, preserves completed rows and staged evidence",
                           "statistical_early_stopping_allowed": False, "outcome_based_N_extension_allowed": False,
                           "partial_sample_confirmatory_interpretation_allowed": False},
        "implementation_scope": "Pure retention component accepts development/QA records only; no final seed generator or B1-E execution entrypoint",
        "development_only": True, "confirmatory": False, "reusable_as_final": False,
    }
    dump(V11 / "TRACE_RETENTION_POLICY.json", policy)
    summaries = measure["summary"]
    a_allowance = 8192  # More than twice the measured max P7 A blob, rounded to FS blocks.
    manifest_allowance = 4096
    p7 = summaries["P7"]
    b = block_bytes(p7["B"]["max_bundle_bytes"])
    full = block_bytes(p7["C_all"]["max_bundle_bytes"])
    diag = block_bytes(p7["C_diagnostic"]["max_bundle_bytes"])
    normal_bundle = b + diag + a_allowance + manifest_allowance
    all_full_bundle = full + a_allowance + manifest_allowance
    audited = math.ceil(n / 1024)
    p7_typical = (n - audited) * normal_bundle + audited * all_full_bundle
    p7_all_failures = n * all_full_bundle
    # P5-v1 has no architecture benchmark. Its small engineering budget is
    # conservatively costed using measured P6 maximum full-trace bytes and an
    # extra 13/12 factor for the additional witness episode. This is declared
    # explicitly as a planning proxy, not as an observed P5 architecture result.
    p6_full = block_bytes(summaries["P6"]["C_all"]["max_bundle_bytes"])
    p5_proxy = block_bytes(summaries["P6"]["C_all"]["max_bundle_bytes"] * 13 / 12)
    engineering_bytes = 32 * (p6_full + p5_proxy + 2 * (a_allowance + manifest_allowance))
    disk_contingency = 1.5
    staging_reserve = 2 * 1024**3
    typical_total = math.ceil((p7_typical + engineering_bytes) * disk_contingency) + staging_reserve
    all_failure_total = math.ceil((p7_all_failures + engineering_bytes) * disk_contingency) + staging_reserve
    disk = shutil.disk_usage(V11)
    # Conservative upper planning charge: apply the entire archived retention
    # audit's P5+P6+P7 conversion CPU/wall time to every single P7 bundle.
    # This includes JSON parsing and duplicate B/C compression. A future direct
    # writer may be faster; no such speedup is assumed.
    p7_episode_seconds = p7["original_episode_wall_seconds"] / 32
    serialization_seconds = measure["wall_seconds"] / 32
    calibration = json.loads((ROOT / "CALIBRATION_TECHNICAL.json").read_text())
    residual = max(0, calibration["wall_seconds"] - sum(summaries[p]["original_episode_wall_seconds"] for p in summaries)) / 32
    p6_episode_seconds = summaries["P6"]["original_episode_wall_seconds"] / 32
    engineering_seconds = 32 * (p6_episode_seconds * (1 + 13 / 12) + 2 * (serialization_seconds + residual))
    serial_seconds = n * (p7_episode_seconds + serialization_seconds + residual) + engineering_seconds
    cpu_contingency = 1.5
    workers = 4
    effective_speedup = 3
    cpu_quota = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
    available_cpus = None if cpu_quota[0] == "max" else int(cpu_quota[0]) / int(cpu_quota[1])
    available_memory = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
    memory_worker_allowance = 512 * 1024**2
    memory_total = workers * memory_worker_allowance + 1024**3
    certificate = {
        "version": "B1-E-v2-prospective-resource-certificate-1", "N": n, "N_v1": stats["N_v1"],
        "N_meaning": "Independent paired P7 bundles; no confirmatory bundles have been generated or executed",
        "future_schedule": stats["future_schedule"],
        "pilots": {"P7": "confirmatory-eligible primary family", "P5": "32 engineering negative-control bundles", "P6": "32 engineering negative-control bundles"},
        "episode_schedule": {"P7_and_P6": ["C0", "C1", "C2_cycle1", "C2_cycle2", "C3", "C4", "C5", "C6", "C1_kappa1_intact", "C1_kappa1_lesion", "C1_kappa2_intact", "C1_kappa2_lesion"],
                             "P5": "Same 12 arms plus C4_FULL_REFILL_CEILING_WITNESS", "source_diagnostic_events_per_pilot_bundle": 24,
                             "epochs_per_episode": 32, "cells_per_episode": 3, "P7_records_per_bundle": 1176},
        "measurement_sources": {"retention/MEASUREMENTS.json": sha(V11 / "retention/MEASUREMENTS.json"),
                                "B1E_STATISTICAL_REDESIGN.json": sha(V11 / "B1E_STATISTICAL_REDESIGN.json"),
                                "RESOURCE_MATCHING_CERTIFICATE.json": sha(V11 / "RESOURCE_MATCHING_CERTIFICATE.json"),
                                "TRACE_RETENTION_POLICY.json": sha(V11 / "TRACE_RETENTION_POLICY.json")},
        "disk": {"units": "bytes; decimal GB reported only as bytes/1e9", "filesystem_block_bytes": 4096,
                 "P7_level_B_max_measured_bytes": p7["B"]["max_bundle_bytes"], "P7_level_C_full_max_measured_bytes": p7["C_all"]["max_bundle_bytes"],
                 "P7_level_A_max_measured_bytes": p7["level_A_max_bundle_bytes"], "level_A_per_bundle_allowance_bytes": a_allowance,
                 "manifest_per_bundle_allowance_bytes": manifest_allowance,
                 "P7_normal_bundle_allocated_bytes": normal_bundle, "P7_full_bundle_allocated_bytes": all_full_bundle,
                 "P7_audit_subset_bundles": audited, "engineering_pilots_full_trace_bytes": engineering_bytes,
                 "P5_estimation_proxy": "P6 observed maximum full trace times 13/12, because v1 P5 had no architecture runs; 32 engineering bundles only",
                 "typical_precontingency_bytes": p7_typical + engineering_bytes,
                 "all_failures_precontingency_bytes": p7_all_failures + engineering_bytes,
                 "contingency_factor": disk_contingency, "staging_and_system_reserve_bytes": staging_reserve,
                 "typical_total_bytes": typical_total, "all_failures_total_bytes": all_failure_total,
                 "available_free_bytes": disk.free, "all_failures_remaining_margin_bytes": disk.free - all_failure_total,
                 "typical_fits": typical_total < disk.free, "all_failures_observed_max_projection_fits": all_failure_total < disk.free,
                 "all_failures_scenario": "All 138688 P7 bundles and all 64 engineering bundles retain full C. Full C directly supplies B, so no duplicate B storage. Model failures are not assumed rare.",
                 "unconditional_worst_case_bound": False,
                 "limits": "Observed maxima times 1.5 are a conservative planning envelope, not a theorem bounding every future trace or compression ratio. Unanticipated trace size or reduced disk triggers the technical storage gate; completed evidence is never dropped."},
        "CPU_and_wall_time": {"P7_original_12_episode_seconds_per_bundle": p7_episode_seconds,
                              "all_pilot_retention_conversion_seconds_charged_per_P7_bundle": serialization_seconds,
                              "diagnostic_and_runner_residual_seconds_per_bundle": residual,
                              "engineering_seconds": engineering_seconds,
                              "serial_seconds_precontingency": serial_seconds,
                              "contingency_factor": cpu_contingency,
                              "CPU_seconds_planning_budget": serial_seconds * cpu_contingency,
                              "CPU_hours_planning_budget": serial_seconds * cpu_contingency / 3600,
                              "worker_processes": workers, "effective_parallel_speedup": effective_speedup,
                              "parallel_efficiency_assumption": effective_speedup / workers,
                              "wall_seconds_planning_budget": serial_seconds * cpu_contingency / effective_speedup,
                              "wall_hours_planning_budget": serial_seconds * cpu_contingency / effective_speedup / 3600,
                              "runtime_guarantee": False,
                              "scope": "Linear same-host estimate from archived runtime and measured retention conversion; 4 single-thread workers, 75% parallel efficiency, no assumed algorithmic speedup. Final CPU allocation must be authorized."},
        "memory": {"retention_measurement_peak_RSS_bytes": measure["max_rss_kib"] * 1024,
                   "per_worker_allowance_bytes": memory_worker_allowance,
                   "worker_processes": workers, "coordinator_allowance_bytes": 1024**3,
                   "total_allowance_bytes": memory_total, "cgroup_limit_bytes": available_memory,
                   "estimated_fit": memory_total < available_memory,
                   "limits": "Measured conversion peaked below 512 MiB; 3 GiB aggregate planning allowance is not a portable worst-case heap bound"},
        "available_infrastructure": {"measured_utc": datetime.now(timezone.utc).isoformat(),
                                     "cgroup_cpu_quota": available_cpus, "logical_cpu_count": os.cpu_count(),
                                     "memory_bytes": available_memory, "disk_free_bytes": disk.free,
                                     "same_host_assumed": True, "future_resources_reserved": False,
                                     "future_CPU_time_budget_authorized": False, "future_storage_allocation_authorized": False},
        "feasible": typical_total < disk.free and all_failure_total < disk.free and memory_total < available_memory and available_cpus >= workers,
        "feasibility_kind": "estimated_feasible_on_observed_host_with_declared_contingencies_and_fail_closed_capacity_gate",
        "unconditional_completion_guaranteed": False,
        "ready_for_b1e_confirmatory_run": False,
        "remaining_run_gates": ["preregistration", "final_code_and_config_freeze", "independent_seed_custody", "resource_authorization"],
        "B1E_executed": False, "final_seeds_generated": False, "historical_results_modified": False,
        "development_only": True, "confirmatory": False, "reusable_as_final": False,
    }
    dump(V11 / "B1E_RESOURCE_CERTIFICATE.json", certificate)
    policy_text = f"""# Prospective trace retention policy v2

Every future bundle keeps Level A seed/commit/config/task identity, comparator outcomes, guardrails, invariant audit and trace hashes. Level B keeps the complete original external task event, including requested/admitted/executed operations, effects and both mediator states, plus complete route payloads and context. The omission is verbose internal memory, not causal task evidence.

Full Level C is selected before outcomes for indices divisible by 1024 (zero-based; {audited} P7 bundles). All controller failures, guardrail violations, failed invariant audits and every deterministic source/sham diagnostic are retained fully. A complete C record contains every B field, so the verified field projection supplies B without storing a duplicate file. This remains true in the all-failure scenario.

Full records are spooled until the final bundle audit. Malformed or incomplete audits preserve staged evidence and reject publication. A failed audit retains A and full C, publishes only FAILED_MANIFEST.json with complete=false, and raises an error. The writer requires a duplicate-free plan of complete event IDs before accepting records, freezes its SHA256, and verifies exact coverage before publication. It reuses the exporter's task-event schema and historical identity adapter, checks development flags and event identity, and requires complete route reports for architecture trials. Duplicate, unexpected, malformed or extra records permanently prevent completion even if a caller later supplies a PASS audit; serializable rejected records remain staged. SHA256 manifests record the plan hash and the successful exact-ID coverage check. The writer accepts development/QA identities only; no confirmatory execution driver exists here.

The measured archive contained {measure['source_records']} records in {measure['source_shards_verified']} hash-verified shards. All causal projections and compressed round trips were checked. Replaying 24 archived episodes (all 12 frozen arms for P6 and P7 at bundle 32) reproduced scientific event hashes, exact loss, guardrails and operation/memory counters. Timing is excluded from bitwise equality. This is a scoped replay check, not a universal cross-platform certification. Feasibility does not depend on replay: required causal evidence is retained directly.

At N={n}, typical storage is {typical_total / 1e9:.6f} GB and the all-full-trace scenario is {all_failure_total / 1e9:.6f} GB, including filesystem blocks, metadata allowances, 1.5 contingency and 2 GiB staging/system reserve. Available disk was {disk.free / 1e9:.6f} GB. The all-failure calculation assumes every bundle needs full C; it does not assume rare failures. Observed maximum trace sizes are an empirical planning basis, not an unconditional compression bound. A pre-bundle capacity gate must stop technical collection as incomplete if the reserved full-trace allowance is unavailable. All completed evidence remains; fixed N cannot be shortened, extended based on outcomes, or interpreted confirmatorily after technical truncation.

The resource certificate estimates {certificate['CPU_and_wall_time']['CPU_hours_planning_budget']:.3f} CPU-hours and {certificate['CPU_and_wall_time']['wall_hours_planning_budget']:.3f} hours elapsed with 4 workers at 75% parallel efficiency. These are same-host projections, not reservations. Future execution remains blocked until preregistration, final freeze, independent seed custody and resource authorization. No final seeds or B1-E runs were produced.

Reproduce arithmetic only: `python -B -m b1.v1_1.retention.certificate`. Run temporary-output QA: `python -B -m unittest b1.v1_1.retention.test_retention -v`. Re-measure archived evidence (overwrites only the new measurements file): `python -B -m b1.v1_1.retention.measure`.
"""
    (V11 / "TRACE_RETENTION_POLICY.md").write_text(policy_text)
    return certificate


if __name__ == "__main__":
    certificate = build()
    print(json.dumps({"feasible": certificate["feasible"], "N": certificate["N"],
                      "typical_disk_GB": certificate["disk"]["typical_total_bytes"] / 1e9,
                      "all_failures_disk_GB": certificate["disk"]["all_failures_total_bytes"] / 1e9,
                      "CPU_hours": certificate["CPU_and_wall_time"]["CPU_hours_planning_budget"],
                      "wall_hours": certificate["CPU_and_wall_time"]["wall_hours_planning_budget"]}, indent=2))
