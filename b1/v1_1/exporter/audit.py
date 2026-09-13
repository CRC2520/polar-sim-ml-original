"""Assemble the separately versioned exporter closure audit from verified files."""
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile

from .core import ROOT, FLAGS, digest, document, export_qa, read_json, verify_v1


def main():
    directory = ROOT / "v1_1/exporter"
    transcript = (directory / "TEST_RESULTS.txt").read_text()
    match = re.search(r"Ran (\d+) tests in ([0-9.]+)s\n\nOK\s*$", transcript)
    if not match or int(match.group(1)) != 14:
        raise RuntimeError("A complete passing exporter test transcript is required")
    historic = read_json(directory / "archive_replay/AUDIT.json")
    incident = read_json(ROOT / "v1_1/EXPORTER_ROOT_CAUSE.json")
    qa = read_json(directory / "qa_run/QA_AUDIT.json")
    with tempfile.TemporaryDirectory() as temporary:
        qdir = directory / "qa_run"
        verified_qa = export_qa(qdir / "traces", qdir / "EXPECTED_PLAN.json",
                               qa["expected_plan_sha256_before_execution"], Path(temporary) / "export",
                               index_sha256=digest(qdir / "traces/INDEX.json"))["audit"]
    if not (historic["coverage_pass"] and qa["export_pass"] and verified_qa["coverage_pass"]):
        raise RuntimeError("Historical or current QA completeness failed")
    source_names = ["__init__.py", "core.py", "qa.py", "test_exporter.py", "investigate.py", "audit.py"]
    result = {"version": "1.1", "status": "PASS", "blocker_id": "B1D-ORIGINAL-EXPORT-PROVENANCE-INCIDENT",
              "blocker_disposition": "RESOLVED_FOR_FUTURE_PIPELINE", "matrix_status": "RESOLVED",
              "scope": "new pipeline only; original failed export and root-cause uncertainty remain immutable provenance",
              "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "change_classification": "exporter_only", "scientific_rerun_required": False,
              "original_export_status": "FAILED_RETAINED_UNCHANGED", "root_cause": incident["root_cause"],
              "root_cause_hypotheses_investigated": len(incident["investigated_hypotheses"]),
              "historical_preservation": verify_v1(), "archive_replay": historic,
              "QA": {"namespace": qa["namespace"], "trial_count": qa["trial_count"],
                     "trace_records": qa["trace_records"], "pipeline": qa["pipeline"],
                     "counts_as_experimental_evidence": False, "C6_checks": qa["C6_checks"],
                     "guardrail_violations": qa["guardrail_violations"], "controller_failures": qa["controller_failures"],
                     "current_exporter_revalidation_of_retained_QA": verified_qa},
              "tests": {"command": "python -B -m unittest b1.v1_1.exporter.test_exporter -v",
                        "count": int(match.group(1)), "seconds": float(match.group(2)), "status": "PASS",
                        "corruptions": ["missing shard", "missing row with rehashed archive", "all nine missing key dimensions",
                            "duplicate complete ID", "changed route/cycle/context", "corrupted shard hash", "missing causal field",
                            "missing state mediator", "malformed channel container", "negative dose", "malformed effect",
                            "missing effect mediator", "wrong namespace", "invalid final row prevents prefix publication"],
                        "order_independence": True, "manifest_hash_verified": True, "preexisting_export_not_overwritten": True},
              "implementation_sha256": {f"b1/v1_1/exporter/{name}": digest(directory / name) for name in source_names},
              "evidence_sha256": {f"b1/v1_1/{name}": digest(ROOT / "v1_1" / name) for name in
                  ("EXPORTER_ROOT_CAUSE.json", "EXPORTER_ROOT_CAUSE.md", "exporter/TEST_RESULTS.txt",
                   "exporter/archive_replay/MANIFEST.json", "exporter/archive_replay/bundle_diagnostics.jsonl",
                   "exporter/qa_run/QA_AUDIT.json", "exporter/qa_run/EXPECTED_PLAN.json", "exporter/qa_run/traces/INDEX.json")},
              "new_scientific_runs_by_exporter": 0, "final_seeds_generated": False, "B1E_executed": False,
              "ready_for_b1e_confirmatory_run": False,
              "remaining_limitations": incident["remaining_limitations"], **FLAGS}
    (ROOT / "v1_1/EXPORTER_V1_1_AUDIT.json").write_bytes(document(result))


if __name__ == "__main__":
    main()
