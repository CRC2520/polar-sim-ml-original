import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from .policy import (causal_record, full_trace_required, storage_gate,
                     RetentionBudgetExceeded, encode_records, decode_records)
from .writer import RetainedBundle
from .validation import record_key


def fixture():
    path = Path(__file__).resolve().parents[2] / "development_results/calibration/traces/trace-00000.jsonl.gz"
    with gzip.open(path, "rb") as stream:
        row = json.loads(stream.readline())
    row["bundle_index"] = 1
    return row


class RetentionTests(unittest.TestCase):
    def test_causal_fields_and_lossless_codec(self):
        row = fixture()
        projected = causal_record(row)
        self.assertEqual(row["event"], projected["event"])
        self.assertEqual(decode_records(encode_records([projected])), [projected])
        bad = copy.deepcopy(row)
        del bad["event"]["channels"]["B"]["executed_operation"]
        with self.assertRaises(ValueError):
            causal_record(bad)

    def test_subset_independent_of_outcome_and_all_exceptions(self):
        self.assertTrue(full_trace_required(0))
        self.assertTrue(full_trace_required(1024))
        self.assertFalse(full_trace_required(1))
        self.assertTrue(full_trace_required(1, failed=True))
        self.assertTrue(full_trace_required(1, guardrails={"unauthorized_executions": 1}))
        self.assertTrue(full_trace_required(1, deterministic_diagnostic=True))

    def test_precollection_budget_gate(self):
        with self.assertRaises(RetentionBudgetExceeded):
            storage_gate(".", next_bundle_worst_case_bytes=100, reserve_bytes=10, available_bytes=109)
        self.assertTrue(storage_gate(".", next_bundle_worst_case_bytes=100, reserve_bytes=10, available_bytes=110))

    def writer(self, folder, records=None):
        records = [fixture()] if records is None else records
        return RetainedBundle(folder, {"pilot": "P5", "bundle_index": 1,
            "source_commit": "0" * 40, "config_sha256": "0" * 64, "task_sha256": "0" * 64,
            "seed_usage": [{"namespace": "PD-B1-D-v1.1-QA", "synthetic_fixture": True}],
            "development_only": True, "confirmatory": False, "reusable_as_final": False},
            expected_records=len(records), expected_ids=[record_key(row) for row in records],
            reserve_bytes=0, worst_case_bundle_bytes=10000)

    def test_duplicate_trace_cannot_replace_missing_expected_id(self):
        first = fixture()
        second = copy.deepcopy(first)
        second["event"]["epoch"] += 1
        for field in ("state_before", "state_after"):
            second["event"][field]["epoch"] += 1
        with tempfile.TemporaryDirectory() as folder:
            writer = self.writer(folder, [first, second])
            writer.append(first)
            with self.assertRaisesRegex(ValueError, "Duplicate complete"):
                writer.append(first)
            with self.assertRaises(ValueError):
                writer.finish([{"loss": 0}], {"status": "PASS", "checks": {"execution": True}})
            self.assertFalse((Path(folder) / "MANIFEST.json").exists())
            self.assertTrue(any(Path(folder).glob("retention-stage-*")))

    def test_corrupt_event_identity_flags_and_missing_dimension_fail_closed(self):
        for corruption in ("event_pilot", "event_flags", "record_flags", "missing_context", "wrong_epoch"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                writer = self.writer(folder)
                row = fixture()
                if corruption == "event_pilot":
                    row["event"]["pilot"] = "P7"
                elif corruption == "event_flags":
                    row["event"]["confirmatory"] = True
                elif corruption == "record_flags":
                    row["confirmatory"] = True
                elif corruption == "missing_context":
                    del row["context"]
                else:
                    row["event"]["epoch"] = 17
                with self.assertRaises((ValueError, RuntimeError)):
                    writer.append(row)
                with self.assertRaises(ValueError):
                    writer.finish([{"loss": 0}], {"status": "PASS", "checks": {"execution": True}})
                self.assertFalse((Path(folder) / "MANIFEST.json").exists())

    def test_rejected_extra_event_poisons_otherwise_complete_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = self.writer(folder)
            writer.append(fixture())
            with self.assertRaises(ValueError):
                writer.append(fixture())
            with self.assertRaises(ValueError):
                writer.finish([{"loss": 0}], {"status": "PASS", "checks": {"execution": True}})
            self.assertFalse((Path(folder) / "MANIFEST.json").exists())

    def test_trial_requires_complete_route_report(self):
        path = Path(__file__).resolve().parents[1] / "p5/results/traces/trace-00000.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            row = json.loads(stream.readline())
        valid_key = record_key(row)
        for missing in ("controller_report", "route_slots", "route_consumed", "joint_manipulation"):
            with self.subTest(missing=missing):
                invalid = copy.deepcopy(row)
                if missing == "controller_report":
                    del invalid[missing]
                else:
                    del invalid["controller_report"][missing]
                with self.assertRaisesRegex(ValueError, "Incomplete causal route"):
                    record_key(invalid)
        with tempfile.TemporaryDirectory() as folder:
            writer = RetainedBundle(folder, {"pilot": row["pilot"], "bundle_index": row["bundle_index"],
                "source_commit": row["source_commit"], "config_sha256": row["config_sha256"],
                "task_sha256": "0" * 64,
                "seed_usage": [{"namespace": "PD-B1-D-v1.1", "seed": 1}],
                "development_only": True, "confirmatory": False, "reusable_as_final": False},
                expected_records=1, expected_ids=[valid_key], reserve_bytes=0, worst_case_bundle_bytes=10000)
            writer.append(row)
            result = writer.finish([{"comparator": row["comparator"], "loss": 0}],
                                   {"status": "PASS", "checks": {"execution": True}})
            self.assertTrue(result["complete_event_ID_coverage_pass"])
            self.assertEqual(len(result["expected_plan_sha256"]), 64)

    def test_frozen_conventional_p5_no_route_schema_is_preserved(self):
        path = Path(__file__).resolve().parents[1] / "p5/results/traces/trace-00000.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            row = next(row for line in stream if (row := json.loads(line))["comparator"] == "C4")
        record_key(row)
        self.assertNotIn("route_use_masked", row["controller_report"])
        row["controller_report"]["route_slots"] = [{"invented": True}]
        with self.assertRaisesRegex(ValueError, "cannot contain a route"):
            record_key(row)

    def test_diagnostics_directly_retained_and_final_failure_keeps_full(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                writer = self.writer(folder)
                writer.append(fixture())
                result = writer.finish([{"comparator": "C1", "loss_exact": [0, 1],
                    "controller_failure": fail}], {"status": "PASS", "checks": {"execution": True}})
                self.assertEqual(result["full_trace_required"], fail)
                self.assertTrue((Path(folder) / ("C_full.jsonl.gz" if fail else "C_diagnostic.jsonl.gz")).exists())
                self.assertTrue((Path(folder) / "A.json.gz").exists())
                self.assertEqual((Path(folder) / "B.jsonl.gz").exists(), not fail)
                self.assertTrue((Path(folder) / "MANIFEST.json").exists())

    def test_partial_output_never_published(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = self.writer(folder)
            with self.assertRaises(ValueError):
                writer.finish([{"loss": 0}], {"status": "PASS", "checks": {"execution": True}})
            self.assertFalse((Path(folder) / "MANIFEST.json").exists())
            self.assertTrue(any(Path(folder).glob("retention-stage-*")))

    def test_failed_invariant_retains_full_and_rejects_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = self.writer(folder)
            writer.append(fixture())
            with self.assertRaises(ValueError):
                writer.finish([{"loss": 0}], {"status": "FAIL", "checks": {"route_audit": False}})
            self.assertFalse((Path(folder) / "MANIFEST.json").exists())
            self.assertTrue((Path(folder) / "C_full.jsonl.gz").exists())
            self.assertFalse(json.loads((Path(folder) / "FAILED_MANIFEST.json").read_text())["complete"])

    def test_malformed_invariant_does_not_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = self.writer(folder)
            writer.append(fixture())
            with self.assertRaises(ValueError):
                writer.finish([{"loss": 0}], {"execution": "FAILED"})
            self.assertFalse((Path(folder) / "MANIFEST.json").exists())
            self.assertTrue(any(Path(folder).glob("retention-stage-*")))

    def test_p5_failure_and_cap_binding_force_full_retention(self):
        for failure in ({"controller_failures": [{"epoch": 0, "cause": "decision failure"}]},
                        {"resource_cap_binding": True}):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                writer = self.writer(folder)
                writer.append(fixture())
                manifest = writer.finish([{"loss": 0, **failure}], {"status": "PASS", "checks": {"execution": True}})
                self.assertTrue(manifest["full_trace_required"])
                self.assertTrue((Path(folder) / "C_full.jsonl.gz").exists())


if __name__ == "__main__":
    unittest.main()
