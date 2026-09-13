"""Scientific-risk gates: archive completeness, deliberate corruption, atomicity."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from . import core
from .qa import run as run_qa


def simple_record(epoch=0):
    # Use an actual retained task event, never a schema-shaped invented outcome.
    fixture = core.read_json(core.ROOT / "development_results/calibration/mechanism_diagnostics.json")
    event = deepcopy(fixture["P6"]["contexts"]["kappa1"]["source"]["trace"][epoch])
    return {"schema_version": core.VERSION, "namespace": "PD-B1-D-v1.1-QA",
            "record_kind": "qa_trial", "pilot": "P6", "bundle_index": 0,
            "comparator": "C1", "context": "kappa1", "route_status": "intact",
            "cycle": 1, "replicate_id": 0, "epoch": event["epoch"],
            "event": event, **core.FLAGS}


def make_inputs(root, records, expected=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    traces = root / "traces"
    traces.mkdir()
    raw = b"".join(core.canonical(r) + b"\n" for r in records)
    data = gzip.compress(raw, mtime=0)
    name = "trace-00000.jsonl.gz"
    (traces / name).write_bytes(data)
    index = {"shards": [{"path": name, "bytes": len(data), "records": len(records),
                         "sha256": sha256(data).hexdigest()}],
             "raw_bytes": len(raw), "compressed_bytes": len(data), "records": len(records)}
    (traces / "INDEX.json").write_bytes(core.document(index))
    if expected is None:
        expected = [tuple(r[k] for k in core.DIMENSIONS) for r in records]
    plan = {"namespace": "PD-B1-D-v1.1-QA", "fixed_before_execution": True,
            "expected_ids": list(expected)}
    (root / "PLAN.json").write_bytes(core.document(plan))
    return traces, root / "PLAN.json", core.digest(root / "PLAN.json"), core.digest(traces / "INDEX.json")


class CorruptionTests(unittest.TestCase):
    def check_failure(self, records, pattern, expected=None, mutate=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces, plan, plan_sha, index_sha = make_inputs(root, records, expected)
            if mutate:
                mutate(traces)
            destination = root / "export"
            with self.assertRaisesRegex(core.ExportError, pattern):
                core.export_qa(traces, plan, plan_sha, destination, index_sha256=index_sha)
            self.assertFalse(destination.exists(), "No partial summary may be published")
            self.assertFalse(list(root.glob(".export-stage-*")))

    def test_missing_shard_is_fail_closed(self):
        self.check_failure([simple_record()], "Missing shard", mutate=lambda p: (p / "trace-00000.jsonl.gz").unlink())

    def test_missing_row_even_rehashed_archive_is_fail_closed(self):
        records = [simple_record(0), simple_record(1)]
        expected = [tuple(r[k] for k in core.DIMENSIONS) for r in records]
        self.check_failure(records[:1], "Missing complete event IDs", expected)

    def test_each_missing_key_dimension_is_fail_closed(self):
        for dimension in core.DIMENSIONS:
            with self.subTest(dimension=dimension):
                record = simple_record()
                expected = [tuple(record[k] for k in core.DIMENSIONS)]
                del record[dimension]
                self.check_failure([record], "schema/key dimension", expected)

    def test_duplicate_complete_id_is_fail_closed(self):
        r = simple_record()
        self.check_failure([r, r], "Duplicate complete event ID", [tuple(r[k] for k in core.DIMENSIONS)])

    def test_changed_route_cycle_context_are_rejected(self):
        for field, changed in (("route_status", "bypassed"), ("cycle", 2), ("context", "kappa2")):
            with self.subTest(field=field):
                record = simple_record()
                expected = [tuple(record[k] for k in core.DIMENSIONS)]
                record[field] = changed
                self.check_failure([record], "Unexpected complete event ID", expected)

    def test_corrupt_shard_hash_is_fail_closed(self):
        def mutate(path):
            p = path / "trace-00000.jsonl.gz"
            data = p.read_bytes()
            p.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        self.check_failure([simple_record()], "Shard SHA256/size mismatch", mutate=mutate)

    def test_missing_causal_field_is_fail_closed(self):
        r = simple_record()
        del r["event"]["channels"]["A"]["external_admitted_dose"]
        self.check_failure([r], "requested/admitted/executed/effect schema")

    def test_nested_state_and_channel_corruption_preserving_ids_fails(self):
        for corruption in ("mediator", "channel_type", "negative_dose", "effect_type", "effect_mediator"):
            with self.subTest(corruption=corruption):
                r = simple_record()
                if corruption == "mediator":
                    del r["event"]["state_after"]["reports"]
                    pattern = "State/mediator schema"
                elif corruption == "channel_type":
                    r["event"]["channels"]["A"] = []
                    pattern = "requested/admitted/executed/effect schema"
                elif corruption == "negative_dose":
                    r["event"]["channels"]["A"]["executed_operation"] = -1
                    pattern = "Channel dose domain"
                elif corruption == "effect_type":
                    r["event"]["channels"]["A"]["environmental_effect"] = None
                    pattern = "Environmental effect schema"
                else:
                    del r["event"]["channels"]["A"]["environmental_effect"]["arriving_reports"]
                    pattern = "Environmental effect/mediator schema"
                self.check_failure([r], pattern)

    def test_final_namespace_rejected(self):
        r = simple_record()
        r["namespace"] = "FORBIDDEN_FINAL_SENTINEL"
        self.check_failure([r], "namespace differs from declared development plan")

    def test_invalid_last_record_does_not_publish_prefix(self):
        rows = [simple_record(0), simple_record(1)]
        rows[-1]["event"]["missed"] += 1
        self.check_failure(rows, "demand accounting")

    def test_record_order_is_irrelevant_and_manifest_hash_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [simple_record(0), simple_record(1)]
            expected = [tuple(r[k] for k in core.DIMENSIONS) for r in rows]
            traces, plan, plan_sha, index_sha = make_inputs(root, rows[::-1], expected)
            manifest = core.export_qa(traces, plan, plan_sha, root / "export", index_sha256=index_sha)
            self.assertEqual(manifest["audit"]["validated_records"], 2)
            self.assertEqual(core.digest(root / "export/MANIFEST.json"),
                             (root / "export/MANIFEST.sha256").read_text().split()[0])
            for name, expected_sha in manifest["artifact_sha256"].items():
                self.assertEqual(core.digest(root / "export" / name), expected_sha)
            with self.assertRaisesRegex(core.ExportError, "overwrite"):
                core.export_qa(traces, plan, plan_sha, root / "export", index_sha256=index_sha)

    def test_diagnostic_cycle_context_arm_recovery_is_exact(self):
        # Full archive test below validates all actual events. This tests that
        # none of the historical envelope dimensions may silently disappear.
        traces = core.ROOT / "development_results/calibration/traces"
        record = json.loads(gzip.decompress((traces / "trace-00000.jsonl.gz").read_bytes()).splitlines()[0])
        for key in ("context", "replicate_id", "diagnostic_arm", "role"):
            corrupted = deepcopy(record)
            del corrupted[key]
            with self.assertRaisesRegex(core.ExportError, "schema/key dimension"):
                core.historic_key(corrupted, {})


class EndToEndTests(unittest.TestCase):
    def test_archive_replay_96_and_all_event_dimensions(self):
        before = core.verify_v1()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "archive"
            audit = core.export_historical(out)["audit"]
            self.assertEqual(audit["validated_records"], 82176)
            self.assertEqual(audit["diagnostic_rows_exported"], 96)
            self.assertEqual(audit["original_rows_byte_identical"], 42)
            self.assertEqual(audit["previously_derived_rows_byte_identical"], 54)
            self.assertEqual(audit["new_seeds"], 0)
            self.assertEqual(audit["new_scientific_runs"], 0)
            self.assertEqual(core.verify_v1(), before)

    def test_new_namespace_task_trace_shard_export_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_qa(Path(temporary) / "qa")
            self.assertEqual(result["namespace"], "PD-B1-D-v1.1-QA")
            self.assertEqual(result["trial_count"], 16)
            self.assertEqual(result["trace_records"], 1536)
            self.assertTrue(result["export_pass"])
            self.assertFalse(result["counts_as_experimental_evidence"])
            self.assertEqual(result["expected_plan_sha256_before_execution"],
                             result["expected_plan_sha256_after_execution"])


if __name__ == "__main__":
    unittest.main()
