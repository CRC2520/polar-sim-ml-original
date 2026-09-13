"""Offline reporting recovery tests using only retained closed artifacts."""
import ast
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from b1.postprocessing import export_recovery as recovery


class ClosedTraceRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = recovery.ROOT
        cls.directory = cls.root / "development_results/calibration"
        cls.technical = recovery.read_json(cls.root / "CALIBRATION_TECHNICAL.json")
        cls.fixtures = recovery.read_json(cls.directory / "mechanism_diagnostics.json")
        cls.original = (cls.directory / "bundle_diagnostics.jsonl").read_bytes().splitlines(keepends=True)
        recovery.verify_reporting_inputs(cls.root)
        cls.archived, cls.totals = recovery.collect_archived_diagnostics(cls.directory, cls.technical)

    def reconstruct(self, archived=None, original=None):
        return recovery.reconstruct_records(
            self.archived if archived is None else archived, self.fixtures,
            self.original if original is None else original)

    def test_all_closed_events_and_original_42_rows_match_exactly(self):
        rows, checks = self.reconstruct()
        self.assertEqual(len(rows), 96)
        self.assertEqual(checks["archive_diagnostic_events_verified"], 2304)
        self.assertEqual(checks["original_rows_byte_identical"], 42)
        self.assertEqual(checks["recovered_derived_rows"], 54)
        self.assertEqual(checks["new_diagnostic_executions"], 0)
        self.assertEqual(checks["new_independent_observations"], 0)
        self.assertEqual(self.original, [recovery.canonical_bytes(r) + b"\n" for r in rows[:42]])
        self.assertEqual({(r["pilot"], r["bundle_index"]) for r in rows},
                         {(pilot, index) for pilot in ("P5", "P6", "P7") for index in range(32, 64)})
        self.assertEqual(self.totals["records"], self.technical["trace_index"]["records"])

    def test_missing_archived_event_blocks(self):
        with self.assertRaisesRegex(recovery.RecoveryError, "Incomplete archived"):
            self.reconstruct(archived=self.archived[1:])

    def test_duplicate_archived_event_blocks(self):
        with self.assertRaisesRegex(recovery.RecoveryError, "Duplicated archived"):
            self.reconstruct(archived=self.archived + [self.archived[0]])

    def test_semantic_event_corruption_blocks_fixture_metadata_reuse(self):
        corrupted = deepcopy(self.archived[0])
        corrupted["event"]["guardrails"]["physical_state_violations"] = 1
        with self.assertRaisesRegex(recovery.RecoveryError, "does not exactly equal"):
            self.reconstruct(archived=[corrupted] + self.archived[1:])

    def test_wrong_role_blocks(self):
        corrupted = dict(self.archived[0], role="tuning")
        with self.assertRaisesRegex(recovery.RecoveryError, "envelope metadata"):
            self.reconstruct(archived=[corrupted] + self.archived[1:])

    def test_duplicate_original_row_blocks(self):
        with self.assertRaisesRegex(recovery.RecoveryError, "Duplicated or unexpected original"):
            self.reconstruct(original=self.original + [self.original[0]])

    def test_nonidentical_original_row_bytes_block(self):
        row = recovery.strict_json(self.original[0].decode("utf-8"))
        altered_format = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        self.assertNotEqual(altered_format, self.original[0])
        with self.assertRaisesRegex(recovery.RecoveryError, "byte-for-byte"):
            self.reconstruct(original=[altered_format] + self.original[1:])

    def test_corrupt_compressed_shard_is_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "traces").mkdir()
            raw = recovery.canonical_bytes(self.archived[0]) + b"\n"
            data = gzip.compress(raw, mtime=0)
            shard = {"path": "trace-00000.jsonl.gz", "records": 1, "bytes": len(data),
                     "sha256": sha256(data).hexdigest()}
            index = {"shards": [shard], "records": 1, "raw_bytes": len(raw), "compressed_bytes": len(data)}
            (root / "traces/INDEX.json").write_text(json.dumps(index))
            (root / "traces/trace-00000.jsonl.gz").write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaisesRegex(recovery.RecoveryError, "checksum/size mismatch"):
                recovery.collect_archived_diagnostics(root, {"trace_index": index})

    def test_frozen_artifact_corruption_and_unsafe_path_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").write_bytes(b"closed")
            expected = sha256(b"closed").hexdigest()
            recovery.verify_snapshot(root, {"evidence": expected})
            (root / "evidence").write_bytes(b"changed")
            with self.assertRaisesRegex(recovery.RecoveryError, "Frozen artifact changed"):
                recovery.verify_snapshot(root, {"evidence": expected})
            with self.assertRaisesRegex(recovery.RecoveryError, "Unsafe snapshot path"):
                recovery.verify_snapshot(root, {"../evidence": expected})

    def test_closed_trial_schedule_rejects_duplicates(self):
        identities = recovery.read_json(self.directory / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json")["identities"]
        rows = [recovery.strict_json(line) for line in (self.directory / "masked_trials.jsonl").read_text().splitlines()]
        for row in rows:
            row.update(identities[row["arm_id"]])
        self.assertEqual(recovery.validate_trials(rows, self.technical), 768)
        with self.assertRaisesRegex(recovery.RecoveryError, "duplicated closed calibration"):
            recovery.validate_trials(rows[:-1] + [rows[0]], self.technical)

    def test_reporting_module_has_no_experimental_import_or_seed_api(self):
        tree = ast.parse(Path(recovery.__file__).read_text())
        banned = ("b1.run", "b1.seeds", "b1.tasks", "b1.controllers",
                  "b1.interventions", "b1.evaluation.runner", "random")
        for node in ast.walk(tree):
            names = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                self.assertFalse(any(name == prefix or name.startswith(prefix + ".") for prefix in banned), name)
            if isinstance(node, ast.Call):
                function = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                self.assertNotIn(function, {"derive", "open_calibration", "run_episode", "sample_bundle",
                                             "mechanism_diagnostic", "all_mechanism_diagnostics", "exec", "eval"})


if __name__ == "__main__":
    unittest.main()
