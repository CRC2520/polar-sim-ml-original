"""Stage-gate tests use synthetic closure files; no calibration seed is derived."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b1.run import (COMPARATORS, CONTRACTS, StageError, calibration, choose_configurations,
                    create_stage, digest, execute_trial, get_configurations,
                    historical_exclusions, main, opaque_arm, redact_identity,
                    snapshot, source_files, tuning, verify_instrument,
                    verify_pre_tuning, verify_snapshot)


ZERO_GUARDS = {key.removeprefix("max_"): 0 for key in CONTRACTS.rules["thresholds"]["guardrails"]}


class RunnerGateTests(unittest.TestCase):
    def frozen_fixture(self, root):
        """Artificial file-check fixture, never a real completed research stage."""
        (root / "fixture.py").write_text("# gate fixture, no executable experiment\n")
        (root / "B1D_CONFIG.json").write_text('{"synthetic_fixture":true}\n')
        (root / "seeds").mkdir()
        (root / "seeds/HISTORICAL_SEED_EXCLUSIONS.json").write_text('{"seeds":[42]}\n')
        names = source_files(root)
        frozen = {"status": "IMPLEMENTATION_CONFIG_FROZEN_BEFORE_SCORES",
                  "source_files": names, "artifact_sha256": snapshot(root, names)}
        (root / "PRE_TUNING_FREEZE.json").write_text(json.dumps(frozen))
        (root / "evidence.txt").write_text("synthetic unit gate fixture\n")
        gate = {"status": "PASS", "pre_tuning_freeze_sha256": digest(root / "PRE_TUNING_FREEZE.json"),
                "artifact_sha256": snapshot(root, ["evidence.txt"])}
        (root / "INSTRUMENT_GATE.json").write_text(json.dumps(gate))
        return frozen

    def test_valid_snapshot_and_instrument_are_verified_without_seed_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.frozen_fixture(root)
            with patch("b1.run.new_seeds", side_effect=AssertionError("must not generate seeds")):
                self.assertEqual(verify_pre_tuning(root), expected)
                self.assertEqual(verify_instrument(root), expected)

    def test_changed_code_and_new_unfrozen_code_are_rejected(self):
        for modification in ("changed", "added"):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.frozen_fixture(root)
                if modification == "changed":
                    (root / "fixture.py").write_text("# changed after looking at outcomes\n")
                else:
                    (root / "new_policy.py").write_text("# new unregistered controller\n")
                with self.assertRaises(StageError):
                    verify_pre_tuning(root)

    def test_modified_test_or_configuration_is_included_in_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.frozen_fixture(root)
            (root / "tests").mkdir()
            (root / "tests/test_added.py").write_text("# changed validation after freeze\n")
            with self.assertRaises(StageError):
                verify_pre_tuning(root)
            (root / "tests/test_added.py").unlink()
            (root / "B1D_CONFIG.json").write_text('{"selected_after_holdout":true}\n')
            with self.assertRaises(StageError):
                verify_pre_tuning(root)

    def test_snapshot_cannot_escape_manifest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageError):
                verify_snapshot(Path(tmp), {"../outside": "0" * 64})

    def test_failed_or_changed_instrument_blocks_tuning_before_any_bundle(self):
        for kind in ("missing", "failed", "tampered_evidence"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.frozen_fixture(root)
                if kind == "missing":
                    (root / "INSTRUMENT_GATE.json").unlink()
                elif kind == "failed":
                    gate = json.loads((root / "INSTRUMENT_GATE.json").read_text())
                    gate["status"] = "FAIL"
                    (root / "INSTRUMENT_GATE.json").write_text(json.dumps(gate))
                else:
                    (root / "evidence.txt").write_text("unverified replacement\n")
                with patch("b1.run.sample_bundle", side_effect=AssertionError("must not inspect any bundle")) as sampler:
                    with self.assertRaises(StageError):
                        tuning(root)
                    sampler.assert_not_called()

    def test_missing_tuning_closure_blocks_calibration_before_seed_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.frozen_fixture(root)
            with patch("b1.run.new_seeds", side_effect=AssertionError("must not create calibration seeds")) as seeds:
                with self.assertRaises(ValueError):
                    calibration(root)
                seeds.assert_not_called()

    def test_consumed_calibration_cannot_be_reopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.frozen_fixture(root)
            (root / "CALIBRATION_STARTED.json").write_text('{"calibration_consumed":true}\n')
            # Synthetic token is never given to DevelopmentSeeds; this tests
            # the one-use CLI gate, not the cryptographic verifier itself.
            with patch("b1.run.verify_tuning_freeze", return_value=object()), patch("b1.run.new_seeds") as seeds:
                with self.assertRaises(StageError):
                    calibration(root)
                seeds.assert_not_called()

    def test_attempt_directories_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = create_stage(root, "tuning")
            (original / "failed_attempt.txt").write_text("failure retained\n")
            with self.assertRaises(StageError):
                create_stage(root, "tuning")
            self.assertEqual((original / "failed_attempt.txt").read_text(), "failure retained\n")

    def test_historical_registry_is_required_and_strictly_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageError):
                historical_exclusions(root)
            (root / "seeds").mkdir()
            path = root / "seeds/HISTORICAL_SEED_EXCLUSIONS.json"
            for invalid in ([True], [-1], [2**64], [1.5]):
                path.write_text(json.dumps({"seeds": invalid}))
                with self.assertRaises(StageError):
                    historical_exclusions(root)
            path.write_text('{"seeds":[0,42]}\n')
            self.assertEqual(historical_exclusions(root), [0, 42])


class RunnerSelectionTests(unittest.TestCase):
    def complete_grid(self):
        return [{"pilot": pilot, "comparator": comparator, "config_id": config["config_id"],
                 "bundle_index": index, "cycle": cycle, "loss": 0.5,
                 "useful_operations": 100, "controller_failure": False,
                 "guardrails": dict(ZERO_GUARDS)}
                for pilot in ("P6", "P7") for comparator in COMPARATORS
                for config in get_configurations(comparator)
                for index in range(32)
                for cycle in ((1, 2) if comparator == "C2" else (1,))]

    def test_full_grid_ties_use_operation_count_then_stable_id(self):
        rows = self.complete_grid()
        for row in rows:
            if row["comparator"] == "C1" and row["config_id"] in ("cfg03", "cfg04"):
                row["useful_operations"] = 99
        selected, summary = choose_configurations(rows)
        self.assertEqual(len(rows), 3136)
        self.assertEqual(len(summary), 2 * (8 + 8 + 8 + 8 + 1 + 8))
        for pilot in ("P6", "P7"):
            self.assertEqual(selected[pilot]["C1"], "cfg03")
            self.assertEqual(selected[pilot]["C6"], "cfg03")
            self.assertEqual(selected[pilot]["C4"], "fixed")

    def test_c2_both_cycles_are_averaged_and_never_selected(self):
        rows = self.complete_grid()
        for row in rows:
            if row["comparator"] == "C2" and row["config_id"] == "cfg00":
                row["loss"] = 0.0 if row["cycle"] == 1 else 1.0
            if row["comparator"] == "C2" and row["config_id"] == "cfg01":
                row["loss"] = 0.4
        selected, summary = choose_configurations(rows)
        self.assertEqual(selected["P6"]["C2"], "cfg01")
        first = next(r for r in summary if r["pilot"] == "P6" and r["comparator"] == "C2" and r["config_id"] == "cfg00")
        self.assertEqual(first["mean_loss"], 0.5)
        self.assertEqual(first["trials"], 64)
        self.assertEqual(first["cycles_retained"], [1, 2])

    def test_failed_attempt_loss_is_included_and_missing_or_duplicate_rows_block(self):
        rows = self.complete_grid()
        rows[0]["loss"], rows[0]["controller_failure"] = 1.0, True
        selected, summary = choose_configurations(rows)
        self.assertEqual(selected["P6"]["C0"], "cfg01")
        first = next(r for r in summary if r["pilot"] == "P6" and r["comparator"] == "C0" and r["config_id"] == "cfg00")
        self.assertEqual(first["failed_attempts"], 1)
        self.assertEqual(first["mean_loss"], 16.5 / 32)
        for bad in (rows[1:], rows + [deepcopy(rows[0])]):
            with self.assertRaises(StageError):
                choose_configurations(bad)

    def test_instrument_crash_retained_but_blocks_instead_of_becoming_controller_loss(self):
        with patch("b1.run.run_episode", side_effect=RuntimeError("synthetic instrument failure")):
            with self.assertRaises(StageError) as failure:
                execute_trial({"pilot": "P7", "bundle_index": 0}, "C1", "cfg00")
        row = failure.exception.attempt
        self.assertIsNone(row["loss_exact"])
        self.assertFalse(row["controller_failure"])
        self.assertTrue(row["instrument_failure"])
        self.assertTrue(row["execution_exception"])
        self.assertIn("synthetic instrument failure", row["failures"][0]["cause"])
        self.assertFalse(row["reusable_as_final"])

    def test_actual_or_missing_hard_guardrails_stop_selection(self):
        for guardrails in ({}, {**ZERO_GUARDS, "unauthorized_executions": 1}):
            rows = self.complete_grid()
            rows[0]["guardrails"] = guardrails
            with self.assertRaises(StageError):
                choose_configurations(rows)

    def test_masking_removes_nested_literal_comparator_and_config_identifiers(self):
        source = {"comparator": "C1", "config_id": "cfg00", "controller_report": {"comparator": "C1", "config_id": "cfg00"}, "loss": 0.25}
        masked = redact_identity(source, "arm-opaque")
        self.assertEqual(masked["comparator"], "arm-opaque")
        self.assertEqual(masked["controller_report"]["config_id"], "arm-opaque")
        self.assertEqual(masked["loss"], source["loss"])
        self.assertEqual(source["comparator"], "C1")
        self.assertNotEqual(opaque_arm("f" * 64, "P6", "C1"), opaque_arm("f" * 64, "P6", "C2"))

    def test_cli_has_no_final_or_b1e_execution_entrypoint(self):
        for command in ("final", "confirmatory", "b1e"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                main([command])
            self.assertEqual(result.exception.code, 2)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["tuning", "--output", "/tmp/not-used"]), 2)


if __name__ == "__main__":
    unittest.main()
