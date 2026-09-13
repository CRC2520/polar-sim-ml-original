"""Independent integration fixtures; no development/calibration RNG is consumed."""
from copy import deepcopy
from fractions import Fraction
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b1.contracts import load_contracts
from b1.controllers import Controller, C6Controller, ConjugatedTask, encode_tree, decode_tree
from b1.controllers.core import ContractConflict, ControllerFailure, Limits
from b1.evaluation.runner import run_episode, run_ceiling_witness
from b1.tasks import P5Task, P6Task, P7Task


CONTRACTS = load_contracts()
CLOCK = CONTRACTS.pilots["shared_contract"]["clock"]
HORIZON = CLOCK["episode_horizon"]
CELL_COUNT = CLOCK["replicated_cells"]


def deterministic_bundle(pilot):
    cells = []
    for cell in range(CELL_COUNT):
        args = {"cell_id": cell}
        if pilot == "P5":
            args["initial_reserve"] = 0 if cell == 0 else CONTRACTS.pilot(pilot)["task_contract"]["constants"]["reserve_capacity"]
        elif pilot == "P6":
            args.update(initial_mapping="q0" if cell == 0 else "q1",
                        flip_epoch=HORIZON // 2, transfer_eligible=cell != 1)
        else:
            args.update(initial_drifts=(1, cell % 2),
                        mode="replacement" if cell == 1 else "state_preserving",
                        version_switch_epoch=HORIZON // 2,
                        drift_events={HORIZON // 4: 0, 3 * HORIZON // 4: 1})
        cells.append(args)
    return {"pilot": pilot, "bundle_index": 0, "role": "deterministic-integration-fixture",
            "cells": cells, "development_only": True, "confirmatory": False,
            "reusable_as_final": False, "synthetic_fixture": True}


class FullHorizonConjugacyAudit(unittest.TestCase):
    def test_full_episode_including_delays_hidden_flip_version_and_drift(self):
        factories = {"P5": P5Task, "P6": P6Task, "P7": P7Task}
        for pilot, factory in factories.items():
            with self.subTest(pilot=pilot):
                bundle = deterministic_bundle(pilot)
                original_tasks = [factory(**args) for args in bundle["cells"]]
                transformed_tasks = [ConjugatedTask(deepcopy(task)) for task in original_tasks]
                original = Controller(pilot, "C1", "cfg00")
                transformed = C6Controller(deepcopy(original))
                event_count = 0
                for epoch in range(HORIZON):
                    observations = [task.observe() for task in original_tasks]
                    transformed_observations = [decode_tree(task.observe_encoded()) for task in transformed_tasks]
                    self.assertEqual(observations, transformed_observations, (pilot, epoch, "observation"))
                    actions = original.act(observations)
                    images = transformed.act_encoded(encode_tree(transformed_observations))
                    self.assertEqual(actions, decode_tree(images), (pilot, epoch, "action"))
                    self.assertEqual(original.__dict__, decode_tree(transformed.state), (pilot, epoch, "controller_state"))
                    for task, image_task, action in zip(original_tasks, transformed_tasks, actions):
                        event = task.step(action)
                        image_event = decode_tree(image_task.step_encoded(encode_tree(action)))
                        self.assertEqual(event, image_event, (pilot, epoch, task.cell_id, "event"))
                        self.assertEqual(task.__dict__, decode_tree(image_task._state), (pilot, epoch, task.cell_id, "task_state"))
                        event_count += 1
                self.assertEqual(event_count, HORIZON * CELL_COUNT)
                total_demand = sum(task.completed_jobs + task.missed_jobs for task in original_tasks)
                self.assertEqual(total_demand, CONTRACTS.rules["thresholds"]["demanded_jobs_per_episode"])

    def test_runner_c6_preserves_full_physical_trace(self):
        for pilot in ("P5", "P6", "P7"):
            bundle = deterministic_bundle(pilot)
            first = run_episode(bundle, "C1")
            recoded = run_episode(bundle, "C6")
            self.assertEqual(first["loss_exact"], recoded["loss_exact"], pilot)
            self.assertEqual(first["event_trace_sha256"], recoded["event_trace_sha256"], pilot)
            self.assertEqual(first["guardrails"], recoded["guardrails"], pilot)

    def test_c6_preserves_partial_state_when_controller_fails(self):
        tasks = [P6Task(**args) for args in deterministic_bundle("P6")["cells"]]
        observations = [task.observe() for task in tasks]
        original = Controller("P6", "C1", limits=Limits(decision_operations=1))
        recoded = C6Controller(deepcopy(original))
        with self.assertRaises(ControllerFailure) as first:
            original.act(observations)
        with self.assertRaises(ControllerFailure) as second:
            recoded.act_encoded(encode_tree(observations))
        self.assertEqual(str(first.exception), str(second.exception))
        self.assertEqual(original.__dict__, decode_tree(recoded.state))


class CeilingAndRunnerAudit(unittest.TestCase):
    def test_p5_c4_is_blocked_and_both_unselected_witnesses_attain_bound(self):
        with self.assertRaises(ContractConflict):
            run_episode(deterministic_bundle("P5"), "C4")
        capacity = CONTRACTS.pilot("P5")["task_contract"]["constants"]
        production = capacity["production_capacity_per_epoch"]
        for initial in (0, capacity["reserve_capacity"]):
            expected = production * (HORIZON - (1 if initial == 0 else 0))
            runs = []
            for witness in ("json_minimal_refill_witness", "protocol_full_refill_witness"):
                result = run_ceiling_witness("P5", witness, {"initial_reserve": initial})
                self.assertEqual(result["completed_jobs"], expected)
                self.assertFalse(result["frozen_C4_selected"])
                self.assertFalse(result["confirmatory"])
                runs.append(result)
            self.assertNotEqual([event["proposal"] for event in runs[0]["events"]],
                                [event["proposal"] for event in runs[1]["events"]])

    def test_ordinary_gated_zero_service_uses_actual_192_denominator(self):
        class ZeroController:
            def __init__(self, *args, **kwargs):
                self.memory = {}
                self.last_decision_report = {}
            def act(self, observations):
                return [{"A": Fraction(0), "B": Fraction(0)} for _ in observations]
        with patch("b1.evaluation.runner.Controller", ZeroController):
            result = run_episode(deterministic_bundle("P5"), "C1")
        self.assertEqual(result["failed_jobs_observed"], 192)
        self.assertEqual(result["loss"], 1)
        self.assertFalse(result["controller_failure"])

    def test_controller_budget_failure_retained_as_episode_loss_one(self):
        result = run_episode(deterministic_bundle("P6"), "C1",
                             limits=Limits(decision_operations=1))
        self.assertTrue(result["controller_failure"])
        self.assertEqual(result["loss"], 1)
        self.assertEqual(len(result["failures"]), HORIZON)

    def test_malformed_action_count_cannot_silently_skip_cell(self):
        class ShortController:
            def __init__(self, *args, **kwargs):
                self.memory = {}
                self.last_decision_report = {}
            def act(self, observations):
                return [{"A": 0, "B": 0}] * (len(observations) - 1)
        with patch("b1.evaluation.runner.Controller", ShortController):
            result = run_episode(deterministic_bundle("P5"), "C1")
        self.assertTrue(result["controller_failure"])
        self.assertEqual(result["loss"], 1)
        self.assertEqual(result["failed_jobs_observed"], 192)

    def test_model_exception_retained_as_loss_one(self):
        class CrashingController:
            def __init__(self, *args, **kwargs):
                self.memory = {}
                self.last_decision_report = {}
            def act(self, observations):
                raise ValueError("synthetic malformed model output")
        with patch("b1.evaluation.runner.Controller", CrashingController):
            result = run_episode(deterministic_bundle("P6"), "C1")
        self.assertTrue(result["controller_failure"])
        self.assertEqual(result["loss"], 1)

    def test_declared_missed_deadline_is_not_silently_accepted(self):
        # Zero deadline deliberately forces failure; no empirical runtime
        # threshold is selected and this does not certify any resource cap.
        result = run_episode(deterministic_bundle("P6"), "C1",
                             limits=Limits(latency_seconds=0.0))
        self.assertTrue(result["controller_failure"])
        self.assertEqual(result["loss"], 1)


class StageBoundaryAudit(unittest.TestCase):
    def test_runner_rejects_final_missing_flags_and_cross_block_before_task_creation(self):
        base = deterministic_bundle("P6")
        base.pop("synthetic_fixture")
        base["role"] = "tuning"
        invalid = []
        final_marked = deepcopy(base)
        final_marked["confirmatory"] = True
        invalid.append(final_marked)
        for flag in ("development_only", "confirmatory", "reusable_as_final"):
            missing = deepcopy(base)
            missing.pop(flag)
            invalid.append(missing)
        final_role = deepcopy(base)
        final_role["role"] = "evaluation-final"
        invalid.append(final_role)
        for role, index in (("tuning", 32), ("calibration", 31),
                            ("evaluation-development", 63), ("calibration", 64)):
            cross_block = deepcopy(base)
            cross_block.update(role=role, bundle_index=index)
            invalid.append(cross_block)
        final_namespace = deepcopy(base)
        final_namespace["namespace"] = "PD-B1-E-v1"
        invalid.append(final_namespace)
        for bundle in invalid:
            with self.subTest(role=bundle.get("role"), index=bundle.get("bundle_index"),
                              flags={k: bundle.get(k) for k in ("development_only", "confirmatory", "reusable_as_final")}):
                with patch("b1.evaluation.runner.P6Task") as task:
                    with self.assertRaises(ValueError):
                        run_episode(bundle, "C1")
                    task.assert_not_called()

    def test_no_confirmatory_cli_command_exists(self):
        from b1.run import main
        with patch("b1.run.run_episode") as execute, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["B1-E"])
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()

    def test_new_source_file_invalidates_pre_tuning_snapshot(self):
        from b1.run import StageError, source_files, snapshot, verify_pre_tuning
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fixture.py").write_text("# synthetic snapshot fixture\n")
            files = source_files(root)
            freeze = {"status": "IMPLEMENTATION_CONFIG_FROZEN_BEFORE_SCORES",
                      "source_files": files, "artifact_sha256": snapshot(root, files)}
            (root / "PRE_TUNING_FREEZE.json").write_text(json.dumps(freeze))
            self.assertEqual(verify_pre_tuning(root)["source_files"], files)
            (root / "later.py").write_text("# unreviewed new code\n")
            with self.assertRaises(StageError):
                verify_pre_tuning(root)

    def test_historical_seed_registry_does_not_accept_boolean_seed(self):
        from b1.run import StageError, historical_exclusions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seeds").mkdir()
            (root / "seeds/HISTORICAL_SEED_EXCLUSIONS.json").write_text('{"seeds":[true]}')
            with self.assertRaises(StageError):
                historical_exclusions(root)


if __name__ == "__main__":
    unittest.main()
