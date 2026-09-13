"""P5 instrument properties use QA fixtures, never the scientific namespace."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b1.controllers.core import Limits
from b1.tasks import P5Task
from b1.v1_1.p5.controllers import P5ConventionalController
from b1.v1_1.p5.benchmark import bundle, episode, protocol, source_freeze_inventory, verify_freeze


class P5Tests(unittest.TestCase):
    def test_c4_all_feasible_stock_values_and_terminal_semantics(self):
        for reserve in range(9):
            observations = [P5Task(reserve, cell_id=c).observe() for c in range(3)]
            minimal = P5ConventionalController()
            expected_service = min(2, reserve)
            expected_refill = min(2, max(0, 2-(reserve-expected_service)))
            for action in minimal.act(observations):
                self.assertEqual(action["A"]*2, expected_service)
                self.assertEqual(action["B"]*2, expected_refill)
            for obs in observations:
                obs["epoch"] = 31
            self.assertTrue(all(a["B"] == 0 for a in minimal.act(observations)))
            witness = P5ConventionalController(full_refill_witness=True)
            self.assertTrue(all(a == {"A": 1, "B": 1} for a in witness.act(observations)))
            self.assertEqual(witness.comparator, "C4_FULL_REFILL_CEILING_WITNESS")

    def test_qa_ceiling_bound_and_distinct_secondary_actions(self):
        fixture = {"pilot": "P5", "bundle_index": 0, "namespace": "PD-B1-D-v1.1-QA",
            "cells": [{"cell_id": c, "initial_reserve": 0 if c == 0 else 8} for c in range(3)],
            "development_only": True, "confirmatory": False, "reusable_as_final": False}
        minimal = episode(fixture, "C4")
        witness = episode(fixture, "C4_FULL_REFILL_CEILING_WITNESS")
        self.assertEqual(minimal["completed_jobs"], 190)
        self.assertEqual(witness["completed_jobs"], 190)
        self.assertEqual(minimal["terminal_refill_units"], 0)
        self.assertEqual(witness["terminal_refill_units"], 6)
        self.assertLess(minimal["replenishment_units"], witness["replenishment_units"])

    def test_qa_c6_complete_exact_conjugacy(self):
        result = episode(bundle(0, "PD-B1-D-v1.1-QA"), "C6")
        self.assertEqual(result["C6_exact_checks"], {"actions": 96, "states": 32, "observations": 96, "events": 96})

    def test_qa_event_replay_is_bitwise_deterministic(self):
        fixture = bundle(4, "PD-B1-D-v1.1-QA")
        first, second = episode(fixture, "C4"), episode(fixture, "C4")
        self.assertEqual(first["event_trace_sha256"], second["event_trace_sha256"])
        self.assertEqual(first["loss_exact"], second["loss_exact"])

    def test_seed_namespaces_reject_final_and_are_deterministic(self):
        self.assertEqual(protocol()["b0_source_commit"], "b4796d881ee2581367fc8dd7396521dac30d827a")
        self.assertEqual(bundle(3, "PD-B1-D-v1.1-QA"), bundle(3, "PD-B1-D-v1.1-QA"))
        for namespace in ("PD-B1-D-v1", "PD-B1-E-v2", "final", "PD-B1-D-v1.1-final"):
            with self.assertRaises(ValueError):
                bundle(0, namespace)

    def test_missing_or_changed_source_freeze_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"freeze.json"
            d = {"status": "P5_V1_1_FROZEN_BEFORE_BENCHMARK", "namespace": "PD-B1-D-v1.1",
                 "artifact_sha256": source_freeze_inventory()}
            path.write_text(json.dumps(d))
            verify_freeze(path, "a"*40)
            d["artifact_sha256"].pop(next(iter(d["artifact_sha256"])))
            path.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                verify_freeze(path, "a"*40)
            with self.assertRaises(ValueError):
                verify_freeze(path, "not-a-commit")

    def test_controller_failure_is_retained_with_loss_one(self):
        controller = P5ConventionalController()
        controller.act = lambda observations: (_ for _ in ()).throw(ValueError("deliberate QA failure"))
        with patch("b1.v1_1.p5.benchmark.controller_for", return_value=controller):
            result = episode(bundle(2, "PD-B1-D-v1.1-QA"), "C4")
        self.assertEqual(result["loss_exact"], [1, 1])
        self.assertEqual(len(result["controller_failures"]), 32)
        self.assertEqual(result["event_count"], 96)


if __name__ == "__main__":
    unittest.main()
