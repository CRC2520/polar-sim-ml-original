"""Meaningful statistical gates and fixed-N regression checks; no final run."""
import ast
import hashlib
import json
import math
import unittest
from pathlib import Path

from .build_design import calibration_audit, design
from .method import (EPSILON, TARGET_HALF_WIDTH, fixed_n, interval,
                     planning_second_moment_upper, planning_variance,
                     positive_gates, radius, utility_status, v1_n)

HERE = Path(__file__).resolve().parent


class StatisticsTests(unittest.TestCase):
    def test_fixed_n_minimal_and_historical_n(self):
        q = planning_second_moment_upper([0] * 6)
        n = fixed_n(q)
        self.assertEqual(n, 138688)
        self.assertEqual(v1_n(), 915501)
        self.assertLessEqual(radius(n, planning_variance(n, q)), TARGET_HALF_WIDTH)
        self.assertGreater(radius(n - 1, planning_variance(n - 1, q)), TARGET_HALF_WIDTH)
        self.assertGreater(fixed_n(1.0), n)

    def test_zero_development_is_not_known_zero_variance(self):
        q = planning_second_moment_upper([0] * 6)
        self.assertGreater(q, 0.13)
        self.assertAlmostEqual((1 - q) ** 32, .05 / 6)
        self.assertEqual(planning_second_moment_upper([1, 0, 0, 0, 0, 0]), 1.0)

    def test_context_range_rescaling(self):
        xs = [-0.5, 0, 0.25, 0.5] * 100
        ordinary = interval(xs)
        context = interval([2 * x for x in xs], (-2.0, 2.0))
        self.assertAlmostEqual(context["mean"], 2 * ordinary["mean"])
        self.assertAlmostEqual(context["sample_variance"], 4 * ordinary["sample_variance"])
        self.assertAlmostEqual(context["half_width"], 2 * ordinary["half_width"])

    def test_finite_sample_constant_term_cannot_disappear(self):
        n = 138688
        ci = interval([0.0] * n, required_n=n)
        self.assertGreater(ci["half_width"], 0)
        self.assertEqual(utility_status(ci), "equivalent")
        self.assertAlmostEqual(ci["half_width"], 14 * math.log(720) / (3 * (n - 1)))

    def test_boundary_never_equivalent_or_superior(self):
        n = 138688
        for mean in (-EPSILON, EPSILON, -1 / 64, 1 / 64):
            self.assertEqual(utility_status(interval([mean] * n)), "inconclusive")

    def test_high_variance_does_not_borrow_planning_bound(self):
        ci = interval([-1.0, 1.0] * (138688 // 2))
        self.assertGreater(ci["half_width"], TARGET_HALF_WIDTH)
        self.assertEqual(utility_status(ci), "inconclusive")

    def test_missing_duplicate_size_and_invalid_input_fail(self):
        for xs in ([0.0], [float("nan"), 0], [-2, 0]):
            with self.assertRaises(ValueError):
                interval(xs)
        with self.assertRaises(ValueError):
            interval([0, 0], required_n=138688)

    def test_gatekeeping_cannot_promote_joint_or_negative_controls(self):
        args = dict(instrument_pass=True, mechanism_kappa1_supported=True,
                    mechanism_kappa2_equivalent=True, source_causal_admissible=True,
                    comparator_status={c: "improved" for c in ("C0", "C2", "C3", "C4")},
                    context_supported=True)
        self.assertTrue(all(positive_gates(**args).values()))
        self.assertFalse(any(positive_gates(**(args | {"source_causal_admissible": False})).values()))
        for pilot in ("P5", "P6"):
            self.assertFalse(any(positive_gates(**args, pilot=pilot).values()))
        args["comparator_status"]["C0"] = "equivalent"
        result = positive_gates(**args)
        self.assertTrue(result["H_mechanism"])
        self.assertFalse(result["H_utility"] or result["H_pairing"] or result["H_context"])

    def test_archived_pair_reduction_exact_and_complete(self):
        audit = calibration_audit()
        self.assertEqual(audit["trial_count"], 384)
        self.assertEqual(audit["nonzero_counts"], [0] * 6)
        self.assertTrue(audit["matches_v1_report"])

    def test_saved_method_results_match_current_sources(self):
        report = json.loads((HERE / "SIMULATION_RESULTS.json").read_text())
        for key, name in (("method_sha256", "method.py"), ("simulator_sha256", "simulate.py"),
                          ("design_sha256", "SIMULATION_DESIGN.json")):
            self.assertEqual(report[key], hashlib.sha256((HERE / name).read_bytes()).hexdigest())
        self.assertEqual(report["model_executions"], 0)
        self.assertEqual(report["scenario_cells"], 40)
        for name in ("method.py", "simulate.py"):
            tree = ast.parse((HERE / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(any(x in (node.module or "") for x in ("tasks", "controllers", "seeds")))

    def test_design_artifact_reproduces_without_writes(self):
        path = HERE.parent / "B1E_STATISTICAL_REDESIGN.json"
        if path.exists():
            self.assertEqual(json.loads(path.read_text()), design())


if __name__ == "__main__":
    unittest.main()
