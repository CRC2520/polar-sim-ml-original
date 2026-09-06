"""Tests for the disclosed post-run interpreter, outside frozen Study 2 sources."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from adjudicate_study2 import adjudicate, PROTOCOL_INPUT, RAW_INPUT


class DecisionAdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(RAW_INPUT.read_text())
        self.protocol = json.loads(PROTOCOL_INPUT.read_text())

    def test_aligned_and_misaligned_benefit_does_not_satisfy_only_aligned(self):
        original = copy.deepcopy(self.raw)
        result = adjudicate(self.raw, self.protocol)
        self.assertEqual(result["raw_decision"], "pivot_to_conditional_structural_prior")
        self.assertTrue(result["evaluated_clauses"]["practical_benefit_by_regime"]["paired"])
        self.assertTrue(result["evaluated_clauses"]["practical_benefit_by_regime"]["misaligned"])
        self.assertFalse(result["evaluated_clauses"]["broader_continue_criteria"])
        self.assertEqual(result["adjudicated_decision"], "suspend_exclusive_polar_advantage_claim")
        self.assertEqual(self.raw, original)

    def test_aligned_only_benefit_permits_conditional_pivot(self):
        self.raw["regimes"]["misaligned"]["ci95"] = [-.0015, -.001]
        result = adjudicate(self.raw, self.protocol)
        self.assertTrue(result["evaluated_clauses"]["only_aligned_regime_has_practical_benefit"])
        self.assertEqual(result["adjudicated_decision"], "pivot_to_conditional_structural_prior")

    def test_broader_criteria_all_pass_continue_precedes_fallback(self):
        self.raw["comparisons"]["shuffled"]["ci95"] = [-.001, -.0005]
        self.raw["dense_transfer"]["ci95"] = [0, .001]
        result = adjudicate(self.raw, self.protocol)
        self.assertTrue(result["evaluated_clauses"]["broader_continue_criteria"])
        self.assertEqual(result["adjudicated_decision"], "continue_bounded_contextual_coupling_research")

    def test_invalid_negative_or_equivalence_controls_override_positive_results(self):
        for failure in ("negative", "equivalence"):
            with self.subTest(failure=failure):
                raw = copy.deepcopy(self.raw)
                raw["comparisons"]["shuffled"]["ci95"] = [-.001, -.0005]
                raw["dense_transfer"]["ci95"] = [0, .001]
                if failure == "negative":
                    raw["negative_control_failures"]["zero"] = 0
                else:
                    raw["max_signed_action_difference"] = 1e-4
                self.assertEqual(adjudicate(raw, self.protocol)["adjudicated_decision"], "evaluator_or_equivalence_invalid")

    def test_exact_practical_threshold_is_not_strict_benefit_and_missing_is_not_zero(self):
        self.raw["regimes"]["paired"]["ci95"] = [-.003, -.002]
        result = adjudicate(self.raw, self.protocol)
        self.assertFalse(result["evaluated_clauses"]["practical_benefit_by_regime"]["paired"])
        del self.raw["guards"]["stockout"]
        with self.assertRaises(KeyError):
            adjudicate(self.raw, self.protocol)


if __name__ == "__main__":
    unittest.main()
