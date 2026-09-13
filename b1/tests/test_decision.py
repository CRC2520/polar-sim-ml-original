"""Synthetic rule tests: invented numbers below are not scientific outcomes."""
from copy import deepcopy
from fractions import Fraction
from math import ceil, log
from pathlib import Path
import shutil
import tempfile
import unittest

from b1.contracts import ContractError, load_contracts
from b1.contracts.loader import FROZEN_DIRECTORY, strict_json, strict_yaml
from b1.evaluation.decision import (DecisionError, PredicateEvaluator, Truth,
    derive_audit, evaluate_development, evaluate_rule, evaluate_synthetic_pilot)
from b1.evaluation.precision import bounded_interval, expanded_registry, precision_plan
from b1.metrics import guardrails_from_counts, primary_loss


CONTRACTS = load_contracts()
RULES = CONTRACTS.rules


def instrument():
    checks = {"known_input": {"observed": 1, "expected": 1}}
    return {"guardrail_counts": {name.removeprefix("max_"): 0 for name in RULES["thresholds"]["guardrails"]},
        "invariants": {name: {"left": {"budget": 1}, "right": {"budget": 1}}
                       for name in RULES["six_invariant_axes"]},
        "manipulation_changed_axes": [], "sensitivity_checks": deepcopy(checks),
        "c6_checks": deepcopy(checks),
        "mediator_checks": {name: {"observed": True, "expected": True}
                            for name in ("delivery", "timing", "instrumentation")}}


def synthetic_records(utility=-0.1):
    n = precision_plan(CONTRACTS)["B1E_required_N"]
    means = {"mechanism.kappa1": 0.9, "mechanism.kappa2": 0.0,
             "context.interaction": 0.9,
             "acute_route.kappa1": -0.1, "acute_route.kappa2": 0.0}
    means.update({"utility." + name: utility for name in RULES["required_pairing_comparators"]})
    return {name: {"mean": mean, "ci_lower": mean - 0.001,
                   "ci_upper": mean + 0.001, "N": n} for name, mean in means.items()}


class ContractTests(unittest.TestCase):
    def test_hash_verified_source(self):
        self.assertEqual(CONTRACTS.source_commit, RULES["source_commit"])
        self.assertEqual(CONTRACTS.pilot("P7")["evidence_status"], "O2")
        self.assertEqual(CONTRACTS.field("P5", 3), CONTRACTS.field("P5", "operational_pole_A"))

    def test_copies_prevent_shared_mutation(self):
        rules = CONTRACTS.rules
        rules["thresholds"]["family_size"] = 1
        self.assertEqual(CONTRACTS.rules["thresholds"]["family_size"], 27)

    def test_duplicate_parsers_reject(self):
        for parser, text in ((strict_json, '{"a":1,"a":2}'), (strict_yaml, 'a: 1\na: 2\n')):
            with self.assertRaises(ContractError):
                parser(text)

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ContractError):
            strict_json('{"a":NaN}')

    def test_corrupt_and_missing_contracts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "frozen"
            shutil.copytree(FROZEN_DIRECTORY, copied)
            (copied / "B1_PROTOCOL_DRAFT_v1.md").write_text("altered")
            with self.assertRaises(ContractError):
                load_contracts(copied)
            (copied / "B1_PROTOCOL_DRAFT_v1.md").unlink()
            with self.assertRaises(ContractError):
                load_contracts(copied)


class PredicateTests(unittest.TestCase):
    def setUp(self):
        self.engine = PredicateEvaluator(CONTRACTS)

    def test_kleene_truth_tables(self):
        fields = {"t": True, "f": False}
        nodes = {name: {"field": name, "op": "eq", "value": True} for name in ("t", "f", "u")}
        for first in ("t", "f", "u"):
            for second in ("t", "f", "u"):
                expected_all = Truth.FALSE if "f" in (first, second) else Truth.UNKNOWN if "u" in (first, second) else Truth.TRUE
                expected_any = Truth.TRUE if "t" in (first, second) else Truth.UNKNOWN if "u" in (first, second) else Truth.FALSE
                self.assertIs(self.engine.evaluate({"all": [nodes[first], nodes[second]]}, fields), expected_all)
                self.assertIs(self.engine.evaluate({"any": [nodes[first], nodes[second]]}, fields), expected_any)
        self.assertIs(self.engine.evaluate({"not": nodes["u"]}, fields), Truth.UNKNOWN)
        self.assertIs(self.engine.evaluate({"not": nodes["t"]}, fields), Truth.FALSE)

    def test_every_operator_and_boundary(self):
        cases = [("eq", 2, True), ("ne", 2, False), ("gt", 2, False),
                 ("gte", 2, True), ("lt", 2, False), ("lte", 2, True),
                 ("in", [1, 2], True), ("not_in", [1, 2], False)]
        for op, value, expected in cases:
            with self.subTest(op=op):
                result = self.engine.evaluate({"field": "x.y", "op": op, "value": value}, {"x": {"y": 2}})
                self.assertIs(result, Truth.TRUE if expected else Truth.FALSE)
        self.assertIs(self.engine.evaluate({"field": "x", "op": "present"}, {}), Truth.FALSE)

    def test_missing_null_nan_and_wrong_type_cannot_support(self):
        node = {"field": "x", "op": "gt", "threshold_ref": "mechanism.delta"}
        for value in (None, float("nan"), True, "1"):
            self.assertIs(self.engine.evaluate(node, {"x": value}), Truth.UNKNOWN)
        self.assertIs(self.engine.evaluate({"field": "x", "op": "eq", "value": True}, {"x": 1}), Truth.UNKNOWN)

    def test_invalid_predicate_rejected(self):
        for node in ({"all": []}, {"field": "x", "op": "execute", "value": 1},
                     {"field": "x", "op": "gt", "value": 1, "threshold_ref": "alpha"}):
            with self.assertRaises(DecisionError):
                self.engine.evaluate(node, {})

    def test_missing_required_is_incomplete_even_if_case_would_match(self):
        r = evaluate_rule("mechanism_effect", {"common_admissible": True,
            "mediator_contract_verified": True, "ci_lower": 0.9},
            data_class="synthetic-rule-test", contracts=CONTRACTS)
        self.assertEqual(r["status"], "inconclusive")
        self.assertFalse(r["aggregate_complete"])

    def test_strict_utility_threshold_not_support_at_equality(self):
        delta = RULES["thresholds"]["utility"]["negative_delta"]
        record = {"common_admissible": True, "mean": delta - 0.01,
                  "ci_lower": delta - 0.02, "ci_upper": delta, "N": 1}
        r = evaluate_rule("practical_utility_comparison", record,
                          data_class="synthetic-rule-test", contracts=CONTRACTS)
        self.assertEqual(r["status"], "inconclusive")
        record["ci_upper"] -= 1e-10
        self.assertEqual(evaluate_rule("practical_utility_comparison", record,
            data_class="synthetic-rule-test", contracts=CONTRACTS)["status"], "improved")

    def test_development_never_receives_confirmatory_rule_status(self):
        record = {"common_admissible": True, "mediator_contract_verified": True,
                  "mean": 0.95, "ci_lower": 0.9, "ci_upper": 1, "N": 1000000}
        self.assertEqual(evaluate_rule("mechanism_effect", record,
            data_class="B1-D", contracts=CONTRACTS)["status"], "inconclusive")
        with self.assertRaises(DecisionError):
            evaluate_rule("mechanism_effect", record, data_class="B1-E", contracts=CONTRACTS)


class PrecisionMetricTests(unittest.TestCase):
    def test_exact_registry_and_support_override(self):
        registry = expanded_registry(CONTRACTS)
        self.assertEqual(len(registry), 27)
        self.assertEqual(registry["P5.utility.C4"]["support"], [0.0, 1.0])
        self.assertEqual(registry["P6.utility.C4"]["support"], [-1.0, 1.0])

    def test_frozen_sample_formula(self):
        plan = precision_plan(CONTRACTS)
        expected = ceil(2**2 * log(2 * 27 / 0.05) / (2 * (1 / 256)**2))
        self.assertEqual(plan["B1E_required_N"], expected)
        self.assertFalse(plan["final_seeds_generated"])

    def test_known_support_clipping_and_invalid_samples(self):
        r = bounded_interval([0] * 2, [0, 1], contracts=CONTRACTS)
        self.assertEqual(r["ci_lower"], 0)
        self.assertEqual(r["ci_upper"], 1)
        for values in ([], [float("nan")], [2], [True]):
            with self.assertRaises(ValueError):
                bounded_interval(values, [0, 1], contracts=CONTRACTS)

    def test_primary_loss_uses_all_demanded_jobs(self):
        for pilot in ("P5", "P6", "P7"):
            self.assertEqual(primary_loss(pilot, 1, contracts=CONTRACTS), Fraction(1, 192))
            self.assertEqual(primary_loss(pilot, 0, controller_failure=True, contracts=CONTRACTS), 1)
            with self.assertRaises(ValueError):
                primary_loss(pilot, 193, contracts=CONTRACTS)

    def test_guardrails_missing_negative_and_bool_fail_closed(self):
        counts = instrument()["guardrail_counts"]
        self.assertTrue(guardrails_from_counts(counts, contracts=CONTRACTS)["guardrails_pass"])
        for invalid in (-1, True, None, 1):
            changed = dict(counts, unauthorized_executions=invalid)
            self.assertFalse(guardrails_from_counts(changed, contracts=CONTRACTS)["guardrails_pass"])


class DecisionGraphTests(unittest.TestCase):
    def test_aggregates_recomputed_and_p7_can_lose(self):
        supported = evaluate_synthetic_pilot("P7", synthetic_records(), instrument(), contracts=CONTRACTS)
        self.assertEqual(supported["aggregate_decisions"]["pairing_specificity"]["status"], "supported_in_domain")
        adverse = evaluate_synthetic_pilot("P7", synthetic_records(utility=0.1), instrument(), contracts=CONTRACTS)
        self.assertEqual(adverse["aggregate_decisions"]["practical_utility"]["status"], "worsened_against_required_alternative")
        self.assertEqual(adverse["aggregate_decisions"]["pairing_specificity"]["status"], "rejected_in_domain")
        self.assertFalse(adverse["scientific_evidence"])

    def test_p6_negative_control_does_not_claim_positive_expected_advantage(self):
        result = evaluate_synthetic_pilot("P6", synthetic_records(), instrument(), contracts=CONTRACTS)
        self.assertEqual(result["aggregate_decisions"]["practical_utility"]["status"], "negative_control_no_positive_claim")
        self.assertEqual(result["aggregate_decisions"]["pairing_specificity"]["status"], "suspended")

    def test_missing_comparator_suspends_aggregate(self):
        records = synthetic_records()
        del records["utility.C4"]
        result = evaluate_synthetic_pilot("P7", records, instrument(), contracts=CONTRACTS)
        self.assertEqual(result["aggregate_decisions"]["pairing_specificity"]["status"], "suspended")

    def test_p5_impossible_advantage_is_invalid_not_supported(self):
        result = evaluate_synthetic_pilot("P5", synthetic_records(), instrument(), contracts=CONTRACTS)
        c4 = result["scalar_decisions"]["utility.C4"]
        self.assertFalse(c4["admissible"])
        self.assertEqual(c4["status"], "inconclusive")

    def test_bad_N_bad_CI_joint_and_C6_all_block(self):
        cases = []
        bad_n = synthetic_records(); bad_n["mechanism.kappa1"]["N"] -= 1
        cases.append((bad_n, instrument()))
        bad_ci = synthetic_records(); bad_ci["mechanism.kappa1"]["ci_upper"] = 1.2
        cases.append((bad_ci, instrument()))
        for field in ("c6_checks", "sensitivity_checks"):
            audit = instrument(); audit[field]["known_input"]["observed"] = 0
            cases.append((synthetic_records(), audit))
        audit = instrument(); audit["manipulation_changed_axes"] = ["available_information"]
        cases.append((synthetic_records(), audit))
        for records, audit in cases:
            r = evaluate_synthetic_pilot("P7", records, audit, contracts=CONTRACTS)
            self.assertEqual(r["scalar_decisions"]["mechanism.kappa1"]["status"], "inconclusive")

    def test_development_preserves_missing_registry_and_blocker(self):
        audit = instrument(); audit["blocked_reasons"] = ["B0-P5-C4-REFILL-CONFLICT"]
        result = evaluate_development("P5", {}, audit, contracts=CONTRACTS)
        self.assertEqual(len(result["endpoints"]), 9)
        self.assertEqual(result["endpoint_count_registered_family"], 27)
        self.assertTrue(result["architecture_comparison_blocked"])
        self.assertFalse(result["aggregate_complete"])
        self.assertEqual(result["hypotheses"]["practical_utility"]["status"], "not_eligible")

    def test_development_blocks_duplicate_or_mixed_holdout_indices(self):
        for indices in ([0, 0], [0, 32], [-1], [64]):
            data = {"mechanism.kappa1": [{"bundle_index": n, "value": 1} for n in indices]}
            with self.assertRaises(DecisionError):
                evaluate_development("P7", data, instrument(), contracts=CONTRACTS)

    def test_primary_p5_loss_below_physical_comparator_fails_support(self):
        result = evaluate_development("P5", {"utility.C4": [{"bundle_index": 0, "value": -0.1}]}, instrument(), contracts=CONTRACTS)
        self.assertEqual(result["endpoints"]["P5.utility.C4"]["status"], "invalid")

    def test_hand_entered_guard_flags_ignored(self):
        audit = {"guardrails_pass": True, "invariants_pass": True,
                 "common_admissible": True, "positive_utility_eligible": True}
        derived = derive_audit(audit, CONTRACTS)
        self.assertFalse(derived["guardrails_pass"])
        self.assertFalse(derived["invariants_pass"])
        self.assertTrue(derived["joint_manipulation"])

    def test_two_missing_instrument_values_do_not_prove_equality(self):
        audit = instrument()
        for name in RULES["six_invariant_axes"]:
            audit["invariants"][name] = {"left": None, "right": None}
        for name in ("sensitivity_checks", "c6_checks", "mediator_checks"):
            for check in audit[name].values():
                check.update(observed=None, expected=None)
        derived = derive_audit(audit, CONTRACTS)
        self.assertFalse(derived["invariants_pass"])
        self.assertFalse(derived["sensitivity_pass"])
        self.assertFalse(derived["C6_conjugacy_pass"])
        self.assertFalse(derived["mediator_contract_verified"])


if __name__ == "__main__":
    unittest.main()
