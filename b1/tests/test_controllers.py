"""Finite instrument fixtures only; no development or final seeds/scores."""
from copy import deepcopy
from fractions import Fraction
import unittest

from b1.controllers import Controller, ContractConflict, ControllerFailure, Limits, get_configurations
from b1.controllers.invariants import profile, audit_contrast, mean_both_c2_cycles
from b1.tasks import P5Task, P6Task, P7Task


class ControllerInstrumentTests(unittest.TestCase):
    def test_frozen_search_and_analytical_exception(self):
        for arm in ("C0", "C1", "C2", "C3", "C5"):
            self.assertEqual(len(get_configurations(arm)), 8)
            self.assertEqual(len({c["config_id"] for c in get_configurations(arm)}), 8)
        self.assertEqual(len(get_configurations("C4")), 1)
        self.assertIn((0, 1, 2), [c["generic_sources"] for c in get_configurations("C3")])

    def test_p5_c4_conflict_is_not_silently_resolved(self):
        with self.assertRaisesRegex(ContractConflict, "B1-CONFLICT-P5-C4-REFILL-001"):
            Controller("P5", "C4")

    def test_c4_p6_complete_binary_feedback(self):
        cells = [P6Task(initial_mapping="q1", cell_id=i) for i in range(3)]
        ctl = Controller("P6", "C4")
        first = ctl.act([t.observe() for t in cells])
        self.assertEqual([a["routine_id"] for a in first], ["q0"] * 3)
        for task, action in zip(cells, first):
            task.step(action)
        second = ctl.act([t.observe() for t in cells])
        self.assertEqual([a["routine_id"] for a in second], ["q1"] * 3)
        self.assertTrue(all(a["A"] == 0 and a["B"] == 1 for a in second))

    def test_c4_p6_conflicting_feedback_rejected(self):
        observations = [P6Task(cell_id=i).observe() for i in range(3)]
        observations[0]["feedback"] = [{"job_id": "0", "routine_id": "q0", "correct": 0},
                                        {"job_id": "1", "routine_id": "q0", "correct": 1}]
        with self.assertRaisesRegex(ControllerFailure, "conflicting_deterministic_feedback"):
            Controller("P6", "C4").act(observations)

    def test_c4_p7_ordinary_drift_is_positive_repair(self):
        observations = [P7Task(initial_drifts=(1, 0), cell_id=i).observe() for i in range(3)]
        actions = Controller("P7", "C4").act(observations)
        self.assertTrue(all(a["A"] == 1 and a["B"] == 0 and a["repair_target"] == 0 for a in actions))

    def test_c4_p7_concurrent_distinct_objects_and_modes(self):
        observations = [P7Task(initial_drifts=(1, 0), version_switch_epoch=0,
                               mode="state_preserving", cell_id=i).observe() for i in range(3)]
        actions = Controller("P7", "C4").act(observations)
        self.assertTrue(all(a["A"] == 1 and a["B"] == 1 and a["repair_target"] == 0
                            and a["deploy_target"] == 1 for a in actions))
        for obs in observations:
            obs["context"]["mode"] = "replacement"
        actions = Controller("P7", "C4").act(observations)
        self.assertTrue(all(a["A"] == 0 and a["B"] == 1 and a["deploy_target"] == 0 for a in actions))

    def test_no_unapproved_deployment_and_clean_noop(self):
        observations = [P7Task(initial_drifts=(0, 0), cell_id=i).observe() for i in range(3)]
        ctl = Controller("P7", "C4")
        self.assertTrue(all(a["A"] == a["B"] == 0 for a in ctl.act(observations)))
        for obs in observations:
            obs["required_version"] = 1
            obs["approved_versions"] = [0]
        self.assertTrue(all(a["B"] == 0 for a in ctl.act(observations)))

    def test_both_permutations_preserve_source_identity_and_raw_observations(self):
        observations = [P6Task(cell_id=i).observe() for i in range(3)]
        untouched = deepcopy(observations)
        maps = []
        for cycle in (1, 2):
            ctl = Controller("P6", "C2", cycle=cycle)
            ctl.act(observations)
            slots = ctl.last_decision_report["route_slots"]
            self.assertEqual(len(slots), 3)
            self.assertEqual(sorted(s["source"] for s in slots), [0, 1, 2])
            self.assertTrue(all(s["source"] != s["recipient"] and
                                s["payload"]["source_cell_id"] == s["source"] for s in slots))
            maps.append([s["source"] for s in slots])
            self.assertEqual(ctl.memory["raw_history"][-1], untouched)
        self.assertNotEqual(*maps)
        self.assertEqual(mean_both_c2_cycles({1: Fraction(1, 4), 2: Fraction(3, 4)}), Fraction(1, 2))
        with self.assertRaises(ValueError):
            mean_both_c2_cycles({1: 0})

    def test_c5_exact_explicit_capacity_multiplier(self):
        observations = [P7Task(cell_id=i).observe() for i in range(3)]
        c3, c5 = Controller("P7", "C3"), Controller("P7", "C5")
        c3.act(observations)
        c5.act(observations)
        self.assertEqual(c5.limits.memory_scalars, 2 * c3.limits.memory_scalars)
        self.assertEqual(c5.limits.decision_operations, 2 * c3.limits.decision_operations)
        self.assertEqual(len(c5.last_decision_report["route_slots"]), 6)
        self.assertEqual(c5.limits.planning_horizon, c3.limits.planning_horizon)

    def test_acute_route_leaves_raw_data_and_parameters_unchanged(self):
        cells = [P6Task(initial_mapping="q1", cell_id=i) for i in range(3)]
        for task in cells:
            task.step({"A": 1, "B": 0, "candidate_id": "q1"})
        observations = [t.observe() for t in cells]
        intact, bypass = Controller("P6", "C1"), Controller("P6", "C1", lesion=True)
        self.assertEqual(intact.act(observations), bypass.act(observations))
        self.assertEqual(intact.parameters, bypass.parameters)
        self.assertEqual(intact.memory["raw_history"], bypass.memory["raw_history"])
        self.assertTrue(intact.last_decision_report["route_consumed"])
        self.assertTrue(bypass.last_decision_report["nominated_route_bypassed"])
        self.assertFalse(bypass.last_decision_report["joint_manipulation"])

    def test_stale_repair_route_does_not_override_current_drift(self):
        observations = [P7Task(initial_drifts=(1, 0), version_switch_epoch=0,
                               cell_id=i).observe() for i in range(3)]
        for obs in observations:
            obs["completion_events"] = [{"operation": "repair", "instance_id": 0,
                "source_cell_id": obs["cell_id"], "target_version": 0,
                "post_repair_drift": 0, "completion_epoch": 0}]
        ctl = Controller("P7", "C1")
        actions = ctl.act(observations)
        self.assertTrue(all(a["repair_target"] == 0 and a["deploy_target"] == 1 for a in actions))
        self.assertTrue(ctl.last_decision_report["route_consumed"])

    def test_budget_enforcement_happens_before_actions_return(self):
        observations = [P5Task(initial_reserve=8, cell_id=i).observe() for i in range(3)]
        with self.assertRaisesRegex(ControllerFailure, "decision_budget_exhausted"):
            Controller("P5", "C0", limits=Limits(decision_operations=1)).act(observations)
        with self.assertRaisesRegex(ControllerFailure, "memory_budget_exhausted"):
            Controller("P5", "C0", limits=Limits(memory_scalars=1)).act(observations)

    def test_matching_pending_is_not_pass_and_joint_change_is_not_excused(self):
        left, right = profile("C1"), profile("C0")
        audit = audit_contrast(left, right)
        self.assertFalse(audit["matching_passed"])
        self.assertTrue(audit["pending_certification"])
        right["available_information"]["raw_cells"] = [0]
        audit = audit_contrast(left, right, contrast_kind="acute_route", gamma_manipulated=True,
                               expected_differences=("available_information",))
        self.assertTrue(audit["joint_manipulation"])
        self.assertFalse(audit["gamma_specific_verdict_allowed"])


if __name__ == "__main__":
    unittest.main()
