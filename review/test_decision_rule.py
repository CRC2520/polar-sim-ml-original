"""Artificial branch/boundary tests; none of these values are scientific results."""
from copy import deepcopy
import unittest
from review.decision_rule import decide, PARAMETERS, VALIDITY_KEYS, BOUND_KEYS, IDENTITY_KEYS


class ReviewDecisionRuleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = dict(
            registered=True, validity={k: True for k in VALIDITY_KEYS},
            seeds=64, trials=6912, upper99={
                'foreign_mse': -.001, 'clean_mse': 0., 'recall_mse': 0.,
                'postshift_mse': 0., 'action_cost': 0.,
            }, identity_errors={k: 0. for k in IDENTITY_KEYS}, hard_violations=0,
        )
    def test_design_is_still_a_draft(self):
        self.assertEqual(PARAMETERS['status'], 'draft_not_registered_for_final_execution')
        self.assertTrue(PARAMETERS['execution_blockers'])
    def test_registration_is_mandatory(self):
        self.assertEqual(decide(**{**self.fixture, 'registered': False}), 'not_registered')
    def test_all_true_synthetic_fixture(self):
        self.assertEqual(decide(**self.fixture), 'provenance_use_supported_in_scope')
    def test_each_validity_condition_is_required(self):
        for name in VALIDITY_KEYS:
            f=deepcopy(self.fixture);f['validity'][name]=False
            self.assertEqual(decide(**f), 'invalid_experiment')
    def test_missing_and_extra_validity_keys(self):
        for operation in ('missing','extra'):
            f=deepcopy(self.fixture)
            if operation=='missing':f['validity'].pop(VALIDITY_KEYS[0])
            else:f['validity']['unregistered_override']=True
            self.assertEqual(decide(**f), 'invalid_experiment')
    def test_exact_seed_and_trial_counts(self):
        for key,value in [('seeds',63),('seeds',65),('trials',6911),('trials',6913),('seeds',64.0)]:
            self.assertEqual(decide(**{**self.fixture,key:value}), 'invalid_experiment')
    def test_primary_threshold_is_strict(self):
        f=deepcopy(self.fixture);f['upper99']['foreign_mse']=-PARAMETERS['practical_mse_margin']
        self.assertEqual(decide(**f), 'provenance_use_not_supported_in_scope')
    def test_guardrail_thresholds_are_inclusive(self):
        mapping={'clean_mse':'clean_mse_noninferiority_margin','recall_mse':'recall_mse_noninferiority_margin','postshift_mse':'postshift_mse_noninferiority_margin','action_cost':'normalized_action_cost_margin'}
        for outcome,param in mapping.items():
            f=deepcopy(self.fixture);f['upper99'][outcome]=PARAMETERS[param]
            self.assertEqual(decide(**f), 'provenance_use_supported_in_scope')
            f['upper99'][outcome]+=1e-12
            self.assertEqual(decide(**f), 'provenance_use_not_supported_in_scope')
    def test_hard_violation_fails_use_not_data_validity(self):
        self.assertEqual(decide(**{**self.fixture,'hard_violations':1}), 'provenance_use_not_supported_in_scope')
        self.assertEqual(decide(**{**self.fixture,'hard_violations':-1}), 'invalid_experiment')
    def test_nonfinite_outcomes_fail(self):
        for value in [float('nan'),float('inf'),True,'0.0']:
            f=deepcopy(self.fixture);f['upper99']['foreign_mse']=value
            self.assertEqual(decide(**f),'invalid_experiment')
    def test_all_coordinate_checks_required(self):
        for name in IDENTITY_KEYS:
            f=deepcopy(self.fixture);f['identity_errors'][name]=1e-8
            self.assertEqual(decide(**f),'invalid_experiment')
    def test_no_unknown_override_fields(self):
        f=deepcopy(self.fixture);f['upper99']['positive_subgroup']=-100.
        self.assertEqual(decide(**f),'invalid_experiment')


if __name__=='__main__':unittest.main()
