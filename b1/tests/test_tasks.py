"""Instrument and task-law tests, with explicit deterministic states; no final seeds."""
import json
import math
import unittest
from copy import deepcopy
from b1.tasks import CLOCK, PILOTS, P5Task, P6Task, P7Task, admit, audit_execution, InstrumentViolation
from b1.interventions.diagnostics import all_mechanism_diagnostics


class AdmissionTests(unittest.TestCase):
    def test_fixed_grid_intermediates_and_saturation(self):
        for capacity in (1, 2):
            for request in (0, 0.01, 0.49, 0.5, 0.75, 0.999, 1):
                with self.subTest(capacity=capacity, request=request):
                    record = admit(request, capacity)
                    self.assertEqual(record['external_requested_dose'], math.floor(capacity * request))
                    self.assertEqual(record['admitted_activation'], math.floor(capacity * request) / capacity)

    def test_invalid_is_rejected_not_clipped(self):
        for request in (-1, 1.01, 10**500, float('nan'), float('inf'), -float('inf'), True, '1'):
            record = admit(request, 2)
            self.assertFalse(record['valid_request'])
            self.assertEqual(record['admitted_activation'], 0)
            self.assertIn('invalid_request', record['reasons'])

    def test_internal_permission_distinct_from_external_feasibility(self):
        task = P5Task(0, internal_permissions={'B': False})
        event = task.step({'A': 1, 'B': 1})
        self.assertEqual(event['channels']['A']['admitted_activation'], 1)
        self.assertEqual(event['channels']['A']['executed_operation'], 0)
        self.assertEqual(event['channels']['B']['admitted_activation'], 0)
        self.assertIn('internal_permission_absent', event['channels']['B']['reasons'])


class SharedInstrumentTests(unittest.TestCase):
    def tasks(self):
        return (P5Task(8), P6Task(), P7Task(initial_drifts=(1, 0)))

    def test_four_regimes_and_coactivation(self):
        for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
            for task in self.tasks():
                with self.subTest(pilot=task.pilot, a=a, b=b):
                    event = task.step({'A': a, 'B': b, 'repair_target': 0, 'deploy_target': 1, 'target_version': 1})
                    for pole, dose in [('A', a), ('B', b)]:
                        ch = event['channels'][pole]
                        self.assertEqual(ch['admitted_activation'], dose)
                        self.assertEqual(ch['executed_operation'], dose * ch['nominal_capacity'])
                    self.assertTrue(all(v == 0 for v in event['guardrails'].values()))

    def test_all_logged_stages_and_units(self):
        for task in self.tasks():
            event = task.step({'A': 0, 'B': 0})
            for channel in event['channels'].values():
                self.assertTrue({'requested_activation', 'admitted_activation', 'external_requested_dose',
                                 'external_admitted_dose', 'executed_operation', 'continuing_operation',
                                 'environmental_effect', 'nominal_capacity'} <= set(channel))
            self.assertEqual(event['demand'], CLOCK['per_cell_service_demand'])
            self.assertEqual(event['completed'] + event['missed'], event['demand'])
            json.dumps(event, allow_nan=False)

    def test_observation_idempotent_and_detached(self):
        for task in self.tasks():
            obs = task.observe()
            self.assertEqual(obs, task.observe())
            obs['context']['corruption'] = 1
            self.assertNotIn('corruption', task.observe()['context'])

    def test_invalid_requests_logged_as_controller_failure_without_bad_execution(self):
        for task in self.tasks():
            event = task.step({'A': float('nan'), 'B': 2})
            self.assertTrue(event['controller_failure'])
            for channel in event['channels'].values():
                self.assertEqual(channel['executed_operation'], 0)
            self.assertTrue(all(v == 0 for v in event['guardrails'].values()))
            json.dumps(event, allow_nan=False)

    def test_determinism(self):
        for factory in (lambda: P5Task(0), lambda: P6Task(initial_mapping='q1', flip_epoch=16),
                        lambda: P7Task(initial_drifts=(1, 0), drift_events={8: 1}, version_switch_epoch=16)):
            a, b = factory(), factory()
            for _ in range(CLOCK['episode_horizon']):
                action = {'A': 0.5, 'B': 1, 'deploy_target': 1, 'target_version': 1}
                self.assertEqual(a.step(action), b.step(action))
            self.assertTrue(a.done)
            with self.assertRaisesRegex(RuntimeError, 'episode_complete'):
                a.step({'A': 0, 'B': 0})


class P5Tests(unittest.TestCase):
    def test_zero_preserves_pending_delivery_and_delay(self):
        task = P5Task(0)
        e0 = task.step({'A': 1, 'B': 1})
        self.assertEqual(e0['completed'], 0)
        self.assertEqual(e0['state_after']['reserve'], 0)
        e1 = task.step({'A': 0, 'B': 0})
        self.assertEqual(e1['channels']['B']['continuing_operation'], 2)
        self.assertEqual(e1['channels']['B']['executed_operation'], 0)
        self.assertEqual(e1['state_after']['reserve'], 2)

    def test_cap_overflow_and_paid_tokens(self):
        task = P5Task(8)
        e0 = task.step({'A': 0, 'B': 1})
        e1 = task.step({'A': 0, 'B': 0})
        self.assertEqual(e0['channels']['B']['environmental_effect']['tokens_consumed'], 2)
        self.assertEqual(e1['channels']['B']['environmental_effect']['overflow'], 2)
        self.assertEqual(e1['channels']['B']['environmental_effect']['delivered'], 0)
        self.assertEqual(e1['state_after']['reserve'], 8)

    def test_intermediate_doses(self):
        event = P5Task(8).step({'A': 0.75, 'B': 0.5})
        self.assertEqual(event['completed'], 1)
        self.assertEqual(event['channels']['A']['admitted_activation'], 0.5)
        self.assertEqual(event['channels']['B']['executed_operation'], 1)

    def test_unauthorized_request_not_execution(self):
        event = P5Task(8, operation_permissions={'A': False}).step({'A': 1, 'B': 0})
        self.assertEqual(event['channels']['A']['admitted_activation'], 1)
        self.assertEqual(event['channels']['A']['executed_operation'], 0)
        self.assertTrue(event['controller_failure'])
        self.assertEqual(event['guardrails']['unauthorized_executions'], 0)

    def test_physical_ceiling_constructive_policy_not_comparator_selection(self):
        # This task-law witness is not a silent choice between conflicting C4 recipes.
        for stock in (0, PILOTS['P5']['task_contract']['constants']['reserve_capacity']):
            task = P5Task(stock)
            for epoch in range(task.horizon):
                task.step({'A': 1, 'B': int(epoch + 1 < task.horizon)})
            unavoidable_first_epoch = task.demand if stock == 0 else 0
            self.assertEqual(task.completed_jobs, task.horizon * task.demand - unavoidable_first_epoch)


class P6Tests(unittest.TestCase):
    def test_assay_never_changes_same_epoch_routine_or_service(self):
        task = P6Task(initial_mapping='q1')
        event = task.step({'A': 1, 'B': 1, 'candidate_id': 'q1'})
        self.assertEqual(event['channels']['A']['executed_operation'], 1)
        self.assertEqual(event['channels']['B']['executed_operation'], 2)
        self.assertEqual(event['completed'], 0)
        self.assertEqual(event['state_after']['current_routine'], 'q0')
        self.assertEqual(event['state_after']['reports'], [])
        self.assertNotIn('report_value', event['state_after']['pending_assays'][0])

    def test_report_arrives_at_next_boundary_persists_under_zero(self):
        task = P6Task(initial_mapping='q1')
        task.step({'A': 1, 'B': 0, 'candidate_id': 'q1'})
        event = task.step({'A': 0, 'B': 0})
        self.assertEqual(event['channels']['A']['continuing_operation'], 1)
        self.assertEqual(event['state_before']['reports'][0]['report_value'], 1)
        self.assertEqual(event['state_after']['current_routine'], 'q0')
        self.assertEqual(task.observe()['reports'][0]['report_age_epochs'], 1)

    def test_boundary_selector_authorization_and_provenance(self):
        task = P6Task(initial_mapping='q1', adoption_permission=False)
        event = task.step({'A': 0, 'B': 1, 'routine_id': 'q1', 'selector_source': 'test'})
        self.assertEqual(event['completed'], 0)
        self.assertFalse(event['selector_event']['authorized'])
        self.assertTrue(event['controller_failure'])
        self.assertEqual(event['state_after']['current_routine'], 'q0')

    def test_no_mapping_or_future_schedule_in_observation(self):
        a = P6Task(initial_mapping='q0', flip_epoch=None)
        b = P6Task(initial_mapping='q1', flip_epoch=16)
        self.assertEqual(a.observe(), b.observe())

    def test_nontransferable_context_and_candidate_permissions(self):
        task = P6Task(initial_mapping='q1', transfer_eligible=False)
        task.step({'A': 1, 'B': 0, 'candidate_id': 'q1'})
        report = task.observe()['reports'][0]
        self.assertEqual(report['report_value'], 0)
        self.assertFalse(report['transfer_eligible'])
        bad = task.step({'A': 1, 'B': 0, 'candidate_id': 'q2'})
        self.assertEqual(bad['channels']['A']['executed_operation'], 0)
        self.assertTrue(bad['controller_failure'])

    def test_intermediate_assay_request_quantizes_to_zero(self):
        event = P6Task().step({'A': 0.999, 'B': 0.5})
        self.assertEqual(event['channels']['A']['admitted_activation'], 0)
        self.assertEqual(event['channels']['B']['executed_operation'], 1)

    def test_expected_feedback_bound_exhaustive_task_law(self):
        # Four equally likely mapping/change states, no RNG or development bundles.
        losses = []
        for initial in PILOTS['P6']['task_contract']['constants']['routine_catalogue']:
            for flip in (None, CLOCK['episode_horizon'] // 2):
                task = P6Task(initial_mapping=initial, flip_epoch=flip)
                while not task.done:
                    obs = task.observe()
                    q = obs['current_routine']
                    if obs['feedback']:
                        f = obs['feedback'][0]
                        q = f['routine_id'] if f['correct'] else next(x for x in obs['routine_catalogue'] if x != f['routine_id'])
                    task.step({'A': 0, 'B': 1, 'routine_id': q, 'selector_source': 'last_live_feedback_task_law'})
                losses.append(task.missed_jobs / (task.horizon * task.demand))
        self.assertEqual(sum(losses) / len(losses), 1 / CLOCK['episode_horizon'])
        self.assertEqual(min(losses), 0)
        self.assertGreater(max(losses), sum(losses) / len(losses))


class P7Tests(unittest.TestCase):
    def test_repair_is_positive_write_retains_target_and_occupies_epoch(self):
        task = P7Task(initial_drifts=(1, 0))
        event = task.step({'A': 1, 'B': 0, 'repair_target': 0})
        self.assertEqual(event['channels']['A']['executed_operation'], 1)
        self.assertEqual(event['completed'], 1)
        self.assertEqual(event['state_after']['instances'][0]['drift'], 0)
        self.assertEqual(event['state_after']['instances'][0]['target_version'], 0)
        self.assertEqual(event['channels']['A']['environmental_effect']['repair_completion']['completion_phase'], 'end')

    def test_deployment_delay_snapshot_and_zero_continuation(self):
        task = P7Task()
        event = task.step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': 1})
        self.assertEqual(event['state_after']['instances'][0]['target_version'], 0)
        self.assertTrue(event['state_after']['instances'][0]['locked'])
        self.assertEqual(event['completed'], 1)
        next_event = task.step({'A': 0, 'B': 0})
        self.assertEqual(next_event['state_before']['instances'][0]['target_version'], 1)
        self.assertEqual(next_event['channels']['B']['continuing_operation'], 1)
        self.assertEqual(next_event['channels']['B']['executed_operation'], 0)
        self.assertEqual(len(next_event['state_before']['snapshot_history']), 1)
        self.assertEqual(next_event['state_before']['rollback_events'], [])

    def test_same_instance_conflict_repair_priority(self):
        event = P7Task(initial_drifts=(1, 0), mode='replacement').step(
            {'A': 1, 'B': 1, 'repair_target': 0, 'deploy_target': 0, 'target_version': 1})
        self.assertEqual(event['channels']['A']['executed_operation'], 1)
        self.assertEqual(event['channels']['B']['admitted_activation'], 1)
        self.assertEqual(event['channels']['B']['executed_operation'], 0)
        self.assertIn('write_conflict', event['channels']['B']['reasons'])
        self.assertEqual(event['guardrails']['invalid_concurrent_writes'], 0)

    def test_modes_same_target_and_authorization(self):
        for mode, expected in [('state_preserving', 0), ('replacement', 1)]:
            event = P7Task(initial_drifts=(1, 0), mode=mode).step(
                {'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': 1})
            self.assertEqual(event['channels']['B']['executed_operation'], expected)
        same = P7Task().step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': 0})
        self.assertIn('same_target_noop', same['channels']['B']['reasons'])
        bad = P7Task().step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': 2})
        self.assertTrue(bad['controller_failure'])
        self.assertEqual(bad['channels']['B']['executed_operation'], 0)
        boolean_version = P7Task().step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': True})
        self.assertTrue(boolean_version['controller_failure'])
        self.assertEqual(boolean_version['channels']['B']['executed_operation'], 0)

    def test_clean_repair_noop(self):
        event = P7Task().step({'A': 1, 'B': 0, 'repair_target': 0})
        self.assertEqual(event['channels']['A']['admitted_activation'], 1)
        self.assertEqual(event['channels']['A']['executed_operation'], 0)
        self.assertIn('healthy_target_noop', event['channels']['A']['reasons'])

    def test_completion_then_external_drift_then_observation(self):
        task = P7Task(mode='replacement', drift_events={1: 0}, version_switch_epoch=1)
        task.step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': 1})
        obs = task.observe()
        self.assertEqual(obs['instances'][0]['target_version'], 1)
        self.assertEqual(obs['instances'][0]['drift'], 1)
        self.assertEqual(obs['required_version'], 1)
        self.assertEqual(obs['completion_events'][0]['post_deployment_drift'], 0)

    def test_hidden_future_not_in_observations(self):
        self.assertEqual(P7Task().observe(), P7Task(version_switch_epoch=16, drift_events={8: 0, 24: 1}).observe())


class DiagnosticTests(unittest.TestCase):
    def test_all_declared_context_effects_and_matched_instrument_costs(self):
        diagnostics = all_mechanism_diagnostics()
        for pilot, result in diagnostics.items():
            with self.subTest(pilot=pilot):
                self.assertTrue(result['task_law_pass'])
                self.assertTrue(result['mediator_contract_verified'])
                self.assertEqual(result['contexts']['kappa1']['effect'], 1)
                self.assertEqual(result['contexts']['kappa2']['effect'], 0)
                self.assertEqual(result['moderation_difference'], 1)
                self.assertTrue(result['joint_manipulation'])
                self.assertFalse(result['pairing_specific_verdict_permitted'])
                self.assertFalse(result['confirmatory'])


class ExecutedGuardrailAuditTests(unittest.TestCase):
    def test_unauthorized_execution_detected_from_start_facts(self):
        event = P5Task(8).step({'A': 1, 'B': 0})
        event['state_before']['permissions']['operation']['A'] = False
        self.assertGreater(audit_execution(event)['unauthorized_executions'], 0)
        # A rejected request alone is a controller failure, with no bad execution.
        rejected = P5Task(8, operation_permissions={'A': False}).step({'A': 1, 'B': 0})
        self.assertEqual(audit_execution(rejected)['unauthorized_executions'], 0)

    def test_execution_beyond_admitted_dose_detected(self):
        event = P6Task().step({'A': 0, 'B': 1})
        event['channels']['B']['external_admitted_dose'] = 1
        self.assertGreater(audit_execution(event)['executed_infeasible_actions'], 0)

    def test_actual_same_object_write_detected_but_request_conflict_is_safe(self):
        event = P7Task(initial_drifts=(1, 0)).step(
            {'A': 1, 'B': 1, 'repair_target': 0, 'deploy_target': 1, 'target_version': 1})
        deploy = next(f for f in event['execution_facts']['started_operations'] if f['channel'] == 'B')
        deploy['instance_id'] = 0
        self.assertGreater(audit_execution(event)['invalid_concurrent_writes'], 0)
        rejected = P7Task(initial_drifts=(1, 0)).step(
            {'A': 1, 'B': 1, 'repair_target': 0, 'deploy_target': 0, 'target_version': 1})
        self.assertEqual(audit_execution(rejected)['invalid_concurrent_writes'], 0)

    def test_physical_state_and_unauthorized_version_corruption_detected(self):
        event = P5Task(8).step({'A': 1, 'B': 0})
        event['state_after']['reserve'] = -1
        self.assertGreater(audit_execution(event)['physical_state_violations'], 0)
        migration = P7Task().step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': 1})
        migration['execution_facts']['started_operations'][0]['target_version'] = 2
        self.assertGreater(audit_execution(migration)['unauthorized_executions'], 0)

    def test_physical_and_decision_budget_overruns_detected(self):
        event = P5Task(8).step({'A': 0, 'B': 1})
        event['execution_facts']['started_operations'][0]['units'] = 3
        self.assertGreater(audit_execution(event)['undeclared_budget_overruns'], 0)
        normal = P5Task(8).step({'A': 0, 'B': 0})
        normal['decision_compute'] = {'used': 11, 'allowance': 10}
        self.assertGreater(audit_execution(normal)['undeclared_budget_overruns'], 0)

    def test_partial_expected_rejection_not_executed_violation(self):
        event = P5Task(1).step({'A': 1, 'B': 0})
        self.assertEqual(event['completed'], 1)
        self.assertIn('insufficient_stock_or_demand', event['channels']['A']['reasons'])
        self.assertEqual(event['guardrails']['executed_infeasible_actions'], 0)
        self.assertFalse(event['controller_failure'])

    def test_instrument_stops_and_preserves_invalid_execution_event(self):
        class BrokenTask(P5Task):
            def _finish(self, *args, **kwargs):
                self.reserve = -1
                return super()._finish(*args, **kwargs)
        task = BrokenTask(8)
        with self.assertRaises(InstrumentViolation) as captured:
            task.step({'A': 1, 'B': 0})
        self.assertGreater(captured.exception.event['guardrails']['physical_state_violations'], 0)
        self.assertEqual(task.epoch, 0)


if __name__ == '__main__':
    unittest.main()
