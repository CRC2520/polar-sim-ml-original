"""Finite-bound and invariant failure tests; no experiments or seed draws."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from b1.controllers.core import Controller, Limits
from b1.v1_1.resources.accounting import (assert_within_envelope, decision_charge,
    limits, maximum_observations, source_hashes, static_bounds)
from b1.v1_1.resources.certify import AXES, configs, controller

ROOT = Path(__file__).resolve().parents[2]


class ResourceCertificateTests(unittest.TestCase):
    def test_every_configuration_maximum_shape_nonbinding_and_behavior_unchanged(self):
        for pilot in ('P5','P6','P7'):
            for comp in ('C0','C1','C2','C3','C4','C5'):
                for cfg in configs(comp):
                    for cycle in ((1,2) if comp == 'C2' else (1,)):
                        with self.subTest(pilot=pilot,comparator=comp,config=cfg,cycle=cycle):
                            wide = controller(pilot,comp,cfg,cycle)
                            old = controller(pilot,comp,cfg,cycle,False)
                            obs = maximum_observations(pilot)
                            for _ in range(3):
                                self.assertEqual(wide.act(obs),old.act(obs))
                                self.assertEqual(wide.memory,old.memory)
                                self.assertTrue(assert_within_envelope(pilot,comp,decision_charge(obs,wide)))

    def test_larger_input_schema_fails_closed(self):
        obs = maximum_observations('P6')
        obs[0]['reports'].append(deepcopy(obs[0]['reports'][0]))
        c = controller('P6','C1','cfg00')
        c.act(obs)
        with self.assertRaisesRegex(AssertionError,'static_bound_exceeded:raw_observation_scalars'):
            assert_within_envelope('P6','C1',decision_charge(obs,c))

    def test_extra_persistent_model_state_fails_closed(self):
        obs = maximum_observations('P7')
        c = controller('P7','C1','cfg01')
        c.act(obs)
        c.undeclared_state = [0]*4000
        with self.assertRaisesRegex(AssertionError,'total_controller_state_scalars'):
            assert_within_envelope('P7','C1',decision_charge(obs,c))

    def test_binding_envelope_is_rejected(self):
        c = controller('P7','C1','cfg01')
        obs = maximum_observations('P7')
        c.act(obs)
        measured = decision_charge(obs,c)
        measured['legacy_operations'] = 65536
        with self.assertRaisesRegex(AssertionError,'resource_cap_binding'):
            assert_within_envelope('P7','C1',measured)

    def test_c4_uses_less_without_dummy_work(self):
        obs = maximum_observations('P5')
        c4 = controller('P5','C4','fixed')
        c1 = controller('P5','C1','cfg00')
        c4.act(obs)
        c1.act(obs)
        self.assertEqual(c4.last_decision_report['route_slots'],[])
        self.assertLess(c4.last_decision_report['useful_operations'],c1.last_decision_report['useful_operations'])
        self.assertLess(c4.last_decision_report['memory_scalars'],c1.last_decision_report['memory_scalars'])
        self.assertEqual(c4.limits,c1.limits)

    def test_old_caps_already_above_static_bounds(self):
        for p in ('P5','P6','P7'):
            for big in (False,True):
                b = static_bounds(p,big)
                self.assertLess(b['persistent_budgeted_scalars'],4096*(2 if big else 1))
                self.assertLess(b['legacy_operations'],8192*(2 if big else 1))

    def test_frozen_source_and_complete_axis_coverage(self):
        spec = json.loads((ROOT/'RESOURCE_MATCHING_SPEC.json').read_text())
        self.assertEqual(source_hashes(),spec['source_hashes'])
        cert = json.loads((ROOT/'RESOURCE_MATCHING_CERTIFICATE.json').read_text())
        self.assertEqual(len(cert['axes']),288)
        allowed = {'exact_match','common_upper_bound','lower_actual_usage',
                   'unmatched_but_quantified','not_applicable','blocked'}
        for pilot in ('P5','P6','P7'):
            for comp in ('C0','C1','C2','C3','C4','C5'):
                rows = [r for r in cert['axes'] if r['pilot'] == pilot and r['comparator'] == comp]
                self.assertEqual({r['axis'] for r in rows},set(AXES))
                for r in rows:
                    self.assertIn(r['status'],allowed)
                    self.assertTrue(all(k in r for k in ('allowed','requested','actual')))
                    if r['status'] == 'unmatched_but_quantified':
                        self.assertTrue(r['claim_limit'])
                self.assertEqual(cert['invariants_pass'][pilot][comp],comp != 'C5')
        self.assertFalse(cert['resource_cap_binding'])
        self.assertEqual(cert['P7_archived_input_replay']['C1_selected'],'cfg01')
        self.assertFalse(cert['P7_archived_input_replay']['scientific_rerun_required'])


if __name__ == '__main__':
    unittest.main()
