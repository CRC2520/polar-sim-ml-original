"""Causal and provenance tests for P1 tension, separate from old study counts."""
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from p1_completion import tension as t


class P1TensionTests(unittest.TestCase):
    def test_exact_upstream_blobs(self):
        t.check_frozen_blobs()

    def test_nonbinding_budget_removes_shared_budget_route(self):
        a = t.pulse_pair(930201, 0, 0, .02, 'no_network')
        r = t.pulse_metrics(a, 0)
        self.assertEqual(r['max_cross_polarity_effect'], 0.)
        self.assertEqual(r['resource_scale_min'], 1.)

    def test_directed_delay_and_isolation(self):
        a = t.pulse_pair(930201, 0, 0, 0., 'full')
        r = t.pulse_metrics(a, 0)
        self.assertEqual(r['onset_steps'][1:7], [1, 2, 3, 4, 5, 6])
        self.assertIsNone(r['onset_steps'][7])
        self.assertEqual(r['off_path_max'], 0.)

    def test_edge_lesion_breaks_only_available_outgoing_route(self):
        a = t.pulse_pair(930201, 0, 0, .02, 'edge_lesion')
        self.assertEqual(t.pulse_metrics(a, 0)['max_cross_polarity_effect'], 0.)

    def test_reversed_changes_first_recipient(self):
        r = t.pulse_metrics(t.pulse_pair(930201, 0, 0, 0., 'reversed'), 0)
        self.assertEqual(r['onset_steps'][6], 1)
        self.assertEqual(r['onset_steps'][1], 6)

    def test_clamped_tension_identical_to_same_time_sham(self):
        a = t.pulse_pair(930201, 0, 0, .02, 'tau_clamp')
        np.testing.assert_array_equal(a['pulse_tension'], a['sham_tension'])
        full = t.pulse_pair(930201, 0, 0, .02, 'full')
        self.assertGreater(np.max(np.abs(full['pulse_action']-a['pulse_action'])), 1e-5)
        self.assertGreater(t.pulse_metrics(a, 0)['max_cross_polarity_effect'], 0.)

    def test_exogenous_inputs_identical_and_intervention_local(self):
        a = t.pulse_pair(930201, 2, 1, .02, 'full')
        b = t.pulse_pair(930201, 2, 1, .02, 'no_network')
        np.testing.assert_array_equal(a['targets'], b['targets'])
        expected = np.zeros((1, 8, 2)); expected[0, 2, 1] = .2
        np.testing.assert_allclose(a['intervened_initial']-a['initial'], expected, atol=1e-15)

    def test_coordinate_and_generic_consistency(self):
        a = t.pulse_pair(930202, 4, 1, .02, 'full')
        for v in ('coordinate', 'generic_flat'):
            b = t.pulse_pair(930202, 4, 1, .02, v)
            np.testing.assert_allclose(a['pulse_action'], b['pulse_action'], atol=1e-12, rtol=0)

    def test_network_terms_reconstruct_proposal(self):
        a = t.pulse_pair(930202, 1, 0, .02, 'full')
        expected = a['pulse_base_proposal']+a['pulse_eta']*(a['pulse_state_term']+a['pulse_tension_term'])
        np.testing.assert_array_equal(expected, a['pulse_proposal'])
        np.testing.assert_allclose(a['pulse_tension'], a['pulse_mismatch']+a['pulse_conflict'])

    def test_hidden_target_does_not_enter_network(self):
        a, b = t.model('full', 930201), t.model('full', 930201)
        mask = np.zeros((1, 8, 2), bool)
        out0 = a.act({'target': np.zeros((1, 8, 2)), 'observed': mask})
        out1 = b.act({'target': np.ones((1, 8, 2)), 'observed': mask})
        np.testing.assert_array_equal(out0, out1)

    def test_permutation_retains_weights_degrees_and_sparsity(self):
        w0, k0 = t.couplings(3, 'full', 930201)
        w1, k1 = t.couplings(3, 'permuted', 930201)
        for a, b in ((w0, w1), (k0, k1)):
            np.testing.assert_array_equal(np.sort(a.ravel()), np.sort(b.ravel()))
            np.testing.assert_array_equal(np.sort((a != 0).sum(axis=0)), np.sort((b != 0).sum(axis=0)))
            np.testing.assert_array_equal(np.sort((a != 0).sum(axis=1)), np.sort((b != 0).sum(axis=1)))

    def test_external_task_constraints_and_exact_replay(self):
        a = t.task_trial(930201, 'gain_resource_shift', 'full')
        b = t.task_trial(930201, 'gain_resource_shift', 'full')
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(t.task_metrics(a)['hard_violations'], 0)
        self.assertEqual(a['action'].shape, (96, 3, 8, 2))

    def test_final_guard_requires_registration_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'blocked'
            with self.assertRaises(ValueError):
                t.run('final', path)
            with self.assertRaises(ValueError):
                t.run('unrecognized', path)
            self.assertFalse(path.exists())

    def test_numpy_metric_scalars_serialize(self):
        pulse = t.pulse_metrics(t.pulse_pair(930201, 0, 0, .02, 'full'), 0)
        json.dumps(t.jsonable(pulse), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
