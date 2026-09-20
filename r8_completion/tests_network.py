"""Engineering tests; synthetic fixtures are not native benefit evidence."""
import json
from pathlib import Path
import unittest
import numpy as np
from r8_completion.learned_network import (
    LearnedNetwork, JointRidgePredictor, operational_tension,
    to_signed_intensity, from_signed_intensity, VARIANTS,
)


def transitions(seed, n=800):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.05, .8, (n, 8, 2))
    goal = rng.uniform(.05, .8, (n, 8, 2))
    tau = operational_tension(p, goal, .2)
    action = np.tile(np.arange(4), (n+3)//4)[:n]
    next_p = .1+.5*p
    for ai in range(4):
        chosen = action == ai
        other = np.roll(p[chosen], ai+1, axis=1)
        other_tau = np.roll(tau[chosen], ai+1, axis=1)[..., None]
        next_p[chosen] += .2*other+.08*other_tau+.01*ai
    return p, tau, action, next_p


class LearnedNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = transitions(940201)
        cls.test = transitions(940202, 240)
        cls.model = LearnedNetwork(seed=940201).fit(*cls.train)

    def test_two_channels_independent_and_coactivation_not_inactivity(self):
        p = np.ones((8, 2))
        q = np.zeros((8, 2))
        self.assertTrue(np.all(to_signed_intensity(p)[0] == to_signed_intensity(q)[0]))
        np.testing.assert_array_equal(operational_tension(p, p, 0), np.zeros(8))
        np.testing.assert_array_equal(operational_tension(p, p, .2), np.full(8, .2))
        self.assertTrue(np.any(self.train[0].sum(axis=-1) > 1))

    def test_forbidden_own_pair_routes_zero(self):
        owner = np.arange(16)//2
        np.testing.assert_array_equal(self.model.W[:, owner[:, None] == owner[None, :]], 0)
        np.testing.assert_array_equal(self.model.K[:, np.arange(16), owner], 0)

    def test_fit_learns_predictive_cross_relations_on_unit_fixture(self):
        p, tau, action, target = self.test
        local = self.model.predict_all(p, tau, variant='off', gate=1)[np.arange(len(p)), action]
        learned = self.model.predict_all(p, tau, gate=1)[np.arange(len(p)), action]
        mse_local, mse_learned = np.mean((local-target)**2), np.mean((learned-target)**2)
        self.assertLess(mse_learned, .2*mse_local)
        self.assertGreater(np.linalg.norm(self.model.W), 0)
        self.assertGreater(np.linalg.norm(self.model.K), 0)

    def test_predict_handles_single_and_multiple_batch_axes(self):
        p, tau = self.test[:2]
        self.assertEqual(self.model.predict_all(p[0], tau[0], gate=1).shape, (4, 8, 2))
        self.assertEqual(self.model.predict_all(p[:12].reshape(2, 6, 8, 2), tau[:12].reshape(2, 6, 8), gate=1).shape,
                         (2, 6, 4, 8, 2))

    def test_fit_and_tuning_refuse_test_and_future_metadata(self):
        model = LearnedNetwork(seed=940201)
        with self.assertRaises(ValueError):
            model.fit(*self.train, split='test')
        with self.assertRaises(ValueError):
            self.model.select_gate({0: 0, .25: 1, .5: 2, 1: 3}, split='test')
        with self.assertRaises(TypeError):
            self.model.predict_all(self.test[0], self.test[1], hidden_next_poles=self.test[-1])

    def test_development_gate_can_select_off_without_mutating_coefficients(self):
        model = LearnedNetwork.from_state_dict(self.model.state_dict())
        w, k = model.W.copy(), model.K.copy()
        self.assertEqual(model.select_gate({0: 1., .25: 1.1, .5: 1.2, 1: 1.3}), 0)
        np.testing.assert_array_equal(w, model.W)
        np.testing.assert_array_equal(k, model.K)
        p, tau = self.test[:2]
        np.testing.assert_array_equal(model.predict_all(p, tau), model.predict_all(p, tau, variant='off', gate=1))
        self.assertEqual(model.select_gate({0: 1., .25: 1., .5: 1., 1: 1.}), 0)

    def test_intervention_and_lesion_isolate_predictive_paths(self):
        model = LearnedNetwork.from_state_dict(self.model.state_dict())
        p, tau = self.test[0][0].copy(), self.test[1][0].copy()
        changed = p.copy(); changed[0, 0] += .05
        full_diff = model.predict_components(changed, tau, gate=1)['unbounded']-model.predict_components(p, tau, gate=1)['unbounded']
        expected = model.W[:, :, 0]*.05
        # Other pairs cannot change through local own-pair regression.
        np.testing.assert_allclose(full_diff.reshape(4, 16)[:, 2:], expected[:, 2:], atol=1e-14)
        off = model.lesioned(remove_state=True)
        off_diff = off.predict_components(changed, tau, gate=1)['unbounded']-off.predict_components(p, tau, gate=1)['unbounded']
        np.testing.assert_allclose(off_diff[:, 1:], 0, atol=1e-14)
        tchanged = tau.copy(); tchanged[0] += .05
        tdiff = model.predict_components(p, tchanged, gate=1)['unbounded']-model.predict_components(p, tau, gate=1)['unbounded']
        np.testing.assert_allclose(tdiff.reshape(4, 16), model.K[:, :, 0]*.05, atol=1e-14)
        self.assertFalse(np.shares_memory(off.W, model.W))

    def test_multihop_delay_unit_graph_not_empirical_learning_claim(self):
        model = LearnedNetwork.from_state_dict(self.model.state_dict())
        model.local.fill(0); model.W.fill(0); model.K.fill(0)
        model.poles_mean.fill(0); model.poles_scale.fill(1)
        model.tension_mean.fill(0)
        for dest in range(16):
            model.local[:, dest, 1+dest%2] = 1
        model.W[:, 2, 0] = .2; model.W[:, 4, 2] = .2
        sham = np.full((8, 2), .2); pulse = sham.copy(); pulse[0, 0] += .1
        tau = np.zeros(8)
        u0, u1 = model.predict_all(sham, tau, gate=1)[0], model.predict_all(pulse, tau, gate=1)[0]
        self.assertAlmostEqual(u1[1, 0]-u0[1, 0], .02)
        self.assertEqual(u1[2, 0]-u0[2, 0], 0)
        v0, v1 = model.predict_all(u0, tau, gate=1)[0], model.predict_all(u1, tau, gate=1)[0]
        self.assertAlmostEqual(v1[2, 0]-v0[2, 0], .004)

    def test_coordinate_and_generic_flat_preserve_predictions(self):
        p, tau = self.test[:2]
        signed, intensity = to_signed_intensity(p)
        np.testing.assert_allclose(from_signed_intensity(signed, intensity), p, atol=1e-14)
        ref = self.model.predict_all(p, tau, gate=1)
        np.testing.assert_allclose(self.model.predict_all_signed(signed, intensity, tau, gate=1), ref, atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.model.predict_all(p, tau, variant='generic_flat', gate=1), ref, atol=1e-12, rtol=0)

    def test_controls_have_bounded_predictions_and_preserved_weight_norm(self):
        for variant in VARIANTS:
            pred = self.model.predict_all(*self.test[:2], variant=variant, gate=1)
            self.assertTrue(np.all((pred >= 0) & (pred <= 1)))
        for ai in range(4):
            actual = np.linalg.norm(np.r_[self.model.W[ai].ravel(), self.model.K[ai].ravel()])
            random = np.linalg.norm(np.r_[self.model.random_W[ai].ravel(), self.model.random_K[ai].ravel()])
            scrambled = np.linalg.norm(np.r_[self.model.scrambled_W[ai].ravel(), self.model.scrambled_K[ai].ravel()])
            self.assertAlmostEqual(actual, random, places=12)
            self.assertAlmostEqual(actual, scrambled, places=12)

    def test_joint_competitor_matches_capacity_inputs_and_is_distinct_fit(self):
        joint = JointRidgePredictor(seed=940201).fit(*self.train)
        for key in ('samples', 'action_counts', 'training_arrays_sha256', 'local_parameters', 'cross_parameters'):
            self.assertEqual(self.model.fit_diagnostics[key], joint.fit_diagnostics[key])
        self.assertEqual(joint.fit_diagnostics['local_parameters']+joint.fit_diagnostics['cross_parameters'], 1536)
        self.assertGreater(np.max(np.abs(joint.W-self.model.W)), 1e-7)
        self.assertEqual(joint.fit_mode, 'joint_ridge')

    def test_state_roundtrip_exact_for_both_learning_rules(self):
        for model in (self.model, JointRidgePredictor(seed=940201).fit(*self.train)):
            restored = LearnedNetwork.from_state_dict(model.state_dict())
            self.assertEqual(model.parameter_digest(), restored.parameter_digest())
            self.assertEqual(model.fit_mode, restored.fit_mode)
            np.testing.assert_array_equal(model.predict_all(*self.test[:2], gate=1),
                                          restored.predict_all(*self.test[:2], gate=1))

    def test_coverage_and_invalid_data_fail_explicitly(self):
        p, tau, action, y = self.train
        with self.assertRaises(ValueError):
            LearnedNetwork().fit(p, tau, np.zeros_like(action), y)
        invalid = p.copy(); invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            LearnedNetwork().fit(invalid, tau, action, y)
        self.assertEqual(sum(self.model.fit_diagnostics['action_counts']), len(p))
        self.assertTrue(np.isfinite(self.model.fit_diagnostics['regularized_network_condition_max']))

    def test_predictions_do_not_adapt_to_evaluation_samples(self):
        before = self.model.parameter_digest()
        self.model.predict_all(self.test[0], self.test[1], gate=1)
        self.assertEqual(before, self.model.parameter_digest())

    def test_existing_p1_tension_source_remains_frozen(self):
        path = Path(__file__).resolve().parents[1]/'p1_completion/FREEZE_P1.json'
        if not path.exists():
            self.skipTest('Historical P1 source not included in this isolated distribution')
        from p1_completion.tension import source_hashes
        registered = json.loads(path.read_text())
        self.assertEqual(registered['tension_source_sha256'], source_hashes())


if __name__ == '__main__':
    unittest.main()
