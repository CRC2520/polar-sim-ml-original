"""Deterministic contract tests, not a pilot or efficacy experiment."""
import copy
import json
import unittest

import numpy as np

from r9_completion.network import SparseActionModel, tension_vector


class SparseActionModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(320, 8))
        self.a = np.arange(320) % 4
        self.y = np.column_stack((2*self.x[:, 0]-self.x[:, 3]+.2*self.a,
                                  -self.x[:, 0]+.5*self.x[:, 3]-.1*self.a))

    def model(self, mode='sparse', **kwargs):
        return SparseActionModel(8, 2, mode=mode, **kwargs).fit(self.x, self.a, self.y)

    def test_objective_descent_and_direct_objective_all_modes(self):
        for mode in ('sparse', 'dense', 'fixed'):
            with self.subTest(mode=mode):
                m = self.model(mode)
                self.assertLessEqual(np.max(np.diff(m.objective_history_, axis=1)), 1e-11)
                self.assertTrue(np.all(m.objective_history_[:, -1] < m.objective_history_[:, 0]))
                self.assertAlmostEqual(m.objective_loss(), m.objective_loss(self.x, self.a, self.y), places=12)
                self.assertEqual(m.fit_diagnostics['gradient_steps_total'], 320)
                self.assertEqual(m.objective_history_.shape, (4, 81))

    def test_correct_descent_sign_and_sparse_support(self):
        m = self.model(l1=.04)
        self.assertTrue(np.all(m.coef_[:, 0, 0] > 0))
        self.assertTrue(np.all(m.coef_[:, 3, 0] < 0))
        self.assertTrue(np.all(m.coef_[:, 0, 1] < 0))
        self.assertTrue(np.all(m.coef_[:, 3, 1] > 0))
        irrelevant = np.array([1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(m.coef_[:, irrelevant], 0.)

    def test_large_penalty_exact_zero_with_unpenalized_intercept(self):
        m = self.model(l1=1e4)
        np.testing.assert_array_equal(m.coef_, 0.)
        for a in range(4):
            np.testing.assert_allclose(m.intercept_[a], self.y[self.a == a].mean(0), atol=1e-12)

    def test_sparse_zero_penalty_matches_dense(self):
        sparse, dense = self.model(l1=0.), self.model('dense', l1=100.)
        np.testing.assert_array_equal(sparse.coef_, dense.coef_)
        np.testing.assert_array_equal(sparse.intercept_, dense.intercept_)
        np.testing.assert_array_equal(sparse.predict_all(self.x), dense.predict_all(self.x))

    def test_fixed_support_is_data_independent_and_enforced(self):
        a = self.model('fixed')
        b = SparseActionModel(8, 2, mode='fixed')
        np.testing.assert_array_equal(a.allowed_support_, b.allowed_support_)
        np.testing.assert_array_equal(a.allowed_support_.sum(axis=1), 4)
        np.testing.assert_array_equal(a.coef_[~a.allowed_support_], 0.)
        b.fit(self.x[::-1], self.a[::-1], self.y[::-1]*3)
        np.testing.assert_array_equal(a.allowed_support_, b.allowed_support_)

    def test_local_ablation_preserves_intercept_center_and_state(self):
        m = self.model()
        digest = m.parameter_digest()
        mask = np.zeros((8, 2), dtype=bool)
        mask[0] = True
        components = m.predict_components(self.x, mask)
        np.testing.assert_allclose(components['local']+components['cross'], components['full'], atol=1e-12)
        np.testing.assert_array_equal(components['local'], m.predict_local_all(self.x, mask))
        np.testing.assert_array_equal(m.predict_local_all(m.feature_mean_, mask), m.intercept_)
        np.testing.assert_array_equal(m.predict_all(m.feature_mean_), m.intercept_)
        np.testing.assert_array_equal(m.predict_local_all(self.x, np.ones((8, 2))), m.predict_all(self.x))
        self.assertEqual(m.parameter_digest(), digest)

    def test_training_only_normalization_and_constant_features(self):
        x = self.x.copy()
        x[:, 7] = 9.
        m = SparseActionModel(8, 2).fit(x, self.a, self.y)
        np.testing.assert_array_equal(m.coef_[:, 7], 0.)
        expected_mean = m.feature_mean_.copy()
        digest = m.parameter_digest()
        m.predict_all(x+100.)
        np.testing.assert_array_equal(expected_mean, m.feature_mean_)
        self.assertEqual(digest, m.parameter_digest())
        with self.assertRaises(ValueError):
            m.fit(x, self.a, self.y, split='test')
        self.assertEqual(digest, m.parameter_digest())

    def test_safe_json_roundtrip_and_invalid_checkpoint(self):
        for mode in ('sparse', 'dense', 'fixed'):
            m = self.model(mode)
            portable = json.loads(json.dumps(m.state_dict(), allow_nan=False))
            recovered = SparseActionModel.from_state_dict(portable)
            np.testing.assert_array_equal(m.predict_all(self.x), recovered.predict_all(self.x))
            self.assertEqual(m.parameter_digest(), recovered.parameter_digest())
        changed = copy.deepcopy(portable)
        changed['feature_scale_'][0] = 0.
        with self.assertRaises(ValueError):
            SparseActionModel.from_state_dict(changed)
        changed = copy.deepcopy(portable)
        changed['allowed_support'][0][0][0] = not changed['allowed_support'][0][0][0]
        with self.assertRaises(ValueError):
            SparseActionModel.from_state_dict(changed)

    def test_input_contracts_and_batched_predictions(self):
        m = self.model()
        self.assertEqual(m.predict_all(self.x[:6].reshape(2, 3, 8)).shape, (2, 3, 4, 2))
        self.assertEqual(m.predict_all(self.x[0]).shape, (4, 2))
        with self.assertRaises(ValueError):
            m.fit(self.x, self.a.astype(float), self.y)
        with self.assertRaises(ValueError):
            m.fit(self.x[self.a != 3], self.a[self.a != 3], self.y[self.a != 3])
        with self.assertRaises(ValueError):
            m.predict_all(np.full((1, 8), np.nan))
        with self.assertRaises(ValueError):
            m.predict_local_all(self.x, np.full((8, 2), .5))
        self.assertFalse(m.fit_diagnostics['causal_graph_identified'])

    def test_observation_targets_only_api_and_input_immutability(self):
        # No simulator/frame/oracle object is accepted by prediction; future Y
        # affects the fit only, and all caller arrays remain untouched.
        x, a, y = self.x.copy(), self.a.copy(), self.y.copy()
        m = SparseActionModel(8, 2).fit(x, a, y)
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(a, self.a)
        np.testing.assert_array_equal(y, self.y)
        with self.assertRaises(TypeError):
            m.predict_all(x, next_observation=y)


class TensionVectorTests(unittest.TestCase):
    def test_components_are_separate_and_opposition_is_predictive(self):
        p = np.array([[.8, .6], [.2, .5]])
        prior = np.array([[.6, .2], [.2, .5]])
        candidates = np.broadcast_to(p, (4, 2, 2)).copy()
        candidates[1, 0] = (.9, .4)
        candidates[2, 0] = (.7, .8)
        candidates[3, 0] = (.9, .7)  # Same-direction response contributes zero.
        tau = tension_vector(p, prior, candidates, [.1, .2], coactivation_weight=.2, n_pairs=2)
        self.assertEqual(tau.shape, (2, 4))
        np.testing.assert_allclose(tau[0], [.3, .096, (.02+.02)/3, .1], atol=1e-14)
        np.testing.assert_allclose(tau[1], [0., .02, 0., .2], atol=1e-14)

    def test_no_predicted_action_difference_means_no_opposition(self):
        p = np.full((2, 3, 8, 2), .5)
        candidates = np.repeat(p[..., None, :, :], 4, axis=-3)
        tau = tension_vector(p, p, candidates, np.full((8,), .1))
        self.assertEqual(tau.shape, (2, 3, 8, 4))
        np.testing.assert_array_equal(tau[..., 0], 0.)
        np.testing.assert_array_equal(tau[..., 2], 0.)
        np.testing.assert_array_equal(tau[..., 3], .1)

    def test_tension_rejects_invalid_proxies(self):
        p = np.full((8, 2), .5)
        all_p = np.repeat(p[None], 4, axis=0)
        for uncertainty in (-1., np.nan):
            with self.assertRaises(ValueError):
                tension_vector(p, p, all_p, uncertainty)
        with self.assertRaises(ValueError):
            tension_vector(p, p, all_p, .1, n_pairs=3)
        with self.assertRaises(ValueError):
            tension_vector(p, p, all_p, .1, coactivation_weight=2.)
        with self.assertRaises(ValueError):
            tension_vector(p, p, all_p+1., .1)


if __name__ == '__main__':
    unittest.main()
