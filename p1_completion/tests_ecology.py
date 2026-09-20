"""Mechanistic contracts for the new P1.E extension, using pilot-only seeds."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import ecology as e
except ModuleNotFoundError:
    from p1_completion import ecology as e


class EcologyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = e.acquisition(930101)
        cls.models, cls.shifted = e.fit_models(cls.record)

    def test_forced_physics_matches_preserved_native_engine(self):
        c = replace(e.config_for(), steps=12, epsilon=0.)
        base = e.BridgeEngine(c, [930101], .5, study="C", variant="full")
        tape = base.random_tape(91)
        actions = np.array([[0, 1, 2, 3, 0, 2, 1, 3], [3, 2, 1, 0, 3, 1, 2, 0]])
        tape["gumbel"][:] = -100.
        for t in range(c.steps):
            for g in range(c.groups):
                for i in range(c.agents):
                    tape["gumbel"][0, t, g, i, actions[g, i]] = 100.
        resource = base.state["resource"][0].copy()
        vitality = base.state["vitality"][0].copy()
        alive = base.state["alive"][0].copy()
        history = base.state["resource_history"][0].copy()
        for t in range(c.steps):
            growth = c.growth_mean + c.growth_amplitude * np.sin(2*np.pi*(t+tape["offset"][0])/c.growth_period)
            resource, vitality, alive, history, _, _ = e.step_physics(resource, vitality, alive, history, actions, growth, c)
        base.episode(91, learn=False, tape=tape)
        np.testing.assert_array_equal(resource, base.state["resource"][0])
        np.testing.assert_array_equal(vitality, base.state["vitality"][0])
        np.testing.assert_array_equal(alive, base.state["alive"][0])
        np.testing.assert_array_equal(history, base.state["resource_history"][0])

    def test_credit_shift_changes_only_outcome_assignment(self):
        before = e.digest_arrays(self.record)
        r, v, a, y = e.public_training_data(self.record, e.config_for())
        shifted = e.shifted_targets(self.record, y)
        for ids in np.split(np.arange(len(y)), e.PROTOCOL["train_anchors"]):
            self.assertEqual(np.unique(r[ids]).size, 1)
            self.assertEqual(np.unique(v[ids]).size, 1)
            np.testing.assert_array_equal(np.sort(y[ids]), np.sort(shifted[ids]))
            np.testing.assert_array_equal(shifted[ids], np.roll(y[ids], 1))
        self.assertEqual(before, e.digest_arrays(self.record))
        self.assertFalse(np.array_equal(y, shifted))

    def test_one_factual_initial_action_per_training_episode(self):
        r = self.record
        np.testing.assert_array_equal(r["own_action"], r["actions_DIAGNOSTIC"][:, 0, 0])
        for ids in np.split(np.arange(len(r["own_action"])), e.PROTOCOL["train_anchors"]):
            np.testing.assert_array_equal(np.bincount(r["own_action"][ids], minlength=4), [8]*4)
        self.assertEqual(r["public_resource"].shape, (4096, 13))

    def test_hidden_states_cannot_change_fitted_model(self):
        changed = {k: v.copy() for k, v in self.record.items()}
        for key in changed:
            if "DIAGNOSTIC" in key:
                changed[key] = np.zeros_like(changed[key])
        fitted, _ = e.fit_models(changed)
        for name in self.models:
            np.testing.assert_array_equal(self.models[name].coefficients, fitted[name].coefficients)

    def test_action_credit_is_identifiable_in_noiseless_mechanism(self):
        r, v, action, _ = e.public_training_data(self.record, e.config_for())
        y = 80 + 30*r - 2*action*r
        aligned = e.ConsequenceModel().fit(r, v, action, y)
        wrong = e.ConsequenceModel().fit(r, v, action, e.shifted_targets(self.record, y))
        truth = 2 * r[:, None] * np.arange(4)
        self.assertLess(np.mean((aligned.costs(r, v) - truth)**2), .001)
        self.assertGreater(np.mean((wrong.costs(r, v) - truth)**2), .1)

    def test_generic_and_ecological_capacity_inputs_and_budget(self):
        r, v, action, y = e.public_training_data(self.record, e.config_for())
        self.assertEqual(e.features(r, v, action, "ecological").shape, (4096, 20))
        self.assertEqual(e.features(r, v, action, "generic").shape, (4096, 20))
        self.assertEqual(len(self.models["aligned"].coefficients), len(self.models["generic"].coefficients))
        self.assertIs(self.models["aligned"], self.models["no_head"])

    def test_zero_action_has_zero_marginal_cost(self):
        for model in self.models.values():
            np.testing.assert_array_equal(model.costs(np.array([.2, .6]), np.array([.3, .8]))[:, 0], [0., 0.])

    def test_train_and_holdout_are_different_anchor_streams(self):
        test = e.acquisition(930101, evaluation=True)
        self.assertNotEqual(self.record["split_stream_id"][0], test["split_stream_id"][0])
        self.assertFalse(np.array_equal(self.record["public_resource"][::32, 0], test["public_resource"][::4, 0]))
        np.testing.assert_array_equal(test["noise_tape_DIAGNOSTIC"][::4], test["noise_tape_DIAGNOSTIC"][1::4])

    def test_no_head_gate_is_exact_and_direct_generic_reexpression_matches(self):
        c = e.config_for()
        base = e.BridgeEngine(c, [930101], .5, study="C", variant="full")
        models = [e.native_engine(base, c, self.models["aligned"], arm) for arm in ("aligned", "no_head", "generic")]
        idx, _ = base.observe(np.zeros((1, 2)))
        for engine in models:
            engine.observe(np.zeros((1, 2)))
        np.testing.assert_array_equal(models[1].decision_scores(idx), base.decision_scores(idx))
        np.testing.assert_array_equal(models[0].decision_scores(idx), models[2].decision_scores(idx))

    def test_native_evaluation_freezes_every_learned_array(self):
        c = replace(e.config_for(), steps=4)
        base = e.BridgeEngine(c, [930101], .5, study="C", variant="full")
        engine = e.native_engine(base, c, self.models["aligned"], "aligned")
        before = {k: engine.state[k].copy() for k in ("qt", "qv", "qg", "visits")}
        coef = engine.ecological_model.coefficients.copy()
        engine.episode(700, learn=False)
        for k in before:
            np.testing.assert_array_equal(before[k], engine.state[k])
        np.testing.assert_array_equal(coef, engine.ecological_model.coefficients)


if __name__ == "__main__":
    unittest.main()
