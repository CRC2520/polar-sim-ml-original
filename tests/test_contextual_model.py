import unittest
from dataclasses import replace
import numpy as np
from polar import ContextualPolarModel, ModelConfig, from_signed_intensity, to_signed_intensity, swap_poles
from polar.workspace import CapabilityModel


class ContextualMechanismTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(agents=3, types=8, seed=11)
        self.shape = (3, 8, 2)

    def obs(self, target, **kwargs):
        return {"target": np.broadcast_to(target, self.shape).copy(), "budget": 48., **kwargs}

    def test_inactivity_and_coactivation_are_not_conflated(self):
        quiet, active = np.zeros(self.shape), np.full(self.shape, .8)
        sq, iq = to_signed_intensity(quiet)
        sa, ia = to_signed_intensity(active)
        np.testing.assert_array_equal(sq, sa)
        self.assertTrue(np.all(iq == 0))
        self.assertTrue(np.all(ia == 1.6))
        np.testing.assert_allclose(from_signed_intensity(sa, ia), active)
        with self.assertRaises(ValueError):
            from_signed_intensity(1., 0.)

    def test_coordinate_control_same_actions_feedback_and_memory(self):
        a = ContextualPolarModel(self.cfg)
        b = ContextualPolarModel(replace(self.cfg, representation="signed_intensity"))
        rng = np.random.default_rng(100)
        for t in range(25):
            target = rng.uniform(0, .7, self.shape)
            observed = rng.random(self.shape) > .25
            observation = self.obs(target, observed=observed, cue=str(t % 3), budget=7., weights=rng.uniform(.2, 2, self.shape))
            aa, ab = a.act(observation), b.act(observation)
            np.testing.assert_allclose(aa, ab, atol=1e-13, rtol=1e-13)
            a.learn({"effect": np.clip(aa * .7, 0, 1)})
            b.learn({"effect": np.clip(ab * .7, 0, 1)})

    def test_sign_convention_is_equivariant_including_learned_state(self):
        a, b = ContextualPolarModel(self.cfg), ContextualPolarModel(self.cfg)
        rng = np.random.default_rng(101)
        for _ in range(4):
            obs = self.obs(rng.uniform(0, .5, self.shape), cue="learn")
            for model in (a, b):
                action = model.act(obs)
                model.learn({"effect": action * .7})
        b.reverse_convention(types=[0, 3, 5])
        for _ in range(5):
            obs = self.obs(rng.uniform(0, .7, self.shape), observed=rng.random(self.shape) > .5,
                           weights=rng.uniform(.1, 1, self.shape), cue="learn", budget=9.)
            reverse_obs = {key: swap_poles(value, [0, 3, 5]) if isinstance(value, np.ndarray) else value for key, value in obs.items()}
            aa, ab = a.act(obs), b.act(reverse_obs)
            np.testing.assert_allclose(ab, swap_poles(aa, [0, 3, 5]), atol=1e-13)
            a.learn({"effect": aa * .7})
            b.learn({"effect": ab * .7})

    def test_hidden_target_cannot_leak_into_action_or_memory(self):
        a, b = ContextualPolarModel(self.cfg), ContextualPolarModel(self.cfg)
        hidden = np.zeros(self.shape, bool)
        aa = a.act(self.obs(0., observed=hidden, cue="unseen"))
        ab = b.act(self.obs(1., observed=hidden, cue="unseen"))
        np.testing.assert_array_equal(aa, ab)
        a.learn({"effect": aa, "target": np.ones(self.shape)})
        values, confidence, _ = a.memory.recall("unseen")
        np.testing.assert_array_equal(values, np.zeros(self.shape))
        np.testing.assert_array_equal(confidence, np.zeros(self.shape))

    def test_memory_erasure_and_edit_change_post_stimulus_decisions(self):
        intact, erased, edited = [ContextualPolarModel(self.cfg) for _ in range(3)]
        for model in (intact, erased, edited):
            for _ in range(5):
                model.learn({"effect": model.act(self.obs(.8, cue="experience"))})
            # Equal action history/control state: only internal memory differs.
            model.q = np.zeros(self.shape)
        erased.memory.erase("experience")
        edited.memory.edit("experience", np.zeros(self.shape))
        cue_only = self.obs(0., observed=np.zeros(self.shape, bool), cue="experience")
        self.assertGreater(intact.act(cue_only).sum(), 1.)
        self.assertEqual(erased.act(cue_only).sum(), 0.)
        self.assertEqual(edited.act(cue_only).sum(), 0.)

    def test_memory_retention_decay_and_learning_are_auditable(self):
        model = ContextualPolarModel(replace(self.cfg, memory_retention=.5))
        model.memory.learn("cue", np.ones(self.shape))
        v1, c1, _ = model.memory.recall("cue")
        model.memory.tick()
        v2, c2, _ = model.memory.recall("cue")
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_allclose(c2, c1 * .5)
        self.assertEqual(model.memory.snapshot()["records"]["cue"]["counts"][0][0][0], 1)

    def test_capability_feedback_and_lesion_affect_inverse_planning(self):
        learned, lesioned = ContextualPolarModel(self.cfg), ContextualPolarModel(self.cfg)
        for model in (learned, lesioned):
            for _ in range(25):
                action = model.act(self.obs(.3))
                model.learn({"effect": action * .5})
            model.q = np.zeros(self.shape)
        self.assertLess(learned.self_model.gain_hat.mean(), .51)
        self.assertTrue(np.all(learned.self_model.counts > 0))
        lesioned.self_model.lesion(gain=1.)
        self.assertGreater(learned.act(self.obs(.3)).sum(), lesioned.act(self.obs(.3)).sum())

    def test_workspace_has_specific_resource_allocation_consequence(self):
        enabled = ContextualPolarModel(self.cfg)
        disabled = ContextualPolarModel(replace(self.cfg, use_workspace=False))
        target = np.zeros(self.shape)
        target[0] = .9
        aa = enabled.act(self.obs(target, budget=3.))
        ab = disabled.act(self.obs(target, budget=3.))
        self.assertAlmostEqual(aa.sum(), 3.)
        self.assertAlmostEqual(ab.sum(), 1.)
        # No resource scarcity: allocation has no artificial state consensus effect.
        enabled.reset(); disabled.reset()
        np.testing.assert_allclose(enabled.act(self.obs(target)), disabled.act(self.obs(target)))

    def test_context_switch_preserves_access_to_both_poles(self):
        model = ContextualPolarModel(self.cfg)
        for pole in (0, 1, 0):
            target = np.zeros(self.shape)
            target[..., pole] = .9
            for _ in range(12):
                action = model.act(self.obs(target))
                model.learn({"effect": action})
            self.assertGreater(action[..., pole].mean(), .89)
            self.assertLess(action[..., 1-pole].mean(), .01)

    def test_stability_admissibility_resources_are_separate(self):
        model = ContextualPolarModel(self.cfg)
        model.self_model.lesion(gain=.1)
        model.self_model.counts.fill(10000)
        allowed = np.ones(self.shape, bool); allowed[..., 0] = False
        action = model.act(self.obs(1., allowed=allowed, budget=2.))
        self.assertGreater(model.last_trace["stability"]["clipped_channels"], 0)
        self.assertGreater(model.last_trace["constraints"]["blocked_channels"], 0)
        self.assertLessEqual(action.sum(), 2. + 1e-12)
        self.assertTrue(np.all(action[..., 0] == 0))

    def test_missing_invalid_and_duplicate_feedback_fail_explicitly(self):
        with self.assertRaises(ValueError): ModelConfig(representation="recurrent")
        model = ContextualPolarModel(self.cfg)
        with self.assertRaises(ValueError): model.act({})
        with self.assertRaises(ValueError): model.act(self.obs(np.nan))
        with self.assertRaises(ValueError): model.act(self.obs(.5, budget=-1))
        with self.assertRaises(ValueError): model.act(self.obs(.5, costs=0))
        with self.assertRaises(ValueError): model.act(self.obs(.5, observed=np.ones(self.shape)))
        with self.assertRaises(RuntimeError): model.learn({"effect": np.zeros(self.shape)})
        action = model.act(self.obs(.5))
        with self.assertRaises(RuntimeError): model.act(self.obs(.8))
        with self.assertRaises(ValueError): model.learn({})
        model.learn({"effect": action})
        with self.assertRaises(RuntimeError): model.learn({"effect": action})
        model.act(self.obs(.4))
        with self.assertRaises(ValueError): model.skip_feedback("")
        model.skip_feedback("sensor unavailable")
        self.assertTrue(model.last_trace["feedback"]["missing"])
        model.act(self.obs(.5))
        model.skip_feedback("test complete")
        cap = CapabilityModel(self.shape)
        with self.assertRaises(ValueError): cap.learn(np.full(self.shape, np.nan), np.zeros(self.shape))

    def test_reset_seed_replays_workspace_shuffle_and_trace(self):
        model = ContextualPolarModel(self.cfg)
        model.workspace.mode = "shuffle"
        target = np.linspace(0, .9, np.prod(self.shape)).reshape(self.shape)
        a = model.act(self.obs(target, budget=3.))
        model.reset(seed=self.cfg.seed)
        model.workspace.mode = "shuffle"
        b = model.act(self.obs(target, budget=3.))
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
