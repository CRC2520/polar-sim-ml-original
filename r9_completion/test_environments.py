"""Physics contracts only; no learned controller or final evaluation seeds."""
import unittest
import numpy as np

from r9_completion.environments import ACTION_FLOW, DOMAIN_SPECS, HORIZON, ScalarEnvironment


def feasibility_witness(env):
    """Privileged hand-fixed witness, not a comparator or a learned controller.

    Uses true state and current demand. Rules are specified before fixture runs;
    no controller outcomes or future tape values select these rules.
    """
    family = env.spec.family
    if family == "ecology":
        return 1 if env.health < .55 else 0
    if family == "inventory":
        return 2 if env.backlog > .025 else 1
    return 2 if env._demand(env.t) > .065 and env.temperature < .50 else 1


class EnvironmentContracts(unittest.TestCase):
    def test_api_and_deterministic_replay(self):
        for domain in DOMAIN_SPECS:
            left, right = ScalarEnvironment(domain), ScalarEnvironment()
            np.testing.assert_array_equal(left.reset(950081), right.reset(950081, domain))
            for t in range(HORIZON):
                action = (t * 7 + t // 11) % 4
                a, b = left.step(action), right.step(action)
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:], b[1:])
                self.assertEqual(a[2], t == HORIZON - 1)
                self.assertEqual(a[0].shape, (3,))
                self.assertTrue(np.all((a[0] >= 0) & (a[0] <= 1)))
                self.assertGreaterEqual(a[1], 0.)
                self.assertLessEqual(a[1], 1.)
            self.assertEqual(left.evaluation_summary(), right.evaluation_summary())
            with self.assertRaises(RuntimeError):
                left.step(0)

    def test_tapes_are_independent_of_actions_and_domain(self):
        digests = []
        for domain in DOMAIN_SPECS:
            env = ScalarEnvironment(domain)
            env.reset(950082)
            before = env.tape_digest()
            for t in range(30):
                env.step(t % 4)
            self.assertEqual(before, env.tape_digest())
            digests.append(before)
        self.assertEqual(len(set(digests)), 1)

    def test_no_future_suffix_leakage(self):
        for domain in DOMAIN_SPECS:
            env = ScalarEnvironment(domain)
            env.reset(950083)
            changed = env.clone()
            for key in changed._tape:
                value = changed._tape[key].copy()
                value[41:] = value[41:][::-1] + .123
                changed._tape[key] = value
            for t in range(40):
                a, b = env.step(t % 3), changed.step(t % 3)
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:], b[1:])

    def test_delays_are_action_causal_and_exact(self):
        for domain, delay in (("ecology_train", 5), ("ecology_delay9", 9), ("inventory_transfer", 7)):
            first, second = ScalarEnvironment(domain), ScalarEnvironment(domain)
            first.reset(950084)
            second.reset(950084)
            differences = []
            for t in range(delay + 1):
                a = first.step(3 if t == 0 else 0)[3]
                b = second.step(0)[3]
                key = "growth" if "ecology" in domain else "arrival"
                differences.append(a[key] - b[key])
            np.testing.assert_array_equal(differences[:delay], np.zeros(delay))
            self.assertNotEqual(differences[delay], 0.)

    def test_identical_observations_can_hide_different_future(self):
        for domain in ("ecology_train", "inventory_transfer", "thermal_transfer"):
            env = ScalarEnvironment(domain)
            env.reset(950085)
            other = env.clone()
            if domain == "ecology_train":
                other.resource_history[0] = .25
            elif domain == "inventory_transfer":
                other.pipeline[0] = .20
            else:
                other.temperature = .90
            np.testing.assert_array_equal(env._observe(), other._observe())
            left, right = env.step(1), other.step(1)
            self.assertNotEqual(left[3]["state_snapshot"], right[3]["state_snapshot"])
            self.assertTrue(left[1] != right[1] or not np.array_equal(left[0], right[0]))

    def test_mass_balance_and_executed_flow(self):
        for domain in DOMAIN_SPECS:
            env = ScalarEnvironment(domain)
            env.reset(950086)
            initial = env.reserve
            net_flow = 0.
            for t in range(HORIZON):
                _, _, _, info = env.step(t % 4)
                self.assertLess(abs(info["balance_residual"]), 1e-12)
                self.assertGreaterEqual(info["executed_flow"], 0.)
                self.assertLessEqual(info["executed_flow"], info["requested_flow"] + 1e-12)
                self.assertTrue(all(info[key] >= 0. for key in ("arrival", "loss", "spill", "charge")))
                net_flow += info["arrival"] + info["growth"] + info["charge"] - info["executed_flow"] - info["loss"] - info["spill"]
            self.assertAlmostEqual(env.reserve, initial + net_flow, places=11)

    def test_thermal_inertia_and_failure_are_not_clipped_away(self):
        env = ScalarEnvironment("thermal_transfer")
        env.reset(950087)
        other = env.clone()
        env.step(3)
        other.step(0)
        initial_gap = env.temperature - other.temperature
        self.assertGreater(initial_gap, 0.)
        env.step(0)
        other.step(0)
        self.assertAlmostEqual(env.temperature - other.temperature, .95 * initial_gap)
        env.temperature = 1.5
        obs, reward, done, info = env.step(0)
        self.assertFalse(info["alive"])
        self.assertGreater(info["temperature"], 1.2)
        self.assertEqual(reward, 0.)
        self.assertFalse(done)
        while not done:
            obs, reward, done, info = env.step(3)
            self.assertFalse(info["alive"])
            self.assertEqual(reward, 0.)
        self.assertEqual(env.evaluation_summary()["steps"], HORIZON)
        # The hazard is also reachable from an ordinary reset, without injecting
        # a state: a fixed 12-rest/12-high-throughput stress tape overheats.
        stress = ScalarEnvironment("thermal_transfer")
        stress.reset(950087)
        reached_overload = False
        for t in range(HORIZON):
            stress.step(0 if t % 24 < 12 else 3)
            reached_overload |= stress.temperature > .65
        self.assertTrue(reached_overload)
        self.assertFalse(stress.alive)

    def test_privileged_feasibility_fixtures(self):
        # Eight fixed development fixtures, chosen independently of any policy.
        for domain in DOMAIN_SPECS:
            for seed in range(950088, 950096):
                env = ScalarEnvironment(domain)
                env.reset(seed)
                for _ in range(HORIZON):
                    env.step(feasibility_witness(env))
                summary = env.evaluation_summary()
                self.assertGreaterEqual(summary["alive_fraction"], .80, (domain, seed, summary))
                self.assertTrue(summary["complete"])
                self.assertLessEqual(summary["constraint_fraction"], 1.)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ScalarEnvironment("unknown")
        env = ScalarEnvironment()
        with self.assertRaises(RuntimeError):
            env.step(0)
        for seed in (-1, True, .5):
            with self.assertRaises(ValueError):
                env.reset(seed)
        env.reset(950096)
        for action in (-1, 4, True, .5):
            with self.assertRaises(ValueError):
                env.step(action)


if __name__ == "__main__":
    unittest.main()
