"""Scientific contracts for new R8 adapters, factual calibration and controls."""
from dataclasses import replace
import unittest

import numpy as np

from collective.bridge_v2.engine import BridgeConfig, BridgeEngine
from r8_completion.population import (fit_arbitration, arbitrate_q, RegulatedBridge,
    FactorialBridge, FactualCalibrationBridge, _history_run, FINAL_SEEDS,
    TRAIN_SEEDS, VALIDATION_SEEDS, POWER_SEEDS, THRESHOLD_PILOT_SEEDS)


def calibration_fixture():
    rng = np.random.default_rng(940091)
    q = rng.normal(size=(2400, 3, 4))
    states = np.where(np.arange(len(q)) % 2, 0, 19)
    actions = rng.integers(0, 4, len(q))
    advantages = q - q.mean(-1, keepdims=True)
    chosen = advantages[np.arange(len(q)), :, actions]
    target = .5 + .1 * np.where(states == 0, chosen[:, 0], chosen[:, 2])
    return q, states, actions, target


class PopulationContracts(unittest.TestCase):
    def test_seed_partitions_disjoint_and_final_untouched_by_protocol(self):
        pools = [set(x) for x in (TRAIN_SEEDS, VALIDATION_SEEDS, POWER_SEEDS, THRESHOLD_PILOT_SEEDS, FINAL_SEEDS)]
        for i, pool in enumerate(pools):
            for other in pools[i + 1:]:
                self.assertTrue(pool.isdisjoint(other))

    def test_factual_state_fit_and_probe_exclusion(self):
        q, states, actions, y = calibration_fixture()
        calibration = fit_arbitration(q, states, actions, y, ridge=16)
        weights = np.asarray(calibration["state_weights"])
        self.assertGreater(weights[0, 0], weights[0, 2])
        self.assertGreater(weights[19, 2], weights[19, 0])
        np.testing.assert_allclose(weights.sum(1), 3.)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_array_equal(weights[1], calibration["global_weights"])
        bad = np.full((10, 3, 4), 1e15)
        extended = fit_arbitration(np.concatenate((q, bad)), np.r_[states, np.zeros(10, dtype=int)],
                                   np.r_[actions, np.zeros(10, dtype=int)], np.r_[y, np.full(10, -1e20)],
                                   np.r_[np.ones(len(q), dtype=bool), np.zeros(10, dtype=bool)], ridge=16)
        self.assertEqual(calibration, extended)

    def test_scale_and_action_common_offset_invariance(self):
        q, states, actions, y = calibration_fixture()
        calibration = fit_arbitration(q, states, actions, y)
        reference = arbitrate_q(q, states, calibration, "state")
        np.testing.assert_allclose(reference, arbitrate_q(q + np.array([7., -3., 11.])[None, :, None], states, calibration, "state"), atol=1e-13)
        altered = dict(calibration, head_scales=(np.asarray(calibration["head_scales"]) * np.array([3., .2, 8.])).tolist())
        np.testing.assert_allclose(reference, arbitrate_q(q * np.array([3., .2, 8.])[None, :, None], states, altered, "state"), atol=1e-13)
        self.assertEqual(sorted(map(tuple, calibration["state_weights"])), sorted(map(tuple, calibration["shuffled_state_weights"])))

    def test_legacy_and_shared_successor_reproduce_frozen_engine(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=3, tail_generations=1, steps=60)
        for arm, variant in (("legacy_full", "full"), ("shared_successor", "generic_shared_target"), ("qv_policy_off", "qv_policy_off")):
            old = BridgeEngine(config, [940081, 940082], .5, "C", variant)
            new = RegulatedBridge(config, [940081, 940082], .5, arm, {})
            a, _ = old.run()
            b, _, _, _, _ = _history_run(new)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{arm}/{key}")

    def test_factorial_single_switches_exact_and_fixed_score_negative(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=4, tail_generations=1, steps=60, warmup=1)
        for score, pool, variant in ((False, False, "base"), (True, False, "copy_score_equal"), (False, True, "resource_pool")):
            old = BridgeEngine(config, [940081, 940082], .5, "A", variant)
            new = FactorialBridge(config, [940081, 940082], .5, score, pool)
            a, tx = old.run()
            b, ntx, maps, _, _ = _history_run(new)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{variant}/{key}")
            for key in tx:
                np.testing.assert_array_equal(tx[key], ntx[key])
            if not score and not pool:
                reference_maps = maps
        for pool in (False, True):
            plain = FactorialBridge(config, [940081, 940082], .5, False, pool)
            equal = FactorialBridge(config, [940081, 940082], .5, True, pool)
            a, _, _, _, _ = _history_run(plain, reference_maps)
            b, _, _, _, _ = _history_run(equal, reference_maps)
            np.testing.assert_array_equal(a["final_state_sha256"], b["final_state_sha256"])

    def test_factual_labels_future_and_probe_budget(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=60)
        engine = FactualCalibrationBridge(config, [940081, 940082])
        data, outcome = engine.collect_generation(0, horizon=10)
        self.assertEqual(data["qheads"].shape[1:], (3, 4))
        self.assertTrue(len(data["returns"]) > 0)
        self.assertTrue(((data["returns"] >= 0) & (data["returns"] <= 1)).all())
        self.assertLessEqual(len(data["returns"]), outcome["exploration_count"].sum())
        native = BridgeEngine(config, [940081, 940082], .5, "C", "blind_exploration")
        expected = native.episode(0)
        for key in outcome:
            np.testing.assert_array_equal(outcome[key], expected[key])

    def test_validation_resource_is_current_public_observation(self):
        class ObservingReference(BridgeEngine):
            def observe(self, noise):
                result = super().observe(noise)
                self.seen.append(result[1].copy())
                return result
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=20)
        reference = ObservingReference(config, [940081, 940082], .5, "C", "generic_shared_target")
        reference.seen = []
        reference.episode(0)
        new = RegulatedBridge(config, [940081, 940082], .5, "shared_successor", {})
        result = new.episode(0)
        current = np.stack(reference.seen[::2], axis=1)
        expected = np.clip(current / reference.capacity, 0, 1).mean((1, 2))
        np.testing.assert_allclose(result["public_resource_fraction"], expected, rtol=0, atol=1e-15)
        self.assertFalse(np.array_equal(result["public_resource_fraction"], result["resource_fraction"]))


if __name__ == "__main__":
    unittest.main()
