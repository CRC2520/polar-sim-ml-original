"""Contract checks for P1 population summaries, interventions and replay."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collective.bridge_v2.engine import BridgeConfig, BridgeEngine
from p1_completion.population import (GRID, FINAL_SEEDS, PILOT_SEEDS, cells,
    conditional_gap, threshold, paired_ci, run_cell, replay_cell, cell_name,
    P1FactorialEngine, FACTORIAL_FLAGS)


class PopulationContracts(unittest.TestCase):
    def test_design_counts_and_seed_separation(self):
        self.assertEqual(len(cells("A")), 48)
        self.assertEqual(len(cells("C")), 192)
        self.assertTrue(set(FINAL_SEEDS["A"]).isdisjoint(FINAL_SEEDS["C"]))
        self.assertTrue(set(PILOT_SEEDS).isdisjoint(FINAL_SEEDS["A"]))
        self.assertTrue(set(range(931001, 931031)).isdisjoint(FINAL_SEEDS["A"]))
        self.assertTrue(all(abs(p * 20 - round(p * 20)) < 1e-10 for p in GRID))

    def test_threshold_no_artificial_extrapolation(self):
        self.assertEqual(threshold(GRID, [0] * 8)["status"], "above_support_or_absent")
        self.assertIsNone(threshold(GRID, [0] * 8)["estimate"])
        self.assertEqual(threshold(GRID, [1] * 8)["status"], "at_or_below_support")
        self.assertEqual(threshold(GRID, [0, 0, .6, .4, .8, .9, 1, 1])["status"], "not_identifiable_nonmonotone")
        self.assertAlmostEqual(threshold(GRID, [0, 0, .2, .4, .6, .8, 1, 1])["estimate"], .525)
        self.assertEqual(threshold(GRID, [np.nan] * 8)["status"], "not_identifiable_missing")

    def test_absent_trait_class_missing_not_zero(self):
        self.assertTrue(np.isnan(conditional_gap([[1, 2]], [[True, True]])).all())
        self.assertEqual(conditional_gap([[3, 1]], [[True, False]])[0], 2)
        self.assertIsNone(paired_ci([np.nan, np.nan])["mean"])

    def test_score_equal_is_not_income_redistribution(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=40, warmup=0)
        base = BridgeEngine(config, PILOT_SEEDS, .5, "A", "base")
        equal = BridgeEngine(config, PILOT_SEEDS, .5, "A", "copy_score_equal")
        a, b = base.episode(0), equal.episode(0)
        for key in ("income_cumulative", "alive_agent_time", "resource_fraction"):
            np.testing.assert_array_equal(a[key], b[key])
        raw = equal.state["lineage"].astype(float) * .6 + .2
        b["alive_agent_time"] = raw
        transmission = equal.evolve(0, b)
        np.testing.assert_array_equal(transmission["raw_fitness"], raw)
        np.testing.assert_allclose(conditional_gap(transmission["copy_score"], transmission["lineage_before"]), 0, atol=1e-14)

    def test_physical_pool_conserves_and_equalizes(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=40)
        pool = BridgeEngine(config, PILOT_SEEDS, .5, "A", "resource_pool")
        outcome = pool.episode(0)
        self.assertLess(outcome["pool_conservation_error"].max(), 1e-12)
        self.assertEqual(outcome["pool_income_spread"].max(), 0)

    def test_coordinate_control_and_replay(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=40, warmup=0)
        with tempfile.TemporaryDirectory() as temporary:
            job = (temporary, "C", "pilot", "success", "coordinate_equivalent", .5, asdict(config))
            run_cell(job)
            path = Path(temporary) / cell_name("success", "coordinate_equivalent", .5)
            result = replay_cell(path)
            self.assertTrue(result["exact"], result)
            self.assertEqual(run_cell(job)["status"], "verified_existing")
            with (path / "outcomes.npz").open("ab") as handle:
                handle.write(b"integrity-test")
            with self.assertRaises(RuntimeError):
                run_cell(job)

    def test_factorial_single_switches_match_frozen_engine(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=60, warmup=0)
        for variant in ("full", "qv_policy_off", "short_vitality_discount", "blind_exploration"):
            reference = BridgeEngine(config, PILOT_SEEDS, .5, "C", variant)
            composite = P1FactorialEngine(config, PILOT_SEEDS, .5, FACTORIAL_FLAGS[variant])
            left, _ = reference.run()
            right, _ = composite.run()
            for key in ("alive_agent_time", "income_cumulative", "resource_fraction", "restraint_agent"):
                np.testing.assert_array_equal(left[key], right[key], err_msg=f"{variant}/{key}")
            for key in ("qt", "qv", "qg", "visits"):
                np.testing.assert_array_equal(reference.state[key], composite.state[key], err_msg=f"{variant}/{key}")

    def test_combined_factorial_replay(self):
        config = BridgeConfig(groups=2, replace_groups=1, generations=2, tail_generations=1, steps=40, warmup=0)
        with tempfile.TemporaryDirectory() as temporary:
            variant = "anchor_off_short_vitality_blind"
            run_cell((temporary, "C", "pilot", "success", variant, .5, asdict(config)))
            result = replay_cell(Path(temporary) / cell_name("success", variant, .5))
            self.assertTrue(result["exact"], result)


if __name__ == "__main__":
    unittest.main()
