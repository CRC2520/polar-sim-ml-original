"""Mathematical contracts only: these tests never instantiate a POLAR model."""

import json
import unittest

import numpy as np

from r8_completion.population_thresholds import (
    GRID_DEV,
    PILOT_PRESSURES,
    choose_pressure_and_grid,
    estimate_threshold,
    viable_seed,
)


def binary_rows(counts, n=100):
    """Construct deterministic seed observations with the requested counts."""
    return (np.arange(n)[:, None] < np.asarray(counts)[None, :]).astype(int)


def pilot_records(curves=None, n=100):
    default = [0, 0, 0, 10, 30, 50, 70, 90, 100, 100, 100]
    curves = curves or {}
    return [
        {
            "metabolism": pressure,
            "rule": rule,
            "seeds": list(range(n)),
            "grid": GRID_DEV,
            "viable": binary_rows(curves.get((pressure, rule), default), n).tolist(),
        }
        for pressure in PILOT_PRESSURES
        for rule in ("success", "conformity")
    ]


class ThresholdContracts(unittest.TestCase):
    def test_inclusive_viability_and_missing_values(self):
        self.assertTrue(viable_seed(0.8, 0.2))
        self.assertEqual(viable_seed([0.8, 0.79, 1, 1], [0.2, 1, 0.19, 1]), [True, False, False, True])
        for alive, resource in (([0.8], 0.2), ([np.nan], [1]), ([1.01], [1])):
            with self.assertRaises(ValueError):
                viable_seed(alive, resource)

    def test_no_extrapolation_outside_support(self):
        for value, status in ((0, "above_support_or_absent"), (1, "at_or_below_support")):
            result = estimate_threshold([0.2, 0.8], np.full((30, 2), value), bootstrap=40)
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["estimate"])
            self.assertEqual(result["bootstrap"]["status_counts"][status], 40)
            self.assertEqual(result["bootstrap"]["valid_fraction"], 0)
            self.assertIsNone(result["bootstrap"]["conditional_in_support_ci95"])
            self.assertFalse(result["reliable_identification"])

    def test_single_crossing_uses_only_adjacent_points(self):
        result = estimate_threshold([0, 0.3, 0.8, 1], binary_rows([0, 20, 80, 100]), bootstrap=0)
        self.assertEqual(result["status"], "within_support")
        self.assertAlmostEqual(result["estimate"], 0.55)
        self.assertEqual(result["crossing_interval"], [0.3, 0.8])
        self.assertFalse(result["reliable_identification"])

    def test_exact_half_is_high_and_plateau_is_one_crossing(self):
        result = estimate_threshold([0, 0.3, 0.6, 1], binary_rows([0, 50, 50, 100]), bootstrap=0)
        self.assertEqual(result["status"], "within_support")
        self.assertAlmostEqual(result["estimate"], 0.3)
        self.assertEqual(estimate_threshold([0, 1], binary_rows([50, 100]), bootstrap=0)["status"], "at_or_below_support")

    def test_reductions_on_one_side_of_half_are_allowed(self):
        result = estimate_threshold([0, 0.2, 0.4, 0.6, 0.8, 1], binary_rows([10, 30, 20, 60, 55, 90]), bootstrap=0)
        self.assertEqual(result["status"], "within_support")
        self.assertAlmostEqual(result["estimate"], 0.55)

    def test_high_low_reversal_rejects_unique_crossing(self):
        result = estimate_threshold([0, 0.3, 0.6, 1], binary_rows([0, 90, 10, 100]), bootstrap=50)
        self.assertEqual(result["status"], "no_unique_crossing")
        self.assertTrue(result["confident_reversal"])
        self.assertEqual(result["confidence_bracket_status"], "no_unique_crossing")
        self.assertIsNone(result["confidence_bracket"])
        self.assertIsNone(result["estimate"])
        self.assertFalse(result["reliable_identification"])

    def test_exact_binomial_known_value_and_simultaneous_wider(self):
        result = estimate_threshold([0, 0.5, 1], binary_rows([0, 50, 100]), bootstrap=0)
        marginal = np.asarray(result["marginal_ci95"])
        simultaneous = np.asarray(result["simultaneous_ci95"])
        self.assertAlmostEqual(marginal[1, 0], 0.398321129503301)
        self.assertAlmostEqual(marginal[1, 1], 0.601678870496699)
        self.assertEqual(marginal[0, 0], 0)
        self.assertEqual(marginal[-1, 1], 1)
        self.assertAlmostEqual(marginal[0, 1], 1 - 0.025 ** (1 / 100))
        self.assertAlmostEqual(marginal[-1, 0], 0.025 ** (1 / 100))
        self.assertIn("Clopper-Pearson", result["interval_method"])
        self.assertTrue((simultaneous[:, 0] <= marginal[:, 0] + 1e-15).all())
        self.assertTrue((simultaneous[:, 1] >= marginal[:, 1] - 1e-15).all())

    def test_confidence_bracket_uses_extreme_confident_points(self):
        result = estimate_threshold([0, 0.2, 0.4, 0.6, 0.8, 1], binary_rows([0, 10, 45, 55, 90, 100]), bootstrap=100)
        self.assertEqual(result["confidence_bracket"], [0.2, 0.8])
        self.assertEqual(result["confidence_bracket_status"], "ordered_confidence_bracket")
        self.assertTrue(result["reliable_identification"])

    def test_observed_crossing_without_confidence_is_not_reliable(self):
        result = estimate_threshold([0, 1], [[0, 1], [0, 1]], bootstrap=50)
        self.assertEqual(result["status"], "within_support")
        self.assertEqual(result["bootstrap"]["valid_fraction"], 1)
        self.assertIsNone(result["confidence_bracket"])
        self.assertFalse(result["reliable_identification"])

    def test_bootstrap_draws_complete_paired_rows(self):
        observations = np.asarray([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])
        result = estimate_threshold([0, 0.5, 1], observations, bootstrap=151, seed=12345)
        generator = np.random.default_rng(12345)
        counts = dict.fromkeys(result["bootstrap"]["status_counts"], 0)
        estimates = []
        for _ in range(151):
            draw = observations[generator.integers(0, 4, size=4)]
            replicate = estimate_threshold([0, 0.5, 1], draw, bootstrap=0)
            counts[replicate["status"]] += 1
            if replicate["estimate"] is not None:
                estimates.append(replicate["estimate"])
        self.assertEqual(result["bootstrap"]["status_counts"], counts)
        self.assertEqual(sum(counts.values()), 151)
        self.assertEqual(result["bootstrap"]["valid_fraction"], len(estimates) / 151)
        np.testing.assert_array_equal(result["bootstrap"]["conditional_in_support_ci95"], np.quantile(estimates, [0.025, 0.975]))
        self.assertEqual(result, estimate_threshold([0, 0.5, 1], observations, bootstrap=151, seed=12345))
        json.dumps(result, allow_nan=False)

    def test_validation_rejects_ambiguous_or_invalid_data(self):
        cases = [
            ([0, 0], [[0, 1]]),
            ([1, 0], [[0, 1]]),
            ([0, 1], [[0, np.nan]]),
            ([0, 1], [[0, 0.2]]),
            ([0, 1], []),
            ([0, 1], [0, 1]),
            ([-0.1, 1], [[0, 1]]),
            ([0, 0.5, 1], [[0, 1]]),
        ]
        for grid, values in cases:
            with self.subTest(grid=grid, values=values), self.assertRaises(ValueError):
                estimate_threshold(grid, values, bootstrap=0)
        for bootstrap in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                estimate_threshold([0, 1], [[0, 1]], bootstrap=bootstrap)


class PilotSelectionContracts(unittest.TestCase):
    def test_tie_selects_lowest_pressure_and_refines_grid(self):
        result = choose_pressure_and_grid(pilot_records())
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["pressure"], 0.3)
        self.assertEqual(result["center"], 0.5)
        self.assertEqual(result["grid"], [0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1])
        self.assertFalse(result["final_diagnostic_only"])
        json.dumps(result, allow_nan=False)

    def test_selects_closest_average_not_lowest_pressure(self):
        right = [0, 0, 0, 0, 0, 10, 30, 50, 70, 90, 100]
        left = [0, 10, 30, 50, 70, 90, 100, 100, 100, 100, 100]
        curves = {(0.3, rule): right for rule in ("success", "conformity")}
        curves.update({(0.6, rule): left for rule in ("success", "conformity")})
        result = choose_pressure_and_grid(pilot_records(curves))
        self.assertEqual(result["pressure"], 0.45)

    def test_both_rules_must_meet_endpoints_and_crossing(self):
        records = pilot_records()
        for record in records:
            if record["rule"] == "conformity":
                record["viable"] = binary_rows([30] * 10 + [100]).tolist()
        result = choose_pressure_and_grid(records)
        self.assertEqual(result["status"], "no_supported_pilot_crossing")
        self.assertEqual(result["pressure"], 0.3)
        self.assertEqual(result["grid"], GRID_DEV)
        self.assertTrue(result["final_diagnostic_only"])
        self.assertTrue(all(not candidate["eligible"] for candidate in result["candidates"]))

    def test_nonunique_crossing_is_ineligible_even_with_valid_endpoints(self):
        curves = {(pressure, rule): [0, 0, 60, 40, 60, 70, 80, 90, 90, 100, 100] for pressure in PILOT_PRESSURES for rule in ("success", "conformity")}
        result = choose_pressure_and_grid(pilot_records(curves))
        self.assertEqual(result["status"], "no_supported_pilot_crossing")
        self.assertTrue(all(candidate["endpoint_criterion_met"] for candidate in result["candidates"]))

    def test_wide_rule_dispersion_retains_full_grid(self):
        left = [0, 10, 30, 50, 70, 90, 100, 100, 100, 100, 100]
        right = [0, 0, 0, 0, 0, 10, 30, 50, 70, 90, 100]
        curves = {(pressure, rule): left if rule == "success" else right for pressure in PILOT_PRESSURES for rule in ("success", "conformity")}
        result = choose_pressure_and_grid(pilot_records(curves))
        self.assertEqual(result["grid"], GRID_DEV)
        self.assertAlmostEqual(result["threshold_spread"], 0.4)

    def test_edge_refinement_never_extrapolates(self):
        curves = {(pressure, rule): [0] + [100] * 10 for pressure in PILOT_PRESSURES for rule in ("success", "conformity")}
        result = choose_pressure_and_grid(pilot_records(curves))
        self.assertEqual(result["center"], 0.05)
        self.assertEqual(result["grid"], [0, 0.05, 0.1, 0.15, 0.2, 1])

    def test_incomplete_duplicate_or_unpaired_pilots_rejected(self):
        records = pilot_records()
        with self.assertRaises(ValueError):
            choose_pressure_and_grid(records[:-1])
        duplicate = records.copy()
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            choose_pressure_and_grid(duplicate)
        records[0]["seeds"] = list(reversed(records[0]["seeds"]))
        with self.assertRaises(ValueError):
            choose_pressure_and_grid(records)


if __name__ == "__main__":
    unittest.main()
