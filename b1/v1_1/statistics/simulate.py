"""Statistical-only simulation; exact multinomial sufficient statistics.

Run full validation with --output PATH; --smoke writes nothing by default.
This module neither imports nor executes a Polar task, controller, or seed API.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

from .method import (ALPHA, DELTA, EPSILON, FAMILY_SIZE, TARGET_HALF_WIDTH,
                     fixed_n, planning_second_moment_upper)

HERE = Path(__file__).resolve().parent


def laws():
    return {
        "zero_effect": ([-0.1, 0.1], [0.5, 0.5]),
        "practical_margin_boundary": ([-0.1 - DELTA, 0.1 - DELTA], [0.5, 0.5]),
        "equivalence_margin_boundary": ([-0.1 + EPSILON, 0.1 + EPSILON], [0.5, 0.5]),
        "negative_equivalence_margin_boundary": ([-0.1 - EPSILON, 0.1 - EPSILON], [0.5, 0.5]),
        "favorable_effect": ([-0.1 - 2 * DELTA, 0.1 - 2 * DELTA], [0.5, 0.5]),
        "adverse_effect": ([-0.1 + 2 * DELTA, 0.1 + 2 * DELTA], [0.5, 0.5]),
        # Rare extreme events violate a zero-variance assumption while remaining bounded.
        "heavy_tail_bounded": ([-1.0, -0.01, 0.01, 1.0], [0.1, 0.4, 0.4, 0.1]),
        "discrete_distribution": ([-1 / 192, 0, 1 / 192], [0.2, 0.6, 0.2]),
        # A latent iid mixture of low/high variance regimes, not independent jobs.
        "heterogeneous_variance": ([-0.5, -0.02, 0.02, 0.5], [0.25] * 4),
    }


def sufficient(rng, n, repetitions, support, probabilities, support_range=2.0):
    x = np.asarray(support, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    counts = rng.multinomial(n, p, size=repetitions)
    means = counts @ x / n
    variances = np.maximum(0, (counts @ (x * x) - n * means * means) / (n - 1))
    t = math.log(4 * FAMILY_SIZE / ALPHA)
    half = np.sqrt(2 * variances * t / n) + 7 * support_range * t / (3 * (n - 1))
    return means, half, float(p @ x), float(p @ (x * x) - (p @ x) ** 2)


def rate(events):
    events = np.asarray(events, dtype=bool)
    total = int(events.size)
    count = int(events.sum())
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, total - count + 1))
    high = 1.0 if count == total else float(beta.ppf(0.975, count + 1, total - count))
    return {"count": count, "denominator": total, "rate": count / total,
            "exact_binomial_95_interval": [low, high]}


def evaluate_scenario(rng, n, repetitions, name, law, dependence,
                      support_range=2.0, delta=DELTA, epsilon=EPSILON,
                      target_half_width=TARGET_HALF_WIDTH):
    cover, improved, worsened, equivalent, precise = [], [], [], [], []
    first = None
    for coordinate in range(FAMILY_SIZE):
        if dependence == "perfectly_dependent_coordinates" and first is not None:
            mean, half, true_mean, true_variance = first
        else:
            mean, half, true_mean, true_variance = sufficient(rng, n, repetitions, *law, support_range)
            first = (mean, half, true_mean, true_variance)
        ok = half <= target_half_width
        cover.append((mean - half <= true_mean) & (true_mean <= mean + half))
        improved.append(ok & (mean + half < -delta))
        worsened.append(ok & (mean - half > delta))
        equivalent.append(ok & (mean - half > -epsilon) & (mean + half < epsilon))
        precise.append(ok)
    cover, improved, worsened, equivalent, precise = map(np.asarray,
        (cover, improved, worsened, equivalent, precise))
    false_improved = np.any(improved, axis=0) if true_mean >= -delta else np.zeros(repetitions, bool)
    false_worsened = np.any(worsened, axis=0) if true_mean <= delta else np.zeros(repetitions, bool)
    false_equivalent = np.any(equivalent, axis=0) if abs(true_mean) >= epsilon else np.zeros(repetitions, bool)
    correct = (improved if true_mean < -delta else worsened if true_mean > delta
               else equivalent if abs(true_mean) < epsilon else np.zeros_like(improved))
    return {
        "scenario": name, "N": n, "dependence": dependence,
        "support_points": law[0], "probabilities": law[1],
        "true_mean": true_mean, "true_variance": true_variance,
        "declared_support_range": support_range, "delta": delta, "epsilon": epsilon,
        "target_half_width": target_half_width,
        "family_coverage": rate(np.all(cover, axis=0)),
        "family_false_improved": rate(false_improved),
        "family_false_worsened": rate(false_worsened),
        "family_false_equivalent": rate(false_equivalent),
        "scalar_correct_classification_power": rate(correct[0]),
        "all_nine_precise": rate(np.all(precise, axis=0)),
        "limitation": "Synthetic coordinates probe arbitrary family dependence; they are not model outcomes or a simulated intervention experiment",
    }


def run(smoke=False):
    design_bytes = (HERE / "SIMULATION_DESIGN.json").read_bytes()
    design = json.loads(design_bytes)
    seed = int.from_bytes(hashlib.sha256(design["random_seed_text"].encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    repetitions = design["smoke_repetitions_per_scenario" if smoke else "full_repetitions_per_scenario"]
    n = fixed_n(planning_second_moment_upper([0] * 6))
    results = []
    for count in (n, 32):
        for dependence in design["dependence_stress"]:
            for name, law in laws().items():
                results.append(evaluate_scenario(rng, count, repetitions, name, law, dependence))
            results.append(evaluate_scenario(rng, count, repetitions, "context_range4_boundary",
                ([-1.0, 2.0], [0.5, 0.5]), dependence, 4.0, 0.5, 0.25, 0.125))
    minimum = min(x["family_coverage"]["rate"] for x in results)
    largest_error = max(x[key]["rate"] for x in results for key in (
        "family_false_improved", "family_false_worsened", "family_false_equivalent"))
    cutoff = 0.01 / (4 * len(results))
    for cell in results:
        counts = [repetitions - cell["family_coverage"]["count"]] + [
            cell[key]["count"] for key in ("family_false_improved", "family_false_worsened", "family_false_equivalent")]
        p_values = [float(binom.sf(k - 1, repetitions, ALPHA)) for k in counts]
        cell["regression_check"] = {"nominal_error": ALPHA, "exact_upper_tail_p_values": p_values,
                                    "Bonferroni_QA_cutoff": cutoff}
        if min(p_values) < cutoff:
            raise RuntimeError("Nominal-error regression; report rather than tune and hide it")
    return {
        "schema_version": "1.1", "status": "PASS", "mode": "smoke" if smoke else "full",
        "QA_revision": design["QA_revision"],
        "development_only": True, "confirmatory": False, "reusable_as_final": False,
        "B1E_executed": False, "final_seeds_generated": False,
        "model_executions": 0, "statistical_simulations_only": True,
        "namespace": design["namespace"], "substream": design["substream"],
        "design_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "method_sha256": hashlib.sha256((HERE / "method.py").read_bytes()).hexdigest(),
        "simulator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "N_v2": n,
        "repetitions_per_scenario": repetitions, "scenario_cells": len(results),
        "minimum_family_coverage": minimum, "maximum_family_false_declaration_rate": largest_error,
        "interpretation": "Analytical coverage is distribution-free under declared iid bundle assumptions. Monte Carlo supports implementation checking only; zero observed errors do not prove zero error probability.",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.smoke)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite an archived statistical validation result")
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, sort_keys=True))


if __name__ == "__main__":
    main()
