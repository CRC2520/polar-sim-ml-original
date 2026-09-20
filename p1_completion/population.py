"""P1-A/C descriptive population coverage on the preserved bridge_v2 engine.

No fitting/calibration is performed. Independent seeds, not agents, are the units
of uncertainty. Historical confirmatory acceptance gates remain out of scope.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from collective.bridge_v2.engine import BridgeConfig, BridgeEngine, VARIANTS_A, VARIANTS_C


GRID = (.35, .40, .45, .50, .55, .60, .65, .70)
PILOT_SEEDS = tuple(range(930001, 930004))
FINAL_SEEDS = {"A": tuple(range(935001, 935031)), "C": tuple(range(932001, 932031))}
FACTORIAL_FLAGS = {
    "full": (False, False, False),
    "qv_policy_off": (True, False, False),
    "short_vitality_discount": (False, True, False),
    "blind_exploration": (False, False, True),
    "anchor_off_short_vitality": (True, True, False),
    "anchor_off_blind": (True, False, True),
    "short_vitality_blind": (False, True, True),
    "anchor_off_short_vitality_blind": (True, True, True),
}
P1_VARIANTS_C = VARIANTS_C + tuple(name for name in FACTORIAL_FLAGS if name not in VARIANTS_C)
SEED_DEVIATION = (
    "Development timing benchmark evaluated one generation at seeds931001–931030; "
    "no outcomes were inspected or saved. Entire range excluded from final A, "
    "replaced before final analysis by935001–935030. C seeds were not benchmarked."
)
ENDPOINTS = (
    "survival_time_fraction", "resource_time_fraction", "terminal_survival_fraction",
    "terminal_resource_fraction", "restraint_behavior", "restrained_mode_fraction",
    "lineage_fraction", "raw_survival_gap", "copy_score_gap", "real_income_rate_gap",
    "real_income_cumulative_gap", "survival_time_gap", "viable",
)


class P1FactorialEngine(BridgeEngine):
    """Compose existing switches without editing the preserved engine.

    Flags are architectural, not acquired/transmitted state. Vitality learning
    remains enabled when its policy contribution is removed.
    """
    def __init__(self, config, seeds, initial_fraction, flags):
        self.anchor_off, self.short_vitality, self.blind = tuple(flags)
        config = replace(config, gamma_vitality=.5) if self.short_vitality else config
        super().__init__(config, seeds, initial_fraction, "C", "blind_exploration" if self.blind else "full")

    def decision_scores(self, state_index):
        if self.anchor_off:
            qt, qv, qg = self.head_values(state_index)
            return qt + 0. * qv + qg
        return super().decision_scores(state_index)


def make_engine(config, seeds, fraction, part, variant):
    if part == "C" and variant not in VARIANTS_C:
        return P1FactorialEngine(config, seeds, fraction, FACTORIAL_FLAGS[variant])
    return BridgeEngine(config, seeds, fraction, part, variant)


def json_write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    from collective.bridge_v2 import engine
    return {"population.py": file_sha(__file__), "engine.py": file_sha(engine.__file__)}


def config_default():
    return BridgeConfig()


def cells(part):
    variants = VARIANTS_A if part == "A" else P1_VARIANTS_C
    return [(rule, variant, fraction) for rule in ("success", "conformity")
            for variant in variants for fraction in GRID]


def cell_name(rule, variant, fraction):
    return f"{rule}__{variant}__p{round(fraction * 100):02d}"


def finite_mean(array, axis=None):
    array = np.asarray(array, dtype=float)
    mask = np.isfinite(array)
    count = mask.sum(axis=axis)
    return np.divide(np.where(mask, array, 0).sum(axis=axis), count,
                     out=np.full(np.shape(count), np.nan, dtype=float), where=count > 0)


def conditional_gap(values, positive):
    """Within-group positive-minus-negative gap; absent classes stay missing."""
    values, positive = np.asarray(values, dtype=float), np.asarray(positive, dtype=bool)
    pos = finite_mean(np.where(positive, values, np.nan), axis=-1)
    neg = finite_mean(np.where(~positive, values, np.nan), axis=-1)
    return pos - neg


def summarize_cell(history, transmission, config, part):
    tail = slice(-config.tail_generations, None)
    # Behavior classes in C are contemporaneously measured, not ancestral traits.
    classes = transmission["lineage_before"].astype(bool) if part == "A" else history["restraint_agent"] >= .5
    raw_gap = conditional_gap(transmission["raw_fitness"], classes)
    score_gap = conditional_gap(transmission["copy_score"], classes)
    rate_gap = conditional_gap(history["income_agent"], classes)
    cumulative_gap = conditional_gap(history["income_cumulative"], classes)
    time_gap = conditional_gap(history["alive_agent_time"] * config.steps, classes)
    mean_tail = lambda x: finite_mean(x[:, tail], axis=tuple(range(1, x.ndim)))
    alive, resource = mean_tail(history["alive_fraction"]), mean_tail(history["resource_fraction"])
    endpoints = {
        "survival_time_fraction": alive,
        "resource_time_fraction": resource,
        "terminal_survival_fraction": mean_tail(history["terminal_survival"]),
        "terminal_resource_fraction": mean_tail(history["terminal_resource"]),
        "restraint_behavior": mean_tail(history["restraint_behavior"]),
        "restrained_mode_fraction": mean_tail((history["restraint_agent"] >= .5).astype(float)),
        "lineage_fraction": mean_tail(history["lineage_fraction"]),
        "raw_survival_gap": mean_tail(raw_gap), "copy_score_gap": mean_tail(score_gap),
        "real_income_rate_gap": mean_tail(rate_gap),
        "real_income_cumulative_gap": mean_tail(cumulative_gap),
        "survival_time_gap": mean_tail(time_gap),
        "viable": ((alive >= config.viable_alive) & (resource >= config.viable_resource)).astype(float),
    }
    generation_metrics = {
        "raw_survival_gap": raw_gap, "copy_score_gap": score_gap,
        "real_income_rate_gap": rate_gap, "real_income_cumulative_gap": cumulative_gap,
        "survival_time_gap": time_gap,
        "conditional_gap_observed": np.isfinite(raw_gap),
    }
    return endpoints, generation_metrics


def threshold(grid, success_probability):
    """A supported first crossing, or an explicit identification failure.

    Nonmonotonicity means an empirical decrease in viability probability. This
    strict descriptive rule does not fit an isotonic curve or hide sampling noise.
    """
    x, p = np.asarray(grid, dtype=float), np.asarray(success_probability, dtype=float)
    if not np.all(np.isfinite(p)):
        return {"status": "not_identifiable_missing", "estimate": None, "bracket": None}
    if np.any(np.diff(p) < -1e-12):
        return {"status": "not_identifiable_nonmonotone", "estimate": None, "bracket": None}
    if p[0] >= .5:
        return {"status": "at_or_below_support", "estimate": None, "bracket": [None, float(x[0])]}
    if p[-1] < .5:
        return {"status": "above_support_or_absent", "estimate": None, "bracket": [float(x[-1]), None]}
    hi = int(np.flatnonzero(p >= .5)[0])
    lo = hi - 1
    value = x[lo] + (.5 - p[lo]) * (x[hi] - x[lo]) / (p[hi] - p[lo])
    return {"status": "within_support", "estimate": float(value), "bracket": [float(x[lo]), float(x[hi])]}


def threshold_bootstrap(success, repetitions=2000, seed=937004):
    success = np.asarray(success, dtype=float)
    point = threshold(GRID, success.mean(axis=0))
    rng = np.random.default_rng(seed)
    counts, values = {}, []
    for _ in range(repetitions):
        result = threshold(GRID, success[rng.integers(0, len(success), len(success))].mean(axis=0))
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        if result["estimate"] is not None:
            values.append(result["estimate"])
    point["bootstrap_status_counts"] = counts
    # Conditional intervals are explicitly labeled; never imply invalid draws
    # were ordinary in-support thresholds.
    point["conditional_in_support_percentile_95"] = np.quantile(values, [.025, .975]).tolist() if values else None
    point["bootstrap_repetitions"] = repetitions
    return point


def threshold_contrast(terms, repetitions=2000, seed=937005):
    """Paired threshold arithmetic; invalid components invalidate the contrast."""
    def evaluate(indices):
        components = [threshold(GRID, success[indices].mean(0)) for success, weight in terms]
        statuses = [component["status"] for component in components]
        if not all(status == "within_support" for status in statuses):
            return None, "|".join(statuses)
        return sum(component["estimate"] * weight for component, (success, weight) in zip(components, terms)), "all_within_support"
    n = len(terms[0][0])
    if any(len(success) != n for success, weight in terms):
        raise ValueError("Threshold contrast requires paired seed counts")
    estimate, point_status = evaluate(np.arange(n))
    rng, values, counts = np.random.default_rng(seed), [], {}
    for _ in range(repetitions):
        value, status = evaluate(rng.integers(0, n, n))
        counts[status] = counts.get(status, 0) + 1
        if value is not None:
            values.append(value)
    return {"estimate": estimate, "component_statuses": point_status,
            "bootstrap_status_counts": counts, "valid_bootstrap_draws": len(values),
            "bootstrap_repetitions": repetitions,
            "conditional_all_components_identified_ci95": np.quantile(values, [.025, .975]).tolist() if values else None,
            "interpretation": "Threshold contrast not identified unless every component has a supported crossing. Survival effects are separate secondary endpoints."}


def paired_ci(difference, repetitions=2000, seed=937003):
    difference = np.asarray(difference, dtype=float)
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return {"mean": None, "ci95": None, "n_seeds": 0}
    rng = np.random.default_rng(seed)
    boot = difference[rng.integers(0, len(difference), (repetitions, len(difference)))].mean(axis=1)
    return {"mean": float(difference.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(), "n_seeds": len(difference)}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_cell(job):
    output, part, phase, rule, variant, fraction, config_values = job
    config = BridgeConfig(**dict(config_values, transmission=rule))
    seeds = PILOT_SEEDS if phase == "pilot" else FINAL_SEEDS[part]
    target = Path(output) / cell_name(rule, variant, fraction)
    expected = {"part": part, "phase": phase, "rule": rule, "variant": variant,
                "fraction": fraction, "seeds": list(seeds), "config": asdict(config), "source_hashes": source_hashes()}
    completion = target / "COMPLETE.json"
    if completion.exists():
        existing = json.loads(completion.read_text())
        if existing["protocol"] != expected:
            raise RuntimeError(f"Existing completed cell has different protocol: {target}")
        for file, digest in existing["sha256"].items():
            if file_sha(target / file) != digest:
                raise RuntimeError(f"Completed artifact failed integrity check: {target / file}")
        return {"cell": target.name, "status": "verified_existing"}
    if target.exists():
        raise RuntimeError(f"Unfinished cell exists; preserve and investigate it: {target}")
    target.mkdir(parents=True)
    json_write_new(target / "PROTOCOL.json", expected)
    start = time.perf_counter()
    engine = make_engine(config, seeds, fraction, part, variant)
    histories, transmissions = [], []
    for generation in range(config.generations):
        trace = target / f"sample_seed_{seeds[0]}_generation_{generation:03d}.npz" if generation in (0, config.generations - 1) else None
        outcome = engine.episode(generation, trace_path=trace, trace_mode="diagnostic" if trace else "reconstruction")
        outcome["terminal_resource"] = engine.state["resource"].copy() / engine.capacity
        outcome["terminal_survival"] = engine.state["alive"].mean(-1)
        histories.append(outcome)
        transmissions.append(engine.evolve(generation, outcome))
    history = {key: np.stack([h[key] for h in histories], axis=1) for key in histories[0]}
    transmission = {key: np.stack([h[key] for h in transmissions], axis=1) for key in transmissions[0]}
    endpoints, generation_metrics = summarize_cell(history, transmission, config, part)
    np.savez_compressed(target / "outcomes.npz", **history)
    np.savez_compressed(target / "transmissions.npz", **transmission)
    np.savez_compressed(target / "endpoints.npz", seeds=np.asarray(seeds), **endpoints, **{"generation_" + k: v for k, v in generation_metrics.items()})
    engine.save_snapshot(target / "final_snapshot.npz", config.generations)
    json_write_new(target / "SNAPSHOT_ARCHITECTURE.json", {
        "requested_variant": variant, "factorial_flags": FACTORIAL_FLAGS.get(variant) if part == "C" else None,
        "note": "Final snapshot stores base engine state; recreate architecture through make_engine using PROTOCOL.json before restoring arrays for a combined factorial condition.",
    })
    mixed = transmission["lineage_before"].min(-1) != transmission["lineage_before"].max(-1)
    copy_gaps = conditional_gap(transmission["copy_score"], transmission["lineage_before"].astype(bool))
    check = {
        "max_pool_conservation_error": float(history["pool_conservation_error"].max()),
        "max_pool_live_income_spread": float(history["pool_income_spread"].max()),
        "copy_score_max_trait_mean_gap_in_mixed_groups": float(np.abs(copy_gaps[mixed]).max()) if mixed.any() else None,
        "mixed_group_generations": int(mixed.sum()),
        "eligible_behavior_is_finite": bool(np.isfinite(history["restraint_agent"]).all()),
        "raw_fitness_is_recorded_separately": True,
    }
    json_write_new(target / "SUMMARY.json", clean_json({"per_seed_endpoints": endpoints, "manipulation_checks": check}))
    digests = {p.name: file_sha(p) for p in sorted(target.iterdir()) if p.is_file()}
    elapsed = time.perf_counter() - start
    json_write_new(completion, {"protocol": expected, "elapsed_seconds": elapsed, "sha256": digests})
    return {"cell": target.name, "status": "complete", "elapsed_seconds": round(elapsed, 3)}


def aggregate(output, part, phase, repetitions=2000):
    output = Path(output)
    variants = VARIANTS_A if part == "A" else P1_VARIANTS_C
    baseline = variants[0]
    summaries, contrasts, treatments = {}, {}, {}
    for rule in ("success", "conformity"):
        for variant in variants:
            data = []
            checks = []
            for fraction in GRID:
                target = output / cell_name(rule, variant, fraction)
                with np.load(target / "endpoints.npz", allow_pickle=False) as z:
                    data.append({k: z[k].copy() for k in ENDPOINTS})
                checks.append(json.loads((target / "SUMMARY.json").read_text())["manipulation_checks"])
            assembled = {k: np.stack([item[k] for item in data], axis=1) for k in ENDPOINTS}
            treatments[(rule, variant)] = assembled
            summaries[f"{rule}/{variant}"] = {
                "means_by_grid": {k: finite_mean(v, 0) for k, v in assembled.items()},
                "seed_count_by_grid": {k: np.isfinite(v).sum(0) for k, v in assembled.items()},
                "p_star": threshold_bootstrap(assembled["viable"], repetitions),
                "manipulation_checks_by_grid": checks,
            }
        for variant in variants[1:]:
            delta = {k: treatments[(rule, variant)][k] - treatments[(rule, baseline)][k] for k in ENDPOINTS}
            contrasts[f"{rule}/{variant}-minus-{baseline}"] = {
                "grid_average": {k: paired_ci(finite_mean(v, 1), repetitions) for k, v in delta.items()},
                "by_grid": {k: [paired_ci(v[:, j], repetitions) for j in range(len(GRID))] for k, v in delta.items()},
                "paired_seed_complete_grid_counts": {k: int(np.isfinite(v).all(1).sum()) for k, v in delta.items()},
                "p_star_difference": threshold_contrast([(treatments[(rule, variant)]["viable"], 1), (treatments[(rule, baseline)]["viable"], -1)], repetitions),
            }
    # Check the score-only intervention against the real-income intervention.
    first_gen_checks = {}
    for rule in ("success", "conformity"):
        for fraction in GRID:
            first = output / cell_name(rule, baseline, fraction)
            other_variant = "copy_score_equal" if part == "A" else "coordinate_equivalent"
            second = output / cell_name(rule, other_variant, fraction)
            with np.load(first / "outcomes.npz") as a, np.load(second / "outcomes.npz") as b:
                physical = ("alive_agent_time", "income_agent", "income_cumulative", "resource_fraction", "restraint_agent")
                first_gen_checks[f"{rule}/{fraction:.2f}"] = {
                    "first_generation_all_physical_endpoints_equal": all(np.array_equal(a[k][:, 0], b[k][:, 0], equal_nan=True) for k in physical),
                    "all_generation_max_physical_endpoint_absolute_difference": max(float(np.nanmax(np.abs(a[k] - b[k]))) for k in physical),
                    "all_generation_state_hashes_exact": bool(np.array_equal(a["final_state_sha256"], b["final_state_sha256"])),
                    "negative_control_expected": "All generations physically exact under conformity, as copying scores are ignored." if part == "A" and rule == "conformity" else None,
                    "interpretation": "Only generation0 is a physical counterfactual: later transmission may differ." if part == "A" else "Coordinate control can differ at floating-point roundoff; substantive trajectories should agree.",
                }
    factorial_contrasts = {}
    if part == "C":
        for rule in ("success", "conformity"):
            for factor_index, factor_name in enumerate(("qv_policy_off", "short_vitality_discount", "blind_exploration")):
                differences = {}
                for metric in ENDPOINTS:
                    enabled = np.stack([treatments[(rule, name)][metric] for name, flags in FACTORIAL_FLAGS.items() if flags[factor_index]])
                    disabled = np.stack([treatments[(rule, name)][metric] for name, flags in FACTORIAL_FLAGS.items() if not flags[factor_index]])
                    delta = finite_mean(enabled, 0) - finite_mean(disabled, 0)
                    differences[metric] = {"grid_average": paired_ci(finite_mean(delta, 1), repetitions), "by_grid": [paired_ci(delta[:, j], repetitions) for j in range(len(GRID))]}
                factorial_contrasts[f"{rule}/marginal_{factor_name}"] = differences
            interaction = {}
            for metric in ENDPOINTS:
                delta = sum(((-1) ** (3 - sum(flags))) * treatments[(rule, name)][metric] for name, flags in FACTORIAL_FLAGS.items())
                interaction[metric] = {"grid_average": paired_ci(finite_mean(delta, 1), repetitions), "by_grid": [paired_ci(delta[:, j], repetitions) for j in range(len(GRID))]}
            factorial_contrasts[f"{rule}/three_way_interaction"] = interaction
    threshold_difference_in_differences = {}
    if part == "A":
        for variant in VARIANTS_A[1:]:
            threshold_difference_in_differences[f"{variant}-minus-base_success-minus-conformity"] = threshold_contrast([
                (treatments[("success", variant)]["viable"], 1), (treatments[("success", baseline)]["viable"], -1),
                (treatments[("conformity", variant)]["viable"], -1), (treatments[("conformity", baseline)]["viable"], 1),
            ], repetitions)
    result = {
        "study": f"P1-{part}", "phase": phase, "grid": GRID,
        "seeds": PILOT_SEEDS if phase == "pilot" else FINAL_SEEDS[part],
        "seed_deviation": SEED_DEVIATION,
        "inference": "Descriptive exploratory extension; seed-paired percentile bootstrap; no confirmatory PASS/FAIL or hypothesis p-values; intervals marginal and unadjusted across all contrasts/endpoints/grid values. Familywise claims are not warranted.",
        "threshold_definition": "Per seed mean(last5generation alive_agent_time)>=.8 AND mean(last5generation resource fraction)>=.2; p* first supported crossing of seed success probability.5. Nonmonotone empirical curve -> not identifiable, no smoothing or extrapolation.",
        "gap_definition": "Within-group positive-minus-negative class means, then average observed groups and last5generations; missing trait classes retained as missing. A classes=lineage; C classes=observed restraint>=.5 and associations are endogenous, not trait-causal effects.",
        "missingness_policy": "Primary survival/resource/viability include all30seeds including extinct populations. Conditional gaps omit only undefined class means, with observed counts preserved in cell endpoints and summary finite seed counts; they are descriptive associations, not all-agent causal effects.",
        "summaries": summaries, "paired_contrasts": contrasts,
        "factorial_contrasts": factorial_contrasts,
        "p_star_difference_in_differences": threshold_difference_in_differences,
        "factorial_flags": FACTORIAL_FLAGS if part == "C" else None,
        "causal_limits": "A copy_score_equal removes conditional trait-mean score advantage while retaining within-class variation and group copying. Pool changes physical income and downstream copying jointly; not a clean factorial separation of physical/social benefits. Pool equalizes each live agent's income within an instant; different survival times can still produce unequal lifetime income/rates. C discount changes evaluate discount factors in the same varying environment, not a finite real-time horizon or constant regime.",
        "counterfactual_checks": first_gen_checks,
    }
    path = output / f"P1_{part}_{phase.upper()}_SUMMARY.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite summary: {path}")
    json_write_new(path, clean_json(result))
    return path


def replay_cell(path, seed_index=0):
    path = Path(path)
    protocol = json.loads((path / "PROTOCOL.json").read_text())
    config = BridgeConfig(**protocol["config"])
    seed = protocol["seeds"][seed_index]
    engine = make_engine(config, [seed], protocol["fraction"], protocol["part"], protocol["variant"])
    histories, transmissions = [], []
    for generation in range(config.generations):
        outcome = engine.episode(generation)
        outcome["terminal_resource"] = engine.state["resource"].copy() / engine.capacity
        outcome["terminal_survival"] = engine.state["alive"].mean(-1)
        histories.append(outcome)
        transmissions.append(engine.evolve(generation, outcome))
    history = {key: np.stack([h[key] for h in histories], axis=1) for key in histories[0]}
    transmission = {key: np.stack([h[key] for h in transmissions], axis=1) for key in transmissions[0]}
    comparisons = {}
    with np.load(path / "outcomes.npz", allow_pickle=False) as original:
        for key, values in history.items():
            if key == "full_transition_sha256" or (key == "diagnostic_transition_sha256" and seed_index != 0):
                continue  # aggregate trace hash is intentionally batch dependent
            expected = original[key][seed_index:seed_index + 1]
            comparisons["outcome/" + key] = bool(np.array_equal(values, expected, equal_nan=True) if values.dtype.kind in "fc" else np.array_equal(values, expected))
    with np.load(path / "transmissions.npz", allow_pickle=False) as original:
        for key, values in transmission.items():
            comparisons["transmission/" + key] = bool(np.array_equal(values, original[key][seed_index:seed_index + 1], equal_nan=True))
    with np.load(path / "final_snapshot.npz", allow_pickle=False) as original:
        for key, values in engine.state.items():
            comparisons["final_snapshot/" + key] = bool(np.array_equal(values, original[key][seed_index:seed_index + 1], equal_nan=True))
    return {"seed": seed, "cell": path.name, "exact": all(comparisons.values()), "checks": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--part", choices=("A", "C"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--replay-cell")
    parser.add_argument("--replay-seed-index", type=int, default=0)
    args = parser.parse_args()
    if args.replay_cell:
        result = replay_cell(args.replay_cell, args.replay_seed_index)
        print(json.dumps(result))
        if not result["exact"]:
            raise SystemExit(2)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selection = cells(args.part)
    if args.cell_index is not None:
        selection = [selection[args.cell_index]]
    jobs = [(str(output), args.part, args.phase, *cell, asdict(config_default())) for cell in selection]
    if not args.aggregate_only:
        if args.workers == 1:
            for job in jobs:
                print(json.dumps(run_cell(job)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for result in executor.map(run_cell, jobs):
                    print(json.dumps(result), flush=True)
    if args.cell_index is None:
        print(str(aggregate(output, args.part, args.phase)), flush=True)


if __name__ == "__main__":
    main()
