"""R8 population corrections: Q regulation, supported thresholds, causal paths.

The P1 engine is imported unchanged. Only new adapters and new result trees are
used. All result publication uses the shared atomic SHA/CRC-verified IO layer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from collective.bridge_v2.engine import BridgeConfig, BridgeEngine, STATE_KEYS
from p1_completion.population import clean_json, finite_mean, paired_ci, summarize_cell
from r8_completion.io import atomic_json, atomic_npz, sha256, verify_npz, write_manifest


TRAIN_SEEDS = tuple(range(940001, 940009))
VALIDATION_SEEDS = tuple(range(940011, 940015))
POWER_SEEDS = tuple(range(940015, 940021))
THRESHOLD_PILOT_SEEDS = tuple(range(940031, 940036))
FINAL_SEEDS = tuple(range(941001, 941031))
RIDGES = (16., 64., 256.)
CONSTANT_PALETTE = ((3., 0., 0.), (0., 3., 0.), (0., 0., 3.), (1., 1., 1.),
                    (1.5, 1.5, 0.), (1.5, 0., 1.5), (0., 1.5, 1.5))
RULES = ("success", "conformity")
Q_ARMS = ("legacy_full", "shared_successor", "normalized_independent",
          "normalized_common", "global_common", "state_common",
          "shuffled_common", "qv_policy_off")
Q_REGIMES = {
    "id": {},
    "higher_metabolism": {"metabolism": .4},
    "lower_regeneration": {"growth_mean": .12, "growth_amplitude": .05},
}
A_FRACTIONS = (.35, .50, .65)


def _simplex_weights(values, fallback=None):
    values = np.maximum(np.asarray(values, dtype=float), 0.)
    total = values.sum()
    return values * (3. / total) if total > 1e-12 else np.ones(3) if fallback is None else np.asarray(fallback).copy()


def _nonnegative_ridge(x, y, ridge, prior=None):
    prior = np.zeros(3) if prior is None else np.asarray(prior, dtype=float)
    matrix = x.T @ x + float(ridge) * np.eye(3)
    target = x.T @ y + float(ridge) * prior
    weights = np.maximum(prior, 0.).copy()
    for _ in range(500):
        old = weights.copy()
        for j in range(3):
            weights[j] = max(0., (target[j] - matrix[j] @ weights + matrix[j, j] * weights[j]) / matrix[j, j])
        if np.max(np.abs(weights - old)) < 1e-11:
            break
    return weights


def fit_arbitration(qheads, state_index, actions, returns, is_uniform_probe=None, ridge=64.0):
    """Fit from factual randomized-action observations, never outcome oracles.

    qheads[N,3,4] must precede the chosen action. returns[N] contains subsequent
    *observed* outcomes from that action's actual trajectory. state is the same
    observable 20-bin state used by the frozen controller. No test data enters.
    Both global and state models consume exactly these same rows and scales.
    """
    qheads = np.asarray(qheads, dtype=float)
    state_index = np.asarray(state_index, dtype=int)
    actions, returns = np.asarray(actions, dtype=int), np.asarray(returns, dtype=float)
    if qheads.shape != (len(returns), 3, 4) or state_index.shape != returns.shape or actions.shape != returns.shape:
        raise ValueError("Expected Q[N,3,4], state/action/return[N]")
    eligible = np.ones(len(returns), dtype=bool) if is_uniform_probe is None else np.asarray(is_uniform_probe, dtype=bool).copy()
    eligible &= np.isfinite(qheads).all((1, 2)) & np.isfinite(returns)
    eligible &= (state_index >= 0) & (state_index < 20) & (actions >= 0) & (actions < 4)
    qheads, states, actions, targets = qheads[eligible], state_index[eligible], actions[eligible], returns[eligible]
    if len(targets) < 12:
        raise ValueError("At least twelve valid factual uniform-probe observations required")
    advantages = qheads - qheads.mean(-1, keepdims=True)
    scales = np.maximum(np.sqrt(np.mean(advantages ** 2, axis=(0, 2))), 1e-8)
    predictors = np.take_along_axis(advantages / scales[None, :, None], actions[:, None, None], axis=-1)[..., 0]
    centered_x, centered_y = predictors.copy(), targets.copy()
    for state in range(20):
        rows = states == state
        if rows.any():
            centered_x[rows] -= predictors[rows].mean(0)
            centered_y[rows] -= targets[rows].mean()
    global_raw = _nonnegative_ridge(centered_x, centered_y, ridge)
    global_weights = _simplex_weights(global_raw)
    state_weights, counts, actions_by_state = [], [], []
    for state in range(20):
        rows = states == state
        counts.append(int(rows.sum()))
        actions_by_state.append(np.bincount(actions[rows], minlength=4).tolist())
        if rows.sum() < 12 or (np.bincount(actions[rows], minlength=4) == 0).any():
            state_weights.append(global_weights.copy())
        else:
            local = _nonnegative_ridge(centered_x[rows], centered_y[rows], ridge, global_raw)
            state_weights.append(_simplex_weights(local, global_weights))
    state_weights = np.asarray(state_weights)
    permutation = np.random.default_rng([940090, 701]).permutation(20)
    return {
        "schema": "r8-factual-arbitration-v1", "head_order": ["qt", "qv", "qg"],
        "head_scales": scales.tolist(), "global_weights": global_weights.tolist(),
        "state_weights": state_weights.tolist(), "shuffled_state_weights": state_weights[permutation].tolist(),
        "state_permutation": permutation.tolist(), "ridge": float(ridge),
        "factual_probe_rows": len(targets), "state_rows": counts,
        "action_counts_by_state": actions_by_state,
        "weights_sum": 3., "state_definition": "vitality_bin*5+noisy_resource_bin",
        "no_test_fitting": True,
    }


def arbitrate_q(qheads, state_index, calibration, mode="state"):
    """Shared actor API for population and integrated R8 agents."""
    qheads = np.asarray(qheads, dtype=float)
    if qheads.shape[-2:] != (3, 4):
        raise ValueError("Last Q axes must be head(3), action(4)")
    if mode == "raw":
        return qheads[..., 0, :] + qheads[..., 1, :] + qheads[..., 2, :]
    centered = qheads - qheads.mean(-1, keepdims=True)
    normalized = centered / np.asarray(calibration["head_scales"])[..., None]
    if mode == "normalized":
        weights = np.ones(3)
    elif mode == "global":
        weights = np.asarray(calibration["global_weights"])
    elif mode in ("state", "shuffled"):
        key = "state_weights" if mode == "state" else "shuffled_state_weights"
        weights = np.asarray(calibration[key])[np.asarray(state_index, dtype=int)]
    else:
        raise ValueError("Unknown arbitration mode")
    return np.sum(normalized * weights[..., None], axis=-2)


class RegulatedBridge(BridgeEngine):
    def __init__(self, config, seeds, fraction, arm, calibration):
        if arm not in Q_ARMS:
            raise ValueError(arm)
        common = arm in ("shared_successor", "normalized_common", "global_common", "state_common", "shuffled_common")
        variant = "generic_shared_target" if common else "qv_policy_off" if arm == "qv_policy_off" else "full"
        super().__init__(config, seeds, fraction, "C", variant)
        self.arm, self.calibration = arm, calibration

    def observe(self, noise):
        result = super().observe(noise)
        if getattr(self, "capture_public_observations", False):
            # Native episode calls observe(current) then observe(next) once per
            # step. Count only current public observations, avoiding duplicates.
            if self.observation_calls % 2 == 0:
                self.public_resource_sum += np.clip(result[1] / self.capacity, 0., 1.)
                self.public_observation_steps += 1
            self.observation_calls += 1
        return result

    def episode(self, *args, **kwargs):
        self.public_resource_sum = np.zeros((self.b, self.config.groups))
        self.public_observation_steps = self.observation_calls = 0
        self.capture_public_observations = True
        try:
            outcome = super().episode(*args, **kwargs)
        finally:
            self.capture_public_observations = False
        outcome["public_resource_fraction"] = self.public_resource_sum.mean(-1) / self.public_observation_steps
        outcome["group_public_resource"] = self.public_resource_sum / self.public_observation_steps
        return outcome

    def decision_scores(self, state_index):
        if self.arm in ("legacy_full", "shared_successor", "qv_policy_off"):
            return super().decision_scores(state_index)
        mode = {"normalized_independent": "normalized", "normalized_common": "normalized",
                "global_common": "global", "state_common": "state", "shuffled_common": "shuffled"}[self.arm]
        return arbitrate_q(np.stack(self.head_values(state_index), axis=-2), state_index, self.calibration, mode)


class FactualCalibrationBridge(BridgeEngine):
    """Observe actual uniform exploration choices and subsequent public feedback."""
    def __init__(self, config, seeds):
        super().__init__(config, seeds, .5, "C", "blind_exploration")
        self.capture, self.records, self.active_tape = False, [], None

    def decision_scores(self, state_index):
        scores = super().decision_scores(state_index)
        if self.capture:
            step = len(self.records)
            tape = self.active_tape
            greedy = (scores / self.config.temperature + tape["gumbel"][:, step]).argmax(-1)
            action = np.where(tape["explore"][:, step], tape["random_action"][:, step], greedy)
            action = np.where(self.state["alive"], action, 0)
            self.records.append((np.stack(self.head_values(state_index), axis=-2), state_index.copy(), action,
                                 (tape["explore"][:, step] & self.state["alive"]).copy(), self.state["alive"].copy(),
                                 np.clip((self.state["resource"] + tape["obs_noise"][:, step]) / self.capacity, 0., 1.)))
        return scores

    def collect_generation(self, generation, horizon=40):
        self.records, self.active_tape = [], self.random_tape(generation)
        self.capture = True
        try:
            outcome = self.episode(generation, tape=self.active_tape)
        finally:
            self.capture = False
        q, state, action, probe, alive, resource = [np.stack([r[j] for r in self.records], axis=1) for j in range(6)]
        usable = self.config.steps - horizon
        alive_cumsum = np.concatenate((np.zeros_like(alive[:, :1], dtype=float), np.cumsum(alive, axis=1)), axis=1)
        resource_cumsum = np.concatenate((np.zeros_like(resource[:, :1]), np.cumsum(resource, axis=1)), axis=1)
        own_future = (alive_cumsum[:, horizon + 1:] - alive_cumsum[:, 1:usable + 1]) / horizon
        observed_future = (resource_cumsum[:, horizon + 1:] - resource_cumsum[:, 1:usable + 1]) / horizon
        target = .5 * own_future + .5 * observed_future[..., None]
        select = probe[:, :usable]
        data = {"qheads": q[:, :usable][select], "state_index": state[:, :usable][select],
                "actions": action[:, :usable][select], "returns": target[select],
                "generation": np.full(int(select.sum()), generation, dtype=np.int16)}
        self.evolve(generation, outcome)
        self.records = []
        return data, outcome


class FactorialBridge(BridgeEngine):
    """Score and physical pooling compose independently; donor maps are explicit."""
    def __init__(self, config, seeds, fraction, score_equal=False, pool=False):
        super().__init__(config, seeds, fraction, "A", "resource_pool" if pool else "base")
        self.score_equal, self.pool = bool(score_equal), bool(pool)

    def evolve_with_maps(self, generation, outcome, fixed_maps=None):
        c, s = self.config, self.state
        fitness = outcome["alive_agent_time"].copy()
        adjusted = fitness.copy()
        if self.score_equal:
            for value in (0, 1):
                mask = s["lineage"] == value
                count = mask.sum(-1, keepdims=True)
                mean = np.divide((fitness * mask).sum(-1, keepdims=True), count,
                                 out=np.zeros_like(count, dtype=float), where=count > 0)
                adjusted += np.where(mask & (count > 0), fitness.mean(-1, keepdims=True) - mean, 0.)
        records = {"raw_fitness": fitness, "copy_score": adjusted,
                   "lineage_before": s["lineage"].copy()}
        maps = {"social_low": np.full((self.b, c.groups), -1, dtype=np.int16),
                "social_high": np.full((self.b, c.groups), -1, dtype=np.int16),
                "group_low": np.full((self.b, c.replace_groups), -1, dtype=np.int16),
                "group_high": np.full((self.b, c.replace_groups), -1, dtype=np.int16),
                "mutation": np.zeros((self.b, c.groups, c.agents), dtype=bool)}
        for b, seed in enumerate(self.seeds):
            social = np.random.default_rng([int(seed), generation, 103]).random((c.groups, c.agents + 3))
            group_rng = np.random.default_rng([int(seed), generation, 104])
            mutation = np.random.default_rng([int(seed), generation, 105]).random((c.groups, c.agents)) < c.mutation
            if generation >= c.warmup:
                for group in range(c.groups):
                    if fixed_maps is not None:
                        low, high = int(fixed_maps["social_low"][b, group]), int(fixed_maps["social_high"][b, group])
                        should_copy = low >= 0
                    elif c.transmission == "conformity":
                        categories = s["lineage"][b, group].astype(bool)
                        majority = bool(categories.mean() > .5) if categories.mean() != .5 else bool(social[group, -1] < .5)
                        candidates = np.flatnonzero(categories == majority)
                        low = int(social[group, -2] * c.agents)
                        high = int(candidates[int(social[group, -3] * len(candidates))])
                        should_copy = categories[low] != majority
                    else:
                        score = adjusted[b, group]
                        jitter = (social[group, :c.agents] - .5) * 2e-12
                        low, high = int(np.argmin(score + jitter)), int(np.argmax(score + jitter))
                        should_copy = score[high] > score[low] + 1e-12
                    if should_copy:
                        maps["social_low"][b, group], maps["social_high"][b, group] = low, high
                        for key in ("qt", "qv", "qg", "visits", "lineage"):
                            s[key][b, group, low] = s[key][b, group, high]
                if fixed_maps is None:
                    order = np.argsort(outcome["group_alive"][b] + group_rng.uniform(-1e-12, 1e-12, c.groups))
                    lows, highs = order[:c.replace_groups], order[-c.replace_groups:]
                else:
                    lows, highs = fixed_maps["group_low"][b], fixed_maps["group_high"][b]
                for index, (low, high) in enumerate(zip(lows, highs)):
                    if low < 0:
                        continue
                    maps["group_low"][b, index], maps["group_high"][b, index] = low, high
                    for key in ("qt", "qv", "qg", "visits", "lineage"):
                        s[key][b, low] = s[key][b, high].copy()
            maps["mutation"][b] = mutation if fixed_maps is None else fixed_maps["mutation"][b]
            s["lineage"][b] ^= maps["mutation"][b].astype(np.uint8)
        records["lineage_after"] = s["lineage"].copy()
        return records, maps


def _history_run(engine, fixed_maps=None):
    histories, transmissions, maps = [], [], []
    for generation in range(engine.config.generations):
        outcome = engine.episode(generation)
        outcome["terminal_resource"] = engine.state["resource"].copy() / engine.capacity
        outcome["terminal_survival"] = engine.state["alive"].mean(-1)
        histories.append(outcome)
        if isinstance(engine, FactorialBridge):
            transmission, mapping = engine.evolve_with_maps(generation, outcome, None if fixed_maps is None else {k: v[:, generation] for k, v in fixed_maps.items()})
            maps.append(mapping)
        else:
            transmission = engine.evolve(generation, outcome)
        transmissions.append(transmission)
    history = {k: np.stack([v[k] for v in histories], axis=1) for k in histories[0]}
    transmission = {k: np.stack([v[k] for v in transmissions], axis=1) for k in transmissions[0]}
    mapped = {k: np.stack([v[k] for v in maps], axis=1) for k in maps[0]} if maps else {}
    endpoints, metrics = summarize_cell(history, transmission, engine.config, "A" if isinstance(engine, FactorialBridge) else "C")
    if "public_resource_fraction" in history:
        endpoints["observed_resource_time_fraction"] = history["public_resource_fraction"][:, -engine.config.tail_generations:].mean(1)
    return history, transmission, mapped, endpoints, metrics


def _source_hashes():
    from collective.bridge_v2 import engine
    return {"population.py": sha256(__file__), "inherited_engine.py": sha256(engine.__file__)}


def _verify_final_gate(calibration=None, selection=None):
    from r8_completion.provenance import verify, FREEZE, ROOT
    status = verify()
    if tuple(status["final_seeds"]["population"]) != FINAL_SEEDS:
        raise ValueError("Final seed list differs from prospective freeze")
    frozen = json.loads(FREEZE.read_text())
    for value, suffix in ((calibration, "CALIBRATION_FROZEN.json"), (selection, "THRESHOLD_SELECTION.json")):
        if value is None:
            continue
        matches = [ROOT / name for name in frozen["source_sha256"] if name.endswith(suffix)]
        if len(matches) != 1 or json.loads(matches[0].read_text()) != value:
            raise ValueError(f"Final input differs from the frozen development artifact: {suffix}")
    return status


def _save_run(target, protocol, engine, fixed_maps=None):
    target = Path(target)
    protocol = dict(protocol, config=asdict(engine.config), seeds=engine.seeds.tolist(), source_sha256=_source_hashes())
    if target.exists():
        complete = target / "COMPLETE.json"
        if not complete.exists():
            raise FileExistsError(f"Partial immutable run exists; preserve and investigate: {target}")
        manifest = json.loads(complete.read_text())
        if manifest["metadata"]["protocol"] != clean_json(protocol):
            raise ValueError(f"Completed protocol differs from requested run: {target}")
        verify_run(target)
        endpoints = _read_endpoints(target)
        maps = {}
        if (target / "donor_maps.npz").exists():
            with np.load(target / "donor_maps.npz", allow_pickle=False) as data:
                maps = {key: data[key].copy() for key in data.files}
        return endpoints, maps
    target.mkdir(parents=True)
    files = [atomic_json(target / "PROTOCOL.json", protocol)]
    start = time.perf_counter()
    history, transmission, maps, endpoints, metrics = _history_run(engine, fixed_maps)
    files += [atomic_npz(target / "outcomes.npz", history), atomic_npz(target / "transmissions.npz", transmission),
              atomic_npz(target / "endpoints.npz", endpoints), atomic_npz(target / "final_state.npz", engine.state)]
    if maps:
        files.append(atomic_npz(target / "donor_maps.npz", maps))
    files.append(atomic_npz(target / "conditional_metrics.npz", metrics))
    write_manifest(target / "COMPLETE.json", files, {"elapsed_seconds": time.perf_counter() - start, "protocol": protocol})
    return endpoints, maps


def verify_run(target):
    target = Path(target)
    manifest = json.loads((target / "COMPLETE.json").read_text())
    for record in manifest["files"]:
        path = target / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Artifact checksum changed: {path}")
        if path.suffix == ".npz":
            verify_npz(path)
    return {"cell": str(target), "artifacts_verified": len(manifest["files"])}


def replay_run(target, seed_index=0):
    target = Path(target)
    verify_run(target)
    protocol = json.loads((target / "PROTOCOL.json").read_text())
    seed = protocol["seeds"][seed_index]
    if seed in FINAL_SEEDS:
        _verify_final_gate()
    config = BridgeConfig(**protocol["config"])
    fixed = None
    if protocol["study"] == "Q":
        engine = RegulatedBridge(config, [seed], protocol["initial_fraction"], protocol["arm"], protocol["calibration"])
    else:
        engine = FactorialBridge(config, [seed], protocol["initial_fraction"], protocol.get("score_equal", False), protocol.get("pool", False))
        if protocol.get("transmission_mode") == "fixed_donors":
            reference = target.parent / protocol["fixed_donor_reference"] / "donor_maps.npz"
            with np.load(reference, allow_pickle=False) as data:
                fixed = {key: data[key][seed_index:seed_index + 1].copy() for key in data.files}
    history, transmissions, maps, endpoints, metrics = _history_run(engine, fixed)
    checks = {}
    for filename, arrays in (("outcomes.npz", history), ("transmissions.npz", transmissions), ("final_state.npz", engine.state),
                             ("endpoints.npz", endpoints), ("conditional_metrics.npz", metrics), ("donor_maps.npz", maps)):
        if not arrays:
            continue
        with np.load(target / filename, allow_pickle=False) as original:
            for key, values in arrays.items():
                if key == "full_transition_sha256" or (key == "diagnostic_transition_sha256" and seed_index != 0):
                    continue
                if key not in original:
                    raise ValueError(f"Replay contains an unrecorded field: {filename}/{key}")
                expected = original[key][seed_index:seed_index + 1]
                checks[f"{filename}/{key}"] = bool(np.array_equal(values, expected, equal_nan=True) if values.dtype.kind in "fc" else np.array_equal(values, expected))
    return {"seed": seed, "cell": str(target), "exact": all(checks.values()), "checks": checks}


def collect_training(output):
    output = Path(output)
    target = output / "calibration_training"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    engine = FactualCalibrationBridge(BridgeConfig(), TRAIN_SEEDS)
    datasets, outcomes = [], []
    for generation in range(engine.config.generations):
        dataset, outcome = engine.collect_generation(generation, horizon=40)
        datasets.append(dataset)
        outcomes.append(outcome)
    data = {k: np.concatenate([r[k] for r in datasets]) for k in datasets[0]}
    files = [atomic_npz(target / "factual_uniform_probes.npz", data),
             atomic_npz(target / "training_outcomes.npz", {k: np.stack([r[k] for r in outcomes], axis=1) for k in outcomes[0]})]
    models = {}
    for ridge in RIDGES:
        model = fit_arbitration(data["qheads"], data["state_index"], data["actions"], data["returns"], ridge=ridge)
        models[str(int(ridge))] = model
        files.append(atomic_json(target / f"calibration_ridge_{int(ridge)}.json", model))
    files.append(atomic_json(target / "PROTOCOL.json", {
        "seeds": TRAIN_SEEDS, "ridge_candidates": RIDGES,
        "label": "half own alive plus half publicly observed noisy resource over subsequent40actual steps; only uniform exploration actions",
        "steps_and_generation_budget": asdict(engine.config), "source": _source_hashes(),
    }))
    write_manifest(target / "COMPLETE.json", files, {"factual_probe_rows": len(data["returns"])})
    return models


def _q_job(job):
    target, seeds, rule, regime, arm, calibration = job
    config = replace(BridgeConfig(), transmission=rule, **Q_REGIMES[regime])
    engine = RegulatedBridge(config, seeds, .5, arm, calibration)
    endpoints, _ = _save_run(target, {"study": "Q", "arm": arm, "rule": rule, "regime": regime, "initial_fraction": .5,
                                    "calibration": calibration}, engine)
    return {"target": str(target), "endpoints": {k: v.tolist() for k, v in endpoints.items()}}


def _map_jobs(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, jobs))


def calibrate_development(output, workers=4, training_source=None):
    output = Path(output)
    if training_source is None:
        ridge_models = collect_training(output)
        training_path = output / "calibration_training/factual_uniform_probes.npz"
    else:
        training_path = Path(training_source) / "calibration_training/factual_uniform_probes.npz"
        verify_npz(training_path)
        with np.load(training_path, allow_pickle=False) as data:
            ridge_models = {str(int(ridge)): fit_arbitration(data["qheads"], data["state_index"], data["actions"], data["returns"], ridge=ridge) for ridge in RIDGES}
        atomic_json(output / "FACTUAL_DATA_REUSE.json", {"path": str(training_path), "sha256": sha256(training_path), "same_training_rows": True,
                    "no_new_training_exposure": True, "training_seeds": TRAIN_SEEDS})
    models = {f"ridge{key}": dict(value, candidate_type="factual_ridge", candidate_name=f"ridge{key}") for key, value in ridge_models.items()}
    for index, weights in enumerate(CONSTANT_PALETTE):
        name = f"constant{index}"
        candidate = dict(ridge_models[str(int(RIDGES[0]))])
        candidate.update(candidate_type="constant_palette", candidate_name=name, ridge=None,
                         global_weights=list(weights), state_weights=[list(weights)] * 20,
                         shuffled_state_weights=[list(weights)] * 20)
        models[name] = candidate
    atomic_json(output / "CANDIDATE_PALETTE.json", {"models": models, "same_ten_candidates_per_family": True,
                "tie_rule": "objective rounded12decimals, constants before ridge, then candidate name",
                "validation_metric": "half alive_agent_time plus half current publicly observed clipped resource; no latent stock for selection"})
    jobs = []
    for model_name, calibration in models.items():
        for arm in ("global_common", "state_common"):
            for rule in RULES:
                target = output / "calibration_validation" / f"{model_name}__{arm}__{rule}"
                jobs.append((target, VALIDATION_SEEDS, rule, "id", arm, calibration))
    results = _map_jobs(_q_job, jobs, workers)
    validation, true_resource_audit = {}, {}
    for model_name in models:
        for arm in ("global_common", "state_common"):
            values = []
            for result in results:
                if Path(result["target"]).name.startswith(f"{model_name}__{arm}__"):
                    ep = result["endpoints"]
                    values.extend((.5 * np.asarray(ep["survival_time_fraction"]) + .5 * np.asarray(ep["observed_resource_time_fraction"])).tolist())
                    true_resource_audit.setdefault(f"{arm}/{model_name}", []).extend(ep["resource_time_fraction"])
            validation[f"{arm}/{model_name}"] = float(np.mean(values))
    selected = {}
    for arm in ("global_common", "state_common"):
        selected[arm] = min(models, key=lambda key: (-round(validation[f"{arm}/{key}"], 12),
                                                   0 if models[key]["candidate_type"] == "constant_palette" else 1, key))
    calibration = dict(models[selected["state_common"]])
    calibration["global_weights"] = models[selected["global_common"]]["global_weights"]
    calibration["selected_ridge_global"] = models[selected["global_common"]]["ridge"]
    calibration["selected_ridge_state"] = models[selected["state_common"]]["ridge"]
    calibration["selected_candidate_global"] = selected["global_common"]
    calibration["selected_candidate_state"] = selected["state_common"]
    calibration["state_is_constant_fallback"] = models[selected["state_common"]]["candidate_type"] == "constant_palette"
    calibration["training_seeds"] = list(TRAIN_SEEDS)
    calibration["validation_seeds"] = list(VALIDATION_SEEDS)
    calibration["validation_scores"] = validation
    calibration["validation_true_resource_audit_only"] = {k: float(np.mean(v)) for k, v in true_resource_audit.items()}
    calibration["selection_objective"] = "mean .5survival_time+.5PUBLIC_OBSERVED_resource over both transmission rules, ID validation only"
    calibration["training_artifact_sha256"] = sha256(training_path)
    calibration["candidate_budget_per_family"] = 10
    atomic_json(output / "CALIBRATION_FROZEN.json", calibration)
    return calibration


def q_campaign(output, phase, calibration, workers=4):
    if phase == "final":
        _verify_final_gate(calibration=calibration)
    seeds = POWER_SEEDS if phase == "pilot" else FINAL_SEEDS
    regimes = ("id",) if phase == "pilot" else tuple(Q_REGIMES)
    jobs = []
    for rule in RULES:
        for regime in regimes:
            for arm in Q_ARMS:
                target = Path(output) / f"{rule}__{regime}__{arm}"
                jobs.append((target, seeds, rule, regime, arm, calibration))
    _map_jobs(_q_job, jobs, workers)
    return summarize_q(output, phase)


def _read_endpoints(path):
    with np.load(Path(path) / "endpoints.npz", allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def summarize_q(output, phase):
    output = Path(output)
    regimes = ("id",) if phase == "pilot" else tuple(Q_REGIMES)
    data, descriptives = {}, {}
    for rule in RULES:
        for arm in Q_ARMS:
            endpoints = [_read_endpoints(output / f"{rule}__{regime}__{arm}") for regime in regimes]
            data[(rule, arm)] = {k: np.mean(np.stack([ep[k] for ep in endpoints]), axis=0) for k in ("survival_time_fraction", "resource_time_fraction")}
            descriptives[f"{rule}/{arm}"] = {"regime_means": {regime: {k: float(np.mean(ep[k])) for k in ("survival_time_fraction", "resource_time_fraction")} for regime, ep in zip(regimes, endpoints)},
                                                   "across_regime_means": {k: float(v.mean()) for k, v in data[(rule, arm)].items()}}
    contrasts = {}
    factorial = {}
    for rule in RULES:
        for arm in Q_ARMS:
            if arm == "legacy_full":
                continue
            contrasts[f"{rule}/{arm}-minus-legacy_full"] = {k: paired_ci(data[(rule, arm)][k] - data[(rule, "legacy_full")][k]) for k in data[(rule, arm)]}
        contrasts[f"{rule}/state-minus-global"] = {k: paired_ci(data[(rule, "state_common")][k] - data[(rule, "global_common")][k]) for k in data[(rule, "state_common")]}
        contrasts[f"{rule}/state-minus-shuffled"] = {k: paired_ci(data[(rule, "state_common")][k] - data[(rule, "shuffled_common")][k]) for k in data[(rule, "state_common")]}
        for metric in data[(rule, "legacy_full")]:
            raw_independent = data[(rule, "legacy_full")][metric]
            raw_common = data[(rule, "shared_successor")][metric]
            normalized_independent = data[(rule, "normalized_independent")][metric]
            normalized_common = data[(rule, "normalized_common")][metric]
            effects = {
                "normalization_given_independent": normalized_independent - raw_independent,
                "normalization_given_common": normalized_common - raw_common,
                "common_successor_given_raw": raw_common - raw_independent,
                "common_successor_given_normalized": normalized_common - normalized_independent,
                "normalization_x_successor_interaction": normalized_common - normalized_independent - raw_common + raw_independent,
            }
            for effect, values in effects.items():
                factorial[f"{rule}/{effect}/{metric}"] = paired_ci(values)
    q1, q2, components = [], [], {}
    for rule in RULES:
        d1 = data[(rule, "normalized_common")]["survival_time_fraction"] - data[(rule, "legacy_full")]["survival_time_fraction"]
        ds = data[(rule, "state_common")]["survival_time_fraction"] - data[(rule, "global_common")]["survival_time_fraction"]
        dr = data[(rule, "state_common")]["resource_time_fraction"] - data[(rule, "global_common")]["resource_time_fraction"]
        q1.append(d1 >= .01)
        q2.append((dr >= .01) & (ds >= -.005))
        components[rule] = {"Q1_delta_alive": d1, "Q2_delta_alive": ds, "Q2_delta_resource": dr}
    wins = {"Q1": np.logical_and.reduce(q1), "Q2": np.logical_and.reduce(q2)}
    results = {"phase": phase, "seeds": POWER_SEEDS if phase == "pilot" else FINAL_SEEDS,
               "regimes": regimes, "descriptives": descriptives, "paired_descriptive_ci": contrasts,
               "factorial_secondary_descriptive_ci": factorial,
               "primary_components_by_seed": components,
               "primary_wins_by_seed": wins, "primary_win_counts": {k: int(v.sum()) for k, v in wins.items()},
               "primary_scope": "Per seed conjunction of both transmission rules; each rule's metric averages the three prespecified regimes. Pilot ID-only is development, not the final primary contrast.",
               "primary_margins": {"Q1_alive": .01, "Q2_resource": .01, "Q2_alive_noninferiority": -.005},
               "inference": "Root combines six preregistered tests with exact binomial tails and Holm; mean/percentile intervals are descriptive."}
    atomic_json(output / f"Q_{phase.upper()}_SUMMARY.json", clean_json(results))
    return results


def _threshold_job(job):
    target, seeds, rule, pressure, fraction = job
    config = replace(BridgeConfig(), transmission=rule, metabolism=pressure)
    engine = FactorialBridge(config, seeds, fraction)
    ep, _ = _save_run(target, {"study": "threshold", "rule": rule, "metabolism": pressure, "initial_fraction": fraction}, engine)
    return {"target": str(target), "metabolism": pressure, "rule": rule, "fraction": fraction, "viable": ep["viable"].tolist()}


def threshold_campaign(output, phase, selection=None, workers=4):
    from r8_completion.population_thresholds import GRID_DEV, choose_pressure_and_grid, estimate_threshold
    output = Path(output)
    if phase == "final":
        _verify_final_gate(selection=selection)
    seeds = THRESHOLD_PILOT_SEEDS if phase == "pilot" else FINAL_SEEDS
    pressures = (.3, .45, .6) if phase == "pilot" else (selection["pressure"],)
    grid = tuple(GRID_DEV) if phase == "pilot" else tuple(selection["grid"])
    jobs = [(output / f"m{round(pressure*100):02d}__{rule}__p{round(fraction*100):02d}", seeds, rule, pressure, fraction)
            for pressure in pressures for rule in RULES for fraction in grid]
    results = _map_jobs(_threshold_job, jobs, workers)
    records = []
    for pressure in pressures:
        for rule in RULES:
            ordered = sorted([row for row in results if row["metabolism"] == pressure and row["rule"] == rule], key=lambda row: row["fraction"])
            matrix = np.asarray([row["viable"] for row in ordered]).T
            records.append({"metabolism": pressure, "rule": rule, "seeds": list(seeds), "grid": list(grid), "viable": matrix.tolist(),
                            "threshold": estimate_threshold(grid, matrix)})
    result = {"phase": phase, "records": records}
    if phase == "pilot":
        result["selection"] = choose_pressure_and_grid(records)
        atomic_json(output / "THRESHOLD_SELECTION.json", result["selection"])
    else:
        result["frozen_pilot_selection"] = selection
    atomic_json(output / f"THRESHOLD_{phase.upper()}_SUMMARY.json", clean_json(result))
    return result


def _a_block(job):
    output, seeds, fraction, rule = job
    output = Path(output)
    config = replace(BridgeConfig(), transmission=rule)
    # Donor selection maps are obtained prospectively from the matched untreated
    # reference, never from the treated arm or final outcome-based selection.
    baseline = FactorialBridge(config, seeds, fraction)
    baseline_target = output / f"{rule}__p{round(fraction*100):02d}__endogenous__s0p0"
    _, reference_maps = _save_run(baseline_target, {"study": "causal_A", "rule": rule, "initial_fraction": fraction,
                                                 "score_equal": False, "pool": False, "transmission_mode": "endogenous"}, baseline)
    variants = [("endogenous", score, pool) for score in (False, True) for pool in (False, True) if score or pool]
    if rule == "success":
        variants += [("fixed_donors", score, pool) for score in (False, True) for pool in (False, True)]
    else:
        variants = [("endogenous", True, False)]
    for mode, score, pool in variants:
        engine = FactorialBridge(config, seeds, fraction, score, pool)
        target = output / f"{rule}__p{round(fraction*100):02d}__{mode}__s{int(score)}p{int(pool)}"
        _save_run(target, {"study": "causal_A", "rule": rule, "initial_fraction": fraction, "score_equal": score,
                           "pool": pool, "transmission_mode": mode, "fixed_donor_reference": baseline_target.name if mode == "fixed_donors" else None},
                  engine, reference_maps if mode == "fixed_donors" else None)
    return {"rule": rule, "fraction": fraction}


def causal_a_campaign(output, phase, workers=4):
    output = Path(output)
    if phase == "final":
        _verify_final_gate()
    seeds = POWER_SEEDS[:3] if phase == "pilot" else FINAL_SEEDS
    _map_jobs(_a_block, [(output, seeds, p, rule) for p in A_FRACTIONS for rule in RULES], workers)
    contrasts, controls, by_fraction = {}, {}, {}
    metrics = ("survival_time_fraction", "resource_time_fraction", "raw_survival_gap", "copy_score_gap", "real_income_rate_gap")
    accumulated = {}
    for p in A_FRACTIONS:
        data = {}
        for mode in ("endogenous", "fixed_donors"):
            for score in (0, 1):
                for pool in (0, 1):
                    key = f"success__p{round(p*100):02d}__{mode}__s{score}p{pool}"
                    data[(mode, score, pool)] = _read_endpoints(output / key)
            for metric in metrics:
                effects = {
                    "score_at_pool0": data[(mode, 1, 0)][metric] - data[(mode, 0, 0)][metric],
                    "score_at_pool1": data[(mode, 1, 1)][metric] - data[(mode, 0, 1)][metric],
                    "pool_at_score0": data[(mode, 0, 1)][metric] - data[(mode, 0, 0)][metric],
                    "pool_at_score1": data[(mode, 1, 1)][metric] - data[(mode, 1, 0)][metric],
                    "interaction": data[(mode, 1, 1)][metric] - data[(mode, 1, 0)][metric] - data[(mode, 0, 1)][metric] + data[(mode, 0, 0)][metric],
                }
                for effect, difference in effects.items():
                    name = f"{mode}/{effect}/{metric}"
                    accumulated.setdefault(name, []).append(difference)
                    by_fraction[f"p{p:.2f}/{name}"] = paired_ci(difference)
        # Comparisons of whole physical trajectories are stronger than endpoints.
        for mode, pool in (("fixed_donors", 0), ("fixed_donors", 1)):
            left = output / f"success__p{round(p*100):02d}__{mode}__s0p{pool}"
            right = output / f"success__p{round(p*100):02d}__{mode}__s1p{pool}"
            with np.load(left / "outcomes.npz") as a, np.load(right / "outcomes.npz") as b:
                controls[f"fixed_donors/p{p:.2f}/pool{pool}"] = bool(np.array_equal(a["final_state_sha256"], b["final_state_sha256"]))
        left = output / f"conformity__p{round(p*100):02d}__endogenous__s0p0"
        right = output / f"conformity__p{round(p*100):02d}__endogenous__s1p0"
        with np.load(left / "outcomes.npz") as a, np.load(right / "outcomes.npz") as b:
            controls[f"conformity/p{p:.2f}/score_negative"] = bool(np.array_equal(a["final_state_sha256"], b["final_state_sha256"]))
    for key, differences in accumulated.items():
        contrasts[key] = paired_ci(finite_mean(np.stack(differences, axis=1), axis=1))
    results = {"phase": phase, "seeds": seeds, "fractions": A_FRACTIONS,
               "contrasts_averaged_over_composition": contrasts, "by_fraction": by_fraction, "negative_controls_exact": controls,
               "interpretation": "Endogenous 2x2 estimates total effects with operational transmission. Fixed-donor maps control social selection decisions; pool effects there are controlled effects relative to that intervention, not natural direct/indirect mediation. Score effects under fixed maps should be exactly zero, and cannot identify an operative score channel.",
               "inference": "Paired seed bootstrap descriptive intervals, no additional confirmatory primary test."}
    atomic_json(output / f"CAUSAL_A_{phase.upper()}_SUMMARY.json", clean_json(results))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("calibration", "Q", "threshold", "A"), required=True)
    parser.add_argument("--phase", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--selection")
    parser.add_argument("--training-source")
    parser.add_argument("--replay-run")
    parser.add_argument("--replay-seed-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "final":
        _verify_final_gate()
    if args.replay_run:
        result = replay_run(args.replay_run, args.replay_seed_index)
        print(json.dumps(result))
        if not result["exact"]:
            raise SystemExit(2)
        return
    if args.part == "calibration":
        if args.phase != "pilot":
            raise ValueError("Calibration is development-only")
        result = calibrate_development(args.output, args.workers, args.training_source)
    elif args.part == "Q":
        if not args.calibration:
            raise ValueError("Q needs frozen development calibration")
        result = q_campaign(args.output, args.phase, json.loads(Path(args.calibration).read_text()), args.workers)
    elif args.part == "threshold":
        selection = json.loads(Path(args.selection).read_text()) if args.selection else None
        if args.phase == "final" and selection is None:
            raise ValueError("Final threshold requires frozen pilot selection")
        result = threshold_campaign(args.output, args.phase, selection, args.workers)
    else:
        result = causal_a_campaign(args.output, args.phase, args.workers)
    print(json.dumps({"part": args.part, "phase": args.phase, "output": str(args.output), "completed": True}), flush=True)


if __name__ == "__main__":
    main()
