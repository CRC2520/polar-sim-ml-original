"""P1.E: learned ecological consequences and trial-level temporal credit.

New, bounded experiment; the preserved BridgeEngine is never modified. The
learner is an ordinary fitted consequence model. Resource desirability is a
designer-selected objective, not discovered valence or evidence of consciousness.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collective"))
from bridge_v2.engine import BridgeConfig, BridgeEngine

PROTOCOL = {
    "version": "P1.E.1",
    "train_anchors": 128,
    "trials_per_anchor": 32,
    "test_anchors": 128,
    "horizon": 12,
    "ecological_weight": 3.0,
    "ridge": 0.01,
    "base_learning_episodes": 10,
    "native_episodes_per_domain": 2,
    "native_steps": 300,
    "pilot_seeds": [930101, 930102, 930103],
    "final_seeds": list(range(933001, 933031)),
    "primary_mechanism": "paired causal-cost MSE: aligned versus zero-cost/action-free",
    "primary_credit": "paired causal-cost MSE: aligned versus shifted trial credit",
    "primary_policy": "alive-agent-time: aligned versus no ecological head",
    "secondary_policy": "mean resource fraction and half-survival plus half-resource",
    "scope": "controlled simulator resets, frozen continuation; objective is designed",
}
ARMS = ("aligned", "no_head", "shifted_credit", "generic", "immediate")
POLYNOMIAL_POWERS = tuple(p for p in itertools.product(range(4), repeat=3) if sum(p) <= 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_arrays(data):
    h = hashlib.sha256()
    for key in sorted(data):
        value = np.ascontiguousarray(data[key])
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def config_for(domain="id"):
    c = BridgeConfig(agents=8, groups=2, generations=10, warmup=4, steps=300,
                     tail_generations=3, restraint_cost=0., initial_restraint_prior=0.,
                     replace_groups=0, mutation=0.)
    if domain == "ood":
        c = replace(c, growth_mean=.10, growth_amplitude=.04, initial_resource_fraction=.35)
    elif domain != "id":
        raise ValueError(domain)
    c.validate()
    return c


def features(resource_observed_fraction, vitality_fraction, action, kind="ecological"):
    """Only public current resource, own vitality and own candidate action."""
    r, v, a = np.broadcast_arrays(resource_observed_fraction, vitality_fraction, action)
    r, v = np.asarray(r, float), np.asarray(v, float)
    if kind == "ecological":
        basis = np.stack((np.ones_like(r), r, r*r, v, r*v), axis=-1)
        action_onehot = np.eye(4)[np.asarray(a, int)]
        return (action_onehot[..., :, None] * basis[..., None, :]).reshape(r.shape + (20,))
    if kind == "generic":
        a = np.asarray(a, float) / 3.
        return np.stack([r**i * v**j * a**k for i, j, k in POLYNOMIAL_POWERS], axis=-1)
    if kind == "no_action":
        return np.stack((np.ones_like(r), r, r*r, v, r*v), axis=-1)
    raise ValueError(kind)


class ConsequenceModel:
    def __init__(self, kind="ecological", coefficients=None):
        self.kind = kind
        self.coefficients = coefficients

    def fit(self, r, v, a, target):
        x = features(r, v, a, self.kind)
        self.coefficients = np.linalg.solve(x.T @ x + PROTOCOL["ridge"] * np.eye(x.shape[1]), x.T @ target)
        return self

    def predict(self, r, v, a):
        return features(r, v, a, self.kind) @ self.coefficients

    def costs(self, r, v):
        r, v = np.broadcast_arrays(r, v)
        predictions = self.predict(r[..., None], v[..., None], np.arange(4))
        return predictions[..., :1] - predictions


def step_physics(resource, vitality, alive, history, action, growth, c):
    """Vectorized forced-action physical update copied algebraically from frozen v2.

    Used only for controlled acquisition/calibration. A parity test compares
    this update to the original engine; native-policy trials use original code.
    """
    capacity = c.capacity_per_agent * c.agents
    action = np.where(alive, action, 0)
    take = action * resource[:, None] / capacity
    demand = take.sum(-1)
    take *= np.minimum(1., np.divide(resource, demand, out=np.ones_like(demand), where=demand > 0))[:, None]
    after_harvest = resource - take.sum(-1)
    lag = history[:, 0] if c.regeneration_delay else after_harvest
    next_resource = np.maximum(after_harvest + growth * lag * (1 - lag / capacity), 0.)
    next_history = np.concatenate((history[:, 1:], resource[:, None]), axis=-1)
    delta_v = np.where(alive, take - c.metabolism, 0.)
    next_vitality = np.clip(vitality + delta_v, -1., c.max_vitality)
    next_alive = alive & (next_vitality > 0)
    return next_resource, next_vitality, next_alive, next_history, take, action


def acquisition(seed, domain="id", evaluation=False):
    """One factual action per reset trial; counterfactual grids only in evaluation.

    Repeated trials share the exact initial anchor/observation, but use separate
    exogenous future tapes. Trial-credit shifting changes labels after physical
    acquisition, so no environmental parameter, transition or X/Y multiset moves.
    """
    c = config_for(domain)
    split = 301 if not evaluation else (401 if domain == "id" else 501)
    rng = np.random.default_rng([int(seed), split])
    anchors = PROTOCOL["test_anchors"] if evaluation else PROTOCOL["train_anchors"]
    repetitions = 4 if evaluation else PROTOCOL["trials_per_anchor"]
    n = anchors * repetitions
    capacity, horizon = c.capacity_per_agent * c.agents, PROTOCOL["horizon"]
    lo, hi = (.25, .90) if domain == "id" else (.10, .55)
    initial_resource = rng.uniform(lo, hi, anchors) * capacity
    initial_vitality = rng.uniform(4., 8., (anchors, c.agents))
    initial_observed_resource = initial_resource + rng.normal(0., c.observation_sd, anchors)
    offsets = rng.integers(0, c.growth_period, anchors)
    if evaluation:
        action0 = np.tile(np.arange(4), anchors)
        # Paired counterfactuals share every environmental random draw.
        noise = np.repeat(rng.normal(0., c.observation_sd, (anchors, horizon)), repetitions, axis=0)
    else:
        action0 = np.concatenate([rng.permutation(np.tile(np.arange(4), repetitions // 4)) for _ in range(anchors)])
        noise = rng.normal(0., c.observation_sd, (n, horizon))
    resource = np.repeat(initial_resource, repetitions)
    vitality = np.repeat(initial_vitality, repetitions, axis=0)
    alive = np.ones((n, c.agents), bool)
    history = np.repeat(resource[:, None], c.regeneration_delay, axis=-1)
    observed = [np.repeat(initial_observed_resource, repetitions)]
    true_resource, resource_history = [resource.copy()], history.copy()
    observed_actions, true_vitality, true_alive = [], [vitality.copy()], [alive.copy()]
    true_take = []
    phase = np.repeat(offsets, repetitions)
    growths = []
    for t in range(horizon):
        action = np.ones((n, c.agents), dtype=np.int8)
        if t == 0:
            action[:, 0] = action0
        growth = c.growth_mean + c.growth_amplitude * np.sin(2 * np.pi * (t + phase) / c.growth_period)
        resource, vitality, alive, history, take, executed = step_physics(resource, vitality, alive, history, action, growth, c)
        observed.append(resource + noise[:, t])
        true_resource.append(resource.copy())
        true_vitality.append(vitality.copy())
        true_alive.append(alive.copy())
        observed_actions.append(executed.copy())
        true_take.append(take.copy())
        growths.append(growth.copy())
    observed = np.stack(observed, axis=1)
    record = {
        "split_stream_id": np.full(n, split, dtype=np.int64),
        "anchor_id": np.repeat(np.arange(anchors), repetitions),
        "trial_id": np.tile(np.arange(repetitions), anchors),
        "own_action": action0,
        "public_resource": observed,
        "own_initial_vitality": np.repeat(initial_vitality[:, 0], repetitions),
        "initial_resource_history": resource_history,
        "true_resource_DIAGNOSTIC": np.stack(true_resource, axis=1),
        "true_vitality_DIAGNOSTIC": np.stack(true_vitality, axis=1),
        "alive_DIAGNOSTIC": np.stack(true_alive, axis=1),
        "actions_DIAGNOSTIC": np.stack(observed_actions, axis=1),
        "take_DIAGNOSTIC": np.stack(true_take, axis=1),
        "growth_DIAGNOSTIC": np.stack(growths, axis=1),
        "noise_tape_DIAGNOSTIC": noise,
        "offset_DIAGNOSTIC": phase,
    }
    return record


def public_training_data(record, c):
    # Explicit allowlist prevents hidden diagnostic arrays entering the learner.
    return (record["public_resource"][:, 0] / (c.capacity_per_agent * c.agents),
            record["own_initial_vitality"] / c.max_vitality,
            record["own_action"], record["public_resource"][:, 1:].mean(axis=1))


def shifted_targets(record, target):
    shifted = np.empty_like(target)
    for anchor in np.unique(record["anchor_id"]):
        ids = np.flatnonzero(record["anchor_id"] == anchor)
        shifted[ids] = np.roll(target[ids], 1)
    return shifted


def fit_models(record):
    c = config_for()
    r, v, a, y = public_training_data(record, c)
    shifted = shifted_targets(record, y)
    models = {
        "aligned": ConsequenceModel().fit(r, v, a, y),
        "shifted_credit": ConsequenceModel().fit(r, v, a, shifted),
        "generic": ConsequenceModel("generic").fit(r, v, a, y),
        "immediate": ConsequenceModel().fit(r, v, a, record["public_resource"][:, 1]),
        "no_action": ConsequenceModel("no_action").fit(r, v, a, y),
    }
    models["no_head"] = models["aligned"]
    return models, shifted


def calibrate(models, record):
    c = config_for()
    r, v, action, y = public_training_data(record, c)
    true_outcome = record["true_resource_DIAGNOSTIC"][:, 1:].mean(axis=1).reshape(-1, 4)
    true_cost = true_outcome[:, :1] - true_outcome
    true_immediate = record["true_resource_DIAGNOSTIC"][:, 1].reshape(-1, 4)
    true_immediate_cost = true_immediate[:, :1] - true_immediate
    records, predictions = {}, {}
    for name in ("aligned", "shifted_credit", "generic", "immediate", "no_action"):
        model = models[name]
        estimate = model.costs(r[::4], v[::4])
        error = estimate[:, 1:] - true_cost[:, 1:]
        flat_truth, flat_est = true_cost[:, 1:].ravel(), estimate[:, 1:].ravel()
        cal = np.linalg.lstsq(np.stack((np.ones_like(flat_est), flat_est), axis=1), flat_truth, rcond=None)[0]
        records[name] = {
            "causal_cost_mse": float(np.mean(error**2)),
            "causal_cost_mae": float(np.mean(np.abs(error))),
            "cost_bias": float(np.mean(error)),
            "calibration_intercept": float(cal[0]),
            "calibration_slope": float(cal[1]),
            "outcome_mse_public": float(np.mean((model.predict(r, v, action) - y)**2)),
            "mean_predicted_cost": float(flat_est.mean()),
        }
        predictions[name] = estimate
    records["zero_cost_or_action_free"] = {"causal_cost_mse": float(np.mean(true_cost[:, 1:]**2))}
    records["physical_immediate_oracle_DIAGNOSTIC"] = {
        "causal_cost_mse": float(np.mean((true_immediate_cost[:, 1:] - true_cost[:, 1:])**2))}
    records["causal_oracle_DIAGNOSTIC"] = {"causal_cost_mse": 0.0}
    records["target"] = {
        "mean_true_horizon_cost": float(true_cost[:, 1:].mean()),
        "mean_true_immediate_cost": float(true_immediate_cost[:, 1:].mean()),
        "mean_delayed_component": float((true_cost - true_immediate_cost)[:, 1:].mean()),
    }
    predictions.update(true_cost=true_cost, true_immediate_cost=true_immediate_cost)
    return records, predictions


class EcologyPolicyBridge(BridgeEngine):
    def attach(self, model, gate=1., generic_score=False):
        self.ecological_model, self.ecological_gate, self.generic_score = model, gate, generic_score
        return self

    def observe(self, noise):
        index, resource_obs = super().observe(noise)
        self.public_resource_for_head = resource_obs
        return index, resource_obs

    def decision_scores(self, state_index):
        base = super().decision_scores(state_index)
        r = np.broadcast_to(self.public_resource_for_head[..., None] / self.capacity, self.state["vitality"].shape)
        v = self.state["vitality"] / self.config.max_vitality
        predictions = self.ecological_model.predict(r[..., None], v[..., None], np.arange(4))
        cost = predictions[..., :1] - predictions
        if self.generic_score:
            # Conventional direct future-resource utility; subtract common
            # action-0 baseline for exact numerical/action equivalence.
            addition = predictions - predictions[..., :1]
            return base + self.ecological_gate * PROTOCOL["ecological_weight"] * addition
        return base - self.ecological_gate * PROTOCOL["ecological_weight"] * cost


def native_engine(base, c, model, arm):
    engine = EcologyPolicyBridge(c, base.seeds.tolist(), .5, study="C", variant="full")
    for key in ("qt", "qv", "qg", "visits", "lineage"):
        engine.state[key] = base.state[key].copy()
    return engine.attach(model, gate=0. if arm == "no_head" else 1., generic_score=arm == "generic")


def scalar_outcome(result):
    keys = ("alive_fraction", "resource_fraction", "action_mean", "restraint_behavior")
    out = {k: float(result[k][0]) for k in keys}
    out["joint_designer_score"] = .5 * (out["alive_fraction"] + out["resource_fraction"])
    out["transition_sha256"] = str(result["full_transition_sha256"][0])
    out["final_state_sha256"] = str(result["final_state_sha256"][0])
    return out


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run_seed(seed, output):
    output = Path(output)
    if (output / "result.json").exists():
        raise ValueError("Refusing to overwrite completed seed output")
    output.mkdir(parents=True, exist_ok=True)
    train = acquisition(seed)
    models, shifted = fit_models(train)
    train["target_shifted_credit"] = shifted
    np.savez_compressed(output / "training_factual_trials.npz", **train)
    model_data = {f"coef_{k}": m.coefficients for k, m in models.items()}
    np.savez_compressed(output / "learned_models.npz", **model_data)
    result = {"seed": seed, "training_trials": len(shifted), "calibration": {}, "native": {}}
    for domain in ("id", "ood"):
        holdout = acquisition(seed, domain, evaluation=True)
        calibration, predictions = calibrate(models, holdout)
        np.savez_compressed(output / f"holdout_{domain}_counterfactual_diagnostic.npz", **holdout)
        np.savez_compressed(output / f"holdout_{domain}_predictions.npz", **predictions)
        result["calibration"][domain] = calibration
    c = config_for()
    base = BridgeEngine(c, [seed], .5, study="C", variant="full")
    base_training = []
    for episode in range(PROTOCOL["base_learning_episodes"]):
        outcome = base.episode(episode)
        base_training.append(scalar_outcome(outcome))
    base.save_snapshot(output / "base_policy_snapshot.npz", PROTOCOL["base_learning_episodes"])
    result["base_training"] = base_training
    for domain in ("id", "ood"):
        config = config_for(domain)
        result["native"][domain] = {}
        for arm in ARMS:
            episodes = []
            for episode in range(PROTOCOL["native_episodes_per_domain"]):
                engine = native_engine(base, config, models[arm], arm)
                generation = 700 + episode
                tape = engine.random_tape(generation)
                trace = output / f"native_{domain}_{arm}_{episode:02d}.npz"
                outcome = engine.episode(generation, learn=False, tape=tape, trace_path=trace, trace_mode="full")
                episodes.append(scalar_outcome(outcome))
                if arm == "aligned" and domain == "id" and episode == 0:
                    replay = native_engine(base, config, models[arm], arm)
                    repeat = replay.episode(generation, learn=False, tape=tape)
                    result["replay_exact"] = bool(all(np.array_equal(outcome[k], repeat[k], equal_nan=True)
                        if outcome[k].dtype.kind in "fc" else np.array_equal(outcome[k], repeat[k]) for k in outcome))
            result["native"][domain][arm] = episodes
    result["acquisition_record_sha256"] = digest_arrays({k: v for k, v in train.items() if k != "target_shifted_credit"})
    result["same_anchor_x_exact"] = all(np.unique(train["public_resource"][ids, 0]).size == 1 and
        np.unique(train["own_initial_vitality"][ids]).size == 1 for ids in np.split(np.arange(len(shifted)), PROTOCOL["train_anchors"]))
    y = train["public_resource"][:, 1:].mean(1)
    result["shift_preserves_y_per_anchor_exact"] = all(np.array_equal(np.sort(y[ids]), np.sort(shifted[ids]))
        for ids in np.split(np.arange(len(shifted)), PROTOCOL["train_anchors"]))
    json_write(output / "result.json", result)
    return result


def paired_summary(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    # Reproducible percentile bootstrap over independent seeds; descriptive CI.
    rng = np.random.default_rng(990017)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean": mean, "bootstrap95": np.quantile(samples, [.025, .975]).tolist(),
            "positive_seeds": int(np.sum(values > 0)), "n": len(values), "per_seed": values.tolist()}


def summarize(results):
    summary = {"seed_count": len(results), "mechanism": {}, "native": {},
               "all_replays_exact": all(r["replay_exact"] for r in results),
               "all_credit_contracts_exact": all(r["same_anchor_x_exact"] and r["shift_preserves_y_per_anchor_exact"] for r in results)}
    for domain in ("id", "ood"):
        summary["mechanism"][domain] = {}
        for comparator in ("zero_cost_or_action_free", "shifted_credit", "generic", "immediate", "physical_immediate_oracle_DIAGNOSTIC"):
            delta = [r["calibration"][domain][comparator]["causal_cost_mse"] - r["calibration"][domain]["aligned"]["causal_cost_mse"] for r in results]
            summary["mechanism"][domain][f"mse_gain_aligned_vs_{comparator}"] = paired_summary(delta)
        summary["mechanism"][domain]["model_mse_means"] = {name: float(np.mean([r["calibration"][domain][name]["causal_cost_mse"] for r in results]))
            for name in ("aligned", "shifted_credit", "generic", "immediate", "zero_cost_or_action_free", "physical_immediate_oracle_DIAGNOSTIC")}
        summary["mechanism"][domain]["calibration_aligned"] = {key: float(np.mean([r["calibration"][domain]["aligned"][key] for r in results]))
            for key in ("calibration_intercept", "calibration_slope", "cost_bias")}
        summary["native"][domain] = {}
        for comparator in ("no_head", "shifted_credit", "generic", "immediate"):
            contrasts = {}
            for metric in ("alive_fraction", "resource_fraction", "joint_designer_score", "action_mean"):
                delta = [np.mean([e[metric] for e in r["native"][domain]["aligned"]]) - np.mean([e[metric] for e in r["native"][domain][comparator]]) for r in results]
                contrasts[metric] = paired_summary(delta)
            summary["native"][domain][f"aligned_minus_{comparator}"] = contrasts
    return summary


def verify_output(output):
    output = Path(output)
    manifest = json.loads((output / "MANIFEST.json").read_text())
    files_ok = all(sha(output / p) == h for p, h in manifest["files"].items())
    run = json.loads((output / "RUN.json").read_text())
    seed = run["seeds"][0]
    seed_dir = output / f"seed_{seed}"
    base, _ = BridgeEngine.load_snapshot(seed_dir / "base_policy_snapshot.npz")
    with np.load(seed_dir / "learned_models.npz", allow_pickle=False) as data:
        model = ConsequenceModel(coefficients=data["coef_aligned"].copy())
    engine = native_engine(base, config_for(), model, "aligned")
    replay = scalar_outcome(engine.episode(700, learn=False))
    expected = json.loads((seed_dir / "result.json").read_text())["native"]["id"]["aligned"][0]
    result = {"files_sha256_match": files_ok, "first_native_replay_matches": replay == expected,
              "source_sha256_match": all(sha(ROOT / path) == digest for path, digest in run["source_sha256"].items())}
    json_write(output / "VERIFICATION.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verification = verify_output(args.output)
        print(json.dumps(verification, sort_keys=True), flush=True)
        if not all(verification.values()):
            raise SystemExit(1)
        return
    if args.phase is None:
        parser.error("--phase required unless --verify-only")
    if (args.output / "RUN.json").exists():
        raise SystemExit("Refusing to overwrite an existing run; choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = PROTOCOL[f"{args.phase}_seeds"]
    source = [Path(__file__), ROOT / "collective/bridge_v2/engine.py"]
    run = {"phase": args.phase, "seeds": seeds, "protocol": PROTOCOL,
           "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source},
           "numpy": np.__version__, "python": sys.version, "config_id": asdict(config_for()), "config_ood": asdict(config_for("ood"))}
    json_write(args.output / "RUN.json", run)
    started, results = time.monotonic(), []
    for seed in seeds:
        results.append(run_seed(seed, args.output / f"seed_{seed}"))
        print(json.dumps({"seed": seed, "complete": len(results), "total": len(seeds), "elapsed_s": round(time.monotonic()-started, 1)}), flush=True)
    summary = summarize(results)
    json_write(args.output / "SUMMARY.json", summary)
    files = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file() and p.name not in ("MANIFEST.json", "VERIFICATION.json")}
    json_write(args.output / "MANIFEST.json", {"files": files})
    print(json.dumps({"done": True, "all_replays_exact": summary["all_replays_exact"], "all_credit_contracts_exact": summary["all_credit_contracts_exact"]}), flush=True)


if __name__ == "__main__":
    main()
