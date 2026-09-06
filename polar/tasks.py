"""Frozen synthetic external tasks; environment truth never leaks through occlusion."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Trial:
    name: str
    split: str
    seed: int
    steps: list


def oracle_action(target, gains, weights, costs, allowed, budget):
    """Constrained weighted least-squares oracle, with known plant gains."""
    weights = np.maximum(np.asarray(weights, float), 1e-12)
    gains = np.asarray(gains, float)
    def at(lam):
        return np.where(allowed, np.clip(target / gains - lam * costs /
                        (2 * weights * gains**2), 0, 1), 0)
    lo, hi = 0., 100.
    if np.sum(at(0) * costs) <= budget:
        return at(0)
    for _ in range(55):
        mid = (lo + hi) / 2
        if np.sum(at(mid) * costs) > budget:
            lo = mid
        else:
            hi = mid
    return at(hi)


def _pattern(rng, shape, high):
    # Includes inactivity, single-pole predominance and genuine coactivation.
    modes = rng.integers(0, 4, shape[:-1])
    x = np.zeros(shape)
    x[..., 0] = np.isin(modes, [1, 3]) * high
    x[..., 1] = np.isin(modes, [2, 3]) * high
    return x


def make_trial(name, seed, split, agents=3, types=8, steps=96):
    if name not in {"switching_memory", "gain_resource_shift"}:
        raise ValueError(name)
    if split not in {"development", "heldout"} or steps != 96:
        raise ValueError("Frozen protocol requires a known split and 96 steps")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 17 if name == "switching_memory" else 29]))
    shape = (agents, types, 2)
    high = .60 if split == "development" else .57
    base = _pattern(rng, shape, high)
    # Force nondegenerate polar comparisons in every seed.
    base[:, 0, :] = (high, 0.)
    base[:, 1, :] = (0., high)
    base[:, 2, :] = (high, high)
    base[:, 3, :] = (0., 0.)
    other = _pattern(rng, shape, high)
    frames = []
    for t in range(steps):
        phase = t // 16
        weights = np.ones(shape)
        costs = np.ones(shape)
        allowed = np.ones(shape, dtype=bool)
        observed = np.ones(shape, dtype=bool)
        gains = np.ones(shape)
        budget = 24.
        horizon = 1.
        if name == "switching_memory":
            target = (base if phase in [0, 3, 4] else
                      base[..., ::-1] if phase in [1, 5] else other).copy()
            cue = ["A", "B", "distractor", "A", "A", "B"][phase]
            if phase == 3:
                observed[:] = False
            if phase == 2:
                horizon = 2.
            label = ["exposure", "pole_reversal", "distraction", "recall", "reacquisition", "second_reversal"][phase]
        else:
            cue = "resource_" + str(phase)
            target = base.copy() if phase % 2 == 0 else base[..., ::-1].copy()
            # New gain combinations and weight profiles in the heldout split.
            low, high_gain = ((.75, 1.25) if split == "development" else (.65, 1.35))
            if phase >= 1:
                gains = np.where(np.indices(shape).sum(0) % 2, low, high_gain)
            if phase >= 3:
                gains = gains[..., ::-1].copy()
            if phase in [2, 3, 4]:
                budget = 7.5
                focal = (phase + seed) % agents
                weights[focal] = 6. if split == "development" else 8.
                target[focal] = np.maximum(target[focal], .45)
                horizon = 2. if phase == 4 else 1.
            if phase == 4:
                allowed[1, -1, 0] = False
            label = ["calibration", "gain_shift", "priority_shift", "gain_and_role_shift", "restriction", "release"][phase]
        noise = rng.normal(0, .002, shape)
        ideal = oracle_action(target, gains, weights, costs, allowed, budget)
        frames.append(dict(t=t, phase=label, cue=cue, target=target, observed=observed,
                           weights=weights, costs=costs, allowed=allowed, gains=gains,
                           budget=budget, horizon=horizon, noise=noise, oracle_action=ideal,
                           switch=bool(t > 0 and t % 16 == 0), recall=(name == "switching_memory" and phase == 3),
                           reacquisition=(name == "switching_memory" and phase == 4)))
    return Trial(name, split, int(seed), frames)


def observation(frame):
    return {"target": np.where(frame["observed"], frame["target"], 0.),
            **{k: frame[k] for k in ["observed", "cue", "weights", "costs", "allowed", "budget", "horizon"]}}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
