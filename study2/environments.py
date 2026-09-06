"""Two internally designed dynamic tasks; truth is separate from observations.

All policies receive the same public variables. Neither the transition matrix,
reference action nor unobserved state components enter the observation mapping.
"""
from dataclasses import dataclass, asdict
import numpy as np

FAMILIES = ("dynamic_tracking", "cooperative_inventory")
REGIMES = ("paired", "diagonal", "misaligned")


@dataclass(frozen=True)
class EnvironmentConfig:
    family: str
    regime: str
    seed: int
    channels: int = 8
    steps: int = 128
    phase_length: int = 32


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def build_environment(config):
    if config.family not in FAMILIES or config.regime not in REGIMES:
        raise ValueError("Unknown environment family/regime")
    if config.channels != 8 or config.steps != 4 * config.phase_length:
        raise ValueError("Protocol requires eight channels and four equal phases")
    # Regime intentionally excluded: each regime shares nuisance random draws.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, FAMILIES.index(config.family), 202609]))
    inventory = config.family == "cooperative_inventory"
    rho = .93 if inventory else .55
    sigma = .002 if inventory else .006
    costs = rng.uniform(.8, 1.2, config.channels)
    base_diagonal = rng.uniform(.85, 1.15, config.channels)
    base_cross = rng.uniform(.20, .35, config.channels)
    initial = rng.uniform(.02, .15, config.channels) if inventory else rng.uniform(-.05, .15, config.channels)
    contexts = []
    for context in range(3):
        diagonal = base_diagonal * (1., .8, 1.15)[context]
        matrix = np.diag(diagonal)
        if config.regime != "diagonal":
            for i in range(config.channels):
                j = i ^ 1 if config.regime == "paired" else (i + 2) % config.channels
                sign = -1. if (i + context) % 3 == 0 else 1.
                matrix[i, j] = sign * base_cross[i]
        weights = np.ones(config.channels)
        # Two agents control indices 0:4 and 4:8; relative urgency reverses.
        weights[4 * (context % 2):4 * (context % 2) + 4] = 3.
        weights *= rng.uniform(.9, 1.1, config.channels)
        allowed = np.ones(config.channels, dtype=bool)
        if context == 2:
            allowed[1] = False
        budget = ((3.6, 2.8, 3.2) if inventory else (4.5, 3.3, 4.0))[context]
        reference = np.zeros(config.channels)
        for pair in range(config.channels // 2):
            levels = ((.04, .04), (.70, .08), (.08, .70), (.60, .60))[(pair + context) % 4]
            reference[2 * pair:2 * pair + 2] = levels
        reference *= rng.uniform(.93, 1.07, config.channels)
        reference[~allowed] = 0.
        reference *= min(1., .85 * budget / float(np.dot(costs, reference)))
        drift = -rng.uniform(.008, .016, config.channels) if inventory else np.zeros(config.channels)
        target = matrix @ reference + drift / (1. - rho)
        bounds = {}
        if inventory:
            lower = np.zeros(config.channels)
            upper = rng.uniform(.45, .65, config.channels)
            target = np.clip(target, lower, upper)
            bounds = dict(state_lower=lower, state_upper=upper)
        contexts.append(dict(matrix=matrix, target=target, weights=weights, costs=costs,
                             allowed=allowed, budget=budget, drift=drift,
                             reference_action=reference, persistence=rho,
                             horizon=8 if inventory else 3, context=f"context_{context}", **bounds))
    masks = rng.random((config.steps + 1, config.channels)) < (.60 if inventory else .80)
    masks[0] = True
    for t in range(config.steps + 1):
        masks[t, t % config.channels] = True
    noise = rng.normal(0., sigma, (config.steps, config.channels))
    frames = []
    for t in range(config.steps):
        phase = t // config.phase_length
        context = (0, 1, 2, 0)[phase]
        frame = dict(contexts[context])
        frame.update(step=t, phase=phase, switch=t in (config.phase_length, 2*config.phase_length, 3*config.phase_length),
                     return_to_initial=phase == 3, observed=masks[t], next_observed=masks[t+1], noise=noise[t])
        frames.append(frame)
    return _plain(dict(schema="polar-study2-environment-1", config=asdict(config),
                       initial_state=initial, frames=frames,
                       semantics="Tracking is affine and unbounded; inventory has finite capacity, stockouts and nonlinear state clipping. Exogenous withdrawals and capacities are public."))


def observation(frame, state):
    state = np.asarray(state, dtype=float)
    mask = np.asarray(frame["observed"], dtype=bool)
    if state.shape != (8,) or not np.isfinite(state).all() or mask.shape != state.shape:
        raise ValueError("Finite eight-channel state and observation mask required")
    public = {name: frame[name] for name in ("target", "weights", "costs", "allowed", "budget", "drift", "persistence", "horizon", "context")}
    public.update({name: frame[name] for name in ("state_lower", "state_upper") if name in frame})
    return {**public, "state": np.where(mask, state, 0.), "observed": mask}


def transition(frame, state, action):
    return transition_details(frame, state, action)["state"]


def transition_details(frame, state, action):
    state, action = np.asarray(state, float), np.asarray(action, float)
    if state.shape != (8,) or action.shape != (8,) or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Finite eight-channel state and action required")
    rho = float(frame["persistence"])
    raw = rho*state + (1-rho)*(np.asarray(frame["matrix"]) @ action) + np.asarray(frame["drift"]) + np.asarray(frame["noise"])
    if "state_lower" in frame:
        lower, upper = np.asarray(frame["state_lower"]), np.asarray(frame["state_upper"])
        result = np.clip(raw, lower, upper)
        return dict(state=result, raw_state=raw, transition_valid=(raw >= lower) & (raw <= upper),
                    stockout=np.maximum(lower-raw, 0.), overflow=np.maximum(raw-upper, 0.))
    return dict(state=raw, raw_state=raw, transition_valid=np.ones(8, bool), stockout=None, overflow=None)


def feedback(frame, next_state, transition_valid=None):
    mask = np.asarray(frame["next_observed"], bool)
    result = dict(state=np.where(mask, next_state, 0.), observed=mask)
    if transition_valid is not None:
        result["transition_valid"] = np.asarray(transition_valid, bool)
    return result
