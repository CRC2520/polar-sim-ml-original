"""Batched tabular collective model, derived from the preserved v1.3 mechanism.

This is a NEW study engine, not a claim of byte-for-byte reproduction of v1.3.
Axes: batch(seed), group, individual, observable state, action. All learners use
only noisy resource, own vitality, their observed rewards and visit counts.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import hashlib
import numpy as np


@dataclass(frozen=True)
class BridgeConfig:
    agents: int = 20
    transmission: str = "success"
    groups: int = 6
    generations: int = 20
    warmup: int = 4
    steps: int = 400
    alpha: float = .1
    gamma_take: float = .5
    gamma_vitality: float = .97
    gamma_group: float = .97
    epsilon: float = .05
    temperature: float = .05
    metabolism: float = .3
    max_vitality: float = 10.
    capacity_per_agent: float = 20.
    initial_resource_fraction: float = .8
    initial_vitality: float = 5.
    growth_mean: float = .165
    growth_amplitude: float = .085
    growth_period: int = 1000
    regeneration_delay: int = 5
    observation_sd: float = 10.
    death_penalty: float = 50.
    restraint_cost: float = 3.
    initial_restraint_prior: float = .5
    mutation: float = .01
    replace_groups: int = 2
    tail_generations: int = 5
    viable_alive: float = .8
    viable_resource: float = .2

    def validate(self):
        if self.transmission not in ("success", "conformity"):
            raise ValueError("Unknown transmission rule")
        if self.agents < 2 or self.groups < 2 or self.steps < 1 or self.generations < 1:
            raise ValueError("Invalid population or horizon")
        if not 0 <= self.replace_groups <= self.groups // 2:
            raise ValueError("Invalid group replacement count")
        if self.regeneration_delay < 0 or self.capacity_per_agent <= 0:
            raise ValueError("Invalid ecological parameter")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon outside [0,1]")
        if not 0 < self.temperature or self.observation_sd < 0:
            raise ValueError("Invalid observation or policy parameter")
        if not 1 <= self.tail_generations <= self.generations:
            raise ValueError("Invalid evaluation window")


VARIANTS_A = ("base", "copy_score_equal", "resource_pool")
VARIANTS_C = ("full", "qv_policy_off", "qv_learning_off", "short_vitality_discount",
              "short_group_discount", "blind_exploration", "generic_shared_target", "coordinate_equivalent")
STATE_KEYS = ("qt", "qv", "qg", "visits", "lineage", "resource", "vitality", "alive", "resource_history")


class BridgeEngine:
    """Public API for interventions, independent replay and history transplants.

    ``state`` is a dict of arrays; cloning gives independent copies. Use
    ``episode(generation, reset_physical=False)`` to continue a transplanted
    physical state. ``learn=False`` freezes Q and visits, not environmental time.
    No random generator state is hidden: tapes are keyed by seed/generation/stream.
    """
    def __init__(self, config, seeds, initial_fraction, study="A", variant="base"):
        config.validate()
        if study not in ("A", "C"):
            raise ValueError("study must be A or C")
        if variant not in (VARIANTS_A if study == "A" else VARIANTS_C):
            raise ValueError("Unknown intervention")
        if not 0 <= initial_fraction <= 1:
            raise ValueError("Invalid initial fraction")
        count = initial_fraction * config.agents
        if not np.isclose(count, round(count)):
            raise ValueError("Initial composition must be realizable without rounding")
        self.config, self.seeds = config, np.asarray(seeds, dtype=np.int64)
        if len(self.seeds) == 0 or len(set(self.seeds.tolist())) != len(self.seeds):
            raise ValueError("Need distinct, nonempty seeds")
        self.initial_fraction, self.study, self.variant = float(initial_fraction), study, variant
        self.b = len(seeds)
        shape = (self.b, config.groups, config.agents)
        self.ix = np.indices(shape)
        self.state = {k: np.zeros(shape + (20, 4), dtype=np.float64) for k in ("qt", "qv", "qg", "visits")}
        lineage = np.zeros(shape, dtype=np.uint8)
        # Random assignment avoids agent-index effects; initial composition is exact per group.
        for b, seed in enumerate(self.seeds):
            rng = np.random.default_rng([int(seed), 101])
            for g in range(config.groups):
                lineage[b, g, rng.permutation(config.agents)[:round(count)]] = 1
        self.state["lineage"] = lineage
        if study == "C":
            self.state["qg"][:] = -config.initial_restraint_prior * lineage[..., None, None] * np.arange(4)
        self.reset_physical()
        self.last_episode = None

    @property
    def capacity(self):
        return self.config.capacity_per_agent * self.config.agents

    def clone(self):
        return copy.deepcopy(self)

    def reset_physical(self):
        c = self.config
        self.state["resource"] = np.full((self.b, c.groups), self.capacity * c.initial_resource_fraction)
        self.state["vitality"] = np.full((self.b, c.groups, c.agents), c.initial_vitality)
        self.state["alive"] = np.ones((self.b, c.groups, c.agents), dtype=bool)
        self.state["resource_history"] = np.repeat(self.state["resource"][..., None], max(c.regeneration_delay, 1), axis=-1)

    def save_snapshot(self, path, generation=0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(config=asdict(self.config), seeds=self.seeds.tolist(), initial_fraction=self.initial_fraction,
                    study=self.study, variant=self.variant, generation=int(generation))
        np.savez_compressed(path, **self.state, metadata=np.array(json.dumps(meta, sort_keys=True)))

    @classmethod
    def load_snapshot(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            engine = cls(BridgeConfig(**meta["config"]), meta["seeds"], meta["initial_fraction"], meta["study"], meta["variant"])
            for key in STATE_KEYS:
                engine.state[key] = data[key].copy()
        return engine, meta

    def random_tape(self, generation, steps=None):
        c = self.config
        t = c.steps if steps is None else steps
        records = []
        for seed in self.seeds:
            rng = np.random.default_rng([int(seed), int(generation), 102])
            shape = (t, c.groups, c.agents)
            records.append(dict(gumbel=rng.gumbel(size=shape + (4,)),
                                explore=rng.random(shape) < c.epsilon,
                                random_action=rng.integers(4, size=shape, dtype=np.int8),
                                obs_noise=rng.normal(0, c.observation_sd, size=(t + 1, c.groups)),
                                offset=rng.integers(0, c.growth_period, size=c.groups)))
        return {k: np.stack([r[k] for r in records]) for k in records[0]}

    def observe(self, noise):
        c, s = self.config, self.state
        noisy_resource = s["resource"] + noise
        eb = np.clip((s["vitality"] / c.max_vitality * 4).astype(int), 0, 3)
        rb = np.clip((noisy_resource / self.capacity * 5).astype(int), 0, 4)
        return eb * 5 + rb[..., None], noisy_resource

    def head_values(self, state_index):
        return [self.state[k][*self.ix, state_index] for k in ("qt", "qv", "qg")]

    def decision_scores(self, state_index):
        qt, qv, qg = self.head_values(state_index)
        wv = 0. if self.variant == "qv_policy_off" else 1.
        if self.variant == "coordinate_equivalent":
            # Change coordinates of two additive heads exactly, retaining both DOF.
            difference, intensity = qv - qg, qv + qg
            qv, qg = (intensity + difference) / 2, (intensity - difference) / 2
        return qt + wv * qv + qg

    def state_digest(self, seed_index=None):
        digest = hashlib.sha256()
        for key in sorted(self.state):
            value = np.ascontiguousarray(self.state[key] if seed_index is None else self.state[key][seed_index:seed_index + 1])
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.dtype.str.encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def episode(self, generation, reset_physical=True, steps=None, learn=True, trace_path=None, tape=None, trace_mode="reconstruction"):
        c, s = self.config, self.state
        steps = c.steps if steps is None else int(steps)
        if reset_physical:
            self.reset_physical()
        tape = self.random_tape(generation, steps) if tape is None else tape
        if trace_mode not in ("reconstruction", "compact", "full", "diagnostic"):
            raise ValueError("Unknown trace mode")
        initial_digest = self.state_digest()
        initial_per_seed = [self.state_digest(b) for b in range(self.b)]
        transition_digest, diagnostic_digest = hashlib.sha256(), hashlib.sha256()
        initial = {k: v.copy() if trace_mode == "full" else v[:1].copy() for k, v in self.state.items()} if trace_mode in ("full", "diagnostic") else {}
        shape = (self.b, c.groups, c.agents)
        alive_steps = np.zeros(shape)
        resource_sum = np.zeros((self.b, c.groups))
        action_sum = np.zeros(shape)
        restrained_steps = np.zeros(shape)
        eligible_steps = np.zeros(shape)
        income_sum = np.zeros(shape)
        take_sum = np.zeros(shape)
        exploration_count = np.zeros(shape, dtype=int)
        exploration_changed = np.zeros(shape, dtype=int)
        first_collapse = np.full((self.b, c.groups), -1, dtype=int)
        max_pool_conservation_error = np.zeros(self.b)
        max_pool_income_spread = np.zeros(self.b)
        trace = {k: [] for k in ("resource_before", "resource_after", "vitality_before", "vitality_after", "alive_before",
                 "alive_after", "state_index", "next_state_index", "resource_observation", "action", "boltzmann_action",
                 "explore_mask", "take", "income", "transfer", "vitality_reward", "group_reward", "growth")}
        for t in range(steps):
            before_alive = s["alive"].copy()
            before_resource = s["resource"].copy()
            before_vitality = s["vitality"].copy()
            state_index, resource_obs = self.observe(tape["obs_noise"][:, t])
            scores = self.decision_scores(state_index)
            greedy = (scores / c.temperature + tape["gumbel"][:, t]).argmax(-1)
            visits = s["visits"][*self.ix, state_index]
            if self.variant == "blind_exploration":
                exploratory = tape["random_action"][:, t]
            else:
                exploratory = (-visits + 1e-9 * tape["gumbel"][:, t]).argmax(-1)
            explore = tape["explore"][:, t]
            action = np.where(explore, exploratory, greedy)
            action = np.where(before_alive, action, 0)
            take = action * before_resource[..., None] / self.capacity
            demand = take.sum(-1)
            take *= np.minimum(1., np.divide(before_resource, demand, out=np.ones_like(demand), where=demand > 0))[..., None]
            if self.variant == "resource_pool":
                count_alive = before_alive.sum(-1)
                income = np.divide(take.sum(-1), count_alive, out=np.zeros_like(demand), where=count_alive > 0)[..., None] * before_alive
            else:
                income = take.copy()
            transfer = income - take
            max_pool_conservation_error = np.maximum(max_pool_conservation_error, np.max(np.abs(transfer.sum(-1)), axis=1))
            spread = np.max(np.where(before_alive, income, -np.inf), -1) - np.min(np.where(before_alive, income, np.inf), -1)
            spread = np.where(before_alive.any(-1), spread, 0.)
            max_pool_income_spread = np.maximum(max_pool_income_spread, np.max(spread, axis=1))
            after_harvest = before_resource - take.sum(-1)
            lag_resource = s["resource_history"][..., 0] if c.regeneration_delay else after_harvest
            growth = c.growth_mean + c.growth_amplitude * np.sin(2 * np.pi * (t + tape["offset"]) / c.growth_period)
            resource_next = np.maximum(after_harvest + growth * lag_resource * (1 - lag_resource / self.capacity), 0.)
            s["resource_history"] = np.concatenate((s["resource_history"][..., 1:], before_resource[..., None]), axis=-1)
            s["resource"] = resource_next
            delta_v = np.where(before_alive, income - c.metabolism, 0.)
            s["vitality"] = np.clip(before_vitality + delta_v, -1., c.max_vitality)
            died = before_alive & (s["vitality"] <= 0)
            s["alive"] = before_alive & ~died
            vitality_reward = delta_v - c.death_penalty * died
            other_vitality = (vitality_reward.sum(-1, keepdims=True) - vitality_reward) / (c.agents - 1)
            if self.study == "A":
                penalty = take * s["lineage"]
            else:
                # Contextual ecological cost is independent of ancestral labels.
                pressure = np.clip(1. - resource_obs / self.capacity, 0., 1.)
                penalty = take * pressure[..., None]
            group_reward = other_vitality - c.restraint_cost * penalty
            next_index, _ = self.observe(tape["obs_noise"][:, t + 1])
            gammas = (c.gamma_take, .5 if self.variant == "short_vitality_discount" else c.gamma_vitality,
                      .5 if self.variant == "short_group_discount" else c.gamma_group)
            if learn:
                # Freeze all targets before any in-place head update.
                next_heads = self.head_values(next_index)
                if self.variant == "generic_shared_target":
                    next_action = self.decision_scores(next_index).argmax(-1)
                    bootstraps = [np.take_along_axis(head, next_action[..., None], axis=-1)[..., 0] for head in next_heads]
                else:
                    bootstraps = [head.max(-1) for head in next_heads]
                for key, reward, gamma, bootstrap in zip(("qt", "qv", "qg"), (income, vitality_reward, group_reward), gammas, bootstraps):
                    if key == "qv" and self.variant == "qv_learning_off":
                        continue
                    q = s[key]
                    old = q[*self.ix, state_index, action]
                    target = reward + gamma * bootstrap * s["alive"]
                    q[*self.ix, state_index, action] = old + c.alpha * (target - old) * before_alive
                s["visits"][*self.ix, state_index, action] += before_alive
            alive_steps += s["alive"]
            resource_sum += resource_next
            eligible_steps += before_alive
            action_sum += action
            restrained_steps += (action <= 1) & before_alive
            income_sum += income
            take_sum += take
            exploration_count += explore & before_alive
            exploration_changed += explore & before_alive & (action != greedy)
            collapse = (resource_next < .025 * self.capacity) | ~s["alive"].any(-1)
            first_collapse = np.where((first_collapse < 0) & collapse, t, first_collapse)
            values = (before_resource, resource_next, before_vitality, s["vitality"], before_alive, s["alive"], state_index, next_index,
                      resource_obs, action, greedy, explore, take, income, transfer, vitality_reward, group_reward, growth)
            for key, value in zip(trace, values):
                transition_digest.update(np.ascontiguousarray(value).tobytes())
                diagnostic_digest.update(np.ascontiguousarray(value[:1]).tobytes())
            if trace_path is not None and trace_mode != "reconstruction":
                for key, value in zip(trace, values):
                    if trace_mode in ("full", "diagnostic") or key in ("resource_before", "resource_after", "resource_observation", "action", "alive_after"):
                        if trace_mode == "diagnostic":
                            value = value[:1]
                        if key == "action":
                            value = np.asarray(value, dtype=np.uint8)
                        elif key == "alive_after" and trace_mode == "compact":
                            value = np.packbits(value, axis=-1)
                        trace[key].append(np.asarray(value).copy())
        def ratio(num, den):
            return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
        outcome = dict(alive_agent_time=alive_steps / steps, alive_fraction=alive_steps.mean((1, 2)) / steps,
                       group_alive=alive_steps.mean(-1) / steps, resource_fraction=resource_sum.mean(-1) / steps / self.capacity,
                       group_resource=resource_sum / steps / self.capacity,
                       action_mean=ratio(action_sum.sum((1, 2)), eligible_steps.sum((1, 2))),
                       restraint_behavior=ratio(restrained_steps.sum((1, 2)), eligible_steps.sum((1, 2))),
                       restraint_agent=ratio(restrained_steps, eligible_steps),
                       income_agent=ratio(income_sum, eligible_steps), take_agent=ratio(take_sum, eligible_steps),
                       income_cumulative=income_sum, eligible_steps=eligible_steps,
                       exploration_count=exploration_count.sum((1, 2)), exploration_changed=exploration_changed.sum((1, 2)),
                       first_collapse=first_collapse, pool_conservation_error=max_pool_conservation_error,
                       pool_income_spread=max_pool_income_spread, lineage_fraction=s["lineage"].mean((1, 2)),
                       initial_state_sha256=np.asarray(initial_per_seed),
                       final_state_sha256=np.asarray([self.state_digest(b) for b in range(self.b)]),
                       full_transition_sha256=np.asarray([transition_digest.hexdigest()] * self.b),
                       diagnostic_transition_sha256=np.asarray([diagnostic_digest.hexdigest()] * self.b))
        self.last_episode = outcome
        if trace_path is not None:
            trace_path = Path(trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: np.stack(v, axis=1) for k, v in trace.items() if v}
            data.update({"initial_" + k: v for k, v in initial.items()})
            if trace_mode in ("full", "diagnostic"):
                data.update({"final_" + k: v if trace_mode == "full" else v[:1] for k, v in self.state.items()})
            data["initial_state_sha256"] = np.array(initial_digest)
            data["final_state_sha256"] = np.array(self.state_digest())
            data["metadata"] = np.array(json.dumps(dict(config=asdict(c), seeds=self.seeds.tolist(), generation=generation,
                    study=self.study, variant=self.variant, initial_fraction=self.initial_fraction, steps=steps, learn=learn, trace_mode=trace_mode), sort_keys=True))
            np.savez_compressed(trace_path, **data)
        return outcome

    def evolve(self, generation, outcome=None):
        c, s = self.config, self.state
        outcome = self.last_episode if outcome is None else outcome
        if outcome is None:
            raise ValueError("No observed episode to drive transmission")
        fitness = outcome["alive_agent_time"].copy()
        adjusted = fitness.copy()
        if self.variant == "copy_score_equal":
            # Equalize trait-conditional copying scores, preserving within-class variation.
            for value in (0, 1):
                mask = s["lineage"] == value
                count = mask.sum(-1, keepdims=True)
                mean = np.divide((fitness * mask).sum(-1, keepdims=True), count,
                                 out=np.zeros_like(count, dtype=float), where=count > 0)
                adjusted += np.where(mask & (count > 0), fitness.mean(-1, keepdims=True) - mean, 0.)
        before_lineage = s["lineage"].copy()
        records = dict(raw_fitness=fitness.copy(), copy_score=adjusted.copy(), lineage_before=before_lineage)
        for b, seed in enumerate(self.seeds):
            rng = np.random.default_rng([int(seed), generation, 103])
            social_uniforms = rng.random((c.groups, c.agents + 3))
            group_rng = np.random.default_rng([int(seed), generation, 104])
            mutation_rng = np.random.default_rng([int(seed), generation, 105])
            if generation >= c.warmup:
                for g in range(c.groups):
                    f = adjusted[b, g]
                    # Same tie randomization in all conditions; no fixed-index advantage.
                    jitter = (social_uniforms[g, :c.agents] - .5) * 2e-12
                    if c.transmission == "conformity":
                        if self.study == "A":
                            categories = s["lineage"][b, g].astype(bool)
                        else:
                            # Classification of measured behavior, never ancestral labels.
                            behavior = outcome["restraint_agent"][b, g]
                            if not np.all(np.isfinite(behavior)):
                                raise ValueError("Behavioral conformity requires observed eligible behavior")
                            categories = behavior >= .5
                        maj = bool(categories.mean() > .5) if categories.mean() != .5 else bool(social_uniforms[g, -1] < .5)
                        candidates = np.flatnonzero(categories == maj)
                        lo = int(social_uniforms[g, -2] * c.agents)
                        hi = candidates[int(social_uniforms[g, -3] * len(candidates))]
                        should_copy = categories[lo] != maj
                    else:
                        lo, hi = np.argmin(f + jitter), np.argmax(f + jitter)
                        should_copy = f[hi] > f[lo] + 1e-12
                    if should_copy:
                        for key in ("qt", "qv", "qg", "visits", "lineage"):
                            s[key][b, g, lo] = s[key][b, g, hi]
                order = np.argsort(outcome["group_alive"][b] + group_rng.uniform(-1e-12, 1e-12, c.groups))
                # Copy simultaneously; winners and losers are disjoint.
                for lo, hi in zip(order[:c.replace_groups], order[-c.replace_groups:]):
                    for key in ("qt", "qv", "qg", "visits", "lineage"):
                        s[key][b, lo] = s[key][b, hi].copy()
            flips = mutation_rng.random((c.groups, c.agents)) < c.mutation
            if self.study == "A":
                s["lineage"][b] ^= flips.astype(np.uint8)
            # C has no inherited behavioral forcing or mutation of an unused label.
        records["lineage_after"] = s["lineage"].copy()
        return records

    def run(self, trace_dir=None, trace_mode="reconstruction", save_final_snapshot=False):
        history = []
        transmissions = []
        for generation in range(self.config.generations):
            trace_path = Path(trace_dir) / f"generation_{generation:03d}.npz" if trace_dir is not None else None
            outcome = self.episode(generation, trace_path=trace_path, trace_mode=trace_mode)
            history.append(outcome)
            transmissions.append(self.evolve(generation, outcome))
        history_arrays = {k: np.stack([o[k] for o in history], axis=1) for k in history[0]}
        transmission_arrays = {k: np.stack([o[k] for o in transmissions], axis=1) for k in transmissions[0]}
        if trace_dir is not None:
            np.savez_compressed(Path(trace_dir) / "transmissions.npz", **transmission_arrays)
            if save_final_snapshot:
                self.save_snapshot(Path(trace_dir) / "final_snapshot.npz", self.config.generations)
            np.savez_compressed(Path(trace_dir) / "outcomes.npz", **history_arrays)
        return history_arrays, transmission_arrays
