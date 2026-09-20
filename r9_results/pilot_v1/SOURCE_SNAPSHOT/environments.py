"""Frozen-design scalar tasks for R9; a new realization, not the R8 engine.

The policy receives only three public sensors and factual reward. Evaluation
metadata, latent queues, climate regimes and tapes must never be policy inputs.
All randomness is allocated at reset, indexed by time and independent of action.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib

import numpy as np


HORIZON = 320
ACTION_FLOW = np.array([0.0, 0.04, 0.08, 0.12], dtype=np.float64)
ACTION_FLOW.flags.writeable = False
OBSERVATION_NAMES = ("reserve", "health", "demand_signal")
REWARD_BOUNDS = (0.0, 1.0)


@dataclass(frozen=True)
class DomainSpec:
    family: str
    delay: int
    initial_reserve: float
    initial_health: float


DOMAIN_SPECS = {
    "ecology_train": DomainSpec("ecology", 5, .70, .60),
    "ecology_delay9": DomainSpec("ecology", 9, .70, .60),
    "inventory_transfer": DomainSpec("inventory", 7, .55, .80),
    "thermal_transfer": DomainSpec("thermal", 0, .60, .80),
}


def _make_tape(seed):
    """Same base tape for all domains; domain transformations are deterministic."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9901]))
    count = HORIZON + 1
    switches = rng.random(count) < .035
    regime = np.logical_xor.accumulate(switches).astype(np.float64)
    storm_start = rng.random(count) < .018
    storm = np.zeros(count)
    durations = rng.integers(6, 17, count)
    for index in np.flatnonzero(storm_start):
        storm[index:min(count, index + durations[index])] = 1.
    phase = rng.uniform(0, 2 * np.pi, 2)
    t = np.arange(count)
    season = np.sin(2 * np.pi * t / 83 + phase[0])
    demand_wave = np.sin(2 * np.pi * t / 47 + phase[1])
    tape = {
        "regime": regime,
        "storm": storm,
        "demand_base": .027 + .008 * regime + .004 * demand_wave + .003 * rng.uniform(-1, 1, count),
        "growth": .27 + .045 * season - .045 * storm,
        "charge": .047 + .015 * season + .006 * regime - .018 * storm + .002 * rng.uniform(-1, 1, count),
        "ambient": .0025 + .0015 * regime + .0005 * rng.uniform(-1, 1, count),
        "fulfilment": .96 + .04 * rng.random(count),
        "sensor_noise": rng.normal(0, .012, (count, 3)),
    }
    for values in tape.values():
        values.flags.writeable = False
    return tape


class ScalarEnvironment:
    """Public reset/step API. Termination means the fixed horizon, never death.

    An action is requested throughput, not guaranteed executed flow. The domain
    label belongs to the experiment harness and is not included in observations.
    ``info_eval`` is evaluation-only and contains privileged physical state.
    """

    horizon = HORIZON
    observation_size = 3
    action_count = 4
    reward_bounds = REWARD_BOUNDS

    def __init__(self, domain="ecology_train"):
        self._set_domain(domain)
        self._has_reset = False

    def _set_domain(self, domain):
        if domain not in DOMAIN_SPECS:
            raise ValueError(f"Unknown domain {domain!r}; expected {tuple(DOMAIN_SPECS)}")
        self.domain = domain
        self.spec = DOMAIN_SPECS[domain]

    def reset(self, seed, domain=None):
        if domain is not None:
            self._set_domain(domain)
        self._tape = _make_tape(seed)
        self.seed, self.t = int(seed), 0
        self.reserve, self.health = self.spec.initial_reserve, self.spec.initial_health
        self.alive = True
        self.resource_history = [self.reserve] * max(1, self.spec.delay)
        self.pipeline = [0.] * self.spec.delay
        self.backlog, self.temperature = 0., .20
        self._totals = {"reward": 0., "alive": 0., "reserve": 0., "service": 0., "constraint": 0.}
        self._has_reset = True
        return self._observe()

    def clone(self):
        return copy.deepcopy(self)

    def _demand(self, t):
        base = self._tape["demand_base"][t]
        if self.spec.family == "inventory":
            return float(base + .008 + .009 * self._tape["storm"][t])
        if self.spec.family == "thermal":
            return float(base + .011 + .018 * self._tape["storm"][t])
        return float(base + .004 * self._tape["storm"][t])

    def _observe(self):
        # Saturation is a public sensor range, not a stabilizer of latent physics.
        signal = np.array([self.reserve, self.health, self._demand(self.t) / .12])
        return np.clip(signal + self._tape["sensor_noise"][self.t], 0., 1.).astype(np.float64)

    def tape_digest(self):
        digest = hashlib.sha256()
        for key in sorted(self._tape):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(self._tape[key]).tobytes())
        return digest.hexdigest()

    def _snapshot(self):
        return {"domain": self.domain, "t": self.t, "reserve": self.reserve, "health": self.health,
                "alive": self.alive, "resource_history": list(self.resource_history),
                "pipeline": list(self.pipeline), "backlog": self.backlog, "temperature": self.temperature}

    def step(self, action):
        if not self._has_reset:
            raise RuntimeError("reset must precede step")
        if self.t >= HORIZON:
            raise RuntimeError("episode already ended; reset explicitly")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError("action must be an integer in 0..3")
        was_alive = self.alive
        effective_action = int(action) if was_alive else 0
        request = float(ACTION_FLOW[effective_action])
        demand = self._demand(self.t)
        cost = (effective_action / 3.) ** 2
        before_reserve, before_health = self.reserve, self.health
        details = {"requested_flow": request, "demand": demand, "arrival": 0., "new_order": 0.,
                   "growth": 0., "charge": 0., "spill": 0., "loss": 0., "executed_flow": 0.}

        if self.spec.family == "ecology":
            take = min(self.reserve, request)
            remaining = self.reserve - take
            lag = self.resource_history[0]
            growth = float(self._tape["growth"][self.t] * lag * (1. - lag))
            proposed = remaining + growth
            spill = max(0., proposed - 1.)
            self.reserve = proposed - spill
            # Inventory/energy capacity saturation discards overflow explicitly.
            available_energy = self.health + .90 * take
            operating_cost = .002 * cost if was_alive else 0.
            service = min(1., available_energy / demand) if was_alive else 0.
            raw_health = available_energy - demand - operating_cost if was_alive else 0.
            energy_spill = max(0., raw_health - 1.)
            self.health = min(1., max(0., raw_health))
            deficit = max(0., demand + operating_cost - available_energy) / demand if was_alive else 1.
            severity = min(1., deficit)
            violation = raw_health <= 0. or (self.reserve < .10 and was_alive)
            failed = raw_health <= 0.
            self.resource_history = self.resource_history[1:] + [self.reserve]
            details.update(executed_flow=take, growth=growth, spill=spill, energy_spill=energy_spill,
                           lag_resource=lag, balance_residual=self.reserve - before_reserve + take - growth + spill)
        elif self.spec.family == "inventory":
            arrival = self.pipeline.pop(0)
            pre_service = self.reserve + arrival
            spill = max(0., pre_service - 1.)
            pre_service -= spill
            due = demand + self.backlog if was_alive else 0.
            shipped = min(pre_service, request, due)
            remaining = pre_service - shipped
            spoilage = .002 * remaining
            self.reserve = remaining - spoilage
            new_order = request if was_alive else 0.
            accepted_order = new_order * float(self._tape["fulfilment"][self.t])
            self.pipeline.append(accepted_order)
            self.backlog = max(0., due - shipped) if was_alive else self.backlog
            service = min(1., shipped / demand) if was_alive else 0.
            unmet = max(0., demand - shipped) / demand
            # Shipping earns liquidity now; procurement is paid before arrival.
            raw_health = self.health + .12 * shipped - .08 * new_order - .0015 - .015 * unmet if was_alive else 0.
            health_spill = max(0., raw_health - 1.)
            self.health = min(1., max(0., raw_health))
            severity = min(1., self.backlog / .30)
            violation = self.backlog > .10 or raw_health <= 0.
            failed = raw_health <= 0. or self.backlog > .60
            details.update(executed_flow=shipped, arrival=arrival, new_order=new_order,
                           accepted_order=accepted_order, procurement_loss=new_order - accepted_order,
                           loss=spoilage, spill=spill, health_spill=health_spill,
                           backlog=self.backlog, balance_residual=self.reserve - before_reserve - arrival + shipped + spoilage + spill)
        else:
            dispatch = min(self.reserve, request)
            delivered = min(dispatch, demand)
            unused_dispatch = dispatch - delivered
            charge = float(self._tape["charge"][self.t])
            remaining = self.reserve - dispatch
            leakage = .001 * remaining
            proposed = remaining - leakage + charge
            spill = max(0., proposed - 1.)
            self.reserve = proposed - spill
            self.temperature = .95 * self.temperature + 4.5 * dispatch ** 2 + float(self._tape["ambient"][self.t])
            overload = max(0., self.temperature - .65)
            service = delivered / demand if was_alive else 0.
            unmet = 1. - service
            raw_health = self.health + .004 - .012 * unmet - .08 * overload - .001 * cost if was_alive else 0.
            health_spill = max(0., raw_health - 1.)
            self.health = min(1., max(0., raw_health))
            severity = min(1., max(unmet, overload / .35))
            violation = unmet > .20 or self.temperature > .65
            failed = raw_health <= 0. or self.temperature > 1.20
            details.update(executed_flow=dispatch, delivered=delivered, unused_dispatch=unused_dispatch,
                           charge=charge, loss=leakage, spill=spill, health_spill=health_spill,
                           temperature=self.temperature, overload=overload,
                           balance_residual=self.reserve - before_reserve + dispatch + leakage - charge + spill)

        self.alive = bool(was_alive and not failed)
        if not self.alive:
            self.health = 0.
        # Analytic range [0,1], no empirical normalization and no future target.
        reward = float(service * (1. - .25 * cost) * (1. - .5 * severity)) if self.alive else 0.
        physical_numbers = [self.reserve, self.health, self.temperature, self.backlog, reward,
                            *self.resource_history, *self.pipeline]
        if not np.isfinite(physical_numbers).all():
            raise FloatingPointError("Nonfinite physical state; no silent repair")
        if self.reserve < -1e-12 or self.reserve > 1 + 1e-12 or not 0 <= reward <= 1 + 1e-12:
            raise ArithmeticError("Physical or reward bound violated")
        self.t += 1
        info_eval = {"alive": self.alive, "reserve": float(self.reserve), "constraint_violation": bool(violation or not self.alive),
                     "service": float(service if self.alive else 0.), "health": float(self.health),
                     "resource": float(self.reserve), "viability": float(self.alive), "task_reward": reward,
                     "constraint": float(violation or not self.alive), "step_index": self.t - 1,
                     "state_snapshot": self._snapshot(), "health_delta": self.health - before_health, **details}
        for key, value in (("reward", reward), ("alive", self.alive), ("reserve", self.reserve),
                           ("service", info_eval["service"]), ("constraint", info_eval["constraint_violation"])):
            self._totals[key] += float(value)
        return self._observe(), reward, self.t == HORIZON, info_eval

    def evaluation_summary(self):
        """Fixed-denominator endpoints; unfinished episodes are explicitly marked."""
        return {"steps": self.t, "complete": self.t == HORIZON, "horizon": HORIZON,
                "normalized_return": self._totals["reward"] / HORIZON,
                "alive_fraction": self._totals["alive"] / HORIZON,
                "reserve_mean": self._totals["reserve"] / HORIZON,
                "service_mean": self._totals["service"] / HORIZON,
                "constraint_fraction": self._totals["constraint"] / HORIZON,
                "tape_sha256": self.tape_digest()}
