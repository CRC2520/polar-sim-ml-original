"""Context-dependent two-channel regulator with explicit computational mechanisms.

No scalar in this module measures phenomenal consciousness, ethics or ASI.
"""
from dataclasses import asdict, dataclass, replace
import numpy as np
from .memory import EpisodicMemory
from .polarities import to_signed_intensity, from_signed_intensity, swap_poles
from .workspace import CapabilityModel, ResourceWorkspace


@dataclass(frozen=True)
class ModelConfig:
    agents: int = 3
    types: int = 8
    seed: int = 2025
    representation: str = "dual_pole"
    step_size: float = 0.65
    use_memory: bool = True
    use_workspace: bool = True
    use_self_model: bool = True
    memory_learning_rate: float = 0.5
    memory_retention: float = 0.995
    capability_learning_rate: float = 0.35

    def __post_init__(self):
        if not isinstance(self.agents, int) or not isinstance(self.types, int) or self.agents < 1 or self.types < 1:
            raise ValueError("agents and types must be positive integers")
        if self.representation not in {"dual_pole", "signed_intensity"}:
            raise ValueError("unknown state representation")
        if not np.isfinite(self.step_size) or not 0 < self.step_size <= 1:
            raise ValueError("step_size must lie in (0,1]")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


class ContextualPolarModel:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.config = replace(self.config, seed=int(seed))
        self.shape = (self.config.agents, self.config.types, 2)
        self.rng = np.random.default_rng(self.config.seed)
        self.memory = EpisodicMemory(self.shape, self.config.memory_learning_rate, self.config.memory_retention)
        self.self_model = CapabilityModel(self.shape, self.config.capability_learning_rate)
        self.workspace = ResourceWorkspace(self.config.agents, self.config.seed,
                                           "normal" if self.config.use_workspace else "disable")
        self.q = np.zeros(self.shape)
        self._planning_gain = np.ones(self.shape)
        self.last_action, self.last_trace, self.traces = None, {}, []
        self._pending_feedback, self.step = False, 0
        self._memory_event_cursor, self._self_event_cursor = 0, 0
        self._interventions = []
        return self

    @property
    def q(self):
        if self.config.representation == "signed_intensity":
            return from_signed_intensity(self._signed, self._intensity)
        return self._q.copy()

    @q.setter
    def q(self, value):
        value = np.asarray(value, float)
        if value.shape != self.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError(f"state must have shape {self.shape} and lie in [0,1]")
        if self.config.representation == "signed_intensity":
            self._signed, self._intensity = to_signed_intensity(value)
        else:
            self._q = value.copy()

    @property
    def signed(self):
        return self.q[..., 0] - self.q[..., 1]

    @property
    def intensity(self):
        return self.q.sum(axis=-1)

    def _array(self, value, name, *, broadcast=False, nonnegative=False, unit=False):
        result = np.asarray(value, float)
        if broadcast:
            if result.shape == (self.config.agents,):
                result = result[:, None, None]
            try:
                result = np.broadcast_to(result, self.shape).copy()
            except ValueError as exc:
                raise ValueError(f"{name} cannot broadcast to {self.shape}") from exc
        if result.shape != self.shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must have shape {self.shape} and finite values")
        if (nonnegative or unit) and np.any(result < 0):
            raise ValueError(f"{name} must be nonnegative")
        if unit and np.any(result > 1):
            raise ValueError(f"{name} must lie in [0,1]")
        return result

    def _mask(self, value, name):
        result = np.asarray(value)
        if result.dtype != np.bool_:
            raise ValueError(f"{name} must be boolean")
        try:
            result = np.broadcast_to(result, self.shape).copy()
        except ValueError as exc:
            raise ValueError(f"{name} cannot broadcast to {self.shape}") from exc
        return result

    def propose_action(self, effective_target, weights, horizon):
        """Overridable recurrence for matched nonpolar update-rule controls.

        Callers must retain act's observation filtering, memory, capability estimates,
        resource projection, feedback timing, and trace construction.
        """
        eta = np.clip(self.config.step_size * horizon, 0, 1)
        return self.q + eta * (effective_target / self._planning_gain - self.q)

    def act(self, observation):
        if self._pending_feedback:
            raise RuntimeError("previous action requires learn(feedback) or explicit skip_feedback(reason)")
        if not isinstance(observation, dict) or "target" not in observation:
            raise ValueError("observation must explicitly supply target; absent state is not zero")
        target = self._array(observation["target"], "target", unit=True)
        observed = self._mask(observation.get("observed", observation.get("observed_mask", True)), "observed")
        allowed = self._mask(observation.get("allowed", True), "allowed")
        weights = self._array(observation.get("weights", 1.0), "weights", broadcast=True, nonnegative=True)
        costs = self._array(observation.get("costs", 1.0), "costs", broadcast=True, nonnegative=True)
        if np.any(costs <= 0):
            raise ValueError("costs must be strictly positive")
        horizon = self._array(observation.get("horizon", 1.0), "horizon", broadcast=True, nonnegative=True)
        budget = float(observation.get("budget", np.prod(self.shape)))
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        cue = observation.get("cue")
        if cue is not None and not isinstance(cue, (str, int)):
            raise ValueError("cue must be a string, integer or None")

        self.memory.tick()
        recalled, confidence, memory_read = self.memory.recall(cue)
        if not self.config.use_memory:
            recalled.fill(0)
            confidence.fill(0)
            memory_read["disabled"] = True
        # Hidden target values are never consulted; only learned cue associations
        # can supply missing channels, and confidence explicitly attenuates recall.
        effective_target = np.where(observed, target, recalled * confidence)
        memory_write = self.memory.learn(cue, target, observed) if self.config.use_memory else {"operation": "disabled", "written_channels": 0}
        uncertainty = self.self_model.uncertainty
        if self.config.use_self_model:
            trust = 1 / (1 + uncertainty)
            self._planning_gain = 1 + trust * (self.self_model.gain_hat - 1)
        else:
            trust = np.zeros(self.shape)
            self._planning_gain = np.ones(self.shape)
        before = self.q
        proposal = np.asarray(self.propose_action(effective_target, weights, horizon), float)
        if proposal.shape != self.shape or not np.isfinite(proposal).all():
            raise ValueError("planner produced invalid state; numerical failure cannot be silently repaired")
        # Stability bound, admissible action set and scarce resources are separate.
        bounded = np.clip(proposal, 0, 1)
        permitted = np.where(allowed, bounded, 0)
        allocations, workspace_trace = self.workspace.allocate(permitted, weights, costs, budget, uncertainty)
        demand = np.sum(permitted * costs, axis=(1, 2))
        scale = np.ones(self.config.agents)
        np.divide(allocations, demand, out=scale, where=demand > 0)
        scale = np.minimum(1, scale)
        action = permitted * scale[:, None, None]
        self.q = action
        self.last_action, self._pending_feedback = action.copy(), True
        self.step += 1
        predicted = action * self._planning_gain
        events = self.memory.events[self._memory_event_cursor:] + self.self_model.events[self._self_event_cursor:] + self._interventions
        self._memory_event_cursor, self._self_event_cursor = len(self.memory.events), len(self.self_model.events)
        self._interventions = []
        self.last_trace = {
            "step": self.step, "config": asdict(self.config), "cue": cue,
            "observed_target": np.where(observed, target, 0).tolist(), "observed": observed.tolist(),
            "weights": weights.tolist(), "costs": costs.tolist(), "horizon": horizon.tolist(),
            "allowed": allowed.tolist(), "effective_target": effective_target.tolist(),
            "q_before": before.tolist(), "q_after": self.q.tolist(), "signed": self.signed.tolist(),
            "intensity": self.intensity.tolist(), "proposal": proposal.tolist(), "action": action.tolist(),
            "effect_predicted": predicted.tolist(), "planning_gain": self._planning_gain.tolist(),
            "self_model": self.self_model.snapshot(), "capability_trust": trust.tolist(),
            "memory_read": {**memory_read, "value": recalled.tolist(), "confidence": confidence.tolist()},
            "memory_write": memory_write, "memory_state": self.memory.snapshot(),
            "workspace": workspace_trace, "interventions": events,
            "stability": {"clipped_channels": int(np.count_nonzero(proposal != bounded))},
            "constraints": {"blocked_channels": int(np.count_nonzero((bounded > 0) & ~allowed)),
                            "resource_scale": scale.tolist(), "resource_cost": float(np.sum(action * costs)),
                            "budget": budget},
            "feedback": None,
        }
        self.traces.append(self.last_trace)
        return action.copy()

    def learn(self, feedback):
        if not self._pending_feedback:
            raise RuntimeError("learn requires one unconsumed own action from act")
        if not isinstance(feedback, dict) or "effect" not in feedback:
            raise ValueError("feedback must explicitly supply effect")
        effect = self._array(feedback["effect"], "effect", unit=True)
        result = self.self_model.learn(self.last_action, effect) if self.config.use_self_model else {"disabled": True, "updated_channels": 0}
        # Deliberately ignore feedback['target']: memory may only learn targets
        # observed through act, never the held-out evaluator's answer key.
        self.last_trace["feedback"] = {"effect": effect.tolist(), "capability_update": result}
        self._pending_feedback = False
        return result

    def skip_feedback(self, reason):
        """Explicitly record absent observations rather than pairing a later effect."""
        if not self._pending_feedback:
            raise RuntimeError("no pending action feedback")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a nonempty reason is required")
        self.last_trace["feedback"] = {"missing": True, "reason": reason}
        self._pending_feedback = False

    def reverse_convention(self, types=None):
        """Also swap all future observation/feedback channels outside the model."""
        if self._pending_feedback:
            raise RuntimeError("reverse convention between complete act/learn cycles")
        self.q = swap_poles(self.q, types)
        self.memory.reverse_convention(types)
        self.self_model.reverse_convention(types)
        self._planning_gain = swap_poles(self._planning_gain, types)
        self._interventions.append({"operation": "reverse_convention", "types": types})

    def snapshot(self):
        return {"step": self.step, "config": asdict(self.config), "q": self.q.tolist(),
                "memory": self.memory.snapshot(), "self_model": self.self_model.snapshot(),
                "workspace_mode": self.workspace.mode}
