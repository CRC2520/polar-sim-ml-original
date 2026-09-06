"""Functional budget broadcast and learned action/effect model; no consciousness claim."""
import numpy as np
from .polarities import swap_poles


class ResourceWorkspace:
    def __init__(self, agents, seed=0, mode="normal"):
        self.agents, self.mode = agents, mode
        self.rng = np.random.default_rng(seed)
        self.events = []

    def allocate(self, proposed_action, weights, costs, budget, uncertainty):
        if self.mode not in {"normal", "disable", "shuffle"}:
            raise ValueError("workspace mode must be normal, disable or shuffle")
        demand = np.sum(proposed_action * weights * costs, axis=(1, 2))
        if self.mode == "disable":
            priorities = np.ones(self.agents)
        elif self.mode == "shuffle":
            priorities = demand[self.rng.permutation(self.agents)]
        else:
            priorities = demand.copy()
        allocations = budget * priorities / priorities.sum() if priorities.sum() > 0 else np.full(self.agents, budget / self.agents)
        trace = {"mode": self.mode, "demand": demand.tolist(), "priorities": priorities.tolist(),
                 "allocations": allocations.tolist(), "budget": float(budget),
                 "mean_uncertainty": np.mean(uncertainty, axis=(1, 2)).tolist()}
        return allocations, trace


class CapabilityModel:
    def __init__(self, shape, learning_rate=0.35):
        if not 0 < learning_rate <= 1:
            raise ValueError("invalid capability learning rate")
        self.shape, self.learning_rate = tuple(shape), learning_rate
        self.gain_hat = np.ones(shape)
        self.counts = np.zeros(shape, int)
        self.error_ema = np.zeros(shape)
        self.events = []

    @property
    def uncertainty(self):
        return 1 / np.sqrt(self.counts + 1) + self.error_ema

    def learn(self, action, effect):
        action, effect = np.asarray(action, float), np.asarray(effect, float)
        if action.shape != self.shape or effect.shape != self.shape or not np.isfinite(effect).all() or not np.isfinite(action).all():
            raise ValueError("feedback effect/action shape or finiteness invalid")
        if np.any((action < 0) | (action > 1)) or np.any((effect < 0) | (effect > 1)):
            raise ValueError("feedback effect/action must lie in [0,1]")
        valid = action > 1e-6
        prediction_error = np.abs(effect - action * self.gain_hat)
        measured = np.ones(self.shape)
        np.divide(effect, action, out=measured, where=valid)
        measured = np.clip(measured, 0.1, 4.0)
        self.gain_hat[valid] += self.learning_rate * (measured[valid] - self.gain_hat[valid])
        self.error_ema[valid] += self.learning_rate * (prediction_error[valid] - self.error_ema[valid])
        self.counts[valid] += 1
        return {"updated_channels": int(valid.sum()), "prediction_error": prediction_error.tolist(),
                "gain_after": self.gain_hat.tolist(), "counts_after": self.counts.tolist()}

    def lesion(self, gain=1.0):
        gain = np.broadcast_to(np.asarray(gain, float), self.shape)
        if not np.isfinite(gain).all() or np.any((gain < 0.1) | (gain > 4)):
            raise ValueError("lesion gain must lie in [0.1,4]")
        self.gain_hat = gain.copy()
        self.counts.fill(0)
        self.error_ema.fill(0)
        self.events.append({"operation": "lesion", "gain": gain.tolist()})

    def reverse_convention(self, types=None):
        self.gain_hat = swap_poles(self.gain_hat, types)
        self.counts = swap_poles(self.counts, types)
        self.error_ema = swap_poles(self.error_ema, types)

    def snapshot(self):
        return {"gain_hat": self.gain_hat.tolist(), "counts": self.counts.tolist(),
                "error_ema": self.error_ema.tolist(), "uncertainty": self.uncertainty.tolist()}
