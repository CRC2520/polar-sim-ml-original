"""Matched mechanism controls plus explicitly noncompetitive negative controls."""
import numpy as np
from .model import ContextualPolarModel, ModelConfig


CONTROLLERS = ["dual_pole", "signed_intensity", "recurrent", "no_memory", "no_workspace",
               "no_self_model", "memory_erase", "memory_shuffle", "workspace_shuffle",
               "self_model_reset", "zero", "uniform", "disconnected"]


class RecurrentController(ContextualPolarModel):
    """One projected-gradient update; shares observation and auxiliary mechanisms.

    This compares update rules, NOT a uniquely polar representation: signed_intensity
    supplies the separate exact coordinate-control comparison.
    """
    def propose_action(self, effective_target, weights, horizon):
        eta = np.clip(self.config.step_size * np.asarray(horizon), 0., 1.)
        w = np.asarray(weights) / max(float(np.max(weights)), 1e-12)
        gain = self._planning_gain
        return np.clip(self.q + eta * w * gain * (effective_target - gain * self.q), 0., 1.)


class NegativeControl:
    """Same observations available, deliberately ignores task-relevant information."""
    def __init__(self, kind, shape, seed):
        self.kind, self.shape = kind, shape
        self.fixed = np.random.default_rng(seed).uniform(0, .6, shape)
        self.last_trace = {}
    def act(self, obs):
        if self.kind == "zero":
            action = np.zeros(self.shape)
        elif self.kind == "uniform":
            action = np.full(self.shape, np.mean(obs["target"][obs["observed"]]) if np.any(obs["observed"]) else .3)
        else:
            action = self.fixed.copy()
        # All negative controls retain exactly the same hard action constraints.
        action = np.where(obs["allowed"], action, 0.)
        total = np.sum(action * obs["costs"])
        action *= min(1., obs["budget"] / max(total, 1e-12))
        self.last_trace = {"negative_control": self.kind}
        return action
    def learn(self, feedback):
        pass


def make_controller(name, seed, agents=3, types=8):
    if name not in CONTROLLERS:
        raise ValueError(name)
    if name in {"zero", "uniform", "disconnected"}:
        return NegativeControl(name, (agents, types, 2), seed)
    config = ModelConfig(agents=agents, types=types, seed=seed,
                         representation=name if name in {"dual_pole", "signed_intensity"} else "dual_pole",
                         use_memory=name != "no_memory", use_workspace=name != "no_workspace",
                         use_self_model=name != "no_self_model")
    model = RecurrentController(config) if name == "recurrent" else ContextualPolarModel(config)
    if name == "workspace_shuffle":
        model.workspace.mode = "shuffle"
    return model


def intervene(model, name, frame, task, seed):
    event = None
    if task == "switching_memory" and frame["t"] == 48:
        if name == "memory_erase":
            model.memory.erase()
            event = "erase_all_memory_before_masked_recall"
        elif name == "memory_shuffle":
            model.memory.shuffle(seed=seed + 991)
            event = "shuffle_memory_before_masked_recall"
    if task == "gain_resource_shift" and frame["t"] == 80 and name == "self_model_reset":
        model.self_model.lesion(gain=1.)
        event = "reset_estimated_gain_after_adaptation"
    return event
