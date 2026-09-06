"""Inspectable structured system identification and constrained regulation.

Only observation and own-action feedback enter the controller. True environment
matrices, evaluation targets beyond the supplied target, and outcome metrics are
never consulted. Existing study-one code is deliberately not imported.
"""
from dataclasses import asdict, dataclass
import numpy as np


MODES = ("paired", "paired_lesion", "diagonal", "shuffled", "dense", "signed_intensity")


@dataclass(frozen=True)
class ControllerConfig:
    channels: int = 8
    mode: str = "paired"
    seed: int = 20260906
    planner_steps: int = 24
    prior_diagonal: float = 0.5
    ridge: float = 1.0
    forgetting: float = 0.98
    action_penalty: float = 0.002
    movement_penalty: float = 0.002
    error_retention: float = 0.95

    def __post_init__(self):
        if not isinstance(self.channels, int) or self.channels < 4 or self.channels % 2:
            raise ValueError("channels must be an even integer of at least four")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.planner_steps, int) or self.planner_steps < 1:
            raise ValueError("planner_steps must be a positive integer")
        for name in ("prior_diagonal", "ridge", "forgetting", "action_penalty", "movement_penalty", "error_retention"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.prior_diagonal < 0 or self.ridge <= 0 or not 0 < self.forgetting <= 1:
            raise ValueError("prior_diagonal >= 0, ridge > 0 and forgetting in (0,1] required")
        if min(self.action_penalty, self.movement_penalty) < 0 or not 0 <= self.error_retention < 1:
            raise ValueError("penalties must be nonnegative and error_retention in [0,1)")


def project_action(proposal, costs, budget, allowed):
    """Exact Euclidean projection onto 0<=u<=1, c.u<=budget, blocked u=0.

    The weighted capped-simplex threshold is found at its piecewise-linear
    breakpoints, avoiding optimizer tolerances and long bisection loops.
    """
    v, c, allowed = np.asarray(proposal, float), np.asarray(costs, float), np.asarray(allowed)
    if v.ndim != 1 or c.shape != v.shape or allowed.shape != v.shape or allowed.dtype != np.bool_:
        raise ValueError("projection requires equally shaped vectors and boolean allowed")
    if not np.isfinite(v).all() or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("projection proposal must be finite and costs finite positive")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("projection budget must be finite and nonnegative")
    v = np.where(allowed, v, 0.0)
    bounded = np.clip(v, 0, 1)
    if float(c @ bounded) <= budget or budget == 0:
        return bounded if budget > 0 else np.zeros_like(v)
    knots = np.sort(np.maximum(0, np.concatenate(([0.0], (v - 1) / c, v / c))))
    totals = np.clip(v[None, :] - knots[:, None] * c[None, :], 0, 1) @ c
    upper = int(np.flatnonzero(totals <= budget)[0])
    lower = upper - 1
    fraction = (totals[lower] - budget) / (totals[lower] - totals[upper])
    threshold = knots[lower] + fraction * (knots[upper] - knots[lower])
    return np.where(allowed, np.clip(v - threshold * c, 0, 1), 0)


class CoupledController:
    """Same estimator, memory and planner infrastructure across all controls."""

    def __init__(self, config=None):
        self.config = config or ControllerConfig()
        self.n = self.config.channels
        self.reset()

    def reset(self):
        n = self.n
        self.edge_mask = np.eye(n, dtype=bool)
        if self.config.mode == "dense":
            self.edge_mask[:] = True
        elif self.config.mode != "diagonal":
            # Adjacent pairs are the hypothesized prior; the shifted partition
            # has exactly the same number of independently learned coefficients.
            order = np.arange(n)
            if self.config.mode == "shuffled":
                order = np.roll(order, 1)
            for first, second in order.reshape(-1, 2):
                self.edge_mask[first, second] = self.edge_mask[second, first] = True
        self.B = np.eye(n) * self.config.prior_diagonal
        self.covariance = np.zeros((n, n, n))
        for row in range(n):
            indices = np.flatnonzero(self.edge_mask[row])
            self.covariance[row][np.ix_(indices, indices)] = np.eye(len(indices)) / self.config.ridge
        self.error_variance = np.full(n, 0.05)
        self.update_counts = np.zeros(n, dtype=int)
        self.memory_state = None
        self.memory_uncertainty = np.ones(n)
        self._previous_action = np.zeros(n)
        self._signed = np.zeros(n // 2)
        self._intensity = np.zeros(n // 2)
        self.coupling_enabled = self.config.mode != "paired_lesion"
        self._pending = None
        self.last_trace = {}
        self.step = 0
        self._interventions = []
        return self

    def _vector(self, value, name, *, positive=False):
        result = np.asarray(value, float)
        if result.shape != (self.n,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector of shape ({self.n},)")
        if positive and np.any(result <= 0):
            raise ValueError(f"{name} must be strictly positive")
        return result.copy()

    def _mask(self, value, name):
        result = np.asarray(value)
        if result.shape != (self.n,) or result.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean vector of shape ({self.n},)")
        return result.copy()

    @property
    def previous_action(self):
        if self.config.mode == "signed_intensity":
            return np.column_stack(((self._intensity + self._signed) / 2,
                                    (self._intensity - self._signed) / 2)).ravel()
        return self._previous_action.copy()

    def _remember_action(self, action):
        self._previous_action = action.copy()
        pairs = action.reshape(-1, 2)
        self._signed = pairs[:, 0] - pairs[:, 1]
        self._intensity = pairs.sum(axis=1)

    @property
    def profile(self):
        widths = self.edge_mask.sum(axis=1).astype(int)
        return {
            "channels": self.n, "estimator_coefficients": int(widths.sum()),
            "estimator_covariance_independent_entries": int(np.sum(widths * (widths + 1) // 2)),
            "state_memory_entries": self.n, "previous_action_entries": self.n,
            "error_variance_entries": self.n, "planner_steps": self.config.planner_steps,
            "planner_dense_matrix_terms_per_iteration": self.n * self.n,
            "rls_dense_submatrix_terms_per_observed_step": int(np.sum(4 * widths ** 2 + 3 * widths)),
            "cost_estimate_scope": "multiplication-term estimates; not measured hardware FLOPs or latency",
            "same_capacity_primary_contrast": "paired versus paired_lesion",
        }

    def set_coupling_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        if self._pending is not None:
            raise RuntimeError("consume pending feedback before changing the planning intervention")
        self.coupling_enabled = enabled
        self._interventions.append({"operation": "planning_coupling", "enabled": enabled})

    def erase_state_memory(self):
        if self._pending is not None:
            raise RuntimeError("consume pending feedback before erasing memory")
        self.memory_state = None
        self.memory_uncertainty.fill(1)
        self._interventions.append({"operation": "erase_state_memory"})

    def act(self, observation):
        if self._pending is not None:
            raise RuntimeError("each action requires exactly one learn(feedback) before the next act")
        required = ("state", "observed", "target", "weights", "costs", "allowed", "budget", "persistence", "horizon")
        if not isinstance(observation, dict) or any(key not in observation for key in required):
            raise ValueError(f"observation must explicitly contain {required}; missing state is not zero")
        state = self._vector(observation["state"], "state")
        observed = self._mask(observation["observed"], "observed")
        target = self._vector(observation["target"], "target")
        weights = self._vector(observation["weights"], "weights", positive=True)
        costs = self._vector(observation["costs"], "costs", positive=True)
        allowed = self._mask(observation["allowed"], "allowed")
        drift = self._vector(observation.get("drift", np.zeros(self.n)), "drift")
        lower = self._vector(observation["state_lower"], "state_lower") if "state_lower" in observation else np.full(self.n, -np.inf)
        upper = self._vector(observation["state_upper"], "state_upper") if "state_upper" in observation else np.full(self.n, np.inf)
        if np.any(lower >= upper):
            raise ValueError("state_lower must be strictly below state_upper")
        budget, rho = float(observation["budget"]), float(observation["persistence"])
        horizon = observation["horizon"]
        if not np.isfinite(budget) or budget < 0 or not np.isfinite(rho) or not 0 <= rho < 1:
            raise ValueError("budget must be finite nonnegative and persistence in [0,1)")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        context = observation.get("context")
        if context is not None and not isinstance(context, (str, int)):
            raise ValueError("context must be a string, integer or None")
        if self.memory_state is None and not observed.all():
            raise ValueError("initial or erased state memory requires all state channels observed")
        memory_before = None if self.memory_state is None else self.memory_state.copy()
        estimate = state.copy() if memory_before is None else np.where(observed, state, memory_before)
        memory_source = np.where(observed, "observation", "predicted_memory")
        self.memory_uncertainty[observed] = 0
        planning_B = self.B.copy()
        if not self.coupling_enabled:
            planning_B *= np.eye(self.n)
        decay = rho ** int(horizon)
        action_scale = 1 - decay
        baseline = decay * estimate + action_scale / (1 - rho) * drift
        A = action_scale * planning_B
        old_action = self.previous_action
        action = project_action(old_action, costs, budget, allowed)
        weight_scale = weights / weights.sum()
        # Shared convex planner; a scalar spectral Lipschitz bound guarantees a
        # valid gradient step without environment-specific optimizer tuning.
        hessian = 2 * (A.T * weight_scale) @ A
        hessian += np.eye(self.n) * (2 * self.config.movement_penalty / self.n)
        lipschitz = max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
        linear = 2 * A.T @ (weight_scale * (baseline - target))
        linear += self.config.action_penalty * costs / self.n
        linear -= 2 * self.config.movement_penalty * old_action / self.n
        for _ in range(self.config.planner_steps):
            gradient = hessian @ action + linear
            action = project_action(action - gradient / lipschitz, costs, budget, allowed)
        raw_predicted_next = rho * estimate + drift + (1 - rho) * self.B @ action
        predicted_next = np.clip(raw_predicted_next, lower, upper)
        planned_next = np.clip(rho * estimate + drift + (1 - rho) * planning_B @ action, lower, upper)
        predicted_horizon = baseline + A @ action
        predictive_variance = self.error_variance * (1 + np.einsum("i,rij,j->r", action, self.covariance, action))
        self._remember_action(action)
        self.step += 1
        self._pending = {
            "state": state.copy(), "observed": observed.copy(), "action": action.copy(),
            "persistence": rho, "drift": drift.copy(), "estimate": estimate.copy(),
            "predicted_next": predicted_next.copy(), "predictive_variance": predictive_variance.copy(),
        }
        self.last_trace = {
            "step": self.step, "config": asdict(self.config), "profile": self.profile,
            "observation": {"state": [float(state[i]) if observed[i] else None for i in range(self.n)],
                            "observed": observed.tolist(), "target": target.tolist(), "weights": weights.tolist(),
                            "costs": costs.tolist(), "allowed": allowed.tolist(), "budget": budget,
                            "persistence": rho, "horizon": int(horizon), "drift": drift.tolist(), "context": context,
                            "state_lower": [float(x) if np.isfinite(x) else None for x in lower],
                            "state_upper": [float(x) if np.isfinite(x) else None for x in upper]},
            "memory_before": None if memory_before is None else memory_before.tolist(),
            "memory_source": memory_source.tolist(), "estimated_state": estimate.tolist(),
            "memory_uncertainty": self.memory_uncertainty.tolist(),
            "previous_action": old_action.tolist(), "action": action.tolist(),
            "signed": self._signed.tolist(), "intensity": self._intensity.tolist(),
            "estimated_matrix": self.B.tolist(), "planning_matrix": planning_B.tolist(),
            "edge_mask": self.edge_mask.tolist(),
            "estimator_covariance_diagonal": np.diagonal(self.covariance, axis1=1, axis2=2).tolist(),
            "error_variance": self.error_variance.tolist(), "predictive_variance": predictive_variance.tolist(),
            "predicted_next_state": predicted_next.tolist(), "planned_next_state": planned_next.tolist(),
            "raw_predicted_next_state": raw_predicted_next.tolist(),
            "predicted_horizon_state": predicted_horizon.tolist(),
            "planning": {"objective": float(np.sum(weight_scale * (predicted_horizon - target) ** 2)
                                               + self.config.action_penalty * (costs @ action) / self.n
                                               + self.config.movement_penalty * np.sum((action - old_action) ** 2) / self.n),
                         "lipschitz": lipschitz, "iterations": self.config.planner_steps,
                         "prediction_type": "affine horizon approximation; one-step state memory respects known bounds",
                         "coupling_enabled": self.coupling_enabled},
            "constraints": {"cost": float(costs @ action), "budget": budget,
                            "blocked_action_max": float(np.max(np.abs(action[~allowed]))) if (~allowed).any() else 0.0,
                            "budget_violation": max(0.0, float(costs @ action) - budget)},
            "interventions": self._interventions, "feedback": None,
        }
        self._interventions = []
        return action.copy()

    def learn(self, feedback):
        if self._pending is None:
            raise RuntimeError("learn requires one unconsumed own action")
        if not isinstance(feedback, dict) or "state" not in feedback or "observed" not in feedback:
            raise ValueError("feedback must explicitly contain state and observed")
        state = self._vector(feedback["state"], "feedback.state")
        observed = self._mask(feedback["observed"], "feedback.observed")
        transition_valid = self._mask(feedback.get("transition_valid", np.ones(self.n, dtype=bool)), "feedback.transition_valid")
        pending = self._pending
        valid = observed & pending["observed"] & transition_valid
        rho, action = pending["persistence"], pending["action"]
        residual = np.full(self.n, np.nan)
        B_before = self.B.copy()
        for row in np.flatnonzero(valid):
            indices = np.flatnonzero(self.edge_mask[row])
            x = action[indices]
            covariance = self.covariance[row][np.ix_(indices, indices)]
            z = (state[row] - rho * pending["state"][row] - pending["drift"][row]) / (1 - rho)
            error = z - float(self.B[row, indices] @ x)
            gain = covariance @ x / (self.config.forgetting + x @ covariance @ x)
            self.B[row, indices] += gain * error
            next_covariance = (covariance - np.outer(gain, x @ covariance)) / self.config.forgetting
            self.covariance[row][np.ix_(indices, indices)] = (next_covariance + next_covariance.T) / 2
            self.error_variance[row] = (self.config.error_retention * self.error_variance[row]
                                        + (1 - self.config.error_retention) * error ** 2)
            self.update_counts[row] += 1
            residual[row] = error
        if not np.isfinite(self.B).all() or not np.isfinite(self.covariance).all() or not np.isfinite(self.error_variance).all():
            raise FloatingPointError("system identification produced nonfinite state")
        self.memory_state = np.where(observed, state, pending["predicted_next"])
        propagated_uncertainty = (rho ** 2 * self.memory_uncertainty
                                  + (1 - rho) ** 2 * pending["predictive_variance"])
        self.memory_uncertainty = np.where(observed, 0, propagated_uncertainty)
        self.last_trace["feedback"] = {
            "state": [float(state[i]) if observed[i] else None for i in range(self.n)],
            "observed": observed.tolist(), "transition_valid": transition_valid.tolist(), "update_valid": valid.tolist(),
            "skipped_rows": np.flatnonzero(~valid).tolist(),
            "skip_reason": "both consecutive state observations and an uncensored affine transition are required for sample pairing",
            "transition_residual": [float(residual[i]) if valid[i] else None for i in range(self.n)],
            "estimated_matrix_before": B_before.tolist(), "estimated_matrix_after": self.B.tolist(),
            "covariance_diagonal_after": np.diagonal(self.covariance, axis1=1, axis2=2).tolist(),
            "error_variance_after": self.error_variance.tolist(),
            "update_counts": self.update_counts.tolist(), "memory_after": self.memory_state.tolist(),
            "memory_uncertainty_after": self.memory_uncertainty.tolist(),
        }
        self._pending = None
        return self.last_trace["feedback"]

    def relabel(self, permutation):
        """Reindex outputs AND actions; caller reindexes all subsequent inputs.

        In particular swapping either pole in a pair changes only its naming.
        The learned operator transforms as P B P^T and RLS covariance on both
        feature axes, rather than changing a sign label alone.
        """
        if self._pending is not None:
            raise RuntimeError("consume pending feedback before relabeling")
        order = np.asarray(permutation)
        if order.shape != (self.n,) or order.dtype.kind not in "iu" or sorted(order.tolist()) != list(range(self.n)):
            raise ValueError("permutation must contain every channel index exactly once")
        self.B = self.B[np.ix_(order, order)]
        self.edge_mask = self.edge_mask[np.ix_(order, order)]
        self.covariance = self.covariance[np.ix_(order, order, order)]
        self.error_variance = self.error_variance[order]
        self.update_counts = self.update_counts[order]
        self.memory_uncertainty = self.memory_uncertainty[order]
        if self.memory_state is not None:
            self.memory_state = self.memory_state[order]
        self._remember_action(self.previous_action[order])
        self._interventions.append({"operation": "channel_relabel", "permutation": order.tolist()})


# A descriptive alias accommodates study clients without a second config type.
ModelConfig = ControllerConfig
