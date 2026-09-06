"""Auditable inter-polarity extension; not a replacement for frozen studies.

Rows are receiving channels, columns are sending channels/polarities. State
shape remains (control units, polarity types, 2). Units are not autonomous agents.
Only declared contextual inputs enter tension; hidden evaluator truth is unused.
"""
from __future__ import annotations
import numpy as np
from polar.model import ContextualPolarModel, ModelConfig


def _finite(value, shape, name):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out.copy()


class NetworkTensionModel(ContextualPolarModel):
    """Explicit state and tension routes, followed by unchanged safety projection.

    The two independent channel activities preserve both poles and coactivation.
    W maps channel activity; K maps an operational tension proxy. These are
    configured hypotheses, not learned psychological relationships. Zero W/K
    reproduces the contextual controller. No frozen study source is modified.
    """
    def __init__(self, config=None, *, state_coupling=None,
                 tension_coupling=None, incompatibility=None):
        super().__init__(config or ModelConfig())
        self.pairs = self.config.agents * self.config.types
        self.channels = 2 * self.pairs
        self._network_trace = {}
        self.set_couplings(state_coupling, tension_coupling)
        self.set_incompatibility(incompatibility)

    def set_couplings(self, state_coupling=None, tension_coupling=None):
        if self._pending_feedback:
            raise RuntimeError("consume pending feedback before changing couplings")
        m, n = self.channels, self.pairs
        W = _finite(np.zeros((m, m)) if state_coupling is None else state_coupling,
                    (m, m), "state_coupling")
        K = _finite(np.zeros((m, n)) if tension_coupling is None else tension_coupling,
                    (m, n), "tension_coupling")
        # This extension isolates BETWEEN-polarity routes, not within-pair terms.
        owner = np.arange(m) // 2
        if np.any(W[owner[:, None] == owner[None, :]] != 0):
            raise ValueError("state_coupling must have zero within-polarity blocks")
        if np.any(K[np.arange(m), owner] != 0):
            raise ValueError("tension_coupling must not target its own polarity")
        self.W, self.K = W, K

    def set_incompatibility(self, value=None):
        if self._pending_feedback:
            raise RuntimeError("consume pending feedback before changing context")
        shape = self.shape[:-1]
        chi = _finite(np.zeros(shape) if value is None else value, shape, "incompatibility")
        if np.any((chi < 0) | (chi > 1)):
            raise ValueError("incompatibility must lie in [0,1]")
        self.chi = chi

    def propose_action(self, effective_target, weights, horizon):
        before = self.q
        desired = np.clip(effective_target / self._planning_gain, 0, 1)
        mismatch = np.mean(np.abs(desired - before), axis=-1)
        conflict = self.chi * before[..., 0] * before[..., 1]
        tension = mismatch + conflict
        # Both contractions consume PRE-update state; no accidental sequential update.
        state_edges = self.W * before.reshape(1, -1)
        tension_edges = self.K * tension.reshape(1, -1)
        state_term = state_edges.sum(axis=1).reshape(self.shape)
        tension_term = tension_edges.sum(axis=1).reshape(self.shape)
        eta = np.clip(self.config.step_size * horizon, 0, 1)
        base = super().propose_action(effective_target, weights, horizon)
        with np.errstate(over="raise", invalid="raise"):
            proposal = base + eta * (state_term + tension_term)
        if not np.isfinite(proposal).all():
            raise ValueError("nonfinite network proposal")
        self._network_trace = {
            "version": "network-tension-0.1", "evidence_status": "engineering_extension",
            "axis_order": "unit,type,pole; rows=recipient; columns=source",
            "p_before": before.tolist(), "desired_effort": desired.tolist(),
            "incompatibility": self.chi.tolist(), "mismatch": mismatch.tolist(),
            "conflict": conflict.tolist(), "tension": tension.tolist(),
            "state_coupling": self.W.tolist(), "tension_coupling": self.K.tolist(),
            "state_edges": state_edges.tolist(), "tension_edges": tension_edges.tolist(),
            "state_term": state_term.tolist(), "tension_term": tension_term.tolist(),
            "eta": np.broadcast_to(eta, self.shape).tolist(),
            "base_proposal": base.tolist(), "network_proposal": proposal.tolist(),
        }
        return proposal

    def act(self, observation):
        action = super().act(observation)
        # Inherited last_trace is the same object stored in traces: feedback joins it.
        self.last_trace["network"] = self._network_trace
        return action

    def reverse_convention(self, types=None):
        """Relabel poles AND both graph axes, keeping physical behavior equivalent."""
        if self._pending_feedback:
            raise RuntimeError("consume pending feedback before relabeling")
        types = list(range(self.config.types)) if types is None else list(types)
        if any(not isinstance(t, (int, np.integer)) or t < 0 or t >= self.config.types for t in types):
            raise ValueError("invalid polarity type")
        if len(set(types)) != len(types):
            raise ValueError("duplicate polarity types")
        order = np.arange(self.channels).reshape(self.shape)
        order[:, types, :] = order[:, types, ::-1].copy()
        flat = order.ravel()
        super().reverse_convention(types=types)
        self.W = self.W[np.ix_(flat, flat)]
        self.K = self.K[flat, :]
