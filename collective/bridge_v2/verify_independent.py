"""Independent algebraic/causal checks; no access to final outcome selection.

These tests were written by a reviewer separate from the engine implementer.
They verify the implementation contract, not scientific superiority.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np

from .engine import BridgeConfig, BridgeEngine


def verify():
    checks = {}
    cfg = BridgeConfig(agents=6, groups=2, generations=6, steps=32,
                       tail_generations=2, replace_groups=1)
    seeds = [867001, 867002]
    with tempfile.TemporaryDirectory() as tmp:
        pool = BridgeEngine(cfg, seeds, .5, study="A", variant="resource_pool")
        target = Path(tmp) / "pool.npz"
        out = pool.episode(0, trace_path=target, trace_mode="full")
        with np.load(target, allow_pickle=False) as z:
            np.testing.assert_allclose(z["transfer"].sum(-1), 0, atol=1e-12)
            np.testing.assert_allclose(z["income"].sum(-1), z["take"].sum(-1), atol=1e-12)
            expected_vitality = np.clip(z["vitality_before"] +
                np.where(z["alive_before"], z["income"] - cfg.metabolism, 0),
                -1, cfg.max_vitality)
            np.testing.assert_array_equal(z["vitality_after"], expected_vitality)
            expected_alive = z["alive_before"] & (expected_vitality > 0)
            np.testing.assert_array_equal(z["alive_after"], expected_alive)
            history = z["initial_resource_history"].copy()
            for t in range(cfg.steps):
                previous = z["resource_before"][:, t]
                extracted = z["take"][:, t].sum(-1)
                lagged = history[..., 0]
                expected = np.maximum(previous - extracted +
                    z["growth"][:, t] * lagged * (1 - lagged / pool.capacity), 0)
                np.testing.assert_array_equal(z["resource_after"][:, t], expected)
                history = np.concatenate((history[..., 1:], previous[..., None]), axis=-1)
            np.testing.assert_allclose(out["alive_fraction"], z["alive_after"].mean((1, 2, 3)))
            np.testing.assert_allclose(out["resource_fraction"], z["resource_after"].mean((1, 2)) / pool.capacity)
        checks["pooling_conserves_income_and_replays_physical_ledger"] = True

        ordinary = BridgeEngine(cfg, seeds, .5, study="A", variant="base")
        signal = BridgeEngine(cfg, seeds, .5, study="A", variant="copy_score_equal")
        ordinary_out = ordinary.episode(0)
        signal_out = signal.episode(0)
        for key in ordinary.state:
            np.testing.assert_array_equal(ordinary.state[key], signal.state[key])
        for key in ordinary_out:
            np.testing.assert_array_equal(ordinary_out[key], signal_out[key])
        # A synthetic fitness intervention checks the copying-score operator
        # without selecting an observed performance contrast.
        signal.last_episode["alive_agent_time"] = np.where(signal.state["lineage"], .25, .75)
        events = signal.evolve(0)
        np.testing.assert_allclose(events["copy_score"], .5, atol=1e-14)
        checks["copy_signal_does_not_change_pre_transmission_physics"] = True

        full = BridgeEngine(cfg, seeds, .5, study="C", variant="full")
        recoded = BridgeEngine(cfg, seeds, .5, study="C", variant="coordinate_equivalent")
        # Nonzero random critic states avoid testing only an all-zero triviality.
        rng = np.random.default_rng(867101)
        for key in ("qt", "qv", "qg"):
            values = rng.normal(size=full.state[key].shape)
            full.state[key] = values.copy()
            recoded.state[key] = values.copy()
        obs, _ = full.observe(np.zeros((len(seeds), cfg.groups)))
        np.testing.assert_allclose(full.decision_scores(obs), recoded.decision_scores(obs), atol=1e-14)
        full.episode(1, trace_path=Path(tmp) / "full.npz", trace_mode="full")
        recoded.episode(1, trace_path=Path(tmp) / "recoded.npz", trace_mode="full")
        with np.load(Path(tmp) / "full.npz") as a, np.load(Path(tmp) / "recoded.npz") as b:
            np.testing.assert_array_equal(a["action"], b["action"])
        checks["invertible_critic_coordinates_preserve_scores_and_actions"] = True

        original = BridgeEngine(cfg, seeds, .5, study="C", variant="full")
        relabeled = original.clone()
        relabeled.state["lineage"] = 1 - relabeled.state["lineage"]
        one, two = original.episode(2), relabeled.episode(2)
        # Whole-state fingerprints deliberately include the altered inert labels.
        # Their inequality is required for provenance, not a behavioral failure.
        state_fingerprints = ("initial_state_sha256", "final_state_sha256")
        for key in state_fingerprints:
            if np.any(one[key] == two[key]):
                raise AssertionError(f"Relabeling was not recorded in {key}")
        for key in one:
            if key not in ("lineage_fraction", *state_fingerprints):
                np.testing.assert_array_equal(one[key], two[key])
        for key in original.state:
            if key != "lineage":
                np.testing.assert_array_equal(original.state[key], relabeled.state[key])
        checks["C_ancestry_is_not_a_hidden_action_factor"] = True

        batched = BridgeEngine(cfg, seeds, .5, study="C", variant="full")
        single = BridgeEngine(cfg, seeds[:1], .5, study="C", variant="full")
        batched.episode(3)
        single.episode(3)
        for key in batched.state:
            np.testing.assert_array_equal(batched.state[key][:1], single.state[key])
        checks["seed_trajectory_is_invariant_to_batch_size"] = True

        off = BridgeEngine(cfg, seeds, .5, study="C", variant="qv_policy_off")
        altered = off.clone()
        altered.state["qv"][:] = rng.normal(size=altered.state["qv"].shape) * 100
        obs, _ = off.observe(np.zeros((len(seeds), cfg.groups)))
        np.testing.assert_array_equal(off.decision_scores(obs), altered.decision_scores(obs))
        checks["policy_ablation_removes_only_Qv_decision_contribution"] = True
    return {"review_status": "independent_contract_checks_passed", "checks": checks,
            "n_checks": len(checks), "seed_namespace": seeds,
            "scope": "implementation invariants; not outcome confirmation or external replication"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
