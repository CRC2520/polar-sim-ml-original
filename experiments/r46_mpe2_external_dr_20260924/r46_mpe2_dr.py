#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

MPE2_COMMIT = "7590d9d52791e321974d4fda6090fb18f34dbf49"
DEV_SEEDS = list(range(2176001, 2176017))
REF_CONFIRM_SEEDS = list(range(2177001, 2177033))
CRYPTO_CONFIRM_SEEDS = list(range(2177101, 2177133))
SMOKE_SEED = 2175000

REF_EPISODES = 6
CRYPTO_EPISODES = 12
MAX_CYCLES = 25

REF_CONDITIONS = (
    "FULL_DR",
    "NO_D",
    "WRONG_R",
    "FROZEN_PREVIOUS_R",
    "GENERIC_ISO",
    "STATIONARY",
    "ORACLE_RELATION",
)
CRYPTO_CONDITIONS = (
    "FULL_DR",
    "NO_D",
    "NO_R_DIRECT",
    "FROZEN_PREVIOUS_R",
    "GENERIC_ISO",
)

REF_MEDIAN_GATES = {
    "full_score": 0.70,
    "d_effect": 0.20,
    "r_binding_effect": 0.20,
    "r_rebind_effect": 0.10,
    "relation_change_fraction": 0.50,
}
REF_SEED_GATES = {
    "full_score": 0.60,
    "d_effect": 0.10,
    "r_binding_effect": 0.10,
    "r_rebind_effect": 0.05,
    "relation_change_fraction": 0.40,
}
ISO_TOL = 1e-12

CRYPTO_MEDIAN_GATES = {
    "full_reward": 0.65,
    "d_effect": 0.40,
    "r_direct_effect": 0.40,
    "r_rebind_effect": 0.30,
}
CRYPTO_SEED_GATES = {
    "full_reward": 0.40,
    "d_effect": 0.20,
    "r_direct_effect": 0.20,
    "r_rebind_effect": 0.20,
}

REF_ISO_PERM = np.array([2, 0, 1], dtype=int)
REF_ISO_INV = np.argsort(REF_ISO_PERM)
BIT_ISO_PERM = np.array([1, 0], dtype=int)
BIT_ISO_INV = np.argsort(BIT_ISO_PERM)


def _median(rows: List[dict], key: str) -> float:
    return float(np.median([float(r[key]) for r in rows]))


def _argmax_active(x: np.ndarray, active: int) -> int:
    x = np.asarray(x, dtype=float)[:active]
    return int(np.argmax(x))


def _decode_comm(comm: np.ndarray, active: int) -> Optional[int]:
    c = np.asarray(comm, dtype=float)
    if c.size == 0 or float(np.max(np.abs(c))) <= 1e-9:
        return None
    return int(np.argmax(c[:active]))


def _move_toward(rel: np.ndarray, vel: np.ndarray) -> int:
    rel = np.asarray(rel, dtype=float)
    vel = np.asarray(vel, dtype=float)
    if float(np.linalg.norm(rel)) <= 0.055 and float(np.linalg.norm(vel)) <= 0.08:
        return 0
    drive = rel - 0.30 * vel
    if abs(float(drive[0])) >= abs(float(drive[1])):
        return 2 if drive[0] >= 0 else 1
    return 4 if drive[1] >= 0 else 3


def _ref_parse(obs: np.ndarray) -> dict:
    o = np.asarray(obs, dtype=float)
    if o.shape != (21,):
        raise RuntimeError(f"SimpleReference observation contract changed: {o.shape}")
    return {
        "vel": o[0:2],
        "landmarks": o[2:8].reshape(3, 2),
        "outgoing_goal": _argmax_active(o[8:11], 3),
        "incoming": _decode_comm(o[11:21], 10),
    }


def _ref_partner(agent: str) -> str:
    return "agent_1" if agent == "agent_0" else "agent_0"


def _ref_iso_vector(positions: np.ndarray, native_target: int) -> np.ndarray:
    latent_positions = np.zeros_like(positions)
    for native in range(3):
        latent_positions[REF_ISO_PERM[native]] = positions[native]
    latent_target = int(REF_ISO_PERM[native_target])
    return latent_positions[latent_target]


def _ref_iso_message(native_goal: int) -> int:
    latent = int(REF_ISO_PERM[native_goal])
    return int(REF_ISO_INV[latent])


def _make_ref_env():
    from mpe2 import simple_reference_v3

    return simple_reference_v3.parallel_env(
        local_ratio=0.5,
        max_cycles=MAX_CYCLES,
        continuous_actions=False,
        dynamic_rescaling=False,
    )


def _ref_episode(
    condition: str,
    episode_seed: int,
    stale_cache: Optional[Dict[str, int]] = None,
    warmup: bool = False,
) -> dict:
    env = _make_ref_env()
    obs, _ = env.reset(seed=int(episode_seed))
    if set(obs) != {"agent_0", "agent_1"}:
        raise RuntimeError(f"SimpleReference agents changed: {sorted(obs)}")
    for a in obs:
        if env.action_space(a).n != 50:
            raise RuntimeError("SimpleReference discrete action contract changed")

    parsed0 = {a: _ref_parse(o) for a, o in obs.items()}
    evaluator_targets = {
        a: parsed0[_ref_partner(a)]["outgoing_goal"] for a in parsed0
    }

    rewards_trace: List[float] = []
    last_seen: Dict[str, Optional[int]] = {a: None for a in obs}

    while env.agents:
        parsed = {a: _ref_parse(obs[a]) for a in obs}
        actions = {}
        for a, p in parsed.items():
            outgoing = p["outgoing_goal"]
            if condition == "GENERIC_ISO":
                msg = _ref_iso_message(outgoing)
            else:
                msg = outgoing

            incoming = p["incoming"]
            if incoming is not None:
                last_seen[a] = incoming

            if condition == "STATIONARY":
                move = 0
            else:
                if condition == "ORACLE_RELATION":
                    target = evaluator_targets[a]
                elif condition == "FROZEN_PREVIOUS_R" and not warmup:
                    if stale_cache is None or a not in stale_cache:
                        raise RuntimeError("missing frozen relation cache")
                    target = int(stale_cache[a])
                elif incoming is None:
                    target = None
                elif condition == "WRONG_R":
                    target = (int(incoming) + 1) % 3
                else:
                    target = int(incoming)

                if target is None:
                    move = 0
                else:
                    positions = p["landmarks"]
                    if condition == "NO_D":
                        mean_pos = positions.mean(axis=0)
                        positions = np.repeat(mean_pos[None, :], 3, axis=0)
                    if condition == "GENERIC_ISO":
                        rel = _ref_iso_vector(positions, target)
                    else:
                        rel = positions[target]
                    move = _move_toward(rel, p["vel"])

            actions[a] = int(move + 5 * msg)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        rewards_trace.append(float(np.mean(list(rewards.values()))))
        if not obs:
            break

    env.close()
    new_cache = {
        a: int(last_seen[a]) if last_seen[a] is not None else int(evaluator_targets[a])
        for a in evaluator_targets
    }
    return {
        "cost": float(-np.mean(rewards_trace)),
        "mean_reward": float(np.mean(rewards_trace)),
        "relation_targets": {k: int(v) for k, v in evaluator_targets.items()},
        "new_cache": new_cache,
    }


def _ref_sequence(condition: str, seed: int) -> dict:
    costs = []
    stale_cache: Optional[Dict[str, int]] = None
    changed = 0
    change_total = 0
    previous_targets: Optional[Dict[str, int]] = None
    episodes = []

    for ep in range(REF_EPISODES):
        ep_seed = int(seed) * 100 + ep
        result = _ref_episode(
            condition,
            ep_seed,
            stale_cache=stale_cache,
            warmup=(ep == 0 and condition == "FROZEN_PREVIOUS_R"),
        )

        current_targets = result["relation_targets"]
        if ep > 0 and previous_targets is not None:
            for a in sorted(current_targets):
                change_total += 1
                if int(current_targets[a]) != int(previous_targets[a]):
                    changed += 1

        if condition == "FROZEN_PREVIOUS_R":
            stale_cache = dict(result["new_cache"])

        previous_targets = dict(current_targets)
        if ep > 0:
            costs.append(float(result["cost"]))
        episodes.append(
            {
                "episode": ep,
                "seed": ep_seed,
                "cost": result["cost"],
                "targets": current_targets,
            }
        )

    return {
        "cost": float(np.mean(costs)),
        "relation_change_fraction": float(changed / change_total) if change_total else 0.0,
        "episodes": episodes,
    }


def evaluate_reference_seed(seed: int) -> dict:
    cond = {name: _ref_sequence(name, seed) for name in REF_CONDITIONS}
    stationary = cond["STATIONARY"]["cost"]
    oracle = cond["ORACLE_RELATION"]["cost"]
    denom = stationary - oracle
    if denom <= 1e-8:
        return {
            "seed": int(seed),
            "valid": False,
            "reason": f"normalization denominator {denom}",
            "conditions": cond,
        }

    scores = {
        name: float((stationary - cond[name]["cost"]) / denom)
        for name in ("FULL_DR", "NO_D", "WRONG_R", "FROZEN_PREVIOUS_R", "GENERIC_ISO")
    }
    row = {
        "seed": int(seed),
        "valid": True,
        "normalization_denominator": float(denom),
        "full_score": scores["FULL_DR"],
        "no_d_score": scores["NO_D"],
        "wrong_r_score": scores["WRONG_R"],
        "frozen_r_score": scores["FROZEN_PREVIOUS_R"],
        "generic_iso_score": scores["GENERIC_ISO"],
        "d_effect": scores["FULL_DR"] - scores["NO_D"],
        "r_binding_effect": scores["FULL_DR"] - scores["WRONG_R"],
        "r_rebind_effect": scores["FULL_DR"] - scores["FROZEN_PREVIOUS_R"],
        "iso_gap": abs(scores["FULL_DR"] - scores["GENERIC_ISO"]),
        "relation_change_fraction": cond["FROZEN_PREVIOUS_R"]["relation_change_fraction"],
        "costs": {k: float(v["cost"]) for k, v in cond.items()},
    }
    row["seed_guard"] = bool(
        row["full_score"] >= REF_SEED_GATES["full_score"]
        and row["d_effect"] >= REF_SEED_GATES["d_effect"]
        and row["r_binding_effect"] >= REF_SEED_GATES["r_binding_effect"]
        and row["r_rebind_effect"] >= REF_SEED_GATES["r_rebind_effect"]
        and row["iso_gap"] <= ISO_TOL
        and row["relation_change_fraction"] >= REF_SEED_GATES["relation_change_fraction"]
    )
    return row


def summarize_reference(rows: List[dict], required_seed_guards: int) -> dict:
    if any(not r.get("valid", False) for r in rows):
        return {
            "pass": False,
            "invalid_seed_count": sum(not r.get("valid", False) for r in rows),
            "seed_guard_count": 0,
            "seed_guard_required": required_seed_guards,
        }

    out = {
        "n": len(rows),
        "median_full_score": _median(rows, "full_score"),
        "median_d_effect": _median(rows, "d_effect"),
        "median_r_binding_effect": _median(rows, "r_binding_effect"),
        "median_r_rebind_effect": _median(rows, "r_rebind_effect"),
        "median_iso_gap": _median(rows, "iso_gap"),
        "median_relation_change_fraction": _median(rows, "relation_change_fraction"),
        "seed_guard_count": sum(bool(r["seed_guard"]) for r in rows),
        "seed_guard_required": required_seed_guards,
    }
    out["pass"] = bool(
        out["median_full_score"] >= REF_MEDIAN_GATES["full_score"]
        and out["median_d_effect"] >= REF_MEDIAN_GATES["d_effect"]
        and out["median_r_binding_effect"] >= REF_MEDIAN_GATES["r_binding_effect"]
        and out["median_r_rebind_effect"] >= REF_MEDIAN_GATES["r_rebind_effect"]
        and out["median_iso_gap"] <= ISO_TOL
        and out["median_relation_change_fraction"] >= REF_MEDIAN_GATES["relation_change_fraction"]
        and out["seed_guard_count"] >= required_seed_guards
    )
    return out


def _crypto_parse(obs: Dict[str, np.ndarray]) -> dict:
    expected = {"eve_0": (4,), "bob_0": (8,), "alice_0": (8,)}
    for a, shape in expected.items():
        if np.asarray(obs[a]).shape != shape:
            raise RuntimeError(f"SimpleCrypto observation contract changed for {a}: {np.asarray(obs[a]).shape}")

    alice = np.asarray(obs["alice_0"], dtype=float)
    bob = np.asarray(obs["bob_0"], dtype=float)
    eve = np.asarray(obs["eve_0"], dtype=float)
    return {
        "goal": _argmax_active(alice[:4], 2),
        "alice_key": _argmax_active(alice[4:8], 2),
        "bob_key": _argmax_active(bob[:4], 2),
        "bob_comm": _decode_comm(bob[4:8], 4),
        "eve_comm": _decode_comm(eve[:4], 4),
    }


def _bit_iso_full(goal: int, key: int) -> int:
    lg = int(BIT_ISO_PERM[goal])
    lk = int(BIT_ISO_PERM[key])
    native_g = int(BIT_ISO_INV[lg])
    native_k = int(BIT_ISO_INV[lk])
    native_cipher = native_g ^ native_k
    latent_cipher = int(BIT_ISO_PERM[native_cipher])
    return int(BIT_ISO_INV[latent_cipher])


def _bit_iso_decode(cipher: int, key: int) -> int:
    lc = int(BIT_ISO_PERM[cipher])
    lk = int(BIT_ISO_PERM[key])
    native_c = int(BIT_ISO_INV[lc])
    native_k = int(BIT_ISO_INV[lk])
    native_goal = native_c ^ native_k
    latent_goal = int(BIT_ISO_PERM[native_goal])
    return int(BIT_ISO_INV[latent_goal])


def _make_crypto_env():
    from mpe2 import simple_crypto_v3

    return simple_crypto_v3.parallel_env(
        max_cycles=MAX_CYCLES,
        continuous_actions=False,
        dynamic_rescaling=False,
    )


def _crypto_episode(
    condition: str,
    episode_seed: int,
    stale_key: Optional[int] = None,
    warmup: bool = False,
) -> dict:
    env = _make_crypto_env()
    obs, _ = env.reset(seed=int(episode_seed))
    if set(obs) != {"eve_0", "bob_0", "alice_0"}:
        raise RuntimeError(f"SimpleCrypto agents changed: {sorted(obs)}")
    for a in obs:
        if env.action_space(a).n != 4:
            raise RuntimeError("SimpleCrypto action contract changed")

    initial = _crypto_parse(obs)
    current_key = int(initial["alice_key"])
    good_rewards: List[float] = []

    while env.agents:
        p = _crypto_parse(obs)
        goal = int(p["goal"])
        key = int(p["alice_key"])

        if condition == "NO_D":
            alice_vec = np.asarray(obs["alice_0"], dtype=float)
            merged = 0.5 * (alice_vec[:4] + alice_vec[4:8])
            merged_symbol = _argmax_active(merged, 2)
            alice_action = int(merged_symbol ^ merged_symbol)
        elif condition == "NO_R_DIRECT":
            alice_action = goal
        elif condition == "FROZEN_PREVIOUS_R" and not warmup:
            if stale_key is None:
                raise RuntimeError("missing stale crypto key")
            alice_action = int(goal ^ int(stale_key))
        elif condition == "GENERIC_ISO":
            alice_action = _bit_iso_full(goal, key)
        else:
            alice_action = int(goal ^ key)

        bob_comm = 0 if p["bob_comm"] is None else int(p["bob_comm"])
        eve_comm = 0 if p["eve_comm"] is None else int(p["eve_comm"])

        if condition == "NO_R_DIRECT":
            bob_action = bob_comm
        elif condition == "GENERIC_ISO":
            bob_action = _bit_iso_decode(bob_comm, int(p["bob_key"]))
        else:
            bob_action = int(bob_comm ^ int(p["bob_key"]))

        eve_action = eve_comm
        actions = {
            "alice_0": int(alice_action),
            "bob_0": int(bob_action),
            "eve_0": int(eve_action),
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        good_rewards.append(float(0.5 * (rewards["alice_0"] + rewards["bob_0"])))
        if not obs:
            break

    env.close()
    return {
        "good_reward": float(np.mean(good_rewards)),
        "current_key": current_key,
    }


def _crypto_sequence(condition: str, seed: int) -> dict:
    rewards = []
    stale_key: Optional[int] = None
    changed = 0
    total = 0
    previous_key: Optional[int] = None

    for ep in range(CRYPTO_EPISODES):
        ep_seed = int(seed) * 100 + ep
        result = _crypto_episode(
            condition,
            ep_seed,
            stale_key=stale_key,
            warmup=(ep == 0 and condition == "FROZEN_PREVIOUS_R"),
        )
        current_key = int(result["current_key"])
        if ep > 0 and previous_key is not None:
            total += 1
            if current_key != previous_key:
                changed += 1
        if condition == "FROZEN_PREVIOUS_R":
            stale_key = current_key
        previous_key = current_key
        if ep > 0:
            rewards.append(float(result["good_reward"]))

    return {
        "reward": float(np.mean(rewards)),
        "key_change_fraction": float(changed / total) if total else 0.0,
    }


def evaluate_crypto_seed(seed: int) -> dict:
    cond = {name: _crypto_sequence(name, seed) for name in CRYPTO_CONDITIONS}
    row = {
        "seed": int(seed),
        "full_reward": cond["FULL_DR"]["reward"],
        "no_d_reward": cond["NO_D"]["reward"],
        "no_r_direct_reward": cond["NO_R_DIRECT"]["reward"],
        "frozen_r_reward": cond["FROZEN_PREVIOUS_R"]["reward"],
        "generic_iso_reward": cond["GENERIC_ISO"]["reward"],
        "d_effect": cond["FULL_DR"]["reward"] - cond["NO_D"]["reward"],
        "r_direct_effect": cond["FULL_DR"]["reward"] - cond["NO_R_DIRECT"]["reward"],
        "r_rebind_effect": cond["FULL_DR"]["reward"] - cond["FROZEN_PREVIOUS_R"]["reward"],
        "iso_gap": abs(cond["FULL_DR"]["reward"] - cond["GENERIC_ISO"]["reward"]),
        "key_change_fraction": cond["FROZEN_PREVIOUS_R"]["key_change_fraction"],
    }
    row["seed_guard"] = bool(
        row["full_reward"] >= CRYPTO_SEED_GATES["full_reward"]
        and row["d_effect"] >= CRYPTO_SEED_GATES["d_effect"]
        and row["r_direct_effect"] >= CRYPTO_SEED_GATES["r_direct_effect"]
        and row["r_rebind_effect"] >= CRYPTO_SEED_GATES["r_rebind_effect"]
        and row["iso_gap"] <= ISO_TOL
    )
    return row


def summarize_crypto(rows: List[dict]) -> dict:
    out = {
        "n": len(rows),
        "median_full_reward": _median(rows, "full_reward"),
        "median_d_effect": _median(rows, "d_effect"),
        "median_r_direct_effect": _median(rows, "r_direct_effect"),
        "median_r_rebind_effect": _median(rows, "r_rebind_effect"),
        "median_iso_gap": _median(rows, "iso_gap"),
        "median_key_change_fraction": _median(rows, "key_change_fraction"),
        "seed_guard_count": sum(bool(r["seed_guard"]) for r in rows),
        "seed_guard_required": 26,
    }
    out["pass"] = bool(
        out["median_full_reward"] >= CRYPTO_MEDIAN_GATES["full_reward"]
        and out["median_d_effect"] >= CRYPTO_MEDIAN_GATES["d_effect"]
        and out["median_r_direct_effect"] >= CRYPTO_MEDIAN_GATES["r_direct_effect"]
        and out["median_r_rebind_effect"] >= CRYPTO_MEDIAN_GATES["r_rebind_effect"]
        and out["median_iso_gap"] <= ISO_TOL
        and out["seed_guard_count"] >= 26
    )
    return out


def source_metadata() -> dict:
    try:
        version = importlib.metadata.version("mpe2")
    except importlib.metadata.PackageNotFoundError:
        version = "UNKNOWN"
    return {
        "package": "Farama-Foundation/MPE2",
        "release": "v1.1.0",
        "commit": MPE2_COMMIT,
        "installed_version": version,
        "environments": ["simple_reference_v3", "simple_crypto_v3"],
    }


def run_smoke(output: str) -> None:
    ref = evaluate_reference_seed(SMOKE_SEED)
    crypto = evaluate_crypto_seed(SMOKE_SEED)
    out = {
        "campaign": "R46 external D/R causal transport",
        "mode": "engineering_smoke",
        "seed": SMOKE_SEED,
        "source": source_metadata(),
        "reference_valid": bool(ref.get("valid", False)),
        "reference_iso_gap": ref.get("iso_gap"),
        "crypto_iso_gap": crypto.get("iso_gap"),
        "scientific_seed": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


def run_development(output: str) -> None:
    rows = [evaluate_reference_seed(s) for s in DEV_SEEDS]
    summary = summarize_reference(rows, required_seed_guards=13)
    resolution = (
        "R46_DEVELOPMENT_AUTHORIZE_CONFIRM"
        if summary["pass"]
        else "R46_REFERENCE_DEV_FAIL_NO_CONFIRM"
    )
    out = {
        "campaign": "R46 external D/R causal transport",
        "mode": "development",
        "resolution": resolution,
        "authorize_confirm": bool(summary["pass"]),
        "source": source_metadata(),
        "seeds": DEV_SEEDS,
        "protocol": {
            "reference_episodes_per_seed": REF_EPISODES,
            "evaluation_episodes_per_seed": REF_EPISODES - 1,
            "median_gates": REF_MEDIAN_GATES,
            "seed_gates": REF_SEED_GATES,
            "seed_guard_required": 13,
            "iso_tolerance": ISO_TOL,
        },
        "summary": summary,
        "records": rows,
        "boundaries": {
            "external_R_learnability": "NOT_TESTED",
            "single_task_external_DCR_conjunction": "NOT_ESTABLISHED",
            "E6b": "OPEN",
            "E7": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
            "core_version": "POLAR Core v1.1 unchanged",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2, sort_keys=True))


def run_confirmatory(output: str) -> None:
    ref_rows = [evaluate_reference_seed(s) for s in REF_CONFIRM_SEEDS]
    crypto_rows = [evaluate_crypto_seed(s) for s in CRYPTO_CONFIRM_SEEDS]
    ref_summary = summarize_reference(ref_rows, required_seed_guards=28)
    crypto_summary = summarize_crypto(crypto_rows)

    if ref_summary["pass"] and crypto_summary["pass"]:
        resolution = "R46_EXTERNAL_D_R_TRANSPORT_PASS_SAME_PROGRAM"
    elif ref_summary["pass"] or crypto_summary["pass"]:
        resolution = "R46_EXTERNAL_D_R_TRANSPORT_PARTIAL"
    else:
        resolution = "R46_EXTERNAL_D_R_TRANSPORT_FAIL"

    out = {
        "campaign": "R46 external D/R causal transport",
        "mode": "confirmatory",
        "resolution": resolution,
        "source": source_metadata(),
        "reference": {
            "seeds": REF_CONFIRM_SEEDS,
            "summary": ref_summary,
            "records": ref_rows,
        },
        "crypto": {
            "seeds": CRYPTO_CONFIRM_SEEDS,
            "summary": crypto_summary,
            "records": crypto_rows,
        },
        "boundaries": {
            "external_D_functional_necessity": "SUPPORTED_IF_REFERENCE_AND_CRYPTO_PASS",
            "external_R_binding_reidentification": "SUPPORTED_IF_REFERENCE_AND_CRYPTO_PASS",
            "external_R_learnability": "NOT_TESTED",
            "single_task_external_DCR_conjunction": "NOT_ESTABLISHED",
            "global_minimality": "OPEN",
            "E6b": "OPEN",
            "E7": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
            "core_version": "POLAR Core v1.1 unchanged",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "resolution": resolution,
                "reference_summary": ref_summary,
                "crypto_summary": crypto_summary,
                "boundaries": out["boundaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "development", "confirmatory"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.mode == "smoke":
        run_smoke(a.output)
    elif a.mode == "development":
        run_development(a.output)
    else:
        run_confirmatory(a.output)


if __name__ == "__main__":
    main()
