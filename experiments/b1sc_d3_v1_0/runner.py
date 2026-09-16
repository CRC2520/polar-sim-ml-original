"""B1-SC-D3 v1.0 scientific runner, gated by immutable initialization preflight.

This file installs the runner but does NOT authorize D3 execution.  Scientific
execution additionally requires a separate immutable START_REQUEST.json.  The
current repository intentionally contains no such authorization.

Ordering invariant for every fit:
    full preflight -> START_REQUEST authorization -> block permit -> load exact
    validated bytes -> optimizer construction -> rollout -> backward -> step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

from b1s.execution.core import log_squashed_gaussian, write_json
from b1s.execution.instrument import NativeEnv, runtime_lock
from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0.execution_preflight import (
    IMMUTABLE_BLOCKS,
    PreflightError,
    SnapshotPermit,
    create_optimizer_after_preflight,
    full_preflight,
    load_validated_models,
)
from experiments.b1sc_d3_v1_0.microfit_qa import qa_suite as microfit_qa_suite

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "FIT_REGISTRY.json"
START_REQUEST_PATH = ROOT / "START_REQUEST.json"
DESIGN_PATH = ROOT / "DESIGN_FREEZE.json"
INIT_FREEZE_PATH = ROOT / "INIT_FREEZE.json"
PROTOCOL_PATH = ROOT / "PROTOCOL_ES.md"

# D3 retains the historical PPO optimizer/update contract; D3 changes actor
# information availability, not the optimization family.  These values are
# pinned here so a later change in the historical B1-S PLAN cannot alter D3.
PPO = {
    "lr": 3e-4,
    "gamma": 0.99,
    "rollout_steps": 512,
    "minibatch": 128,
    "epochs": 4,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "adam_eps": 1e-5,
}

CONDITION_MAP = {
    "GLOBAL-G0": (d3.G0, d3.GLOBAL),
    "GLOBAL-S6": (d3.S6, d3.GLOBAL),
    "LOCAL-G0": (d3.G0, d3.LOCAL),
    "LOCAL-S6": (d3.S6, d3.LOCAL),
}


class AuthorizationError(RuntimeError):
    """D3 execution is structurally ready but not explicitly authorized."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child_seed(root: int, namespace: str, index: int) -> int:
    """Prospectively fixed expansion of a frozen D3 panel/environment seed root."""
    raw = f"B1-SC-D3-v1|{int(root)}|{namespace}|{int(index)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def registry() -> list[dict]:
    rows = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if len(rows) != d3.FITS or len({row["id"] for row in rows}) != d3.FITS:
        raise RuntimeError("D3 FIT_REGISTRY must contain exactly 32 unique fits")
    expected_conditions = set(CONDITION_MAP)
    if {row["condition"] for row in rows} != expected_conditions:
        raise RuntimeError("D3 FIT_REGISTRY condition set changed")
    for row in rows:
        if row["block"] not in range(d3.BLOCKS):
            raise RuntimeError(f"Invalid D3 block in registry: {row}")
        if row["initial_snapshot_block"] != row["block"]:
            raise RuntimeError(f"D3 fit does not point to its paired block snapshot: {row['id']}")
        mask, mode = CONDITION_MAP[row["condition"]]
        if row["mask"] != mask or row["information_mode"] != mode:
            raise RuntimeError(f"D3 registry condition contract changed: {row['id']}")
        if row["native_steps"] != d3.NATIVE_STEPS:
            raise RuntimeError(f"D3 training budget changed: {row['id']}")
    return rows


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise AuthorizationError("Cannot resolve D3 runner source commit") from exc


def expected_snapshot_hashes() -> dict[str, str]:
    return {str(row[0]): row[2] for row in IMMUTABLE_BLOCKS}


def require_start_request(path: Path = START_REQUEST_PATH) -> dict:
    """Require explicit immutable authorization after full preflight.

    Absence is intentional and fail-closed.  A request must be created in a
    separate reviewed step; this runner never creates or repairs one.
    """
    if not path.is_file():
        raise AuthorizationError(
            "D3 scientific execution is NOT authorized: START_REQUEST.json is absent"
        )
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorizationError(f"Cannot parse D3 START_REQUEST.json: {exc}") from exc

    if request.get("schema") != "B1-SC-D3-START-REQUEST-1.0.0-20260916":
        raise AuthorizationError("Unexpected D3 START_REQUEST schema")
    if request.get("status") != "D3_SCIENTIFIC_EXECUTION_AUTHORIZED" or request.get("authorized") is not True:
        raise AuthorizationError("D3 START_REQUEST does not explicitly authorize execution")
    if request.get("snapshot_sha256s") != expected_snapshot_hashes():
        raise AuthorizationError("D3 START_REQUEST snapshot authority differs from immutable bytes")
    if request.get("design_sha256") != sha256_file(DESIGN_PATH):
        raise AuthorizationError("D3 START_REQUEST design hash mismatch")
    if request.get("init_freeze_sha256") != sha256_file(INIT_FREEZE_PATH):
        raise AuthorizationError("D3 START_REQUEST initialization-freeze hash mismatch")
    if request.get("fit_registry_sha256") != sha256_file(REGISTRY_PATH):
        raise AuthorizationError("D3 START_REQUEST fit-registry hash mismatch")
    if request.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise AuthorizationError("D3 START_REQUEST protocol hash mismatch")
    if request.get("source_commit") != current_commit():
        raise AuthorizationError("D3 START_REQUEST does not authorize this exact source commit")
    if request.get("allow_seed_reconstruction") is not False:
        raise AuthorizationError("D3 START_REQUEST must explicitly forbid seed reconstruction")
    return request


def _set_fit_rng(row: dict) -> None:
    seed = int(row["policy_sampling_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _env_seed(row: dict, episode: int) -> int:
    return child_seed(row["training_environment_seed_root"], "training-environment", episode)


def _panel_seed(row: dict, split: str, episode: int) -> int:
    key = {
        "diagnostic": "diagnostic_panel_seed",
        "endpoint": "endpoint_panel_seed",
        "causal": "causal_panel_seed",
    }[split]
    return child_seed(row[key], split, episode)


@torch.no_grad()
def evaluate_policy(actor: d3.D3RoutingActor, row: dict, split: str, episodes: int, lesion: tuple[int, ...] = ()) -> list[dict]:
    if split not in {"diagnostic", "endpoint", "causal"}:
        raise ValueError("Forbidden D3 evaluation split")
    env = NativeEnv()
    rows = []
    try:
        for episode in range(int(episodes)):
            seed = _panel_seed(row, split, episode)
            z, _ = env.reset(seed=seed)
            total_return = 0.0
            request_abs_sum = 0.0
            while True:
                tensor = torch.from_numpy(np.asarray(z, dtype=np.float32)[None])
                mu, _ = actor(tensor, lesion=lesion)
                action = mu.tanh()[0].cpu().numpy()
                z, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                request_abs_sum += float(np.abs(action).sum())
                if terminated or truncated:
                    break
                if env.t >= 500:
                    raise RuntimeError("Native D3 evaluation horizon was not enforced")
            rows.append({
                "fit_id": row["id"],
                "block": row["block"],
                "condition": row["condition"],
                "split": split,
                "episode": episode,
                "seed": seed,
                "return": total_return,
                "loss": -total_return,
                "cycles": env.t,
                "package_displacement": info["package_displacement"],
                "fallen_count": info["fallen_count"],
                "package_dropped": info["package_dropped"],
                "motor_request_abs_sum": request_abs_sum,
                "lesion": list(lesion),
            })
    finally:
        env.close()
    return rows


def _checkpoint(actor, critic, optimizer, row: dict, step: int, out: Path) -> None:
    torch.save({
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fit": row,
        "steps": int(step),
    }, out / f"checkpoint-{step}.pt")
    diagnostic = evaluate_policy(actor, row, "diagnostic", d3.DIAGNOSTIC_EPISODES)
    write_json(out / f"diagnostic-{step}.json", {
        "status": "D3_DIAGNOSTIC_COMPLETE",
        "steps": int(step),
        "rows": diagnostic,
        "mean_loss": float(np.mean([r["loss"] for r in diagnostic])),
        "scientific_primary_endpoint": False,
    })


def train_fit(row: dict, permit: SnapshotPermit, out: Path) -> dict:
    """Run one authorized D3 PPO fit from its already validated block bytes."""
    permit.assert_valid(row["block"])
    _set_fit_rng(row)
    mask, information_mode = CONDITION_MAP[row["condition"]]
    actor, critic = load_validated_models(permit, mask, information_mode)

    # This is the only scientific optimizer construction path in D3.
    optimizer = create_optimizer_after_preflight(
        permit,
        actor,
        critic,
        lr=PPO["lr"],
        eps=PPO["adam_eps"],
    )

    steps = int(row["native_steps"])
    rollout = int(PPO["rollout_steps"])
    if steps % rollout:
        raise RuntimeError("D3 native training budget must divide exactly by PPO rollout")

    env = NativeEnv()
    episode = 0
    z, _ = env.reset(seed=_env_seed(row, episode))
    episode_return = 0.0
    episode_length = 0
    completed_episodes = []
    learning_curve = []
    optimizer_updates = 0
    checkpoints = set(d3.CHECKPOINTS)
    t0 = time.perf_counter()

    try:
        for offset in range(0, steps, rollout):
            observations = []
            pre_tanh = []
            old_log_probs = []
            values = []
            rewards = []
            dones = []

            for k in range(rollout):
                tensor = torch.from_numpy(np.asarray(z, dtype=np.float32)[None])
                with torch.no_grad():
                    mu, log_std = actor(tensor)
                    u0 = mu + log_std.exp() * torch.randn_like(mu)
                    action = u0.tanh()
                    log_prob = log_squashed_gaussian(u0, mu, log_std)
                    value = critic(tensor)
                observations.append(np.asarray(z, dtype=np.float32).copy())
                pre_tanh.append(u0[0].cpu().numpy())
                old_log_probs.append(float(log_prob.item()))
                values.append(float(value.item()))

                z, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                done = bool(terminated or truncated)
                rewards.append(float(reward))
                dones.append(done)
                episode_return += float(reward)
                episode_length += 1
                if done:
                    completed_episodes.append({
                        "episode": episode,
                        "seed": env.episode_seed,
                        "return": episode_return,
                        "cycles": episode_length,
                        "step": offset + k + 1,
                        "package_displacement": info["package_displacement"],
                        "fall_count": info["fallen_count"],
                    })
                    episode += 1
                    episode_return = 0.0
                    episode_length = 0
                    z, _ = env.reset(seed=_env_seed(row, episode))

            with torch.no_grad():
                last_value = float(critic(torch.from_numpy(np.asarray(z, dtype=np.float32)[None]))[0])
            advantages = np.zeros(rollout, dtype=np.float32)
            last_gae = 0.0
            for k in reversed(range(rollout)):
                nonterminal = 1.0 - float(dones[k])
                next_value = last_value if k == rollout - 1 else values[k + 1]
                delta = rewards[k] + PPO["gamma"] * next_value * nonterminal - values[k]
                last_gae = delta + PPO["gamma"] * PPO["gae_lambda"] * nonterminal * last_gae
                advantages[k] = last_gae
            returns = advantages + np.asarray(values, dtype=np.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_t = torch.from_numpy(np.stack(observations))
            pre_t = torch.from_numpy(np.stack(pre_tanh))
            old_t = torch.tensor(old_log_probs)
            adv_t = torch.from_numpy(advantages)
            ret_t = torch.from_numpy(returns)
            actor_losses = []
            value_losses = []
            grad_norms = []

            for _ in range(PPO["epochs"]):
                order = np.random.permutation(rollout)
                for lo in range(0, rollout, PPO["minibatch"]):
                    idx = order[lo : lo + PPO["minibatch"]]
                    mu, log_std = actor(obs_t[idx])
                    new_log_prob = log_squashed_gaussian(pre_t[idx], mu, log_std)
                    log_ratio = new_log_prob - old_t[idx]
                    ratio = log_ratio.exp()
                    actor_loss = -torch.min(
                        ratio * adv_t[idx],
                        ratio.clamp(1 - PPO["clip"], 1 + PPO["clip"]) * adv_t[idx],
                    ).mean()
                    value_loss = (critic(obs_t[idx]) - ret_t[idx]).square().mean()
                    loss = actor_loss + PPO["value_coef"] * value_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite D3 PPO objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()),
                        PPO["max_grad_norm"],
                    )
                    if not torch.isfinite(grad):
                        raise RuntimeError("Non-finite D3 PPO gradient")
                    optimizer.step()
                    optimizer_updates += 1
                    actor_losses.append(float(actor_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    grad_norms.append(float(grad.detach()))

            step = offset + rollout
            learning_curve.append({
                "steps": step,
                "actor_loss": float(np.mean(actor_losses)),
                "value_loss": float(np.mean(value_losses)),
                "grad_norm": float(np.mean(grad_norms)),
                "episodes_completed": len(completed_episodes),
                "optimizer_updates": optimizer_updates,
                "elapsed_seconds": time.perf_counter() - t0,
            })
            if step in checkpoints:
                _checkpoint(actor, critic, optimizer, row, step, out)

        endpoint = evaluate_policy(actor, row, "endpoint", d3.ENDPOINT_EPISODES)
        write_json(out / "ENDPOINT.json", {
            "status": "D3_ENDPOINT_COMPLETE",
            "rows": endpoint,
            "mean_loss": float(np.mean([r["loss"] for r in endpoint])),
            "mean_package_displacement": float(np.mean([r["package_displacement"] for r in endpoint])),
        })
        write_json(out / "LEARNING_CURVE.json", learning_curve)
        write_json(out / "TRAINING_EPISODES.json", completed_episodes)
        torch.save({
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "fit": row,
            "steps": steps,
        }, out / "model.pt")
        return {
            "status": "D3_FIT_COMPLETE",
            "fit_id": row["id"],
            "block": row["block"],
            "condition": row["condition"],
            "snapshot_sha256": permit.sha256,
            "snapshot_validated_before_optimizer": True,
            "seed_reconstruction_used": False,
            "environment_steps": steps,
            "optimizer_updates": optimizer_updates,
            "endpoint_episodes": len(endpoint),
        }
    finally:
        env.close()


def run_scientific(out: Path, *, only_fit: str | None = None) -> dict:
    """Execute D3 only after all structural and explicit authorization gates pass."""
    # Full eight-block validation happens before authorization and before any
    # optimizer can exist.  No training side effect occurs in full_preflight().
    preflight_report, permits = full_preflight(range(d3.BLOCKS))
    request = require_start_request()
    rows = registry()
    if only_fit is not None:
        rows = [row for row in rows if row["id"] == only_fit]
        if len(rows) != 1:
            raise ValueError(f"Unknown D3 fit id: {only_fit}")

    if out.exists():
        raise RuntimeError(f"Refuse to overwrite D3 scientific output: {out}")
    out.mkdir(parents=True)
    write_json(out / "RUN_STARTED.json", {
        "status": "D3_SCIENTIFIC_RUN_STARTED",
        "source_commit": current_commit(),
        "start_request_sha256": sha256_file(START_REQUEST_PATH),
        "preflight": preflight_report,
        "runtime": runtime_lock(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fits_requested": [row["id"] for row in rows],
        "authorization": {k: request[k] for k in ("schema", "status", "authorized")},
    })

    receipts = []
    for row in rows:
        fit_out = out / row["id"]
        fit_out.mkdir()
        write_json(fit_out / "STARTED.json", {
            "fit": row,
            "snapshot": permits[row["block"]].public_receipt(),
            "source_commit": current_commit(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        result = train_fit(row, permits[row["block"]], fit_out)
        receipts.append(result)
        write_json(fit_out / "COMPLETE.json", result)

    summary = {
        "status": "D3_REQUESTED_FITS_COMPLETE",
        "fits_complete": len(receipts),
        "fits": receipts,
        "B1E_executed": False,
        "H_CAT": "NOT_EVALUABLE",
        "H_TRANSFER": "NOT_EVALUATED",
    }
    write_json(out / "RUN_COMPLETE.json", summary)
    return summary


def cli() -> None:
    parser = argparse.ArgumentParser(description="B1-SC-D3 v1.0 gated runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--microfit-qa", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--condition", choices=sorted(CONDITION_MAP), default="LOCAL-S6")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only-fit")
    args = parser.parse_args()

    if args.preflight:
        report, _ = full_preflight(range(d3.BLOCKS))
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.microfit_qa:
        print(json.dumps(microfit_qa_suite(args.block, args.condition), indent=2, sort_keys=True))
        return
    if args.out is None:
        parser.error("--execute requires --out")
    print(json.dumps(run_scientific(args.out, only_fit=args.only_fit), indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
