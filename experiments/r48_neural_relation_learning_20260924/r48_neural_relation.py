#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

POPGYM_COMMIT = "e397e5eac9965f9963d18c9f455cd1983bca14fb"
HIDDEN_SIZE = 128
LR = 3e-4
ENTROPY_COEF = 0.01
GRAD_CLIP = 1.0

DEV_TRAIN_SEEDS = [2196001, 2196002, 2196003]
DEV_EVAL_SEEDS = list(range(2196101, 2196113))
CONF_TRAIN_SEEDS = [2197001, 2197002, 2197003]
CONF_EVAL_SEEDS = list(range(2197101, 2197113))

DEV_EPISODES = 12000
CONF_EPISODES = 18000
EVAL_EPISODES_PER_SEED = 2
ISO_TOL = 1e-12


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class RecurrentPolicy(nn.Module):
    def __init__(self, n_identities: int, n_actions: int, arch: str):
        super().__init__()
        self.n_identities = int(n_identities)
        self.n_actions = int(n_actions)
        self.arch = arch.upper()
        input_dim = 2 * self.n_identities
        if self.arch == "LSTM":
            self.core = nn.LSTMCell(input_dim, HIDDEN_SIZE)
        elif self.arch == "GRU":
            self.core = nn.GRUCell(input_dim, HIDDEN_SIZE)
        else:
            raise ValueError(f"unknown arch {arch}")
        self.head = nn.Linear(HIDDEN_SIZE, self.n_actions)

    def initial_state(self, device=None):
        device = device or next(self.parameters()).device
        h = torch.zeros(1, HIDDEN_SIZE, device=device)
        if self.arch == "LSTM":
            c = torch.zeros(1, HIDDEN_SIZE, device=device)
            return (h, c)
        return h

    def encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        v, q = int(obs[0]), int(obs[1])
        x = torch.zeros(1, 2 * self.n_identities, dtype=torch.float32)
        x[0, v] = 1.0
        x[0, self.n_identities + q] = 1.0
        return x

    def forward_step(self, x: torch.Tensor, state):
        if self.arch == "LSTM":
            h, c = self.core(x, state)
            logits = self.head(h)
            return logits, (h, c)
        h = self.core(x, state)
        logits = self.head(h)
        return logits, h


def make_env(level: str):
    import popgym

    key = level.lower()
    if key == "easy":
        return popgym.CountRecallEasy()
    if key == "medium":
        return popgym.CountRecallMedium()
    if key == "hard":
        return popgym.CountRecallHard()
    raise ValueError(level)


def env_contract(level: str) -> Tuple[int, int, int]:
    env = make_env(level)
    n_id = int(env.num_distinct_cards)
    n_actions = int(env.action_space.n)
    episode_length = int(env.max_episode_length)
    env.close()
    return n_id, n_actions, episode_length


def transform_obs(obs: np.ndarray, condition: str, n_id: int) -> np.ndarray:
    o = np.asarray(obs, dtype=np.int64).copy()
    if condition == "NO_D_COLLAPSE":
        o[:] = 0
    elif condition == "WRONG_R_QUERY_CYCLE":
        o[1] = (int(o[1]) + 1) % n_id
    elif condition == "GENERIC_ISO":
        o[0] = (n_id - 1) - int(o[0])
        o[1] = (n_id - 1) - int(o[1])
    elif condition in {"INTACT", "NO_C_STEP_RESET"}:
        pass
    else:
        raise ValueError(condition)
    return o


def make_iso_model(model: RecurrentPolicy) -> RecurrentPolicy:
    iso = copy.deepcopy(model)
    k = model.n_identities
    perm = [(k - 1) - i for i in range(k)]
    with torch.no_grad():
        old = model.core.weight_ih.detach().clone()
        new = old.clone()
        for native, recoded in enumerate(perm):
            new[:, recoded] = old[:, native]
            new[:, k + recoded] = old[:, k + native]
        iso.core.weight_ih.copy_(new)
    return iso


def train_checkpoint(
    arch: str,
    level: str,
    train_seed: int,
    episodes: int,
    checkpoint_path: str,
    log_every: int = 1000,
) -> dict:
    set_global_seed(train_seed)
    torch.set_num_threads(1)
    n_id, n_actions, episode_length = env_contract(level)
    model = RecurrentPolicy(n_id, n_actions, arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    recent_correct: List[float] = []
    history = []

    for ep in range(episodes):
        env = make_env(level)
        obs = np.asarray(env.reset(seed=train_seed * 100000 + ep), dtype=np.int64)
        state = model.initial_state()
        logps = []
        rewards = []
        entropies = []
        correct = 0
        steps = 0
        done = False

        while not done:
            x = model.encode_obs(obs)
            logits, state = model.forward_step(x, state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, done, _ = env.step(int(action.item()))
            scaled_reward = 1.0 if float(reward) > 0 else -1.0
            correct += int(scaled_reward > 0)
            steps += 1
            logps.append(dist.log_prob(action).squeeze())
            entropies.append(dist.entropy().squeeze())
            rewards.append(torch.tensor(scaled_reward, dtype=torch.float32))
            obs = np.asarray(next_obs, dtype=np.int64)

        env.close()
        r = torch.stack(rewards)
        lp = torch.stack(logps)
        ent = torch.stack(entropies)
        loss = -(r.detach() * lp).mean() - ENTROPY_COEF * ent.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        acc = correct / max(steps, 1)
        recent_correct.append(acc)
        if len(recent_correct) > 200:
            recent_correct.pop(0)

        if (ep + 1) % log_every == 0 or ep == 0 or ep + 1 == episodes:
            rec = {
                "episode": ep + 1,
                "recent_accuracy": float(np.mean(recent_correct)),
                "loss": float(loss.detach().item()),
            }
            history.append(rec)
            print(json.dumps(rec), flush=True)

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch.upper(),
            "level": level.lower(),
            "train_seed": int(train_seed),
            "episodes": int(episodes),
            "n_identities": int(n_id),
            "n_actions": int(n_actions),
            "episode_length": int(episode_length),
            "hyperparameters": {
                "hidden_size": HIDDEN_SIZE,
                "lr": LR,
                "entropy_coef": ENTROPY_COEF,
                "grad_clip": GRAD_CLIP,
            },
            "training_log": history,
        },
        checkpoint_path,
    )
    return {"checkpoint": checkpoint_path, "training_log": history}


def load_model(path: str) -> Tuple[RecurrentPolicy, dict]:
    payload = torch.load(path, map_location="cpu")
    model = RecurrentPolicy(
        payload["n_identities"],
        payload["n_actions"],
        payload["arch"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def eval_condition(
    model: RecurrentPolicy,
    level: str,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
    condition: str,
) -> dict:
    total_correct = 0
    total_steps = 0
    per_episode = []
    eval_model = make_iso_model(model) if condition == "GENERIC_ISO" else model

    for seed in eval_seeds:
        for ep in range(episodes_per_seed):
            env = make_env(level)
            episode_seed = int(seed) * 100 + ep
            obs = np.asarray(env.reset(seed=episode_seed), dtype=np.int64)
            state = eval_model.initial_state()
            correct = 0
            steps = 0
            done = False
            while not done:
                if condition == "NO_C_STEP_RESET":
                    state = eval_model.initial_state()
                o = transform_obs(obs, condition, eval_model.n_identities)
                x = eval_model.encode_obs(o)
                logits, state = eval_model.forward_step(x, state)
                action = int(torch.argmax(logits, dim=-1).item())
                next_obs, reward, done, _ = env.step(action)
                correct += int(float(reward) > 0)
                steps += 1
                obs = np.asarray(next_obs, dtype=np.int64)
            env.close()
            total_correct += correct
            total_steps += steps
            per_episode.append(
                {
                    "eval_seed": int(seed),
                    "episode": int(ep),
                    "episode_seed": int(episode_seed),
                    "accuracy": correct / max(steps, 1),
                    "correct": int(correct),
                    "steps": int(steps),
                }
            )
    return {
        "condition": condition,
        "accuracy": total_correct / max(total_steps, 1),
        "correct": int(total_correct),
        "steps": int(total_steps),
        "episodes": per_episode,
    }


@torch.no_grad()
def eval_iso_action_agreement(
    model: RecurrentPolicy,
    level: str,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
) -> dict:
    iso = make_iso_model(model)
    agree = 0
    total = 0

    for seed in eval_seeds:
        for ep in range(episodes_per_seed):
            episode_seed = int(seed) * 100 + ep
            env_a = make_env(level)
            env_b = make_env(level)
            obs_a = np.asarray(env_a.reset(seed=episode_seed), dtype=np.int64)
            obs_b = np.asarray(env_b.reset(seed=episode_seed), dtype=np.int64)
            state_a = model.initial_state()
            state_b = iso.initial_state()
            done_a = False
            done_b = False
            while not done_a:
                xa = model.encode_obs(obs_a)
                la, state_a = model.forward_step(xa, state_a)
                aa = int(torch.argmax(la, dim=-1).item())

                ob = transform_obs(obs_b, "GENERIC_ISO", iso.n_identities)
                xb = iso.encode_obs(ob)
                lb, state_b = iso.forward_step(xb, state_b)
                ab = int(torch.argmax(lb, dim=-1).item())

                agree += int(aa == ab)
                total += 1
                next_a, _, done_a, _ = env_a.step(aa)
                next_b, _, done_b, _ = env_b.step(ab)
                if done_a != done_b:
                    raise RuntimeError("paired iso environments diverged in termination")
                obs_a = np.asarray(next_a, dtype=np.int64)
                obs_b = np.asarray(next_b, dtype=np.int64)
            env_a.close()
            env_b.close()

    return {
        "action_agreement": agree / max(total, 1),
        "agree": int(agree),
        "total": int(total),
    }


def evaluate_checkpoint(
    checkpoint: str,
    level: str,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
    output: str,
) -> dict:
    model, payload = load_model(checkpoint)
    conditions = {}
    for cond in (
        "INTACT",
        "NO_D_COLLAPSE",
        "NO_C_STEP_RESET",
        "WRONG_R_QUERY_CYCLE",
        "GENERIC_ISO",
    ):
        conditions[cond] = eval_condition(
            model, level, eval_seeds, episodes_per_seed, cond
        )
    iso_agreement = eval_iso_action_agreement(
        model, level, eval_seeds, episodes_per_seed
    )

    intact = conditions["INTACT"]["accuracy"]
    row = {
        "campaign": "R48 neural relation discovery from reward",
        "mode": "checkpoint",
        "source": {
            "repository": "proroklab/popgym",
            "commit": POPGYM_COMMIT,
            "environment": f"CountRecall{level.title()}",
        },
        "arch": payload["arch"],
        "train_seed": int(payload["train_seed"]),
        "training_episodes": int(payload["episodes"]),
        "eval_seeds": [int(x) for x in eval_seeds],
        "eval_episodes_per_seed": int(episodes_per_seed),
        "conditions": conditions,
        "intact_accuracy": float(intact),
        "d_effect": float(intact - conditions["NO_D_COLLAPSE"]["accuracy"]),
        "c_effect": float(intact - conditions["NO_C_STEP_RESET"]["accuracy"]),
        "r_effect": float(intact - conditions["WRONG_R_QUERY_CYCLE"]["accuracy"]),
        "iso_gap": float(abs(intact - conditions["GENERIC_ISO"]["accuracy"])),
        "iso_action_agreement": float(iso_agreement["action_agreement"]),
        "iso_action_agreement_counts": iso_agreement,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in row.items() if k != "conditions"}, indent=2, sort_keys=True))
    return row


def checkpoint_eligible(row: dict, phase: str) -> bool:
    if phase == "development":
        return bool(
            row["intact_accuracy"] >= 0.90
            and row["d_effect"] >= 0.25
            and row["c_effect"] >= 0.25
            and row["r_effect"] >= 0.20
            and row["iso_gap"] <= ISO_TOL
            and row["iso_action_agreement"] >= 1.0 - ISO_TOL
        )
    if phase == "confirmatory":
        return bool(
            row["intact_accuracy"] >= 0.80
            and row["d_effect"] >= 0.15
            and row["c_effect"] >= 0.30
            and row["r_effect"] >= 0.15
            and row["iso_gap"] <= ISO_TOL
            and row["iso_action_agreement"] >= 1.0 - ISO_TOL
        )
    raise ValueError(phase)


def aggregate_development(inputs: Sequence[str], output: str) -> dict:
    rows = [json.loads(Path(p).read_text()) for p in inputs]
    rows = sorted(rows, key=lambda x: int(x["train_seed"]))
    for r in rows:
        r["eligible"] = checkpoint_eligible(r, "development")
    n = sum(bool(r["eligible"]) for r in rows)
    authorize = n >= 2
    out = {
        "campaign": "R48 neural relation discovery from reward",
        "phase": "development",
        "training_seeds": DEV_TRAIN_SEEDS,
        "eligible_checkpoints": int(n),
        "required_eligible_checkpoints": 2,
        "authorize_confirm": bool(authorize),
        "resolution": (
            "R48_DEVELOPMENT_AUTHORIZE_CONFIRM"
            if authorize
            else "R48_DEVELOPMENT_FAIL_NO_CONFIRM"
        ),
        "checkpoints": rows,
        "boundaries": {
            "cross_domain_transport": "NOT_TESTED",
            "global_minimality": "OPEN",
            "E6b": "OPEN",
            "E7": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in out.items() if k != "checkpoints"}, indent=2))
    return out


def aggregate_confirmatory(inputs: Sequence[str], output: str) -> dict:
    rows = [json.loads(Path(p).read_text()) for p in inputs]
    for r in rows:
        r["eligible"] = checkpoint_eligible(r, "confirmatory")
    by_arch: Dict[str, List[dict]] = {}
    for r in rows:
        by_arch.setdefault(r["arch"], []).append(r)

    summaries = {}
    for arch in ("LSTM", "GRU"):
        rs = sorted(by_arch.get(arch, []), key=lambda x: int(x["train_seed"]))
        eligible = sum(bool(r["eligible"]) for r in rs)
        summaries[arch] = {
            "n": len(rs),
            "eligible_checkpoints": int(eligible),
            "required_eligible_checkpoints": 2,
            "confirm": bool(len(rs) == 3 and eligible >= 2),
            "checkpoints": rs,
        }

    n_confirm = sum(int(summaries[a]["confirm"]) for a in ("LSTM", "GRU"))
    if n_confirm == 2:
        resolution = "R48_NEURAL_RELATION_LEARNABILITY_PASS"
    elif n_confirm == 1:
        resolution = "R48_NEURAL_RELATION_LEARNABILITY_PARTIAL"
    else:
        resolution = "R48_NEURAL_RELATION_LEARNABILITY_FAIL"

    out = {
        "campaign": "R48 neural relation discovery from reward",
        "phase": "confirmatory",
        "resolution": resolution,
        "architectures": summaries,
        "boundaries": {
            "claim_scope": "neural relation learnability in tested CountRecall variants",
            "cross_domain_transport": "NOT_ESTABLISHED",
            "unique_internal_representation_of_R": "NOT_ESTABLISHED",
            "global_minimality": "OPEN",
            "E6b": "OPEN",
            "E7": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "AGI_ASI": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "resolution": resolution,
        "LSTM": {k:v for k,v in summaries["LSTM"].items() if k != "checkpoints"},
        "GRU": {k:v for k,v in summaries["GRU"].items() if k != "checkpoints"},
    }, indent=2))
    return out


def engineering_smoke(output: str) -> None:
    set_global_seed(2195000)
    n_id, n_actions, ep_len = env_contract("easy")
    model = RecurrentPolicy(n_id, n_actions, "LSTM")
    iso = make_iso_model(model)
    # Exact conjugacy on random observations/states.
    state_a = model.initial_state()
    state_b = iso.initial_state()
    agreements = []
    for i in range(20):
        obs = np.array([i % n_id, (i * 3) % n_id], dtype=np.int64)
        xa = model.encode_obs(obs)
        la, state_a = model.forward_step(xa, state_a)
        oi = transform_obs(obs, "GENERIC_ISO", n_id)
        xb = iso.encode_obs(oi)
        lb, state_b = iso.forward_step(xb, state_b)
        agreements.append(float(torch.max(torch.abs(la - lb)).item()))
    out = {
        "campaign": "R48 neural relation discovery from reward",
        "mode": "engineering_smoke",
        "scientific_seed": False,
        "source_commit": POPGYM_COMMIT,
        "easy_contract": {
            "n_identities": n_id,
            "n_actions": n_actions,
            "episode_length": ep_len,
        },
        "max_iso_logit_gap": max(agreements),
        "pass": max(agreements) <= 1e-6,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("smoke")
    s.add_argument("--output", required=True)

    t = sp.add_parser("train")
    t.add_argument("--arch", required=True, choices=["LSTM", "GRU"])
    t.add_argument("--level", required=True, choices=["medium", "hard"])
    t.add_argument("--train-seed", required=True, type=int)
    t.add_argument("--episodes", required=True, type=int)
    t.add_argument("--checkpoint", required=True)

    e = sp.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--level", required=True, choices=["medium", "hard"])
    e.add_argument("--phase", required=True, choices=["development", "confirmatory"])
    e.add_argument("--output", required=True)

    a = sp.add_parser("aggregate-dev")
    a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--output", required=True)

    c = sp.add_parser("aggregate-confirm")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--output", required=True)

    args = p.parse_args()
    if args.cmd == "smoke":
        engineering_smoke(args.output)
    elif args.cmd == "train":
        train_checkpoint(args.arch, args.level, args.train_seed, args.episodes, args.checkpoint)
    elif args.cmd == "eval":
        seeds = DEV_EVAL_SEEDS if args.phase == "development" else CONF_EVAL_SEEDS
        evaluate_checkpoint(args.checkpoint, args.level, seeds, EVAL_EPISODES_PER_SEED, args.output)
    elif args.cmd == "aggregate-dev":
        aggregate_development(args.inputs, args.output)
    elif args.cmd == "aggregate-confirm":
        aggregate_confirmatory(args.inputs, args.output)


if __name__ == "__main__":
    main()
