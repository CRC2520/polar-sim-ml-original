#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

POPGYM_COMMIT = "e397e5eac9965f9963d18c9f455cd1983bca14fb"
TASKS = ("ConcentrationHard", "CountRecallHard")
PERTURBATIONS = ("CLEAN", "SUBSTITUTE10", "STALE_BURST", "MIXED")
CONDITIONS = ("FULL_DCR", "NO_D_COLLAPSE", "NO_C", "WRONG_R", "GENERIC_ISO")

SMOKE_SEED = 2215000
DEV_SEEDS = list(range(2216001, 2216013))
CONF_SEEDS = list(range(2217001, 2217033))
EPISODES_PER_SEED = 3
ISO_TOL = 1e-12

N_POSITIONS = 52
N_SYMBOLS = 13
N_PAIRS = 26

CLEAN_GATES = {
    "full_metric": 0.90,
    "d_effect": 0.20,
    "c_effect": 0.30,
    "r_effect": 0.20,
}
PERT_GATES = {
    "full_metric": 0.55,
    "robustness_ratio": 0.55,
    "d_effect": 0.10,
    "c_effect": 0.15,
    "r_effect": 0.10,
}


def stable_seed(*parts) -> int:
    raw = hashlib.sha256(("R51|" + "|".join(map(str, parts))).encode()).digest()
    return int.from_bytes(raw[:8], "little") & 0xFFFFFFFF


class SensorPerturber:
    def __init__(self, task: str, perturbation: str, episode_seed: int, max_events: int):
        self.task = task
        self.perturbation = perturbation
        self.max_events = int(max_events)
        rng = np.random.default_rng(stable_seed(task, perturbation, episode_seed))
        self.u = rng.random((self.max_events + 4, 2))
        self.offset = rng.integers(1, N_SYMBOLS, size=(self.max_events + 4, 2))
        burst_len = 4 if perturbation == "STALE_BURST" else 2
        if perturbation in {"STALE_BURST", "MIXED"}:
            lo = min(8, max(1, self.max_events // 10))
            hi = max(lo + 1, self.max_events - burst_len - 2)
            starts = sorted(rng.choice(np.arange(lo, hi), size=2, replace=False).tolist())
            self.bursts = [(int(s), int(s + burst_len)) for s in starts]
        else:
            self.bursts = []
        self.last_scalar: Optional[int] = None
        self.last_vector: Optional[np.ndarray] = None

    def _stale(self, idx: int) -> bool:
        return any(a <= idx < b for a, b in self.bursts)

    def scalar(self, symbol: int, idx: int) -> int:
        s = int(symbol)
        if self.perturbation == "CLEAN":
            out = s
        elif self.perturbation == "SUBSTITUTE10":
            out = (s + int(self.offset[idx, 0])) % N_SYMBOLS if self.u[idx, 0] < 0.10 else s
        elif self.perturbation == "STALE_BURST":
            out = self.last_scalar if self._stale(idx) and self.last_scalar is not None else s
        elif self.perturbation == "MIXED":
            if self._stale(idx) and self.last_scalar is not None:
                out = self.last_scalar
            elif self.u[idx, 0] < 0.05:
                out = (s + int(self.offset[idx, 0])) % N_SYMBOLS
            else:
                out = s
        else:
            raise ValueError(self.perturbation)
        self.last_scalar = int(out)
        return int(out)

    def vector(self, obs: np.ndarray, idx: int) -> np.ndarray:
        x = np.asarray(obs, dtype=np.int64).copy()
        if self.perturbation == "CLEAN":
            out = x
        elif self.perturbation == "SUBSTITUTE10":
            out = x.copy()
            for j in range(2):
                if self.u[idx, j] < 0.10:
                    out[j] = (int(out[j]) + int(self.offset[idx, j])) % N_SYMBOLS
        elif self.perturbation == "STALE_BURST":
            out = self.last_vector.copy() if self._stale(idx) and self.last_vector is not None else x
        elif self.perturbation == "MIXED":
            if self._stale(idx) and self.last_vector is not None:
                out = self.last_vector.copy()
            else:
                out = x.copy()
                for j in range(2):
                    if self.u[idx, j] < 0.05:
                        out[j] = (int(out[j]) + int(self.offset[idx, j])) % N_SYMBOLS
        else:
            raise ValueError(self.perturbation)
        self.last_vector = np.asarray(out, dtype=np.int64).copy()
        return np.asarray(out, dtype=np.int64)


def cyclic_pick(candidates: Sequence[int], last_action: int, n: int) -> int:
    if not candidates:
        raise RuntimeError("no candidates")
    start = (int(last_action) + 1) % n if last_action >= 0 else 0
    return min((int(i) for i in candidates), key=lambda i: ((i - start) % n, i))


class ConcentrationAgent:
    def __init__(self, condition: str):
        self.condition = condition
        self.memory: Dict[int, int] = {}
        self.failed_pairs: Set[Tuple[int, int]] = set()
        self.solved: Set[int] = set()
        self.pair_actions: List[int] = []
        self.last_action = -1

    def encode(self, native_symbol: int) -> int:
        s = int(native_symbol)
        if self.condition == "NO_D_COLLAPSE":
            return 0
        if self.condition == "GENERIC_ISO":
            return (N_SYMBOLS - 1) - s
        return s

    def target_identity(self, first_internal: int) -> int:
        if self.condition == "WRONG_R":
            return (int(first_internal) + 1) % N_SYMBOLS
        return int(first_internal)

    @staticmethod
    def pair_key(i: int, j: int) -> Tuple[int, int]:
        return tuple(sorted((int(i), int(j))))

    def unsolved(self, exclude: Optional[Set[int]] = None) -> List[int]:
        exclude = exclude or set()
        return [i for i in range(N_POSITIONS) if i not in self.solved and i not in exclude]

    def known_candidates(self, target: int, exclude: Set[int], first: Optional[int] = None) -> List[int]:
        out = []
        for i in self.unsolved(exclude):
            if i not in self.memory or int(self.memory[i]) != int(target):
                continue
            if first is not None and self.pair_key(first, i) in self.failed_pairs:
                continue
            out.append(i)
        return out

    def unknown_candidates(self, exclude: Set[int]) -> List[int]:
        return [i for i in self.unsolved(exclude) if i not in self.memory]

    def pairable_first(self) -> List[int]:
        known = [i for i in self.unsolved() if i in self.memory]
        out = []
        for i in known:
            target = self.target_identity(self.memory[i])
            if any(
                j != i
                and self.memory[j] == target
                and self.pair_key(i, j) not in self.failed_pairs
                for j in known
            ):
                out.append(i)
        return out

    def select_action(self) -> int:
        if not self.pair_actions:
            pairable = self.pairable_first()
            if pairable:
                return cyclic_pick(pairable, self.last_action, N_POSITIONS)
            unknown = self.unknown_candidates(set())
            if unknown:
                return cyclic_pick(unknown, self.last_action, N_POSITIONS)
            return cyclic_pick(self.unsolved(), self.last_action, N_POSITIONS)

        first = int(self.pair_actions[0])
        if first not in self.memory:
            unknown = self.unknown_candidates({first})
            return cyclic_pick(unknown or self.unsolved({first}), self.last_action, N_POSITIONS)
        target = self.target_identity(self.memory[first])
        known = self.known_candidates(target, {first}, first)
        if known:
            return cyclic_pick(known, self.last_action, N_POSITIONS)
        unknown = self.unknown_candidates({first})
        if unknown:
            return cyclic_pick(unknown, self.last_action, N_POSITIONS)
        remaining = [
            i for i in self.unsolved({first})
            if self.pair_key(first, i) not in self.failed_pairs
        ]
        return cyclic_pick(remaining or self.unsolved({first}), self.last_action, N_POSITIONS)

    def observe(self, action: int, perceived_symbol: int, reward: float) -> None:
        action = int(action)
        self.memory[action] = self.encode(perceived_symbol)
        self.pair_actions.append(action)
        self.last_action = action

        if len(self.pair_actions) == 2:
            i, j = self.pair_actions
            if reward > 0:
                self.solved.update((i, j))
                self.memory.pop(i, None)
                self.memory.pop(j, None)
                self.failed_pairs = {p for p in self.failed_pairs if i not in p and j not in p}
            elif reward < 0:
                self.failed_pairs.add(self.pair_key(i, j))
            self.pair_actions = []
            if self.condition == "NO_C":
                self.memory.clear()
                self.failed_pairs.clear()


class CountRecallAgent:
    def __init__(self, condition: str, n_actions: int):
        self.condition = condition
        self.counts = np.zeros(N_SYMBOLS, dtype=np.int64)
        self.n_actions = int(n_actions)

    def act(self, perceived_obs: np.ndarray) -> int:
        v, q = int(perceived_obs[0]), int(perceived_obs[1])
        if self.condition == "NO_C":
            self.counts[:] = 0
        if self.condition == "NO_D_COLLAPSE":
            v = 0
            q = 0
        elif self.condition == "GENERIC_ISO":
            v = (N_SYMBOLS - 1) - v
            q = (N_SYMBOLS - 1) - q
        self.counts[v] += 1
        if self.condition == "WRONG_R":
            q = (q + 1) % N_SYMBOLS
        return int(min(int(self.counts[q]), self.n_actions - 1))


def make_concentration():
    import popgym
    env = popgym.ConcentrationHard()
    if int(env.action_space.n) != 52 or int(env.facedown_card) != 13:
        raise RuntimeError("ConcentrationHard contract changed")
    return env


def make_count_recall():
    import popgym
    env = popgym.CountRecallHard()
    if env.observation_space.nvec.tolist() != [13, 13]:
        raise RuntimeError("CountRecallHard observation contract changed")
    return env


@dataclass
class EpisodeMetric:
    metric: float
    native_return: float
    actions: List[int]


def concentration_episode(condition: str, perturbation: str, episode_seed: int) -> EpisodeMetric:
    env = make_concentration()
    obs = np.asarray(env.reset(seed=int(episode_seed)))
    agent = ConcentrationAgent(condition)
    sensor = SensorPerturber("ConcentrationHard", perturbation, episode_seed, int(env.episode_length))
    total_return = 0.0
    matches = 0
    actions: List[int] = []
    done = False
    event = 0
    while not done and event < int(env.episode_length):
        action = agent.select_action()
        obs_after, reward, done, _ = env.step(int(action))
        true_symbol = int(np.asarray(obs_after)[action])
        if not (0 <= true_symbol < N_SYMBOLS):
            raise RuntimeError(f"unexpected revealed symbol {true_symbol}")
        perceived = sensor.scalar(true_symbol, event)
        agent.observe(action, perceived, float(reward))
        total_return += float(reward)
        matches += int(float(reward) > 0)
        actions.append(int(action))
        obs = np.asarray(obs_after)
        event += 1
    env.close()
    return EpisodeMetric(float(matches / N_PAIRS), float(total_return), actions)


def count_recall_episode(condition: str, perturbation: str, episode_seed: int) -> EpisodeMetric:
    env = make_count_recall()
    obs = np.asarray(env.reset(seed=int(episode_seed)), dtype=np.int64)
    agent = CountRecallAgent(condition, int(env.action_space.n))
    sensor = SensorPerturber("CountRecallHard", perturbation, episode_seed, int(env.max_episode_length) + 1)
    total_return = 0.0
    correct = 0
    steps = 0
    actions: List[int] = []
    perceived = sensor.vector(obs, 0)
    done = False
    while not done:
        action = agent.act(perceived)
        next_obs, reward, done, _ = env.step(action)
        total_return += float(reward)
        correct += int(float(reward) > 0)
        steps += 1
        actions.append(int(action))
        if not done:
            perceived = sensor.vector(np.asarray(next_obs, dtype=np.int64), steps)
    env.close()
    return EpisodeMetric(float(correct / max(steps, 1)), float(total_return), actions)


def run_episode(task: str, condition: str, perturbation: str, episode_seed: int) -> EpisodeMetric:
    if task == "ConcentrationHard":
        return concentration_episode(condition, perturbation, episode_seed)
    if task == "CountRecallHard":
        return count_recall_episode(condition, perturbation, episode_seed)
    raise ValueError(task)


def mean_episode_metrics(rows: List[EpisodeMetric]) -> dict:
    return {
        "metric": float(np.mean([r.metric for r in rows])),
        "native_return": float(np.mean([r.native_return for r in rows])),
    }


def action_agreement(a: List[EpisodeMetric], b: List[EpisodeMetric]) -> float:
    agree = 0
    total = 0
    if len(a) != len(b):
        return 0.0
    for x, y in zip(a, b):
        n = max(len(x.actions), len(y.actions))
        total += n
        agree += sum(i < len(x.actions) and i < len(y.actions) and x.actions[i] == y.actions[i] for i in range(n))
    return float(agree / max(total, 1))


def evaluate_seed(seed: int, episodes_per_seed: int = EPISODES_PER_SEED) -> dict:
    cells = {}
    for task_i, task in enumerate(TASKS):
        for perturb in PERTURBATIONS:
            cond_episode_rows = {}
            cond_summary = {}
            for cond in CONDITIONS:
                eps = []
                for ep in range(episodes_per_seed):
                    episode_seed = int(seed) * 1000 + task_i * 100 + ep
                    eps.append(run_episode(task, cond, perturb, episode_seed))
                cond_episode_rows[cond] = eps
                cond_summary[cond] = mean_episode_metrics(eps)

            full = cond_summary["FULL_DCR"]["metric"]
            cell = {
                "task": task,
                "perturbation": perturb,
                "full_metric": float(full),
                "full_native_return": float(cond_summary["FULL_DCR"]["native_return"]),
                "d_effect": float(full - cond_summary["NO_D_COLLAPSE"]["metric"]),
                "c_effect": float(full - cond_summary["NO_C"]["metric"]),
                "r_effect": float(full - cond_summary["WRONG_R"]["metric"]),
                "iso_gap": float(abs(full - cond_summary["GENERIC_ISO"]["metric"])),
                "iso_action_agreement": action_agreement(
                    cond_episode_rows["FULL_DCR"], cond_episode_rows["GENERIC_ISO"]
                ),
                "condition_metrics": cond_summary,
            }
            cells[f"{task}:{perturb}"] = cell

    for task in TASKS:
        clean = cells[f"{task}:CLEAN"]["full_metric"]
        for perturb in PERTURBATIONS:
            key = f"{task}:{perturb}"
            cells[key]["robustness_ratio"] = float(cells[key]["full_metric"] / clean) if clean > 1e-12 else 0.0
            c = cells[key]
            if perturb == "CLEAN":
                c["cell_pass"] = bool(
                    c["full_metric"] >= CLEAN_GATES["full_metric"]
                    and c["d_effect"] >= CLEAN_GATES["d_effect"]
                    and c["c_effect"] >= CLEAN_GATES["c_effect"]
                    and c["r_effect"] >= CLEAN_GATES["r_effect"]
                    and c["iso_gap"] <= ISO_TOL
                    and c["iso_action_agreement"] >= 1.0 - ISO_TOL
                )
            else:
                c["cell_pass"] = bool(
                    c["full_metric"] >= PERT_GATES["full_metric"]
                    and c["robustness_ratio"] >= PERT_GATES["robustness_ratio"]
                    and c["d_effect"] >= PERT_GATES["d_effect"]
                    and c["c_effect"] >= PERT_GATES["c_effect"]
                    and c["r_effect"] >= PERT_GATES["r_effect"]
                    and c["iso_gap"] <= ISO_TOL
                    and c["iso_action_agreement"] >= 1.0 - ISO_TOL
                )

    return {
        "seed": int(seed),
        "cells": cells,
        "seed_guard": bool(all(c["cell_pass"] for c in cells.values())),
    }


def summarize(records: List[dict], required_seed_guards: int) -> dict:
    cell_summaries = {}
    for task in TASKS:
        for perturb in PERTURBATIONS:
            key = f"{task}:{perturb}"
            rows = [r["cells"][key] for r in records]
            med = lambda field: float(np.median([x[field] for x in rows]))
            s = {
                "task": task,
                "perturbation": perturb,
                "median_full_metric": med("full_metric"),
                "median_robustness_ratio": med("robustness_ratio"),
                "median_d_effect": med("d_effect"),
                "median_c_effect": med("c_effect"),
                "median_r_effect": med("r_effect"),
                "median_iso_gap": med("iso_gap"),
                "median_iso_action_agreement": med("iso_action_agreement"),
            }
            if perturb == "CLEAN":
                s["pass"] = bool(
                    s["median_full_metric"] >= CLEAN_GATES["full_metric"]
                    and s["median_d_effect"] >= CLEAN_GATES["d_effect"]
                    and s["median_c_effect"] >= CLEAN_GATES["c_effect"]
                    and s["median_r_effect"] >= CLEAN_GATES["r_effect"]
                    and s["median_iso_gap"] <= ISO_TOL
                    and s["median_iso_action_agreement"] >= 1.0 - ISO_TOL
                )
            else:
                s["pass"] = bool(
                    s["median_full_metric"] >= PERT_GATES["full_metric"]
                    and s["median_robustness_ratio"] >= PERT_GATES["robustness_ratio"]
                    and s["median_d_effect"] >= PERT_GATES["d_effect"]
                    and s["median_c_effect"] >= PERT_GATES["c_effect"]
                    and s["median_r_effect"] >= PERT_GATES["r_effect"]
                    and s["median_iso_gap"] <= ISO_TOL
                    and s["median_iso_action_agreement"] >= 1.0 - ISO_TOL
                )
            cell_summaries[key] = s

    guard_count = sum(bool(r["seed_guard"]) for r in records)
    passed = bool(
        all(s["pass"] for s in cell_summaries.values())
        and guard_count >= required_seed_guards
    )
    return {
        "cell_summaries": cell_summaries,
        "seed_guard_count": int(guard_count),
        "seed_guard_required": int(required_seed_guards),
        "pass": passed,
    }


def source_metadata() -> dict:
    try:
        version = importlib.metadata.version("popgym")
    except importlib.metadata.PackageNotFoundError:
        version = "UNKNOWN"
    return {
        "repository": "proroklab/popgym",
        "commit": POPGYM_COMMIT,
        "installed_version": version,
        "tasks": list(TASKS),
    }


def run_panel(mode: str, output: str) -> None:
    if mode == "smoke":
        seeds = [SMOKE_SEED]
        episodes = 1
        required = 0
    elif mode == "development":
        seeds = DEV_SEEDS
        episodes = EPISODES_PER_SEED
        required = 10
    elif mode == "confirmatory":
        seeds = CONF_SEEDS
        episodes = EPISODES_PER_SEED
        required = 26
    else:
        raise ValueError(mode)

    records = [evaluate_seed(s, episodes) for s in seeds]
    summary = summarize(records, required) if mode != "smoke" else summarize(records, 0)

    if mode == "development":
        resolution = "R51_DEVELOPMENT_AUTHORIZE_CONFIRM" if summary["pass"] else "R51_DEVELOPMENT_FAIL_NO_CONFIRM"
    elif mode == "confirmatory":
        resolution = "R51_EXTERNAL_ROBUSTNESS_PASS_SAME_PROGRAM" if summary["pass"] else "R51_EXTERNAL_ROBUSTNESS_FAIL"
    else:
        resolution = "R51_ENGINEERING_SMOKE"

    out = {
        "campaign": "R51 fresh external robustness battery",
        "mode": mode,
        "resolution": resolution,
        "authorize_confirm": bool(summary["pass"]) if mode == "development" else None,
        "source": source_metadata(),
        "seeds": seeds,
        "episodes_per_seed": episodes,
        "perturbations": list(PERTURBATIONS),
        "thresholds": {
            "clean": CLEAN_GATES,
            "perturbed": PERT_GATES,
            "iso_tolerance": ISO_TOL,
            "required_seed_guards": required,
        },
        "summary": summary,
        "records": records,
        "boundaries": {
            "R48_neural_learnability": "UNCHANGED_ADVERSE",
            "functional_DCR_robustness": "SUPPORTED_IF_CONFIRMATORY_PASS",
            "arbitrary_corruption_robustness": "NOT_ESTABLISHED",
            "global_minimality": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "E6b": "OPEN",
            "E7": "OPEN",
            "consciousness": "NOT_ESTABLISHED",
            "core_version": "POLAR Core v1.1 unchanged",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "resolution": resolution,
        "seed_guard_count": summary["seed_guard_count"],
        "seed_guard_required": summary["seed_guard_required"],
        "pass": summary["pass"],
        "cell_summaries": summary["cell_summaries"],
    }, indent=2, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "development", "confirmatory"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run_panel(a.mode, a.output)


if __name__ == "__main__":
    main()
