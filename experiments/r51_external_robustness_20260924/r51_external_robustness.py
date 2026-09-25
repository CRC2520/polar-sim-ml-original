#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

POPGYM_COMMIT = "e397e5eac9965f9963d18c9f455cd1983bca14fb"
CONDITIONS = ("FULL", "NO_D_COLLAPSE", "NO_C_RESET", "WRONG_R")
PERTURBATIONS = ("CLEAN", "NOISE_05", "OCCLUSION_10", "STALE_05", "SEMANTIC_SHIFT_HALF")
EPISODES_PER_SEED = 2


def stable_seed(seed: int, *labels: str) -> int:
    raw = "|".join(["R51", str(int(seed)), *labels]).encode()
    h = hashlib.sha256(raw).digest()
    return int.from_bytes(h[:8], "little") & 0xFFFFFFFF


def phi(symbol: int) -> int:
    return 3 - int(symbol)


class SymbolPerturber:
    def __init__(self, seed: int, task: str, perturbation: str, episode: int):
        self.perturbation = perturbation
        self.rng = np.random.default_rng(stable_seed(seed, task, perturbation, str(episode)))
        self.prev_raw: Optional[int] = None

    def transform(self, symbol: int, index: int, total_inputs: int) -> Optional[int]:
        s = int(symbol)
        out: Optional[int] = s
        if self.perturbation == "CLEAN":
            out = s
        elif self.perturbation == "NOISE_05":
            if self.rng.random() < 0.05:
                out = (s + int(self.rng.integers(1, 4))) % 4
        elif self.perturbation == "OCCLUSION_10":
            if self.rng.random() < 0.10:
                out = None
        elif self.perturbation == "STALE_05":
            if self.rng.random() < 0.05 and self.prev_raw is not None:
                out = int(self.prev_raw)
        elif self.perturbation == "SEMANTIC_SHIFT_HALF":
            out = phi(s) if index >= total_inputs // 2 else s
        else:
            raise ValueError(self.perturbation)
        self.prev_raw = s
        return out


class AutoencodeController:
    def __init__(self, condition: str):
        self.condition = condition
        self.memory: List[Optional[int]] = []
        self.play_started = False
        self.input_count = 0

    def _encode(self, symbol: Optional[int]) -> Optional[int]:
        if symbol is None:
            return None
        if self.condition == "NO_D_COLLAPSE":
            return 0
        if self.condition == "GENERIC_ISO":
            return phi(symbol)
        return int(symbol)

    def _decode_action(self, internal: Optional[int]) -> int:
        if internal is None:
            return 0
        if self.condition == "GENERIC_ISO":
            return phi(internal)
        return int(internal)

    def consume(self, symbol: Optional[int]) -> None:
        z = self._encode(symbol)
        if self.condition == "NO_C_RESET":
            self.memory = [z]
        else:
            self.memory.append(z)
        self.input_count += 1

    def act(self, obs: Tuple[int, Optional[int]]) -> int:
        mode, symbol = int(obs[0]), obs[1]
        if mode == 1:  # WATCH
            self.consume(symbol)
            return 0

        if not self.play_started:
            # The historical environment switches the mode flag on the final
            # watched symbol; consume that last input exactly once.
            self.consume(symbol)
            self.play_started = True

        if not self.memory:
            return 0

        if self.condition == "WRONG_R":
            z = self.memory.pop(0)
        else:
            z = self.memory.pop(-1)
        return self._decode_action(z)


class RepeatPreviousController:
    def __init__(self, condition: str, k: int):
        self.condition = condition
        self.k = int(k)
        self.memory: List[Optional[int]] = []
        self.input_count = 0

    def _encode(self, symbol: Optional[int]) -> Optional[int]:
        if symbol is None:
            return None
        if self.condition == "NO_D_COLLAPSE":
            return 0
        if self.condition == "GENERIC_ISO":
            return phi(symbol)
        return int(symbol)

    def _decode_action(self, internal: Optional[int]) -> int:
        if internal is None:
            return 0
        if self.condition == "GENERIC_ISO":
            return phi(internal)
        return int(internal)

    def act(self, symbol: Optional[int]) -> int:
        z = self._encode(symbol)
        if self.condition == "NO_C_RESET":
            self.memory = [z]
        else:
            self.memory.append(z)
        self.input_count += 1

        lag = self.k - 1 if self.condition == "WRONG_R" else self.k
        lag = max(1, lag)
        if len(self.memory) < lag:
            return 0
        return self._decode_action(self.memory[-lag])


def make_env(task: str, level: str):
    import popgym
    level = level.lower()
    if task == "autoencode":
        return getattr(popgym, f"Autoencode{level.title()}")()
    if task == "repeat_previous":
        return getattr(popgym, f"RepeatPrevious{level.title()}")()
    raise ValueError(task)


def run_auto_episode(seed: int, level: str, perturbation: str, condition: str, episode: int) -> dict:
    env = make_env("autoencode", level)
    episode_seed = stable_seed(seed, "autoencode", level, "env", str(episode))
    obs = env.reset(seed=episode_seed)
    controller = AutoencodeController(condition)
    pert = SymbolPerturber(seed, "autoencode", perturbation, episode)
    total_inputs = int(env.deck.num_cards)
    correct = scored = 0
    total_reward = 0.0
    actions: List[int] = []
    done = False

    while not done:
        mode, raw_symbol = int(obs[0]), int(obs[1])
        if mode == 1 or not controller.play_started:
            symbol = pert.transform(raw_symbol, controller.input_count, total_inputs)
        else:
            symbol = raw_symbol
        action = controller.act((mode, symbol))
        actions.append(int(action))
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        if abs(float(reward)) > 1e-15:
            scored += 1
            correct += int(float(reward) > 0)
    env.close()
    return {
        "accuracy": correct / max(scored, 1),
        "correct": int(correct),
        "scored": int(scored),
        "return": float(total_reward),
        "actions": actions,
    }


def run_repeat_episode(seed: int, level: str, perturbation: str, condition: str, episode: int) -> dict:
    env = make_env("repeat_previous", level)
    episode_seed = stable_seed(seed, "repeat_previous", level, "env", str(episode))
    obs = env.reset(seed=episode_seed)
    controller = RepeatPreviousController(condition, int(env.k))
    pert = SymbolPerturber(seed, "repeat_previous", perturbation, episode)
    total_inputs = int(env.deck.num_cards)
    correct = scored = 0
    total_reward = 0.0
    actions: List[int] = []
    done = False

    while not done:
        symbol = pert.transform(int(obs), controller.input_count, total_inputs)
        action = controller.act(symbol)
        actions.append(int(action))
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        if abs(float(reward)) > 1e-15:
            scored += 1
            correct += int(float(reward) > 0)
    env.close()
    return {
        "accuracy": correct / max(scored, 1),
        "correct": int(correct),
        "scored": int(scored),
        "return": float(total_reward),
        "actions": actions,
    }


def run_episode(task: str, seed: int, level: str, perturbation: str, condition: str, episode: int) -> dict:
    if task == "autoencode":
        return run_auto_episode(seed, level, perturbation, condition, episode)
    return run_repeat_episode(seed, level, perturbation, condition, episode)


def aggregate_episode_rows(rows: List[dict]) -> dict:
    return {
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "return": float(np.mean([r["return"] for r in rows])),
        "episodes": [{k:v for k,v in r.items() if k != "actions"} for r in rows],
    }


def evaluate_seed(seed: int, level: str) -> dict:
    tasks: Dict[str, dict] = {}
    for task in ("autoencode", "repeat_previous"):
        task_result: Dict[str, dict] = {}
        for pert_name in PERTURBATIONS:
            cond_results = {}
            raw_actions = {}
            for condition in CONDITIONS:
                episodes = [
                    run_episode(task, seed, level, pert_name, condition, ep)
                    for ep in range(EPISODES_PER_SEED)
                ]
                cond_results[condition] = aggregate_episode_rows(episodes)
                raw_actions[condition] = [r["actions"] for r in episodes]

            full = cond_results["FULL"]["accuracy"]
            item = {
                "full_accuracy": float(full),
                "no_d_accuracy": float(cond_results["NO_D_COLLAPSE"]["accuracy"]),
                "no_c_accuracy": float(cond_results["NO_C_RESET"]["accuracy"]),
                "wrong_r_accuracy": float(cond_results["WRONG_R"]["accuracy"]),
                "d_effect": float(full - cond_results["NO_D_COLLAPSE"]["accuracy"]),
                "c_effect": float(full - cond_results["NO_C_RESET"]["accuracy"]),
                "r_effect": float(full - cond_results["WRONG_R"]["accuracy"]),
                "condition_details": cond_results,
            }

            if pert_name == "CLEAN":
                iso_eps = [
                    run_episode(task, seed, level, pert_name, "GENERIC_ISO", ep)
                    for ep in range(EPISODES_PER_SEED)
                ]
                iso = aggregate_episode_rows(iso_eps)
                agreements = []
                total = 0
                same = 0
                for ep in range(EPISODES_PER_SEED):
                    a = raw_actions["FULL"][ep]
                    b = iso_eps[ep]["actions"]
                    if len(a) != len(b):
                        agreements.append(0.0)
                        continue
                    same += sum(int(x == y) for x,y in zip(a,b))
                    total += len(a)
                item["iso_accuracy"] = float(iso["accuracy"])
                item["iso_gap"] = float(abs(full - iso["accuracy"]))
                item["iso_action_agreement"] = float(same / max(total,1))
            task_result[pert_name] = item
        tasks[task] = task_result

    return {
        "seed": int(seed),
        "level": level,
        "tasks": tasks,
    }


def run_panel(lo: int, hi: int, level: str, output: str) -> None:
    rows=[evaluate_seed(seed,level) for seed in range(lo,hi+1)]
    out={
        "campaign":"R51 fresh external robustness battery",
        "source":{
            "repository":"proroklab/popgym",
            "commit":POPGYM_COMMIT,
            "tasks":[f"Autoencode{level.title()}",f"RepeatPrevious{level.title()}"],
        },
        "level":level,
        "seeds":list(range(lo,hi+1)),
        "episodes_per_seed_task_perturbation":EPISODES_PER_SEED,
        "perturbations":list(PERTURBATIONS),
        "records":rows,
        "boundaries":{
            "neural_learnability":"NOT_TESTED",
            "global_minimality":"OPEN",
            "E6b":"OPEN",
            "E7":"OPEN",
            "higher_order_integration":"NOT_TESTED",
            "consciousness":"NOT_ESTABLISHED",
        }
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "level":level,
        "n_seeds":len(rows),
        "tasks":out["source"]["tasks"],
        "perturbations":out["perturbations"],
    },indent=2))


def smoke(output: str) -> None:
    row=evaluate_seed(2215000,"medium")
    for task in row["tasks"]:
        clean=row["tasks"][task]["CLEAN"]
        if clean["iso_gap"] > 1e-12 or clean["iso_action_agreement"] < 1.0-1e-12:
            raise RuntimeError(f"isomorphism failed for {task}")
        if clean["full_accuracy"] < 0.99:
            raise RuntimeError(f"clean full controller failed engineering smoke for {task}")
    out={
        "campaign":"R51 fresh external robustness battery",
        "mode":"engineering_smoke",
        "scientific_seed":False,
        "record":row,
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        t:{
            "clean":row["tasks"][t]["CLEAN"]["full_accuracy"],
            "iso_gap":row["tasks"][t]["CLEAN"]["iso_gap"],
            "iso_action_agreement":row["tasks"][t]["CLEAN"]["iso_action_agreement"],
        } for t in row["tasks"]
    },indent=2))


def main():
    p=argparse.ArgumentParser()
    sp=p.add_subparsers(dest="cmd",required=True)
    s=sp.add_parser("smoke"); s.add_argument("--output",required=True)
    r=sp.add_parser("run")
    r.add_argument("--seeds",required=True)
    r.add_argument("--level",choices=["medium","hard"],required=True)
    r.add_argument("--output",required=True)
    a=p.parse_args()
    if a.cmd=="smoke":
        smoke(a.output)
    else:
        lo,hi=map(int,a.seeds.split(":"))
        run_panel(lo,hi,a.level,a.output)

if __name__=="__main__":
    main()
