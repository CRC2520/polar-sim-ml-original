#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

POPGYM_COMMIT = "e397e5eac9965f9963d18c9f455cd1983bca14fb"
SMOKE_SEED = 2185000
DEV_SEEDS = list(range(2186001, 2186017))
CONF_SEEDS = list(range(2187001, 2187033))
EPISODES_PER_SEED = 4
N_POSITIONS = 52
N_SYMBOLS = 13
N_PAIRS = 26
ISO_TOL = 1e-12

CONDITIONS = (
    "FULL_DCR",
    "NO_D_COLLAPSE",
    "C1_ONLY",
    "WRONG_R_CYCLIC",
    "GENERIC_ISO",
)

MEDIAN_GATES = {
    "full_completion": 0.95,
    "full_return": 0.30,
    "d_effect": 0.35,
    "c_effect": 0.50,
    "r_effect": 0.25,
}
SEED_GATES = {
    "full_completion": 0.90,
    "full_return": 0.20,
    "d_effect": 0.20,
    "c_effect": 0.30,
    "r_effect": 0.15,
}


def _cyclic_pick(candidates: Sequence[int], last_action: int, n: int) -> int:
    if not candidates:
        raise RuntimeError("no candidates")
    start = (int(last_action) + 1) % n if last_action >= 0 else 0
    return min((int(i) for i in candidates), key=lambda i: ((i - start) % n, i))


@dataclass
class EpisodeRecord:
    total_return: float
    matches: int
    completion_fraction: float
    actions_used: int
    done: bool


class MatcherAgent:
    """Deterministic functional Concentration controller.

    All scientific conditions share action scan order, pair-local state and
    failed-pair handling. Conditions differ only in the frozen D/C/R lesion.
    """

    def __init__(self, condition: str, facedown_card: int):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        self.facedown_card = int(facedown_card)
        self.memory: Dict[int, int] = {}
        self.failed_pairs: Set[Tuple[int, int]] = set()
        self.solved: Set[int] = set()
        self.pair_actions: List[int] = []
        self.last_action = -1

    def _encode(self, native_symbol: int) -> int:
        s = int(native_symbol)
        if self.condition == "NO_D_COLLAPSE":
            return 0
        if self.condition == "GENERIC_ISO":
            return (N_SYMBOLS - 1) - s
        return s

    def _target_identity(self, first_internal: int) -> int:
        if self.condition == "WRONG_R_CYCLIC":
            return (int(first_internal) + 1) % N_SYMBOLS
        return int(first_internal)

    @staticmethod
    def _pair_key(i: int, j: int) -> Tuple[int, int]:
        return tuple(sorted((int(i), int(j))))

    def _unsolved(self, exclude: Optional[Set[int]] = None) -> List[int]:
        exclude = exclude or set()
        return [
            i
            for i in range(N_POSITIONS)
            if i not in self.solved and i not in exclude
        ]

    def _known_candidates(
        self, target_identity: int, exclude: Set[int], first: Optional[int] = None
    ) -> List[int]:
        out = []
        for i in self._unsolved(exclude):
            if i not in self.memory:
                continue
            if int(self.memory[i]) != int(target_identity):
                continue
            if first is not None and self._pair_key(first, i) in self.failed_pairs:
                continue
            out.append(i)
        return out

    def _unknown_candidates(self, exclude: Set[int]) -> List[int]:
        return [i for i in self._unsolved(exclude) if i not in self.memory]

    def _known_pair_first_candidates(self) -> List[int]:
        known = [i for i in self._unsolved() if i in self.memory]
        candidates = []
        for i in known:
            target = self._target_identity(self.memory[i])
            for j in known:
                if i == j:
                    continue
                if self.memory[j] != target:
                    continue
                if self._pair_key(i, j) in self.failed_pairs:
                    continue
                candidates.append(i)
                break
        return candidates

    def select_action(self, obs: np.ndarray) -> int:
        # The historical benchmark declares previous action as required context.
        # Every condition retains self.last_action; only persistent hidden card
        # identity/location memory is lesioned in C1_ONLY.
        if len(self.pair_actions) == 0:
            pairable = self._known_pair_first_candidates()
            if pairable:
                return _cyclic_pick(pairable, self.last_action, N_POSITIONS)
            unknown = self._unknown_candidates(set())
            if unknown:
                return _cyclic_pick(unknown, self.last_action, N_POSITIONS)
            remaining = self._unsolved()
            return _cyclic_pick(remaining, self.last_action, N_POSITIONS)

        first = int(self.pair_actions[0])
        if first not in self.memory:
            raise RuntimeError("first card identity missing after first flip")
        target = self._target_identity(self.memory[first])
        known = self._known_candidates(target, {first}, first=first)
        if known:
            return _cyclic_pick(known, self.last_action, N_POSITIONS)

        unknown = self._unknown_candidates({first})
        if unknown:
            return _cyclic_pick(unknown, self.last_action, N_POSITIONS)

        remaining = [
            i
            for i in self._unsolved({first})
            if self._pair_key(first, i) not in self.failed_pairs
        ]
        if not remaining:
            remaining = self._unsolved({first})
        return _cyclic_pick(remaining, self.last_action, N_POSITIONS)

    def observe(self, action: int, obs_after: np.ndarray, reward: float) -> None:
        action = int(action)
        symbol = int(np.asarray(obs_after)[action])
        if symbol == self.facedown_card:
            raise RuntimeError("selected card was not visible after action")
        if not (0 <= symbol < N_SYMBOLS):
            raise RuntimeError(f"unexpected native symbol {symbol}")

        self.memory[action] = self._encode(symbol)
        self.pair_actions.append(action)
        self.last_action = action

        if len(self.pair_actions) == 2:
            i, j = self.pair_actions
            key = self._pair_key(i, j)
            if reward > 0:
                self.solved.update((i, j))
                self.memory.pop(i, None)
                self.memory.pop(j, None)
                self.failed_pairs = {p for p in self.failed_pairs if i not in p and j not in p}
            elif reward < 0:
                self.failed_pairs.add(key)

            self.pair_actions = []
            if self.condition == "C1_ONLY":
                # Preserve externally observable solved status and previous action,
                # but erase hidden cross-pair identity/location memory.
                self.memory.clear()
                self.failed_pairs.clear()

        if len(self.pair_actions) > 2:
            raise RuntimeError("invalid pair phase")


def _make_env():
    import popgym

    env = popgym.ConcentrationHard()
    if int(env.action_space.n) != N_POSITIONS:
        raise RuntimeError(f"action contract changed: {env.action_space}")
    if tuple(env.observation_space.shape) != (N_POSITIONS,):
        raise RuntimeError(f"observation contract changed: {env.observation_space}")
    if int(env.facedown_card) != N_SYMBOLS:
        raise RuntimeError(f"identity contract changed: facedown={env.facedown_card}")
    if int(env.episode_length) != 104:
        raise RuntimeError(f"episode length changed: {env.episode_length}")
    return env


def run_agent_episode(condition: str, episode_seed: int) -> EpisodeRecord:
    env = _make_env()
    obs = np.asarray(env.reset(seed=int(episode_seed)))
    agent = MatcherAgent(condition, int(env.facedown_card))
    total_return = 0.0
    matches = 0
    actions = 0
    done = False

    while not done and actions < int(env.episode_length):
        action = agent.select_action(obs)
        obs_after, reward, done, _ = env.step(int(action))
        obs_after = np.asarray(obs_after)
        agent.observe(action, obs_after, float(reward))
        total_return += float(reward)
        if reward > 0:
            matches += 1
        actions += 1
        obs = obs_after

    env.close()
    return EpisodeRecord(
        total_return=float(total_return),
        matches=int(matches),
        completion_fraction=float(matches / N_PAIRS),
        actions_used=int(actions),
        done=bool(done),
    )


def run_oracle_episode(episode_seed: int) -> EpisodeRecord:
    env = _make_env()
    obs = np.asarray(env.reset(seed=int(episode_seed)))
    solved: Set[int] = set()
    pending: List[int] = []
    total_return = 0.0
    matches = 0
    actions = 0
    done = False
    last_action = -1

    while not done and actions < int(env.episode_length):
        state = np.asarray(env.get_state()[0], dtype=int)
        if len(pending) == 0:
            remaining = [i for i in range(N_POSITIONS) if i not in solved]
            pairable = []
            for i in remaining:
                for j in remaining:
                    if i < j and state[i] == state[j]:
                        pairable.append((i, j))
            if not pairable:
                raise RuntimeError("oracle found no remaining pair")
            flat = sorted({i for p in pairable for i in p})
            action = _cyclic_pick(flat, last_action, N_POSITIONS)
        else:
            first = pending[0]
            candidates = [
                i
                for i in range(N_POSITIONS)
                if i not in solved and i != first and state[i] == state[first]
            ]
            action = _cyclic_pick(candidates, last_action, N_POSITIONS)

        obs_after, reward, done, _ = env.step(int(action))
        pending.append(int(action))
        last_action = int(action)
        total_return += float(reward)
        if reward > 0:
            matches += 1
        actions += 1
        obs = np.asarray(obs_after)

        if len(pending) == 2:
            if reward > 0:
                solved.update(pending)
            pending = []

    env.close()
    return EpisodeRecord(
        total_return=float(total_return),
        matches=int(matches),
        completion_fraction=float(matches / N_PAIRS),
        actions_used=int(actions),
        done=bool(done),
    )


def _mean_records(records: List[EpisodeRecord]) -> dict:
    return {
        "return": float(np.mean([r.total_return for r in records])),
        "matches": float(np.mean([r.matches for r in records])),
        "completion": float(np.mean([r.completion_fraction for r in records])),
        "actions": float(np.mean([r.actions_used for r in records])),
        "done_fraction": float(np.mean([r.done for r in records])),
    }


def evaluate_seed(seed: int) -> dict:
    episode_seeds = [int(seed) * 100 + ep for ep in range(EPISODES_PER_SEED)]
    cond = {}
    raw = {}
    for name in CONDITIONS:
        rs = [run_agent_episode(name, s) for s in episode_seeds]
        cond[name] = _mean_records(rs)
        raw[name] = [r.__dict__ for r in rs]

    oracle_rs = [run_oracle_episode(s) for s in episode_seeds]
    cond["ORACLE_STATE"] = _mean_records(oracle_rs)
    raw["ORACLE_STATE"] = [r.__dict__ for r in oracle_rs]

    full = cond["FULL_DCR"]
    d = cond["NO_D_COLLAPSE"]
    c = cond["C1_ONLY"]
    rr = cond["WRONG_R_CYCLIC"]
    iso = cond["GENERIC_ISO"]

    row = {
        "seed": int(seed),
        "episode_seeds": episode_seeds,
        "conditions": cond,
        "full_completion": full["completion"],
        "full_return": full["return"],
        "d_effect": full["completion"] - d["completion"],
        "c_effect": full["completion"] - c["completion"],
        "r_effect": full["completion"] - rr["completion"],
        "iso_completion_gap": abs(full["completion"] - iso["completion"]),
        "iso_return_gap": abs(full["return"] - iso["return"]),
        "oracle_completion": cond["ORACLE_STATE"]["completion"],
        "raw_episodes": raw,
    }
    row["seed_guard"] = bool(
        row["full_completion"] >= SEED_GATES["full_completion"]
        and row["full_return"] >= SEED_GATES["full_return"]
        and row["d_effect"] >= SEED_GATES["d_effect"]
        and row["c_effect"] >= SEED_GATES["c_effect"]
        and row["r_effect"] >= SEED_GATES["r_effect"]
        and row["iso_completion_gap"] <= ISO_TOL
        and row["iso_return_gap"] <= ISO_TOL
    )
    return row


def summarize(rows: List[dict], required: int) -> dict:
    med = lambda k: float(np.median([r[k] for r in rows]))
    out = {
        "n": len(rows),
        "median_full_completion": med("full_completion"),
        "median_full_return": med("full_return"),
        "median_d_effect": med("d_effect"),
        "median_c_effect": med("c_effect"),
        "median_r_effect": med("r_effect"),
        "median_iso_completion_gap": med("iso_completion_gap"),
        "median_iso_return_gap": med("iso_return_gap"),
        "median_oracle_completion": med("oracle_completion"),
        "seed_guard_count": sum(bool(r["seed_guard"]) for r in rows),
        "seed_guard_required": int(required),
    }
    out["pass"] = bool(
        out["median_full_completion"] >= MEDIAN_GATES["full_completion"]
        and out["median_full_return"] >= MEDIAN_GATES["full_return"]
        and out["median_d_effect"] >= MEDIAN_GATES["d_effect"]
        and out["median_c_effect"] >= MEDIAN_GATES["c_effect"]
        and out["median_r_effect"] >= MEDIAN_GATES["r_effect"]
        and out["median_iso_completion_gap"] <= ISO_TOL
        and out["median_iso_return_gap"] <= ISO_TOL
        and out["seed_guard_count"] >= required
    )
    return out


def source_metadata() -> dict:
    try:
        version = importlib.metadata.version("popgym")
    except importlib.metadata.PackageNotFoundError:
        version = "UNKNOWN"
    return {
        "repository": "proroklab/popgym",
        "commit": POPGYM_COMMIT,
        "installed_version": version,
        "environment": "ConcentrationHard",
    }


def write_output(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(output: str) -> None:
    row = evaluate_seed(SMOKE_SEED)
    out = {
        "campaign": "R47 external integrated D+C+R",
        "mode": "engineering_smoke",
        "scientific_seed": False,
        "seed": SMOKE_SEED,
        "source": source_metadata(),
        "summary": {
            k: row[k]
            for k in (
                "full_completion",
                "full_return",
                "d_effect",
                "c_effect",
                "r_effect",
                "iso_completion_gap",
                "iso_return_gap",
                "oracle_completion",
                "seed_guard",
            )
        },
    }
    write_output(output, out)
    print(json.dumps(out, indent=2, sort_keys=True))


def run_development(output: str) -> None:
    rows = [evaluate_seed(s) for s in DEV_SEEDS]
    s = summarize(rows, required=13)
    resolution = (
        "R47_DEVELOPMENT_AUTHORIZE_CONFIRM"
        if s["pass"]
        else "R47_DEVELOPMENT_FAIL_NO_CONFIRM"
    )
    out = {
        "campaign": "R47 external integrated D+C+R",
        "mode": "development",
        "resolution": resolution,
        "authorize_confirm": bool(s["pass"]),
        "source": source_metadata(),
        "seeds": DEV_SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "thresholds": {
            "median": MEDIAN_GATES,
            "seed": SEED_GATES,
            "iso_tolerance": ISO_TOL,
            "seed_guard_required": 13,
        },
        "summary": s,
        "records": rows,
        "boundaries": {
            "single_task_external_DCR_conjunction": "SUPPORTED_IF_CONFIRMATORY_PASS",
            "external_online_R_acquisition": "SUPPORTED_IF_CONFIRMATORY_PASS_BOUNDED_TASK",
            "neural_R_learnability_from_reward": "NOT_TESTED",
            "global_minimality": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "E6b": "OPEN",
            "E7": "OPEN",
            "consciousness": "NOT_ESTABLISHED",
            "core_version": "POLAR Core v1.1 unchanged",
        },
    }
    write_output(output, out)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2, sort_keys=True))


def run_confirmatory(output: str) -> None:
    rows = [evaluate_seed(s) for s in CONF_SEEDS]
    s = summarize(rows, required=28)
    resolution = (
        "R47_EXTERNAL_INTEGRATED_DCR_PASS_SAME_PROGRAM"
        if s["pass"]
        else "R47_EXTERNAL_INTEGRATED_DCR_FAIL"
    )
    out = {
        "campaign": "R47 external integrated D+C+R",
        "mode": "confirmatory",
        "resolution": resolution,
        "source": source_metadata(),
        "seeds": CONF_SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "thresholds": {
            "median": MEDIAN_GATES,
            "seed": SEED_GATES,
            "iso_tolerance": ISO_TOL,
            "seed_guard_required": 28,
        },
        "summary": s,
        "records": rows,
        "boundaries": {
            "single_task_external_DCR_conjunction": (
                "SUPPORTED_SAME_PROGRAM" if s["pass"] else "NOT_ESTABLISHED"
            ),
            "external_online_R_acquisition": (
                "SUPPORTED_BOUNDED_TASK" if s["pass"] else "NOT_ESTABLISHED"
            ),
            "neural_R_learnability_from_reward": "NOT_TESTED",
            "global_minimality": "OPEN",
            "POLAR_superiority": "NOT_ESTABLISHED",
            "E6b": "OPEN",
            "E7": "OPEN",
            "AGI_ASI": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
            "core_version": "POLAR Core v1.1 unchanged",
        },
    }
    write_output(output, out)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2, sort_keys=True))


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
