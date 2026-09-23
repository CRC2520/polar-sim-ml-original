#!/usr/bin/env python3
"""R45 official POPGym LSTM reproduction and paired checkpoint qualification.

Scientific hyperparameters are frozen in PREREG_R45.md.  This runner imports the
historical upstream POPGym LSTM implementation and reproduces the upstream PPO
configuration with one declared infrastructure adaptation: 2 rollout workers x
32 envs/worker instead of 4 x 16, keeping the aggregate train batch unchanged.
"""
from __future__ import annotations

import argparse
import copy
import glob
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_STEPS = 1_048_576
DEV_TRAIN_SEEDS = [2166001, 2166002, 2166003]
CONF_TRAIN_SEEDS = [2167001, 2167002, 2167003]
DEV_EVAL_SEEDS = list(range(2166101, 2166113))
CONF_EVAL_SEEDS = list(range(2167101, 2167113))
EPISODES = 4
LEVELS = ("Easy", "Medium", "Hard")

COMPETENCE_SCORE = 0.90
HISTORY_BENEFIT = 0.15
CORE_NONINFERIORITY_MARGIN = -0.05
ISO_TOL = 1e-12


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def package_versions() -> Dict[str, str]:
    names = ["ray", "torch", "gym", "numpy", "scipy", "popgym"]
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "UNKNOWN"
    return out


def build_algorithm(level: str, train_seed: int):
    import ray
    import torch
    import popgym
    from popgym import wrappers
    from popgym.baselines.ray_models.ray_lstm import LSTM
    from ray.tune.registry import register_env

    try:
        from ray.rllib.algorithms.ppo import PPO
    except Exception:
        from ray.rllib.agents.ppo import PPOTrainer as PPO

    cls = getattr(popgym, f"StatelessCartPole{level}")
    env_id = popgym.ALL_ENVS[cls]["id"]

    def make_wrapped(_cfg=None):
        return wrappers.Antialias(wrappers.PreviousAction(cls()))

    register_env(env_id, make_wrapped)

    # Upstream ppo.py config at proroklab/popgym@e397e5e.
    h = 128
    h_memory = 256
    bptt_cutoff = 1024

    # Infrastructure adaptation frozen before execution:
    # 2 workers * 32 envs == upstream aggregate 4 * 16 envs.
    num_workers = 2
    num_envs_per_worker = 32
    train_batch_size = bptt_cutoff * num_workers * num_envs_per_worker

    config = {
        "env": env_id,
        "framework": "torch",
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "num_gpus": 0,
        "vf_loss_coeff": 1.0,
        "train_batch_size": train_batch_size,
        "rollout_fragment_length": bptt_cutoff,
        "sgd_minibatch_size": 8 * bptt_cutoff,
        "gamma": 0.99,
        "horizon": bptt_cutoff,
        "batch_mode": "complete_episodes",
        "min_sample_timesteps_per_iteration": train_batch_size,
        "seed": int(train_seed),
        "model": {
            "max_seq_len": bptt_cutoff,
            "custom_model": LSTM,
            "custom_model_config": {
                "preprocessor_input_size": h,
                "preprocessor": torch.nn.Sequential(
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "preprocessor_output_size": h,
                "hidden_size": h_memory,
                "postprocessor": torch.nn.Identity(),
                "actor": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "critic": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "postprocessor_output_size": h,
            },
        },
    }

    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            ignore_reinit_error=True,
            num_cpus=max(3, (os.cpu_count() or 2)),
            log_to_driver=True,
        )

    algo = PPO(config=config)
    return algo, cls, wrappers, env_id, config


def clone_state(state):
    return [np.array(s, copy=True) for s in state]


def compute_action(algo, obs, state):
    result = algo.compute_single_action(
        obs,
        state=state,
        explore=False,
        full_fetch=True,
    )
    if isinstance(result, tuple) and len(result) == 3:
        action, state_out, info = result
        return action, state_out, info
    # Defensive compatibility with older RLlib return shapes.
    if isinstance(result, tuple) and len(result) == 2:
        action, state_out = result
        return action, state_out, {}
    return result, state, {}


def run_lstm_episode(algo, cls, wrappers, episode_seed: int, reset_each_step: bool):
    env = wrappers.Antialias(wrappers.PreviousAction(cls()))
    raw = getattr(env, "unwrapped", env)
    max_steps = int(getattr(raw, "max_episode_length", 0) or 0)
    if max_steps <= 0:
        raise RuntimeError("historical max_episode_length unavailable")

    q = env.reset(seed=int(episode_seed))
    # Historical Gym 0.24 wrappers return a tuple-structured observation here.
    # Only treat a 2-tuple with a dict in position 1 as the newer (obs, info)
    # reset API; otherwise preserve the wrapper observation structure intact.
    if isinstance(q, tuple) and len(q) == 2 and isinstance(q[1], dict):
        obs = q[0]
    else:
        obs = q
    policy = algo.get_policy()
    initial = clone_state(policy.get_initial_state())
    state = clone_state(initial)

    total_reward = 0.0
    steps = 0
    while True:
        use_state = clone_state(initial) if reset_each_step else state
        action, new_state, _ = compute_action(algo, obs, use_state)
        q = env.step(action)
        if len(q) == 4:
            nxt, reward, done, _info = q
        elif len(q) == 5:
            nxt, reward, terminated, truncated, _info = q
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"unexpected step tuple length {len(q)}")
        steps += 1
        total_reward += float(reward)
        obs = nxt
        if not reset_each_step:
            state = clone_state(new_state)
        if done or steps >= max_steps:
            break
    env.close()
    return {
        "steps": int(steps),
        "max_steps": int(max_steps),
        "score": float(steps / max_steps),
        "reward": float(total_reward),
    }


def run_core_episode(level: str, controller: str, episode_seed: int):
    from experiments.r44_published_external_reference_20260923 import r44_reference
    return r44_reference.run_episode(level, controller, int(episode_seed))


def evaluate_checkpoint(algo, cls, wrappers, level: str, eval_seeds: Iterable[int]):
    records = []
    for seed in eval_seeds:
        episode_rows = []
        for ep in range(EPISODES):
            episode_seed = int(seed) * 1000 + ep
            intact = run_lstm_episode(algo, cls, wrappers, episode_seed, False)
            reset = run_lstm_episode(algo, cls, wrappers, episode_seed, True)
            core = run_core_episode(level, "CORE_C", episode_seed)
            iso = run_core_episode(level, "GENERIC_ISO", episode_seed)
            episode_rows.append(
                {
                    "episode_seed": episode_seed,
                    "lstm_intact": intact,
                    "lstm_step_reset": reset,
                    "core_c": core,
                    "generic_iso": iso,
                }
            )

        mean = lambda cond: float(np.mean([r[cond]["score"] for r in episode_rows]))
        intact_score = mean("lstm_intact")
        reset_score = mean("lstm_step_reset")
        core_score = mean("core_c")
        iso_score = mean("generic_iso")
        records.append(
            {
                "seed": int(seed),
                "level": level,
                "lstm_intact_score": intact_score,
                "lstm_step_reset_score": reset_score,
                "history_benefit": intact_score - reset_score,
                "core_score": core_score,
                "iso_score": iso_score,
                "core_minus_lstm": core_score - intact_score,
                "iso_gap": abs(core_score - iso_score),
                "episodes": episode_rows,
            }
        )
    return records


def summarize_records(records: List[Dict[str, Any]]):
    med = lambda k: float(np.median([r[k] for r in records]))
    summary = {
        "n_eval_seeds": len(records),
        "median_lstm_intact_score": med("lstm_intact_score"),
        "median_lstm_step_reset_score": med("lstm_step_reset_score"),
        "median_history_benefit": med("history_benefit"),
        "median_core_score": med("core_score"),
        "median_core_minus_lstm": med("core_minus_lstm"),
        "median_iso_gap": med("iso_gap"),
    }
    summary["eligible"] = bool(
        summary["median_lstm_intact_score"] >= COMPETENCE_SCORE
        and summary["median_history_benefit"] >= HISTORY_BENEFIT
    )
    return summary


def train_eval(level: str, train_seed: int, steps: int, output: str, checkpoint_dir: str):
    if level not in LEVELS:
        raise ValueError(level)
    if steps != TRAIN_STEPS:
        raise RuntimeError(f"R45 frozen training budget is {TRAIN_STEPS}, got {steps}")

    eval_seeds = DEV_EVAL_SEEDS if level == "Easy" else CONF_EVAL_SEEDS

    algo = None
    try:
        algo, cls, wrappers, env_id, config = build_algorithm(level, train_seed)
        last = {}
        while int(last.get("timesteps_total", 0)) < steps:
            last = algo.train()
            print(
                json.dumps(
                    {
                        "level": level,
                        "train_seed": train_seed,
                        "timesteps_total": int(last.get("timesteps_total", 0)),
                        "episode_reward_mean": float(last.get("episode_reward_mean", float("nan"))),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        saved = algo.save(checkpoint_dir)
        saved_path = str(saved)

        records = evaluate_checkpoint(algo, cls, wrappers, level, eval_seeds)
        summary = summarize_records(records)
        out = {
            "campaign": "R45 official POPGym LSTM reproduction",
            "mode": "train_eval",
            "level": level,
            "train_seed": int(train_seed),
            "training_budget_timesteps": int(steps),
            "external_source": {
                "repository": "proroklab/popgym",
                "commit": "e397e5e",
                "model": "popgym.baselines.ray_models.ray_lstm.LSTM",
                "launcher_reference": "popgym/baselines/ppo.py",
                "published_training_budget_timesteps": 15_000_000,
                "r45_is_full_paper_budget_replication": False,
            },
            "env_id": env_id,
            "rllib_config_guard": {
                "num_workers": config["num_workers"],
                "num_envs_per_worker": config["num_envs_per_worker"],
                "train_batch_size": config["train_batch_size"],
                "sgd_minibatch_size": config["sgd_minibatch_size"],
                "gamma": config["gamma"],
                "horizon": config["horizon"],
                "batch_mode": config["batch_mode"],
                "max_seq_len": config["model"]["max_seq_len"],
                "hidden_size": config["model"]["custom_model_config"]["hidden_size"],
            },
            "versions": package_versions(),
            "checkpoint_path_runtime": saved_path,
            "last_training_result": {
                "timesteps_total": int(last.get("timesteps_total", 0)),
                "episode_reward_mean": _jsonable(last.get("episode_reward_mean")),
                "episodes_total": _jsonable(last.get("episodes_total")),
                "training_iteration": _jsonable(last.get("training_iteration")),
            },
            "eval_seeds": list(eval_seeds),
            "episodes_per_eval_seed": EPISODES,
            "thresholds": {
                "competence_score": COMPETENCE_SCORE,
                "history_benefit": HISTORY_BENEFIT,
            },
            "summary": summary,
            "records": records,
            "boundaries": {
                "author_provided_paper_checkpoint": False,
                "full_15M_paper_endpoint_reproduced": False,
                "POLAR_superiority": "NOT_ESTABLISHED",
                "E6b": "OPEN",
                "E7": "OPEN",
                "consciousness": "NOT_ESTABLISHED",
            },
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(_jsonable(out), indent=2, sort_keys=True) + "\n")
        print(json.dumps({"summary": summary, "output": output}, indent=2, sort_keys=True))
    finally:
        if algo is not None:
            try:
                algo.stop()
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


def collect_results(input_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(str(Path(input_dir) / "**" / "R45_*_RESULT.json"), recursive=True))
    rows = []
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        if d.get("campaign") == "R45 official POPGym LSTM reproduction":
            d["_artifact_path"] = p
            rows.append(d)
    return rows


def aggregate(phase: str, input_dir: str, output: str):
    rows = collect_results(input_dir)
    if phase == "development":
        dev = [r for r in rows if r.get("level") == "Easy" and r.get("train_seed") in DEV_TRAIN_SEEDS]
        by_seed = {int(r["train_seed"]): r for r in dev}
        missing = [s for s in DEV_TRAIN_SEEDS if s not in by_seed]
        if missing:
            raise RuntimeError(f"missing development results for seeds {missing}")
        ordered = [by_seed[s] for s in DEV_TRAIN_SEEDS]
        eligible = sum(bool(r["summary"]["eligible"]) for r in ordered)
        authorize = eligible >= 2
        resolution = (
            "R45_DEVELOPMENT_AUTHORIZE_CONFIRM"
            if authorize
            else "R45_OFFICIAL_LSTM_REPRODUCTION_DEV_FAIL_NO_CONFIRM"
        )
        out = {
            "campaign": "R45 official POPGym LSTM reproduction",
            "phase": phase,
            "resolution": resolution,
            "authorize_confirm": authorize,
            "eligible_checkpoints": eligible,
            "required_eligible_checkpoints": 2,
            "checkpoints": [
                {
                    "train_seed": r["train_seed"],
                    "summary": r["summary"],
                    "artifact_path": r["_artifact_path"],
                }
                for r in ordered
            ],
            "boundaries": {
                "confirmation_seeds_opened": bool(authorize),
                "author_provided_paper_checkpoint": False,
                "full_15M_paper_endpoint_reproduced": False,
                "POLAR_superiority": "NOT_ESTABLISHED",
                "E6b": "OPEN",
                "E7": "OPEN",
                "consciousness": "NOT_ESTABLISHED",
            },
        }
    elif phase == "confirmatory":
        levels = {}
        all_pass = True
        for level in ("Medium", "Hard"):
            sub = [
                r for r in rows
                if r.get("level") == level and r.get("train_seed") in CONF_TRAIN_SEEDS
            ]
            by_seed = {int(r["train_seed"]): r for r in sub}
            missing = [s for s in CONF_TRAIN_SEEDS if s not in by_seed]
            if missing:
                raise RuntimeError(f"missing {level} confirmatory results for seeds {missing}")
            ordered = [by_seed[s] for s in CONF_TRAIN_SEEDS]
            eligible = sum(bool(r["summary"]["eligible"]) for r in ordered)
            flat = [rec for r in ordered for rec in r["records"]]
            med_core = float(np.median([x["core_score"] for x in flat]))
            med_iso = float(np.median([x["iso_gap"] for x in flat]))
            med_gap = float(np.median([x["core_minus_lstm"] for x in flat]))
            passed = bool(
                eligible >= 2
                and med_core >= COMPETENCE_SCORE
                and med_iso <= ISO_TOL
                and med_gap >= CORE_NONINFERIORITY_MARGIN
            )
            all_pass = all_pass and passed
            levels[level] = {
                "pass": passed,
                "eligible_checkpoints": eligible,
                "required_eligible_checkpoints": 2,
                "median_core_score": med_core,
                "median_iso_gap": med_iso,
                "median_core_minus_lstm": med_gap,
                "noninferiority_margin": CORE_NONINFERIORITY_MARGIN,
                "checkpoints": [
                    {"train_seed": r["train_seed"], "summary": r["summary"]}
                    for r in ordered
                ],
            }
        resolution = (
            "R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_PASS"
            if all_pass
            else "R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_FAIL"
        )
        out = {
            "campaign": "R45 official POPGym LSTM reproduction",
            "phase": phase,
            "resolution": resolution,
            "levels": levels,
            "interpretation": (
                "paired same-episode comparison against newly trained official-code "
                "POPGym LSTM checkpoints at the frozen bounded R45 training budget"
            ),
            "boundaries": {
                "author_provided_paper_checkpoint": False,
                "full_15M_paper_endpoint_reproduced": False,
                "POLAR_superiority": "NOT_ESTABLISHED",
                "external_D_or_R": "NOT_TESTED",
                "E6b": "OPEN",
                "E7": "OPEN",
                "global_minimality": "OPEN",
                "consciousness": "NOT_ESTABLISHED",
                "core_version": "POLAR Core v1.1 unchanged",
            },
        }
    else:
        raise ValueError(phase)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(_jsonable(out), indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train-eval")
    t.add_argument("--level", choices=LEVELS, required=True)
    t.add_argument("--train-seed", type=int, required=True)
    t.add_argument("--steps", type=int, default=TRAIN_STEPS)
    t.add_argument("--output", required=True)
    t.add_argument("--checkpoint-dir", required=True)

    a = sub.add_parser("aggregate")
    a.add_argument("--phase", choices=["development", "confirmatory"], required=True)
    a.add_argument("--input-dir", required=True)
    a.add_argument("--output", required=True)

    args = p.parse_args()
    if args.cmd == "train-eval":
        train_eval(args.level, args.train_seed, args.steps, args.output, args.checkpoint_dir)
    else:
        aggregate(args.phase, args.input_dir, args.output)


if __name__ == "__main__":
    main()
