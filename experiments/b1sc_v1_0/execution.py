"""B1-SC v1.0 guarded scientific execution pipeline.

The public entry points in this module are inert unless a START_REQUEST.json exists
and passes the immutable execution guard. QA helpers use only the ``qa`` seed
namespace and are not scientific evidence.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import time
import traceback
from typing import Iterable, Sequence

import numpy as np

from experiments.b1sc_v1_0 import implementation as sc

ROOT = Path(__file__).resolve().parent
REPO_ROOT = sc.REPO_ROOT
EXEC_BRANCH = "research/b1sc-v1.0-execution-readiness-20260915"
QUALIFIED_BASE_COMMIT = "5ea44a037bae663ec2885bf8b36d2595c08940c4"
START_REQUEST = ROOT / "START_REQUEST.json"
EXECUTION_FREEZE = ROOT / "run" / "EXECUTION_IMPLEMENTATION_FREEZE.json"
EXECUTION_WORKFLOW = ".github/workflows/b1sc-v1-0-execution.yml"

GRAPH_HP = dict(
    lr=.0003, gamma=.99, rollout_steps=512, minibatch=128, epochs=4,
    gae_lambda=.95, clip=.2, value_coef=.5, max_grad_norm=.5, adam_eps=1e-5,
)
PPO_HP = dict(
    GRAPH_HP, rollout_steps=2048, minibatch=64, epochs=10,
)

def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)

def write_json(path: Path | str, obj, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(data)
    else:
        path.write_text(data, encoding="utf-8", newline="\n")

def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def runtime():
    global torch, nn, RoutingActor, GenericActor, Value, log_squashed_gaussian
    global model_digest, NativeEnv, runtime_lock
    import torch
    from torch import nn
    from b1s.execution.core import (
        RoutingActor, GenericActor, Value, log_squashed_gaussian, model_digest,
    )
    from b1s.execution.instrument import NativeEnv, runtime_lock
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return runtime_lock()

def frozen_source_hashes(freeze: dict) -> dict:
    return dict(freeze.get("source_hashes", {}))

def execution_guard() -> dict:
    """Authorize exactly one scientific run attempt.

    The authorization commit must add only START_REQUEST.json on top of the
    branch state reviewed by the author. A rerun is rejected before any fit.
    """
    require(os.environ.get("GITHUB_REPOSITORY") == sc.REPOSITORY, "Wrong repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + EXEC_BRANCH, "Wrong execution branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Scientific reruns are not authorized")
    require(START_REQUEST.exists(), "START_REQUEST.json is absent")
    require(EXECUTION_FREEZE.exists(), "Execution implementation freeze is absent")
    req = read_json(START_REQUEST)
    freeze = read_json(EXECUTION_FREEZE)

    require(req.get("schema") == sc.SCHEMA, "Wrong request schema")
    require(req.get("authorize_b1sc_v1_0") is True, "Execution not explicitly authorized")
    require(req.get("authorized_run_attempt") == 1, "Only run attempt 1 can be authorized")
    require(req.get("B1E_executed") is False, "B1-E boundary violated")
    require(req.get("final_seeds_generated") is False, "Final seeds boundary violated")
    require(req.get("qualified_source_commit") == freeze["qualified_source_commit"], "Qualified source mismatch")
    require(req.get("execution_freeze_sha256") == sha256_file(EXECUTION_FREEZE), "Execution freeze mismatch")
    require(req.get("registry_sha256") == freeze["registry_sha256"], "Registry mismatch")
    require(req.get("design_freeze_sha256") == freeze["design_freeze_sha256"], "Design freeze mismatch")
    require(req.get("implementation_freeze_sha256") == freeze["implementation_freeze_sha256"], "Implementation freeze mismatch")
    require(req.get("workflow_sha256") == freeze["source_hashes"][EXECUTION_WORKFLOW], "Execution workflow mismatch")

    current = git("rev-parse", "HEAD")
    require(current == os.environ.get("GITHUB_SHA"), "Unexpected checkout")
    parent = git("rev-parse", "HEAD^")
    require(req.get("authorization_base_commit") == parent, "Authorization base is not parent commit")
    changed = git("diff", "--name-only", parent, current).splitlines()
    require(changed == ["experiments/b1sc_v1_0/START_REQUEST.json"], "Authorization commit contains other changes")

    for rel, expected in freeze["source_hashes"].items():
        require((REPO_ROOT / rel).is_file(), "Frozen source missing: " + rel)
        require(sha256_file(REPO_ROOT / rel) == expected, "Frozen source changed: " + rel)

    require(freeze["status"] == "EXECUTION_PIPELINE_QUALIFIED_NOT_AUTHORIZED", "Wrong execution freeze status")
    require(freeze["training_performed"] is False and freeze["scientific_evaluation_performed"] is False,
            "Qualification freeze contains scientific execution")
    return req

def config_for(role: str, block: int) -> dict:
    require(role in sc.ALL_ROLES, "Unknown role")
    require(0 <= int(block) < sc.BLOCKS, "Invalid block")
    row = next(x for x in sc.registry() if x["role"] == role and x["block"] == int(block))
    hp = PPO_HP.copy() if role == "PPO" else GRAPH_HP.copy()
    return dict(row, hp=hp, algorithm="PPO")

def model_for(role: str):
    if role == "PPO":
        return GenericActor()
    return RoutingActor(sc.GRAPH_MASK[role])

def model_state_digest(actor) -> str:
    return model_digest(actor)

def gae_returns(rewards, values, dones, last, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    g = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        nt = 1 - dones[t].astype(float)
        nv = last if t == len(rewards) - 1 else values[t + 1]
        g = rewards[t] + gamma * nv * nt - values[t] + gamma * lam * nt * g
        adv[t] = g
    return adv, adv + values

class TrainingEnv:
    """Native environment with B1-SC block/slot seed identity."""
    def __init__(self, block: int, slot: int, ledger: list, split: str = "training"):
        require(split in ("training", "qa"), "TrainingEnv accepts only training or qa split")
        self.env = NativeEnv()
        self.block = int(block)
        self.slot = int(slot)
        self.ledger = ledger
        self.split = split
        self.counter = 0
        self.total_steps = 0
        self.total_return = 0.0
        self.length = 0

    def reset(self):
        sd = sc.seed(self.split, "environment", self.block, self.slot, self.counter, bits=32)
        self.counter += 1
        z, info = self.env.reset(seed=sd)
        self.total_return = 0.0
        self.length = 0
        return z, info

    def step(self, action):
        z, r, term, trunc, info = self.env.step(action)
        self.total_steps += 1
        self.total_return += r
        self.length += 1
        if term or trunc:
            self.ledger.append(dict(
                block=self.block, slot=self.slot, episode=self.counter - 1,
                seed=self.env.episode_seed, slot_steps=self.total_steps,
                return_value=self.total_return, cycles=self.length,
                displacement=info["package_displacement"],
                fallen_count=info["fallen_count"],
                package_dropped=info["package_dropped"],
                terminated=term, truncated=trunc,
            ))
        return z, r, term, trunc, info

    def close(self):
        self.env.close()

@contextlib.contextmanager
def isolated_rng():
    p, n, t = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(p)
        np.random.set_state(n)
        torch.set_rng_state(t)

def deterministic_action(actor, z: np.ndarray, lesion: tuple[int, ...] = ()) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(z, dtype=np.float32)[None])
        if isinstance(actor, GenericActor):
            require(not lesion, "PPO has no graph-edge lesion")
            mu, _ = actor(x)
        else:
            mu, _ = actor(x, lesion)
        return mu.tanh()[0].detach().cpu().numpy().astype(np.float32)

def summarize(rows: Sequence[dict]) -> dict:
    require(bool(rows), "Cannot summarize absent episodes")
    for row in rows:
        for key in ("loss", "displacement", "cycles"):
            require(math.isfinite(float(row[key])), "Nonfinite episode metric")
    return dict(
        loss=statistics.mean(float(x["loss"]) for x in rows),
        displacement=statistics.mean(float(x["displacement"]) for x in rows),
        fall_fraction=statistics.mean(int(x["fallen_count"] > 0) for x in rows),
        drop_fraction=statistics.mean(int(bool(x["package_dropped"])) for x in rows),
        cycles=statistics.mean(int(x["cycles"]) for x in rows),
        episodes=len(rows),
    )

def _eval_seed(split: str, block: int, episode: int) -> int:
    require(split in ("admission", "causal", "qa"), "Invalid evaluation split")
    return sc.seed(split, "environment", int(block), int(episode), bits=32)

def evaluate_policy(actor, block: int, *, split: str, episodes: int,
                    lesion: tuple[int, ...] = (), condition: str = "intact") -> list[dict]:
    require(1 <= int(episodes) <= sc.EPISODES_PER_BLOCK, "Evaluation episode count outside frozen cap")
    rows = []
    env = NativeEnv()
    before = model_state_digest(actor)
    with isolated_rng():
        try:
            for ep in range(int(episodes)):
                sd = _eval_seed(split, block, ep)
                z, _ = env.reset(seed=sd)
                ret = 0.0
                trace = hashlib.sha256()
                while True:
                    active_lesion = lesion if (split != "causal" or sc.in_causal_window(env.t)) else ()
                    action = deterministic_action(actor, z, active_lesion)
                    nxt, reward, term, trunc, audit = env.step(action)
                    ret += reward
                    trace.update(np.asarray(z, np.float32).tobytes())
                    trace.update(np.asarray(action, np.float32).tobytes())
                    trace.update(np.float64(reward).tobytes())
                    trace.update(np.asarray(nxt, np.float32).tobytes())
                    z = nxt
                    if term or trunc:
                        break
                rows.append(dict(
                    block=int(block), episode=ep, seed=sd, condition=condition,
                    loss=-ret, return_value=ret,
                    displacement=float(audit["package_displacement"]),
                    cycles=int(env.t), fallen_count=int(audit["fallen_count"]),
                    package_dropped=bool(audit["package_dropped"]),
                    terminated=bool(term), truncated=bool(trunc),
                    trace_sha256=trace.hexdigest(),
                ))
        finally:
            env.close()
    require(model_state_digest(actor) == before, "Evaluation altered actor weights")
    return rows

def evaluate_no_action(block: int, *, split: str = "admission",
                       episodes: int = sc.EPISODES_PER_BLOCK) -> list[dict]:
    require(split in ("admission", "qa"), "No-action witness is only admission/QA")
    rows = []
    env = NativeEnv()
    try:
        for ep in range(int(episodes)):
            sd = _eval_seed(split, block, ep)
            z, _ = env.reset(seed=sd)
            ret = 0.0
            trace = hashlib.sha256()
            while True:
                action = np.zeros(12, dtype=np.float32)
                nxt, reward, term, trunc, audit = env.step(action)
                ret += reward
                trace.update(np.asarray(z, np.float32).tobytes())
                trace.update(np.asarray(action, np.float32).tobytes())
                trace.update(np.float64(reward).tobytes())
                trace.update(np.asarray(nxt, np.float32).tobytes())
                z = nxt
                if term or trunc:
                    break
            rows.append(dict(
                block=int(block), episode=ep, seed=sd, condition="no_action",
                loss=-ret, return_value=ret,
                displacement=float(audit["package_displacement"]),
                cycles=int(env.t), fallen_count=int(audit["fallen_count"]),
                package_dropped=bool(audit["package_dropped"]),
                terminated=bool(term), truncated=bool(trunc),
                trace_sha256=trace.hexdigest(),
            ))
    finally:
        env.close()
    return rows

def _train_ppo(cfg: dict, out: Path, *, budget: int, split: str = "training") -> tuple:
    """One uninterrupted PPO fit. Scientific callers pass exactly frozen budget.

    QA may call this with split='qa' and a small budget; those outputs are never
    accepted by scientific aggregation.
    """
    hp = cfg["hp"]
    batch = int(hp["rollout_steps"])
    require(budget > 0 and budget % batch == 0, "Budget must be positive multiple of rollout")
    if split == "training":
        require(budget == sc.NATIVE_STEPS, "Scientific budget differs from freeze")
    else:
        require(split == "qa" and budget <= 2048, "QA microfit budget exceeded")
    n_env = 8
    horizon = batch // n_env
    require(batch % n_env == 0, "Rollout not divisible by sampling parallelism")

    actor = model_for(cfg["role"])
    critic = Value()
    initial = model_state_digest(actor)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=hp["lr"], eps=hp["adam_eps"])
    ledger, curves = [], []
    updates = 0
    envs = [TrainingEnv(cfg["block"], slot, ledger, split=split) for slot in range(n_env)]
    z = np.stack([e.reset()[0] for e in envs])
    started = time.monotonic()
    try:
        for offset in range(0, budget, batch):
            obs, pre, logps, values, rewards, dones = [], [], [], [], [], []
            for _ in range(horizon):
                with torch.no_grad():
                    mu, ls = actor(torch.from_numpy(z))
                    p = mu + ls.exp() * torch.randn_like(mu)
                    actions = p.tanh()
                    lp = log_squashed_gaussian(p, mu, ls)
                    v = critic(torch.from_numpy(z))
                obs.append(z.copy()); pre.append(p.numpy()); logps.append(lp.numpy()); values.append(v.numpy())
                rs, ds, nz = [], [], []
                for i, env in enumerate(envs):
                    nxt, r, term, trunc, _ = env.step(actions[i].numpy())
                    done = bool(term or trunc)
                    rs.append(r); ds.append(done)
                    nz.append(env.reset()[0] if done else nxt)
                rewards.append(rs); dones.append(ds); z = np.stack(nz)
            with torch.no_grad():
                last = critic(torch.from_numpy(z)).numpy()
            adv, target = gae_returns(np.asarray(rewards), np.asarray(values), np.asarray(dones),
                                      last, hp["gamma"], hp["gae_lambda"])
            adv = adv.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ot = torch.from_numpy(np.asarray(obs).reshape(batch, 93))
            pt = torch.from_numpy(np.asarray(pre).reshape(batch, 12))
            old = torch.from_numpy(np.asarray(logps).reshape(-1))
            at = torch.from_numpy(adv)
            rt = torch.from_numpy(target.reshape(-1))
            losses = []
            for _ in range(hp["epochs"]):
                order = np.random.permutation(batch)
                for j in range(0, batch, hp["minibatch"]):
                    ix = order[j:j + hp["minibatch"]]
                    mu, ls = actor(ot[ix])
                    lp = log_squashed_gaussian(pt[ix], mu, ls)
                    logr = lp - old[ix]
                    ratio = logr.exp()
                    a_loss = -torch.min(
                        ratio * at[ix],
                        ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * at[ix]
                    ).mean()
                    v_loss = (critic(ot[ix]) - rt[ix]).square().mean()
                    loss = a_loss + hp["value_coef"] * v_loss
                    require(bool(torch.isfinite(loss)), "Nonfinite PPO objective")
                    opt.zero_grad()
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
                    require(bool(torch.isfinite(grad)), "Nonfinite PPO gradient")
                    opt.step()
                    updates += 1
                    losses.append(float(loss.detach()))
            steps = offset + batch
            if steps % 32768 == 0 or steps == budget:
                curves.append(dict(
                    native_steps=steps, optimizer_updates=updates,
                    mean_objective=statistics.mean(losses),
                    completed_episodes=len(ledger),
                ))
        require(initial != model_state_digest(actor), "Actor parameters did not update")
        resources = dict(
            algorithm="PPO", environment_steps=budget, optimizer_updates=updates,
            sampling_parallelism=8, actor_parameters=sum(p.numel() for p in actor.parameters()),
            critic_parameters=sum(p.numel() for p in critic.parameters()),
            initial_actor_digest=initial, final_actor_digest=model_state_digest(actor),
            elapsed_seconds=time.monotonic() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            split=split,
        )
        return actor, critic, opt, resources, ledger, curves
    finally:
        for env in envs:
            env.close()

def save_model(path: Path, actor, critic, optimizer, cfg: dict) -> None:
    torch.save(dict(
        actor_state=actor.state_dict(),
        critic_state=critic.state_dict(),
        optimizer_state=optimizer.state_dict(),
        configuration=cfg,
        native_steps=sc.NATIVE_STEPS,
    ), path)

def load_actor(fit_dir: Path, role: str):
    runtime()
    payload = torch.load(fit_dir / "MODEL.pt", map_location="cpu")
    require(payload["configuration"]["role"] == role, "Model role mismatch")
    actor = model_for(role)
    actor.load_state_dict(payload["actor_state"], strict=True)
    actor.eval()
    return actor

def fit(role: str, block: int, out_root: Path) -> dict:
    execution_guard()
    runtime()
    cfg = config_for(role, block)
    fit_dir = Path(out_root) / cfg["id"]
    fit_dir.mkdir(parents=True, exist_ok=False)
    random.seed(cfg["initial_seed"])
    np.random.seed(cfg["initial_seed"] % (2**32))
    torch.manual_seed(cfg["initial_seed"])
    write_json(fit_dir / "STARTED.json", dict(
        status="SCIENTIFIC_FIT_STARTED", configuration=cfg,
        workflow_run_id=int(os.environ["GITHUB_RUN_ID"]), run_attempt=1,
        source_commit=os.environ["GITHUB_SHA"], **sc.BOUNDARY,
    ), exclusive=True)
    try:
        actor, critic, optimizer, resources, ledger, curves = _train_ppo(
            cfg, fit_dir, budget=sc.NATIVE_STEPS, split="training")
        save_model(fit_dir / "MODEL.pt", actor, critic, optimizer, cfg)
        before = model_state_digest(actor)
        admission_rows = evaluate_policy(
            actor, block, split="admission", episodes=sc.EPISODES_PER_BLOCK, condition="admission")
        require(before == model_state_digest(actor), "Admission altered weights")
        write_json(fit_dir / "ADMISSION_EPISODES.json", admission_rows)
        write_json(fit_dir / "ADMISSION_SUMMARY.json", summarize(admission_rows))
        write_json(fit_dir / "TRAINING_RESOURCES.json", resources)
        write_json(fit_dir / "TRAINING_EPISODES.json", ledger)
        write_json(fit_dir / "LEARNING_CURVE.json", curves)
        files = {p.name: sha256_file(p) for p in fit_dir.iterdir() if p.is_file()}
        write_json(fit_dir / "COMPLETE.json", dict(
            status="COMPLETE", configuration=cfg, files=files,
            replacement_seed=False, scientific_retry=False, **sc.BOUNDARY,
        ))
        return dict(status="COMPLETE", fit_id=cfg["id"])
    except Exception:
        write_json(fit_dir / "FAILED.json", dict(
            status="FAILED_RETAINED", configuration=cfg,
            error=traceback.format_exc(), replacement_seed=False,
            scientific_retry=False, **sc.BOUNDARY,
        ))
        raise

def verify_fit_dir(fit_dir: Path, cfg: dict) -> dict:
    complete = read_json(fit_dir / "COMPLETE.json")
    require(complete["status"] == "COMPLETE", "Fit is not complete")
    require(complete["configuration"] == cfg, "Fit configuration mismatch")
    for name, digest in complete["files"].items():
        require(sha256_file(fit_dir / name) == digest, "Fit file hash mismatch: " + name)
    resources = read_json(fit_dir / "TRAINING_RESOURCES.json")
    require(resources["environment_steps"] == sc.NATIVE_STEPS, "Incomplete scientific budget")
    require(resources["split"] == "training", "Non-scientific fit cannot enter aggregation")
    rows = read_json(fit_dir / "ADMISSION_EPISODES.json")
    require(len(rows) == sc.EPISODES_PER_BLOCK, "Wrong admission episode count")
    require(read_json(fit_dir / "ADMISSION_SUMMARY.json") == summarize(rows), "Admission summary mismatch")
    return complete

def find_fit_dir(root: Path, role: str, block: int) -> Path:
    fid = f"{role}-b{int(block)}"
    candidates = [root / fid, root / ("b1sc-fit-" + fid), root / ("b1sc-fit-" + fid) / fid]
    for c in candidates:
        if c.is_dir() and (c / "COMPLETE.json").is_file():
            return c
    hits = list(root.rglob(fid))
    hits = [x for x in hits if x.is_dir()]
    require(len(hits) == 1, f"Cannot resolve unique fit directory {fid}")
    return hits[0]

def aggregate_admission(fits_root: Path, out: Path) -> dict:
    execution_guard()
    runtime()
    out.mkdir(parents=True, exist_ok=False)
    cells = {}
    witnesses = {}
    witness_rows = {}
    for block in range(sc.BLOCKS):
        rows = evaluate_no_action(block, split="admission", episodes=sc.EPISODES_PER_BLOCK)
        witness_rows[str(block)] = rows
        witnesses[str(block)] = summarize(rows)
    for role in sc.ALL_ROLES:
        by_block, pooled = [], []
        for block in range(sc.BLOCKS):
            cfg = config_for(role, block)
            fit_dir = find_fit_dir(fits_root, role, block)
            verify_fit_dir(fit_dir, cfg)
            rows = read_json(fit_dir / "ADMISSION_EPISODES.json")
            pooled.extend(rows)
            s = summarize(rows)
            c = sc.block_competence(
                s["loss"], s["displacement"],
                witnesses[str(block)]["loss"], witnesses[str(block)]["displacement"])
            by_block.append(dict(block=block, summary=s, competence=c))
        role_summary = summarize(pooled)
        witness_pooled = [row for b in range(sc.BLOCKS) for row in witness_rows[str(b)]]
        witness_summary = summarize(witness_pooled)
        agg = sc.block_competence(
            role_summary["loss"], role_summary["displacement"],
            witness_summary["loss"], witness_summary["displacement"])
        gate = sc.admission_gate(
            [{"pass": x["competence"]["pass"]} for x in by_block], {"pass": agg["pass"]})
        cells[role] = dict(summary=role_summary, competence=agg, by_block=by_block, gate=gate)
    admitted = [r for r in sc.ALL_ROLES if cells[r]["gate"]["admitted"]]
    result = dict(
        status="ADMISSION_COMPLETE", admitted_roles=admitted,
        admitted_graph_roles=[r for r in sc.GRAPH_ROLES if r in admitted],
        witness_by_block=witnesses, cells=cells,
        primary_causal_ready=True, **sc.BOUNDARY,
    )
    write_json(out / "ADMISSION.json", result)
    write_json(out / "WITNESS_EPISODES.json", witness_rows)
    return result

def _run_causal_condition(actor, role: str, block: int, episode: int,
                          condition: str, lesion: tuple[int, ...],
                          *, split: str = "causal") -> dict:
    require(split in ("causal", "qa"), "Invalid causal split")
    env = NativeEnv()
    sd = _eval_seed(split, block, episode)
    z, _ = env.reset(seed=sd)
    ret = 0.0
    trace = hashlib.sha256()
    prefix = hashlib.sha256()
    decision9_action = None
    decision9_state = None
    try:
        while True:
            active = lesion if sc.in_causal_window(env.t) else ()
            action = deterministic_action(actor, z, active)
            if env.t == sc.CAUSAL_WINDOW[0]:
                decision9_action = action.tolist()
                decision9_state = hashlib.sha256(np.asarray(z, np.float32).tobytes()).hexdigest()
            nxt, reward, term, trunc, audit = env.step(action)
            payload = (
                np.asarray(z, np.float32).tobytes()
                + np.asarray(action, np.float32).tobytes()
                + np.float64(reward).tobytes()
                + np.asarray(nxt, np.float32).tobytes()
            )
            trace.update(payload)
            if env.t <= sc.CAUSAL_WINDOW[0]:
                prefix.update(payload)
            ret += reward
            z = nxt
            if term or trunc:
                break
        return dict(
            block=int(block), episode=int(episode), seed=sd, condition=condition,
            loss=-ret, return_value=ret,
            displacement=float(audit["package_displacement"]), cycles=int(env.t),
            fallen_count=int(audit["fallen_count"]), package_dropped=bool(audit["package_dropped"]),
            terminated=bool(term), truncated=bool(trunc),
            trace_sha256=trace.hexdigest(), prefix_sha256=prefix.hexdigest(),
            decision9_action=decision9_action, decision9_state_sha256=decision9_state,
        )
    finally:
        env.close()

def _same_episode(a: dict, b: dict) -> bool:
    return a["trace_sha256"] == b["trace_sha256"]

def primary_causal(role: str, block: int, fit_root: Path, admission_file: Path,
                   out: Path, *, qa_episodes: int | None = None) -> dict:
    execution_guard() if qa_episodes is None else None
    runtime()
    admission = read_json(admission_file)
    out.mkdir(parents=True, exist_ok=False)
    if role not in sc.GRAPH_ROLES:
        raise RuntimeError("Primary causal stage only accepts graph roles")
    if role not in admission.get("admitted_graph_roles", []):
        result = dict(status="SKIPPED_NOT_ADMITTED", role=role, block=int(block), **sc.BOUNDARY)
        write_json(out / "SKIPPED.json", result)
        return result
    fit_dir = find_fit_dir(fit_root, role, block)
    verify_fit_dir(fit_dir, config_for(role, block))
    actor = load_actor(fit_dir, role)
    before = model_state_digest(actor)
    n = int(qa_episodes or sc.EPISODES_PER_BLOCK)
    split = "qa" if qa_episodes is not None else "causal"
    primary_lesion = sc.primary_lesion(role)
    inactive = sc.inactive_edge_control(role)
    rows = []
    local_changes = []
    controls_exact = True
    for ep in range(n):
        intact = _run_causal_condition(actor, role, block, ep, "intact", (), split=split)
        lesion = _run_causal_condition(actor, role, block, ep, "all_active_lesion", primary_lesion, split=split)
        sham = _run_causal_condition(actor, role, block, ep, "sham", (), split=split)
        require(_same_episode(intact, sham), "Sham diverged from intact")
        controls_exact = controls_exact and _same_episode(intact, sham)
        if inactive is not None:
            ctl = _run_causal_condition(actor, role, block, ep, "inactive_edge_control", (inactive,), split=split)
            require(_same_episode(intact, ctl), "Inactive-edge control diverged")
            rows.append(ctl)
        if role == "G0":
            g0 = _run_causal_condition(
                actor, role, block, ep, "g0_all_unavailable_control",
                sc.g0_negative_control_lesion(), split=split)
            require(_same_episode(intact, g0), "G0 unavailable-route control diverged")
            rows.append(g0)
        if intact["decision9_action"] is not None and lesion["decision9_action"] is not None:
            require(intact["decision9_state_sha256"] == lesion["decision9_state_sha256"],
                    "Intervention prefixes differ before decision 9")
            delta = float(np.max(np.abs(
                np.asarray(intact["decision9_action"]) - np.asarray(lesion["decision9_action"]))))
            local_changes.append(delta)
        rows.extend([intact, lesion, sham])
    intact_rows = [x for x in rows if x["condition"] == "intact"]
    lesion_rows = [x for x in rows if x["condition"] == "all_active_lesion"]
    si, sl = summarize(intact_rows), summarize(lesion_rows)
    loss_harm = sl["loss"] - si["loss"]
    displacement_harm = si["displacement"] - sl["displacement"]
    result = dict(
        status="PRIMARY_CAUSAL_BLOCK_COMPLETE", role=role, block=int(block),
        intact=si, lesion=sl, loss_harm=loss_harm,
        displacement_harm=displacement_harm,
        practical_harm=sc.practical_harm(loss_harm, displacement_harm),
        local_action_change_at_9=max(local_changes) if local_changes else None,
        controls_exact=controls_exact, episodes=rows,
        model_digest=model_state_digest(actor), **sc.BOUNDARY,
    )
    require(before == model_state_digest(actor), "Causal evaluation altered actor")
    write_json(out / "PRIMARY.json", result)
    return result

def _bootstrap_mean(values: Sequence[float], label: str) -> dict:
    vals = np.asarray(list(values), dtype=np.float64)
    require(vals.shape == (sc.BLOCKS,), "Bootstrap requires exactly 8 block effects")
    rng = np.random.default_rng(sc.seed("bootstrap", label, bits=32))
    reps = np.empty(sc.BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for i in range(sc.BOOTSTRAP_RESAMPLES):
        reps[i] = vals[rng.integers(0, len(vals), len(vals))].mean()
    return dict(
        mean=float(vals.mean()), median=float(np.median(vals)),
        ci95=[float(np.quantile(reps, .025)), float(np.quantile(reps, .975))],
        by_block=vals.tolist(), resamples=sc.BOOTSTRAP_RESAMPLES,
    )

def find_stage_dir(root: Path, prefix: str, role: str, block: int) -> Path:
    target = f"{prefix}-{role}-b{int(block)}"
    candidates = [root / target, root / role / f"b{block}"]
    for c in candidates:
        if c.is_dir():
            return c
    hits = [x for x in root.rglob(target) if x.is_dir()]
    require(len(hits) == 1, "Cannot resolve stage directory " + target)
    return hits[0]

def aggregate_primary(primary_root: Path, admission_file: Path, out: Path) -> dict:
    execution_guard()
    admission = read_json(admission_file)
    out.mkdir(parents=True, exist_ok=False)
    roles = admission.get("admitted_graph_roles", [])
    role_results = {}
    passed = []
    for role in roles:
        blocks = []
        for block in range(sc.BLOCKS):
            d = find_stage_dir(primary_root, "b1sc-primary", role, block)
            row = read_json(d / "PRIMARY.json")
            require(row["status"] == "PRIMARY_CAUSAL_BLOCK_COMPLETE", "Missing primary block")
            require(row["controls_exact"] is True, "Primary control failed")
            blocks.append(row)
        loss = [x["loss_harm"] for x in blocks]
        disp = [x["displacement_harm"] for x in blocks]
        pooled_intact = summarize([e for x in blocks for e in x["episodes"] if e["condition"] == "intact"])
        pooled_lesion = summarize([e for x in blocks for e in x["episodes"] if e["condition"] == "all_active_lesion"])
        al = pooled_lesion["loss"] - pooled_intact["loss"]
        ad = pooled_intact["displacement"] - pooled_lesion["displacement"]
        aggregate_harm = sc.practical_harm(al, ad)
        gate = sc.causal_gate([x["practical_harm"] for x in blocks], aggregate_harm)
        role_results[role] = dict(
            blocks=blocks, aggregate=dict(
                intact=pooled_intact, lesion=pooled_lesion,
                loss_harm=al, displacement_harm=ad,
                practical_harm=aggregate_harm,
            ),
            gate=gate,
            loss_harm_bootstrap=_bootstrap_mean(loss, f"primary-loss-{role}"),
            displacement_harm_bootstrap=_bootstrap_mean(disp, f"primary-disp-{role}"),
        )
        if role != "G0" and gate["causal_gate_pass"]:
            passed.append(role)
    result = dict(
        status="PRIMARY_CAUSAL_AGGREGATE_COMPLETE",
        passed_roles=passed, roles=role_results,
        secondary_ready=True, **sc.BOUNDARY,
    )
    write_json(out / "PRIMARY_CAUSAL.json", result)
    return result

def secondary_causal(role: str, block: int, fit_root: Path, primary_file: Path,
                     out: Path) -> dict:
    execution_guard()
    runtime()
    primary = read_json(primary_file)
    out.mkdir(parents=True, exist_ok=False)
    if role not in primary.get("passed_roles", []):
        result = dict(status="SKIPPED_PRIMARY_GATE_NOT_PASSED", role=role, block=int(block), **sc.BOUNDARY)
        write_json(out / "SKIPPED.json", result)
        return result
    fit_dir = find_fit_dir(fit_root, role, block)
    verify_fit_dir(fit_dir, config_for(role, block))
    actor = load_actor(fit_dir, role)
    edges = sc.role_active_edges(role)
    require(bool(edges), "Secondary localization requires active edges")
    results = {}
    for edge in edges:
        intact_rows, lesion_rows = [], []
        for ep in range(sc.EPISODES_PER_BLOCK):
            intact_rows.append(_run_causal_condition(actor, role, block, ep, "intact", (), split="causal"))
            lesion_rows.append(_run_causal_condition(actor, role, block, ep, f"edge_{edge}", (edge,), split="causal"))
        si, sl = summarize(intact_rows), summarize(lesion_rows)
        lh = sl["loss"] - si["loss"]
        dh = si["displacement"] - sl["displacement"]
        results[str(edge)] = dict(
            edge=edge, pair=list(sc.EDGE_ORDER[edge]),
            intact=si, lesion=sl, loss_harm=lh, displacement_harm=dh,
            practical_harm=sc.practical_harm(lh, dh),
        )
    result = dict(
        status="SECONDARY_CAUSAL_BLOCK_COMPLETE", role=role, block=int(block),
        edges=results, **sc.BOUNDARY,
    )
    write_json(out / "SECONDARY.json", result)
    return result

def aggregate_secondary(secondary_root: Path, primary_file: Path, out: Path) -> dict:
    execution_guard()
    primary = read_json(primary_file)
    out.mkdir(parents=True, exist_ok=False)
    roles = {}
    for role in primary.get("passed_roles", []):
        blocks = []
        for block in range(sc.BLOCKS):
            d = find_stage_dir(secondary_root, "b1sc-secondary", role, block)
            row = read_json(d / "SECONDARY.json")
            require(row["status"] == "SECONDARY_CAUSAL_BLOCK_COMPLETE", "Missing secondary block")
            blocks.append(row)
        edges = {}
        for edge in sc.role_active_edges(role):
            loss = [b["edges"][str(edge)]["loss_harm"] for b in blocks]
            disp = [b["edges"][str(edge)]["displacement_harm"] for b in blocks]
            practical = [b["edges"][str(edge)]["practical_harm"] for b in blocks]
            # Secondary localization is descriptive/hierarchical, not a new primary gate.
            edges[str(edge)] = dict(
                edge=edge, pair=list(sc.EDGE_ORDER[edge]),
                blocks_with_practical_harm=sum(bool(x) for x in practical),
                loss_harm_bootstrap=_bootstrap_mean(loss, f"secondary-loss-{role}-{edge}"),
                displacement_harm_bootstrap=_bootstrap_mean(disp, f"secondary-disp-{role}-{edge}"),
            )
        roles[role] = dict(edges=edges)
    result = dict(status="SECONDARY_CAUSAL_AGGREGATE_COMPLETE", roles=roles, **sc.BOUNDARY)
    write_json(out / "SECONDARY_CAUSAL.json", result)
    return result

def utility_from_admission(admission: dict, role: str) -> dict:
    require(role in sc.GRAPH_ROLES and role != "G0", "Utility candidate must be non-G0 graph role")
    cand = admission["cells"][role]
    g0 = admission["cells"]["G0"]
    if not cand["gate"]["admitted"] or not g0["gate"]["admitted"]:
        return dict(eligible=False, gate=None)
    block_flags, loss_vals, disp_vals = [], [], []
    for block in range(sc.BLOCKS):
        cs = cand["by_block"][block]["summary"]
        gs = g0["by_block"][block]["summary"]
        la = gs["loss"] - cs["loss"]
        da = cs["displacement"] - gs["displacement"]
        block_flags.append(sc.practical_advantage(la, da))
        loss_vals.append(la); disp_vals.append(da)
    la = g0["summary"]["loss"] - cand["summary"]["loss"]
    da = cand["summary"]["displacement"] - g0["summary"]["displacement"]
    agg = sc.practical_advantage(la, da)
    gate = sc.utility_gate(block_flags, agg)
    return dict(
        eligible=True, aggregate=dict(loss_advantage=la, displacement_advantage=da,
                                      practical_advantage=agg),
        blocks=[dict(block=i, loss_advantage=loss_vals[i],
                     displacement_advantage=disp_vals[i],
                     practical_advantage=block_flags[i]) for i in range(sc.BLOCKS)],
        gate=gate,
        loss_advantage_bootstrap=_bootstrap_mean(loss_vals, f"utility-loss-{role}"),
        displacement_advantage_bootstrap=_bootstrap_mean(disp_vals, f"utility-disp-{role}"),
    )

def final_aggregate(admission_file: Path, primary_file: Path, secondary_file: Path,
                    out: Path) -> dict:
    execution_guard()
    out.mkdir(parents=True, exist_ok=False)
    admission = read_json(admission_file)
    primary = read_json(primary_file)
    secondary = read_json(secondary_file)
    classifications = {}
    utility = {}
    for role in sc.GRAPH_ROLES:
        if role == "G0":
            classifications[role] = "REFERENCE_COMPETENT" if admission["cells"][role]["gate"]["admitted"] else "NOT_COMPETENT"
            continue
        admitted = admission["cells"][role]["gate"]["admitted"]
        if not admitted:
            classifications[role] = "NOT_COMPETENT"
            continue
        causal = bool(primary.get("roles", {}).get(role, {}).get("gate", {}).get("causal_gate_pass", False))
        u = utility_from_admission(admission, role)
        utility[role] = u
        if not causal:
            pr = primary.get("roles", {}).get(role, {})
            harm_blocks = int(pr.get("gate", {}).get("blocks_with_practical_harm", 0))
            aggregate_harm = bool(pr.get("gate", {}).get("aggregate_practical_harm", False))
            classifications[role] = (
                "COMPETENT_RELATIONALLY_REDUNDANT"
                if (not aggregate_harm and harm_blocks <= 2)
                else "INCONCLUSIVE_OR_MIXED"
            )
        elif not u.get("eligible"):
            classifications[role] = "INCONCLUSIVE_OR_MIXED"
        elif not u["gate"]["utility_gate_pass"]:
            advantage_blocks = int(u["gate"]["blocks_with_practical_advantage"])
            aggregate_advantage = bool(u["gate"]["aggregate_practical_advantage"])
            classifications[role] = (
                "CAUSALLY_DEPENDENT_WITHOUT_STRUCTURAL_UTILITY"
                if (not aggregate_advantage and advantage_blocks <= 2)
                else "INCONCLUSIVE_OR_MIXED"
            )
        else:
            classifications[role] = "COMPETENT_CAUSAL_STRUCTURE"
    ppo = admission["cells"]["PPO"]["gate"]["admitted"]
    candidates = [r for r, s in classifications.items() if s == "COMPETENT_CAUSAL_STRUCTURE"]
    result = dict(
        status="B1SC_V1_0_DEVELOPMENT_COMPLETE_FOR_REVIEW",
        classifications=classifications,
        competent_causal_structures=candidates,
        no_forced_single_winner=True,
        ppo_reference_competent=ppo,
        admission=admission,
        primary_causal=primary,
        secondary_causal=secondary,
        utility_vs_g0=utility,
        automatic_b1e_progression=False,
        **sc.BOUNDARY,
    )
    write_json(out / "SUMMARY.json", result)
    report = [
        "# B1-SC v1.0 — cierre de desarrollo",
        "",
        "**B1SC_V1_0_DEVELOPMENT_COMPLETE_FOR_REVIEW**",
        "",
        "Este cierre no habilita B1-E automáticamente.",
        "",
        "| Rol | Estado |",
        "|---|---|",
    ]
    for role in sc.GRAPH_ROLES:
        report.append(f"| {role} | {classifications[role]} |")
    report += [
        "",
        f"PPO referencia competente: `{ppo}`.",
        f"Estructuras `COMPETENT_CAUSAL_STRUCTURE`: `{candidates}`.",
        "",
        "B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",
        "B1E_executed=false",
        "H_CAT=NOT_EVALUABLE",
        "H_TRANSFER=NOT_EVALUATED",
        "",
    ]
    (out / "COMPLETION_REPORT_ES.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    write_json(out / "RESULTS_MANIFEST.json", dict(
        status="RESULTS_PREPARED_FOR_REVIEW",
        files={p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
        automatic_publication=False, **sc.BOUNDARY,
    ))
    return result

def qa_microfit(out: Path) -> dict:
    """Technical microfit only: QA namespace, 512 native transitions, no evidence."""
    runtime()
    cfg = config_for("G0", 0)
    cfg = dict(cfg)
    cfg["initial_seed"] = sc.seed("qa", "microfit", bits=32)
    random.seed(cfg["initial_seed"]); np.random.seed(cfg["initial_seed"]); torch.manual_seed(cfg["initial_seed"])
    out.mkdir(parents=True, exist_ok=False)
    actor, critic, opt, resources, ledger, curves = _train_ppo(cfg, out, budget=512, split="qa")
    result = dict(
        status="QA_MICROFIT_PASS_NOT_SCIENTIFIC",
        native_steps=resources["environment_steps"],
        split=resources["split"],
        actor_updated=resources["initial_actor_digest"] != resources["final_actor_digest"],
        scientific_registry_consumed=False,
        scientific_results=False,
        **sc.BOUNDARY,
    )
    write_json(out / "QA_MICROFIT.json", result)
    return result
