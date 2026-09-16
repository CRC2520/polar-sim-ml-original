"""B1-SC-D3 v1.0 scientific runner.

This runner consumes the eight repository-persisted trainable snapshots byte-for-byte.
QA microfits are permitted only in the qa namespace. Scientific training remains
blocked unless a separately authorized START_REQUEST.json is the only change after
a qualified runner commit.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from b1s.execution.core import Value, log_squashed_gaussian, model_digest
from experiments.b1sc_d2_v1_0 import execution as d2exec
from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d3_v1_0 import init_format

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
FROZEN_BASE_HEAD = "8e55ac2e270de81f44aca2f4e3987eb4c864ffa8"
RUNNER_BRANCH = "research/b1sc-d3-v1.0-runner-readiness-20260916"
START_REQUEST = ROOT / "START_REQUEST.json"
RUNNER_CONTRACT = ROOT / "RUNNER_CONTRACT.json"
SNAPSHOT_DIR = ROOT / "frozen_init"

HP = dict(d2exec.HP)
N_ENVS = int(d2exec.N_ENVS)
EXPECTED_HP = dict(
    lr=0.0003,
    gamma=0.99,
    rollout_steps=512,
    minibatch=128,
    epochs=4,
    gae_lambda=0.95,
    clip=0.2,
    value_coef=0.5,
    max_grad_norm=0.5,
    adam_eps=1e-5,
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def runtime() -> dict:
    from b1s.execution.instrument import runtime_lock

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d2exec.runtime()
    return runtime_lock()


def _registry() -> list[dict]:
    rows = read_json(d3.REGISTRY_PATH)
    require(len(rows) == d3.FITS, "D3 registry cardinality changed")
    return rows


def config_for(condition: str, block: int) -> dict:
    require(condition in d3.CONDITIONS, "Unknown D3 condition")
    require(0 <= int(block) < d3.BLOCKS, "Invalid D3 block")
    row = next(x for x in _registry() if x["condition"] == condition and int(x["block"]) == int(block))
    require(int(row["native_steps"]) == d3.NATIVE_STEPS, "D3 fit budget changed")
    require(row["mask"] in (d3.G0, d3.S6), "D3 support mask changed")
    require(row["information_mode"] in d3.MODES, "D3 information mode changed")
    require(int(row["initial_snapshot_block"]) == int(block), "D3 snapshot pairing changed")
    return dict(row, hp=HP.copy(), algorithm="PPO", initialization="D3-BYTE-FROZEN")


def _init_freeze() -> dict:
    freeze = read_json(ROOT / "INIT_FREEZE.json")
    require(freeze["status"] == "D3_FROZEN_INITIALIZATION_QUALIFIED_NOT_EXECUTED", "D3 init freeze changed")
    c = freeze["snapshot_contract"]
    require(c["seed_reconstruction_at_scientific_runtime_forbidden"] is True, "Runtime seed reconstruction became allowed")
    require(c["shared_byte_for_byte_across_four_conditions"] is True, "D3 pairing contract changed")
    require(c["support_buffer_included"] is False and c["trainable_only"] is True, "D3 snapshot format changed")
    return freeze


def snapshot_record(block: int) -> dict:
    freeze = _init_freeze()
    row = next(x for x in freeze["blocks"] if int(x["block"]) == int(block))
    require(row["shared_conditions"] == list(d3.CONDITIONS), "Frozen D3 condition sharing changed")
    return row


def snapshot_path(block: int) -> Path:
    return SNAPSHOT_DIR / f"block-{int(block)}.bin"


def derive_child_seed(root: int, *parts: object) -> int:
    raw = json.dumps([int(root), *parts], sort_keys=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def qa_seed(block: int, purpose: str, **extra: object) -> int:
    identity = {"block": int(block), "purpose": str(purpose), **extra}
    return d3.seed("qa", identity)


@dataclasses.dataclass(frozen=True)
class InitAttestation:
    condition: str
    block: int
    snapshot_sha256: str
    snapshot_bytes: int
    source_seed: int
    actor_trainable_digest: str
    critic_trainable_digest: str
    mask: str
    information_mode: str
    validated: bool = True


@dataclasses.dataclass
class PreparedModules:
    actor: nn.Module
    critic: nn.Module
    attestation: InitAttestation
    audit: list[dict]


def _event(audit: list[dict], name: str, **extra) -> None:
    audit.append({"index": len(audit), "event": name, **extra})


def verify_snapshot_bytes(block: int) -> dict:
    rec = snapshot_record(block)
    path = snapshot_path(block)
    require(path.is_file(), f"Missing frozen D3 snapshot: block {block}")
    require(path.stat().st_size == int(rec["bytes"]), f"Frozen D3 byte length mismatch: block {block}")
    actual = sha256_file(path)
    require(actual == rec["sha256"], f"Frozen D3 SHA-256 mismatch: block {block}")
    return dict(block=int(block), bytes=int(rec["bytes"]), sha256=actual, source_seed=int(rec["source_seed"]))


def verify_all_snapshots() -> list[dict]:
    rows = [verify_snapshot_bytes(b) for b in range(d3.BLOCKS)]
    require(len({x["sha256"] for x in rows}) == d3.BLOCKS, "D3 snapshots are not eight distinct block bytes")
    return rows


def validate_and_load_frozen(condition: str, block: int) -> PreparedModules:
    require(HP == EXPECTED_HP and N_ENVS == 8, "Historical PPO profile changed")
    d3.validate_design()
    cfg = config_for(condition, block)
    rec = snapshot_record(block)
    audit: list[dict] = []
    _event(audit, "validation_started", condition=condition, block=int(block))

    path = snapshot_path(block)
    verify_snapshot_bytes(block)
    actor = d3.D3RoutingActor(cfg["mask"], cfg["information_mode"])
    critic = Value()
    header, actual = init_format.load_trainable_into(
        path,
        actor,
        critic,
        expected_sha256=rec["sha256"],
    )
    require(int(header["block"]) == int(block), "D3 snapshot block identity mismatch")
    require(int(header["source_seed"]) == int(rec["source_seed"]), "D3 snapshot source lineage mismatch")

    actor_digest = init_format.trainable_digest(actor)
    critic_digest = init_format.trainable_digest(critic)
    require(actor_digest == rec["actor_trainable_digest"], "D3 actor trainable digest mismatch")
    require(critic_digest == rec["critic_trainable_digest"], "D3 critic trainable digest mismatch")

    att = InitAttestation(
        condition=condition,
        block=int(block),
        snapshot_sha256=actual,
        snapshot_bytes=int(rec["bytes"]),
        source_seed=int(rec["source_seed"]),
        actor_trainable_digest=actor_digest,
        critic_trainable_digest=critic_digest,
        mask=cfg["mask"],
        information_mode=cfg["information_mode"],
    )
    _event(
        audit,
        "snapshot_validated",
        snapshot_sha256=actual,
        snapshot_bytes=int(rec["bytes"]),
        source_seed=int(rec["source_seed"]),
        actor_trainable_digest=actor_digest,
        critic_trainable_digest=critic_digest,
    )
    return PreparedModules(actor=actor, critic=critic, attestation=att, audit=audit)


def create_optimizer(prepared: PreparedModules):
    require(prepared.attestation.validated, "Optimizer requires validated frozen initialization")
    names = [x["event"] for x in prepared.audit]
    require(names == ["validation_started", "snapshot_validated"], "Unexpected pre-optimizer audit order")
    opt = torch.optim.Adam(
        list(prepared.actor.parameters()) + list(prepared.critic.parameters()),
        lr=HP["lr"],
        eps=HP["adam_eps"],
    )
    _event(prepared.audit, "optimizer_created")
    return opt


def guarded_optimizer_step(opt, prepared: PreparedModules) -> None:
    require(prepared.attestation.validated, "optimizer.step blocked before frozen validation")
    names = [x["event"] for x in prepared.audit]
    require("snapshot_validated" in names and "optimizer_created" in names, "optimizer.step missing preconditions")
    require(names.index("snapshot_validated") < names.index("optimizer_created"), "Optimizer created before snapshot validation")
    _event(prepared.audit, "optimizer_step")
    opt.step()


class TrainingEnv:
    def __init__(self, block: int, slot: int, ledger: list[dict], seed_root: int):
        from b1s.execution.instrument import NativeEnv

        self.env = NativeEnv()
        self.block = int(block)
        self.slot = int(slot)
        self.ledger = ledger
        self.seed_root = int(seed_root)
        self.counter = 0
        self.total_steps = 0
        self.ret = 0.0
        self.length = 0

    def reset(self):
        sd = derive_child_seed(self.seed_root, "slot", self.slot, "episode", self.counter)
        self.counter += 1
        z, info = self.env.reset(seed=sd)
        self.ret = 0.0
        self.length = 0
        return z, info

    def step(self, action):
        z, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.ret += reward
        self.length += 1
        if terminated or truncated:
            self.ledger.append(
                dict(
                    block=self.block,
                    slot=self.slot,
                    episode=self.counter - 1,
                    seed=self.env.episode_seed,
                    slot_steps=self.total_steps,
                    return_value=self.ret,
                    cycles=self.length,
                    displacement=float(info["package_displacement"]),
                    fallen_count=int(info["fallen_count"]),
                    package_dropped=bool(info["package_dropped"]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
        return z, reward, terminated, truncated, info

    def close(self):
        self.env.close()


def gae_returns(rewards, values, dones, last, gamma, lam):
    adv = np.empty_like(rewards, dtype=np.float32)
    g = np.empty(rewards.shape[1], dtype=np.float64)
    g.fill(0.0)
    for t in range(len(rewards) - 1, -1, -1):
        nt = 1 - dones[t].astype(float)
        nv = last if t == len(rewards) - 1 else values[t + 1]
        g = rewards[t] + gamma * nv * nt - values[t] + gamma * lam * nt * g
        adv[t] = g
    return adv, adv + values


@contextlib.contextmanager
def isolated_rng():
    pstate = random.getstate()
    nstate = np.random.get_state()
    tstate = torch.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(pstate)
        np.random.set_state(nstate)
        torch.set_rng_state(tstate)


def summarize(rows: Sequence[dict]) -> dict:
    require(bool(rows), "Cannot summarize absent episodes")
    return dict(
        loss=statistics.mean(float(x["loss"]) for x in rows),
        displacement=statistics.mean(float(x["displacement"]) for x in rows),
        fall_fraction=statistics.mean(int(x["fallen_count"] > 0) for x in rows),
        drop_fraction=statistics.mean(int(bool(x["package_dropped"])) for x in rows),
        cycles=statistics.mean(int(x["cycles"]) for x in rows),
        episodes=len(rows),
    )


def deterministic_action(actor, z, lesion: tuple[int, ...]):
    with torch.no_grad():
        mu, _ = actor(torch.from_numpy(np.asarray(z, dtype=np.float32)[None]), lesion)
        return mu.tanh()[0].cpu().numpy().astype(np.float32)


def _panel_root(cfg: dict, split: str) -> int:
    key = {
        "diagnostic": "diagnostic_panel_seed",
        "endpoint": "endpoint_panel_seed",
        "causal": "causal_panel_seed",
    }[split]
    return int(cfg[key])


def evaluation_lesion(mode: str) -> tuple[int, ...]:
    if mode in ("intact", "sham"):
        return ()
    if mode == "permanent_all_active_lesion":
        return tuple(d3.ACTIVE_EDGES)
    if mode == "inactive_edge_control":
        return (d3.INACTIVE_CONTROL_EDGE,)
    raise RuntimeError("Unknown D3 evaluation mode")


def evaluate_actor(actor, cfg: dict, split: str, episodes: int, mode: str) -> list[dict]:
    from b1s.execution.instrument import NativeEnv

    cap = d3.DIAGNOSTIC_EPISODES if split == "diagnostic" else d3.ENDPOINT_EPISODES if split == "endpoint" else d3.CAUSAL_EPISODES
    require(split in ("diagnostic", "endpoint", "causal"), "Invalid D3 evaluation split")
    require(1 <= int(episodes) <= int(cap), "D3 evaluation count outside cap")
    if split == "causal":
        require(cfg["condition"] == "LOCAL-S6", "Primary D3 causal panel is LOCAL-S6 only")
    root = _panel_root(cfg, split)
    rows: list[dict] = []
    env = NativeEnv()
    before = model_digest(actor)
    with isolated_rng():
        try:
            for ep in range(int(episodes)):
                sd = derive_child_seed(root, "episode", ep)
                z, _ = env.reset(seed=sd)
                ret = 0.0
                trace = hashlib.sha256()
                while True:
                    lesion = evaluation_lesion(mode)
                    action = deterministic_action(actor, z, lesion)
                    nz, reward, terminated, truncated, audit = env.step(action)
                    trace.update(
                        np.asarray(z, np.float32).tobytes()
                        + np.asarray(action, np.float32).tobytes()
                        + np.float64(reward).tobytes()
                        + np.asarray(nz, np.float32).tobytes()
                    )
                    ret += reward
                    z = nz
                    if terminated or truncated:
                        break
                rows.append(
                    dict(
                        condition=cfg["condition"],
                        mode=mode,
                        block=int(cfg["block"]),
                        episode=ep,
                        seed=sd,
                        loss=-ret,
                        return_value=ret,
                        displacement=float(audit["package_displacement"]),
                        cycles=int(env.t),
                        fallen_count=int(audit["fallen_count"]),
                        package_dropped=bool(audit["package_dropped"]),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        trace_sha256=trace.hexdigest(),
                    )
                )
        finally:
            env.close()
    require(model_digest(actor) == before, "D3 evaluation altered actor state")
    return rows


def evaluate_no_action(cfg: dict, episodes: int = d3.ENDPOINT_EPISODES) -> list[dict]:
    from b1s.execution.instrument import NativeEnv

    require(1 <= int(episodes) <= d3.ENDPOINT_EPISODES, "Witness count outside D3 endpoint cap")
    root = int(cfg["endpoint_panel_seed"])
    rows: list[dict] = []
    env = NativeEnv()
    try:
        for ep in range(int(episodes)):
            sd = derive_child_seed(root, "episode", ep)
            z, _ = env.reset(seed=sd)
            ret = 0.0
            trace = hashlib.sha256()
            while True:
                action = np.full(12, 0.0, dtype=np.float32)
                nz, reward, terminated, truncated, audit = env.step(action)
                trace.update(
                    np.asarray(z, np.float32).tobytes()
                    + action.tobytes()
                    + np.float64(reward).tobytes()
                    + np.asarray(nz, np.float32).tobytes()
                )
                ret += reward
                z = nz
                if terminated or truncated:
                    break
            rows.append(
                dict(
                    condition="no_action",
                    mode="no_action",
                    block=int(cfg["block"]),
                    episode=ep,
                    seed=sd,
                    loss=-ret,
                    return_value=ret,
                    displacement=float(audit["package_displacement"]),
                    cycles=int(env.t),
                    fallen_count=int(audit["fallen_count"]),
                    package_dropped=bool(audit["package_dropped"]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    trace_sha256=trace.hexdigest(),
                )
            )
    finally:
        env.close()
    return rows


def _checkpoint_payload(actor, critic, opt, cfg, steps, prepared):
    return dict(
        actor_state=actor.state_dict(),
        critic_state=critic.state_dict(),
        optimizer_state=opt.state_dict(),
        configuration=cfg,
        native_steps=int(steps),
        condition=cfg["condition"],
        initialization=dataclasses.asdict(prepared.attestation),
    )


def _seed_plan(cfg: dict, split: str) -> dict:
    block = int(cfg["block"])
    if split == "training":
        env_root = int(cfg["training_environment_seed_root"])
        policy = int(cfg["policy_sampling_seed"])
        minibatches = derive_child_seed(policy, "ppo-minibatches")
    else:
        require(split == "qa", "Unknown D3 runner split")
        env_root = qa_seed(block, "environment-root")
        policy = qa_seed(block, "policy-sampling")
        minibatches = qa_seed(block, "ppo-minibatches")
    return dict(environment_seed_root=env_root, policy_sampling_seed=policy, minibatch_seed=minibatches)


def _train(condition: str, block: int, out: Path, budget: int, split: str) -> dict:
    require(split in ("training", "qa"), "Unknown D3 runner split")
    require(HP == EXPECTED_HP and N_ENVS == 8, "Historical PPO profile changed")
    cfg = config_for(condition, block)
    if split == "training":
        require(int(budget) == d3.NATIVE_STEPS, "Scientific budget differs from frozen D3")
    else:
        require(0 < int(budget) <= 2048 and int(budget) % HP["rollout_steps"] == 0, "QA microfit budget exceeded")
    prepared = validate_and_load_frozen(condition, block)
    actor, critic = prepared.actor, prepared.critic
    initial_actor = init_format.trainable_digest(actor)
    initial_critic = init_format.trainable_digest(critic)

    out.mkdir(parents=True, exist_ok=True)
    if split == "training":
        write_json(
            out / "STARTED.json",
            dict(
                status="D3_SCIENTIFIC_FIT_STARTED",
                configuration=cfg,
                source_commit=os.environ.get("GITHUB_SHA"),
                initialization=dataclasses.asdict(prepared.attestation),
                scientific_evidence=True,
                B1E_executed=False,
                final_seeds_generated=False,
            ),
        )

    opt = create_optimizer(prepared)
    hp = cfg["hp"]
    batch = int(hp["rollout_steps"])
    horizon = batch // N_ENVS
    require(batch % N_ENVS == 0 and budget % batch == 0, "D3 PPO rollout geometry changed")
    seeds = _seed_plan(cfg, split)

    ledger: list[dict] = []
    curves: list[dict] = []
    diagnostics: dict[str, dict] = {}
    envs = [TrainingEnv(block, i, ledger, seeds["environment_seed_root"]) for i in range(N_ENVS)]
    z = np.stack([env.reset()[0] for env in envs])
    torch.manual_seed(seeds["policy_sampling_seed"])
    rng = np.random.default_rng(seeds["minibatch_seed"])
    updates = 0
    first_step_verified = False
    t0 = time.monotonic()
    try:
        for offset in range(0, int(budget), batch):
            obs, pre, logps, values, rewards, dones = [], [], [], [], [], []
            for _ in range(horizon):
                with torch.no_grad():
                    mu, ls = actor(torch.from_numpy(z), ())
                    p = mu + ls.exp() * torch.randn_like(mu)
                    action = p.tanh()
                    logp = log_squashed_gaussian(p, mu, ls)
                    value = critic(torch.from_numpy(z))
                obs.append(z.copy())
                pre.append(p.numpy())
                logps.append(logp.numpy())
                values.append(value.numpy())
                rs, ds, nzs = [], [], []
                for i, env in enumerate(envs):
                    nz, reward, terminated, truncated, _ = env.step(action[i].numpy())
                    done = terminated or truncated
                    rs.append(reward)
                    ds.append(done)
                    nzs.append(env.reset()[0] if done else nz)
                rewards.append(rs)
                dones.append(ds)
                z = np.stack(nzs)

            with torch.no_grad():
                last = critic(torch.from_numpy(z)).numpy()
            adv, target = gae_returns(
                np.asarray(rewards),
                np.asarray(values),
                np.asarray(dones),
                last,
                hp["gamma"],
                hp["gae_lambda"],
            )
            adv = adv.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ot = torch.from_numpy(np.asarray(obs).reshape(batch, 93))
            pt = torch.from_numpy(np.asarray(pre).reshape(batch, 12))
            old = torch.from_numpy(np.asarray(logps).reshape(-1))
            at = torch.from_numpy(adv)
            rt = torch.from_numpy(target.reshape(-1))
            actor_losses, value_losses = [], []

            for _ in range(int(hp["epochs"])):
                order = rng.permutation(batch)
                for j in range(0, batch, int(hp["minibatch"])):
                    ix = order[j : j + int(hp["minibatch"])]
                    mu, ls = actor(ot[ix], ())
                    logp = log_squashed_gaussian(pt[ix], mu, ls)
                    ratio = (logp - old[ix]).exp()
                    actor_loss = -torch.min(
                        ratio * at[ix],
                        ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * at[ix],
                    ).mean()
                    value_loss = (critic(ot[ix]) - rt[ix]).square().mean()
                    loss = actor_loss + hp["value_coef"] * value_loss
                    require(bool(torch.isfinite(loss)), "Nonfinite D3 PPO objective")
                    opt.zero_grad()
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()),
                        hp["max_grad_norm"],
                    )
                    require(bool(torch.isfinite(grad)), "Nonfinite D3 gradient")
                    if not first_step_verified:
                        require(
                            [x["event"] for x in prepared.audit]
                            == ["validation_started", "snapshot_validated", "optimizer_created"],
                            "Unexpected D3 pre-step audit order",
                        )
                        first_step_verified = True
                    guarded_optimizer_step(opt, prepared)
                    updates += 1
                    actor_losses.append(float(actor_loss.detach()))
                    value_losses.append(float(value_loss.detach()))

            steps = offset + batch
            if steps % 32768 == 0 or steps == budget:
                curves.append(
                    dict(
                        native_steps=steps,
                        optimizer_updates=updates,
                        actor_loss=float(np.mean(actor_losses)),
                        value_loss=float(np.mean(value_losses)),
                        completed_episodes=len(ledger),
                    )
                )
            if split == "training" and steps in d3.CHECKPOINTS:
                torch.save(_checkpoint_payload(actor, critic, opt, cfg, steps, prepared), out / f"model-{steps}.pt")
                rows = evaluate_actor(actor, cfg, "diagnostic", d3.DIAGNOSTIC_EPISODES, "intact")
                payload = dict(
                    status="D3_DIAGNOSTIC_CHECKPOINT_COMPLETE",
                    condition=condition,
                    block=int(block),
                    native_steps=int(steps),
                    summary=summarize(rows),
                    episodes=rows,
                    initialization=dataclasses.asdict(prepared.attestation),
                )
                write_json(out / f"DIAGNOSTIC_{steps}.json", payload)
                diagnostics[str(steps)] = payload["summary"]

        require(first_step_verified and updates > 0, "D3 runner never reached a guarded optimizer step")
        result = dict(
            status="D3_QA_MICROFIT_COMPLETE" if split == "qa" else "D3_SCIENTIFIC_FIT_COMPLETE",
            split=split,
            condition=condition,
            block=int(block),
            environment_steps=int(budget),
            optimizer_updates=int(updates),
            sampling_parallelism=N_ENVS,
            seed_plan=seeds,
            initialization=dataclasses.asdict(prepared.attestation),
            initialization_audit=prepared.audit,
            initial_actor_trainable_digest=initial_actor,
            initial_critic_trainable_digest=initial_critic,
            final_actor_trainable_digest=init_format.trainable_digest(actor),
            final_critic_trainable_digest=init_format.trainable_digest(critic),
            seconds=time.monotonic() - t0,
            scientific_evidence=bool(split == "training"),
            scientific_training_performed=bool(split == "training"),
            B1E_executed=False,
            final_seeds_generated=False,
        )
        write_json(out / "RUN_RESULT.json", result)

        if split == "training":
            require(set(map(int, diagnostics)) == set(d3.CHECKPOINTS), "Missing D3 diagnostic checkpoint")
            endpoint_rows = evaluate_actor(actor, cfg, "endpoint", d3.ENDPOINT_EPISODES, "intact")
            torch.save(
                dict(
                    actor_state=actor.state_dict(),
                    critic_state=critic.state_dict(),
                    configuration=cfg,
                    native_steps=d3.NATIVE_STEPS,
                    condition=condition,
                    initialization=dataclasses.asdict(prepared.attestation),
                ),
                out / "model-final.pt",
            )
            write_json(
                out / "ENDPOINT.json",
                dict(
                    status="D3_ENDPOINT_COMPLETE",
                    condition=condition,
                    block=int(block),
                    summary=summarize(endpoint_rows),
                    episodes=endpoint_rows,
                    initialization=dataclasses.asdict(prepared.attestation),
                ),
            )
            if condition == "LOCAL-S6":
                witness = evaluate_no_action(cfg)
                write_json(
                    out / "NO_ACTION_WITNESS.json",
                    dict(status="D3_NO_ACTION_WITNESS_COMPLETE", block=int(block), summary=summarize(witness), episodes=witness),
                )
                panels = {}
                for mode in ("intact", "sham", "inactive_edge_control", "permanent_all_active_lesion"):
                    rows = evaluate_actor(actor, cfg, "causal", d3.CAUSAL_EPISODES, mode)
                    panels[mode] = dict(summary=summarize(rows), episodes=rows)
                write_json(
                    out / "CAUSAL.json",
                    dict(
                        status="D3_CAUSAL_PANEL_COMPLETE",
                        condition=condition,
                        block=int(block),
                        active_edges=list(d3.ACTIVE_EDGES),
                        inactive_control_edge=int(d3.INACTIVE_CONTROL_EDGE),
                        panels=panels,
                    ),
                )
        return result
    finally:
        for env in envs:
            env.close()


def execution_guard() -> dict:
    require(START_REQUEST.exists(), "D3 START_REQUEST.json is absent")
    require(os.environ.get("GITHUB_REPOSITORY") == "CRC2520/polar-sim-ml-original", "Wrong D3 repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + RUNNER_BRANCH, "Wrong D3 execution branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "D3 scientific reruns are not authorized")
    require(os.environ.get("GITHUB_SHA") == git("rev-parse", "HEAD"), "Unexpected D3 checkout")
    req = read_json(START_REQUEST)
    require(req.get("schema") == "B1-SC-D3-START-1.0.0-20260916", "Wrong D3 START_REQUEST schema")
    require(req.get("authorize_b1sc_d3_v1_0") is True, "D3 execution not explicitly authorized")
    require(req.get("authorized_run_attempt") == 1, "Only D3 run attempt 1 may be authorized")
    require(req.get("B1E_executed") is False and req.get("final_seeds_generated") is False, "B1-E boundary violated")
    parent = git("rev-parse", "HEAD^")
    require(req.get("authorization_base_commit") == parent, "D3 authorization base mismatch")
    require(req.get("qualified_runner_commit") == parent, "D3 qualified runner mismatch")
    require(
        git("diff", "--name-only", parent, "HEAD").splitlines() == ["experiments/b1sc_d3_v1_0/START_REQUEST.json"],
        "D3 authorization commit contains changes beyond START_REQUEST.json",
    )
    require(req.get("runner_contract_sha256") == sha256_file(RUNNER_CONTRACT), "D3 runner contract mismatch")
    verify_all_snapshots()
    return req


def qa_microfit(condition: str, block: int, out: Path, purpose: str, budget: int = 512) -> dict:
    require(str(purpose).startswith("qa_"), "D3 microfit purpose must remain QA-only")
    require(not START_REQUEST.exists(), "QA branch must not contain D3 START_REQUEST.json")
    return _train(condition, block, out, int(budget), "qa")


def scientific_fit(condition: str, block: int, out: Path) -> dict:
    execution_guard()
    return _train(condition, block, out, d3.NATIVE_STEPS, "training")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=d3.CONDITIONS, required=True)
    p.add_argument("--block", type=int, choices=range(d3.BLOCKS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qa-microfit", action="store_true")
    p.add_argument("--purpose", default="")
    p.add_argument("--budget", type=int, default=512)
    args = p.parse_args(argv)
    runtime()
    if args.qa_microfit:
        payload = qa_microfit(args.condition, args.block, args.out, args.purpose, args.budget)
    else:
        payload = scientific_fit(args.condition, args.block, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
