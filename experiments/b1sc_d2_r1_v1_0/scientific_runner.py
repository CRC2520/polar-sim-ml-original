"""B1-SC-D2-R1 scientific fit runner with byte-frozen initialization.

The only path to an optimizer object or optimizer.step() is through a
ValidatedInit produced by validating and loading the canonical D2-R1 snapshot.
Scientific entry points remain guarded by START_REQUEST.json; QA uses a bounded
microfit and is not scientific evidence.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from b1s.execution.core import log_squashed_gaussian, model_digest
from b1s.execution.instrument import NativeEnv, runtime_lock
from experiments.b1sc_d2_v1_0 import implementation as d2
from experiments.b1sc_d2_v1_0 import execution as d2exec
from experiments.b1sc_d2_r1_v1_0 import initialization as frozen_init
from experiments.b1sc_d2_r1_v1_0 import execution_contract as contract

ROOT = Path(__file__).resolve().parent
REPO_ROOT = d2.REPO_ROOT
RUNNER_BRANCH = "research/b1sc-d2-r1-v1.0-runner-readiness-20260915"
START_REQUEST = ROOT / "START_REQUEST.json"
RUNNER_FREEZE = ROOT / "run" / "RUNNER_IMPLEMENTATION_FREEZE.json"
SCIENTIFIC_WORKFLOW = ".github/workflows/b1sc-d2-r1-v1-0-scientific.yml"
HP = dict(d2exec.HP)
N_ENVS = d2exec.N_ENVS


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
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return runtime_lock()


def config_for(condition: str, block: int) -> dict:
    require(condition in d2.CONDITIONS and 0 <= int(block) < d2.BLOCKS, "Invalid D2-R1 condition/block")
    row = next(x for x in d2.registry() if x["condition"] == condition and x["block"] == int(block))
    return dict(row, hp=HP.copy(), algorithm="PPO", initialization="D2-R1-BYTE-FROZEN")


@dataclasses.dataclass(frozen=True)
class InitAttestation:
    condition: str
    block: int
    snapshot_sha256: str
    actor_digest: str
    critic_digest: str
    source_seed: int
    artifact_id: int
    validated: bool = True


@dataclasses.dataclass
class PreparedModules:
    actor: nn.Module
    critic: nn.Module
    attestation: InitAttestation
    audit: list[dict]


def _event(audit: list[dict], name: str, **extra) -> None:
    audit.append({"index": len(audit), "event": name, **extra})


def validate_and_load_frozen(condition: str, block: int) -> PreparedModules:
    contract.manifest()
    cfg = config_for(condition, block)
    audit: list[dict] = []
    _event(audit, "validation_started", condition=condition, block=int(block))
    actor, critic, meta = frozen_init.build_initial_modules(condition, int(block))
    _event(
        audit,
        "snapshot_validated",
        snapshot_sha256=meta["initial_state_sha256"],
        actor_digest=meta["actor_digest"],
        critic_digest=meta["critic_digest"],
        artifact_id=int(meta["artifact_id"]),
    )
    require(meta["condition"] == condition and int(meta["block"]) == int(block), "Frozen init identity mismatch")
    require(meta["actor_digest"] == model_digest(actor), "Actor digest changed after frozen load")
    require(meta["critic_digest"] == model_digest(critic), "Critic digest changed after frozen load")
    att = InitAttestation(
        condition=condition,
        block=int(block),
        snapshot_sha256=meta["initial_state_sha256"],
        actor_digest=meta["actor_digest"],
        critic_digest=meta["critic_digest"],
        source_seed=int(meta["source_seed"]),
        artifact_id=int(meta["artifact_id"]),
    )
    require(cfg["initial_seed"] == att.source_seed, "Frozen source-seed lineage mismatch")
    return PreparedModules(actor=actor, critic=critic, attestation=att, audit=audit)


def create_optimizer(prepared: PreparedModules):
    require(prepared.attestation.validated, "Optimizer requires validated frozen initialization")
    require(any(x["event"] == "snapshot_validated" for x in prepared.audit), "Snapshot validation event missing")
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
    require(names.index("snapshot_validated") < names.index("optimizer_created"), "Optimizer created before validation")
    _event(prepared.audit, "optimizer_step")
    opt.step()


def paired_seed(split: str, block: int, purpose: str, **extra) -> int:
    return d2.seed(split, d2.paired_seed_identity(int(block), str(purpose), **extra))


class TrainingEnv:
    def __init__(self, block: int, slot: int, ledger: list[dict], split: str):
        require(split in ("training", "qa"), "Invalid training split")
        self.env = NativeEnv()
        self.block = int(block)
        self.slot = int(slot)
        self.ledger = ledger
        self.split = split
        self.counter = 0
        self.total_steps = 0
        self.ret = 0.0
        self.length = 0

    def reset(self):
        sd = paired_seed(self.split, self.block, "environment", slot=self.slot, episode=self.counter)
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
            self.ledger.append(dict(block=self.block, slot=self.slot, episode=self.counter - 1, seed=self.env.episode_seed, slot_steps=self.total_steps, return_value=self.ret, cycles=self.length, displacement=float(info["package_displacement"]), fallen_count=int(info["fallen_count"]), package_dropped=bool(info["package_dropped"]), terminated=bool(terminated), truncated=bool(truncated)))
        return z, reward, terminated, truncated, info

    def close(self):
        self.env.close()


def gae_returns(rewards, values, dones, last, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    g = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        nt = 1 - dones[t].astype(float)
        nv = last if t == len(rewards) - 1 else values[t + 1]
        g = rewards[t] + gamma * nv * nt - values[t] + gamma * lam * nt * g
        adv[t] = g
    return adv, adv + values


def _train(condition: str, block: int, out: Path, budget: int, split: str) -> dict:
    require(split in ("training", "qa"), "Unknown runner split")
    if split == "training":
        require(budget == d2.NATIVE_STEPS, "Scientific budget differs from frozen D2")
    else:
        require(0 < budget <= 2048 and budget % HP["rollout_steps"] == 0, "QA microfit budget exceeded")
    cfg = config_for(condition, block)
    prepared = validate_and_load_frozen(condition, block)
    initial_actor = model_digest(prepared.actor)
    initial_critic = model_digest(prepared.critic)
    initial_A = d2exec.module_digest(prepared.actor.A)
    opt = create_optimizer(prepared)
    actor, critic = prepared.actor, prepared.critic
    hp = cfg["hp"]
    batch = hp["rollout_steps"]
    horizon = batch // N_ENVS
    require(budget % batch == 0, "Budget must be an integer number of PPO rollouts")
    ledger: list[dict] = []
    envs = [TrainingEnv(block, i, ledger, split) for i in range(N_ENVS)]
    z = np.stack([e.reset()[0] for e in envs])
    if split == "training":
        torch.manual_seed(int(cfg["policy_sampling_seed"]))
    else:
        torch.manual_seed(paired_seed("qa", block, "policy-sampling"))
    rng = np.random.default_rng(paired_seed(split, block, "minibatches"))
    lesion = d2.training_lesion(condition)
    updates = 0
    t0 = time.monotonic()
    first_step_verified = False
    try:
        for offset in range(0, budget, batch):
            obs, pre, logps, values, rewards, dones = [], [], [], [], [], []
            for _ in range(horizon):
                with torch.no_grad():
                    mu, ls = actor(torch.from_numpy(z), lesion)
                    p = mu + ls.exp() * torch.randn_like(mu)
                    a = p.tanh()
                    lp = log_squashed_gaussian(p, mu, ls)
                    v = critic(torch.from_numpy(z))
                obs.append(z.copy()); pre.append(p.numpy()); logps.append(lp.numpy()); values.append(v.numpy())
                rs, ds, nzs = [], [], []
                for i, env in enumerate(envs):
                    nz, reward, terminated, truncated, _ = env.step(a[i].numpy())
                    done = terminated or truncated
                    rs.append(reward); ds.append(done); nzs.append(env.reset()[0] if done else nz)
                rewards.append(rs); dones.append(ds); z = np.stack(nzs)
            with torch.no_grad():
                last = critic(torch.from_numpy(z)).numpy()
            adv, target = gae_returns(np.asarray(rewards), np.asarray(values), np.asarray(dones), last, hp["gamma"], hp["gae_lambda"])
            adv = adv.reshape(-1); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ot = torch.from_numpy(np.asarray(obs).reshape(batch, 93)); pt = torch.from_numpy(np.asarray(pre).reshape(batch, 12)); old = torch.from_numpy(np.asarray(logps).reshape(-1)); at = torch.from_numpy(adv); rt = torch.from_numpy(target.reshape(-1))
            for _ in range(hp["epochs"]):
                order = rng.permutation(batch)
                for j in range(0, batch, hp["minibatch"]):
                    ix = order[j:j + hp["minibatch"]]
                    mu, ls = actor(ot[ix], lesion); lp = log_squashed_gaussian(pt[ix], mu, ls); logr = lp - old[ix]; ratio = logr.exp()
                    al = -torch.min(ratio * at[ix], ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * at[ix]).mean()
                    vl = (critic(ot[ix]) - rt[ix]).square().mean(); loss = al + hp["value_coef"] * vl
                    require(bool(torch.isfinite(loss)), "Nonfinite PPO objective"); opt.zero_grad(); loss.backward()
                    grad = nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), hp["max_grad_norm"]); require(bool(torch.isfinite(grad)), "Nonfinite gradient")
                    if not first_step_verified:
                        names = [x["event"] for x in prepared.audit]
                        require(names == ["validation_started", "snapshot_validated", "optimizer_created"], "Unexpected pre-step audit order")
                        first_step_verified = True
                    guarded_optimizer_step(opt, prepared); updates += 1
        require(first_step_verified and updates > 0, "Runner never reached a guarded optimizer step")
        result = dict(status="QA_MICROFIT_COMPLETE" if split == "qa" else "SCIENTIFIC_FIT_COMPLETE", split=split, condition=condition, block=int(block), environment_steps=int(budget), optimizer_updates=int(updates), initialization=dataclasses.asdict(prepared.attestation), initialization_audit=prepared.audit, initial_actor_digest=initial_actor, initial_critic_digest=initial_critic, initial_message_A_digest=initial_A, final_actor_digest=model_digest(actor), final_critic_digest=model_digest(critic), final_message_A_digest=d2exec.module_digest(actor.A), training_lesion=list(lesion), seconds=time.monotonic() - t0, scientific_evidence=bool(split == "training"), B1E_executed=False, final_seeds_generated=False)
        if condition == "S6-OFF-TRAIN":
            require(result["initial_message_A_digest"] == result["final_message_A_digest"], "OFF message A changed")
        write_json(out / "RUN_RESULT.json", result)
        return result
    finally:
        for env in envs:
            env.close()


def qa_microfit(condition: str, block: int, init_dir: str, out_root: str, budget: int = 512) -> dict:
    os.environ["D2R1_INIT_DIR"] = str(init_dir); runtime(); out = Path(out_root); out.mkdir(parents=True, exist_ok=False)
    return _train(condition, int(block), out, int(budget), "qa")


def scientific_guard() -> dict:
    require(os.environ.get("GITHUB_REPOSITORY") == d2.REPOSITORY, "Wrong repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + RUNNER_BRANCH, "Wrong D2-R1 runner branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Scientific reruns are not authorized")
    require(START_REQUEST.is_file(), "START_REQUEST.json is absent")
    require(RUNNER_FREEZE.is_file(), "Runner implementation freeze is absent")
    req, freeze = read_json(START_REQUEST), read_json(RUNNER_FREEZE)
    require(req.get("schema") == contract.SCHEMA, "Wrong D2-R1 request schema")
    require(req.get("authorize_b1sc_d2_r1_v1_0") is True, "D2-R1 scientific execution not authorized")
    require(req.get("authorized_run_attempt") == 1, "Only attempt 1 may be authorized")
    require(int(req.get("authorized_run_number", -1)) == int(os.environ.get("GITHUB_RUN_NUMBER", "-2")), "Unauthorized workflow run number")
    require(freeze.get("status") == "D2_R1_RUNNER_QUALIFIED_NOT_AUTHORIZED", "Runner freeze status changed")
    require(req.get("qualified_source_commit") == freeze.get("qualified_source_commit"), "Qualified runner source mismatch")
    require(req.get("runner_freeze_sha256") == sha256_file(RUNNER_FREEZE), "Runner freeze mismatch")
    require(req.get("workflow_sha256") == freeze["source_hashes"][SCIENTIFIC_WORKFLOW], "Scientific workflow hash mismatch")
    require(req.get("fit_registry_sha256") == freeze["source_hashes"]["experiments/b1sc_d2_r1_v1_0/FIT_REGISTRY.json"], "Fit registry hash mismatch")
    require(req.get("init_freeze_sha256") == freeze["source_hashes"]["experiments/b1sc_d2_r1_v1_0/INIT_FREEZE.json"], "Init freeze hash mismatch")
    require(req.get("canonical_init_bundle_sha256") == contract.CANONICAL_BUNDLE_SHA256, "Canonical init bundle mismatch")
    require(req.get("B1E_executed") is False and req.get("final_seeds_generated") is False, "B1-E boundary violated")
    for rel, expected in freeze.get("source_hashes", {}).items():
        path = REPO_ROOT / rel; require(path.is_file() and sha256_file(path) == expected, "Frozen runner source changed: " + rel)
    current = git("rev-parse", "HEAD"); require(current == os.environ.get("GITHUB_SHA"), "Unexpected checkout")
    parent = git("rev-parse", "HEAD^"); require(req.get("authorization_base_commit") == parent, "Authorization base mismatch")
    require(git("diff", "--name-only", parent, current).splitlines() == ["experiments/b1sc_d2_r1_v1_0/START_REQUEST.json"], "Authorization commit contains other changes")
    return req


def scientific_fit(condition: str, block: int, init_dir: str, out_root: str) -> dict:
    scientific_guard(); os.environ["D2R1_INIT_DIR"] = str(init_dir); runtime(); out = Path(out_root) / f"{condition}-b{int(block)}"; out.mkdir(parents=True, exist_ok=False)
    try:
        return _train(condition, int(block), out, d2.NATIVE_STEPS, "training")
    except Exception:
        write_json(out / "FAILED.json", {"status": "FAILED_RETAINED", "error": traceback.format_exc(), "replacement_seed": False}); raise
