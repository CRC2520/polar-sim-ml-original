"""B1-SC-D4 v1.0 scientific runner candidate.

Consumes only the twelve repository-persisted, double-verified D4 trainable
snapshots. QA microfits use only the ``qa`` namespace and are never scientific
evidence. Scientific entry points remain blocked until a separate immutable
START_REQUEST authorization is added after runner qualification.
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
from experiments.b1sc_d4_v1_0 import implementation as d4
from experiments.b1sc_d4_v1_0 import init_format

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
FROZEN_BASE_HEAD = "255c520d626440dd20b8b0d626530b6aa9e09f90"
RUNNER_BRANCH = "research/b1sc-d4-v1.0-runner-readiness-20260916"
START_REQUEST = ROOT / "START_REQUEST.json"
RUNNER_CONTRACT = ROOT / "RUNNER_CONTRACT.json"
SNAPSHOT_DIR = ROOT / "frozen_init"
INIT_FREEZE = ROOT / "INIT_FREEZE.json"
INIT_VERIFY_RECEIPT = ROOT / "INIT_VERIFY_RECEIPT.json"

HP = dict(d2exec.HP)
N_ENVS = int(d2exec.N_ENVS)
EXPECTED_HP = dict(lr=0.0003, gamma=0.99, rollout_steps=512, minibatch=128, epochs=4,
                   gae_lambda=0.95, clip=0.2, value_coef=0.5, max_grad_norm=0.5,
                   adam_eps=1e-5)


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
    rows = d4.registry()
    require(len(rows) == d4.FITS, "D4 registry cardinality changed")
    return rows


def config_for(condition: str, block: int) -> dict:
    require(condition in d4.CONDITIONS, "Unknown D4 condition")
    require(0 <= int(block) < d4.BLOCKS, "Invalid D4 block")
    row = next(x for x in _registry() if x["condition"] == condition and int(x["block"]) == int(block))
    require(int(row["native_steps"]) == d4.NATIVE_STEPS, "D4 fit budget changed")
    require(row["support_mask"] == d4.S6 and row["information_mode"] == d4.LOCAL, "D4 actor contract changed")
    require(int(row["initial_snapshot_block"]) == int(block), "D4 snapshot pairing changed")
    return dict(row, hp=HP.copy(), algorithm="PPO", initialization="D4-BYTE-FROZEN")


def _init_authority() -> tuple[dict, dict]:
    freeze = read_json(INIT_FREEZE)
    receipt = read_json(INIT_VERIFY_RECEIPT)
    require(freeze["status"] == "D4_FROZEN_INITIALIZATION_PERSISTED_PENDING_DOUBLE_VERIFICATION",
            "D4 persisted init freeze changed")
    require(receipt["status"] == "D4_FROZEN_INITIALIZATION_DOUBLE_VERIFIED_NOT_EXECUTED",
            "D4 double-verification receipt changed")
    require(receipt["persistence"]["commit"] == "ab8a80f7f44bee6a2b44d5671440e99fba05d36c",
            "D4 persistence authority changed")
    require(receipt["persistence"]["init_freeze_git_blob"] == "dbf0a7238acbbd41ac6d6c0a7d69762095f3ca25",
            "D4 INIT_FREEZE authority changed")
    require(receipt["double_verification"]["status"] == "D4_PERSISTED_INITIALIZATION_DOUBLE_VERIFICATION_PASS",
            "D4 persisted bytes are not double verified")
    require(freeze["scientific_training_performed"] is False and freeze["scientific_evaluation_performed"] is False,
            "Initialization freeze contains science")
    c = receipt["persistence"]
    require(c["snapshot_count"] == d4.BLOCKS and c["trainable_only"] is True, "D4 snapshot count/format changed")
    require(c["support_buffer_excluded"] is True and c["temporal_gate_state_excluded"] is True,
            "D4 excluded state entered snapshots")
    require(c["condition_excluded"] is True and c["shared_byte_for_byte_across_three_conditions"] is True,
            "D4 condition pairing changed")
    require(c["seed_reconstruction_at_scientific_runtime_forbidden"] is True,
            "D4 seed reconstruction became allowed")
    return freeze, receipt


def snapshot_record(block: int) -> dict:
    _, receipt = _init_authority()
    return next(x for x in receipt["snapshots"] if int(x["block"]) == int(block))


def snapshot_path(block: int) -> Path:
    return SNAPSHOT_DIR / f"block-{int(block)}.bin"


def verify_snapshot_bytes(block: int) -> dict:
    rec = snapshot_record(block)
    p = snapshot_path(block)
    require(p.is_file(), f"Missing frozen D4 snapshot: block {block}")
    require(p.stat().st_size == int(rec["bytes"]), f"D4 snapshot byte length mismatch: block {block}")
    actual = sha256_file(p)
    require(actual == rec["sha256"], f"D4 snapshot SHA mismatch: block {block}")
    return dict(block=int(block), bytes=int(rec["bytes"]), sha256=actual, source_seed=int(rec["source_seed"]))


def verify_all_snapshots() -> list[dict]:
    rows = [verify_snapshot_bytes(b) for b in range(d4.BLOCKS)]
    require(len({x["sha256"] for x in rows}) == d4.BLOCKS, "D4 snapshots are not twelve distinct block bytes")
    return rows


def derive_child_seed(root: int, *parts: object) -> int:
    raw = json.dumps([int(root), *parts], separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def qa_seed(block: int, purpose: str, **extra: object) -> int:
    return d4.seed("qa", d4.paired_seed_identity(int(block), str(purpose), **extra))


@dataclasses.dataclass(frozen=True)
class InitAttestation:
    condition: str
    block: int
    snapshot_sha256: str
    snapshot_bytes: int
    source_seed: int
    actor_trainable_digest: str
    critic_trainable_digest: str
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
    require(HP == EXPECTED_HP and N_ENVS == 8, "Historical PPO profile changed")
    d4.validate_design()
    config_for(condition, block)
    rec = snapshot_record(block)
    audit: list[dict] = []
    _event(audit, "validation_started", condition=condition, block=int(block))
    verify_snapshot_bytes(block)
    actor = d4.D4RoutingActor()
    critic = Value()
    header, actual = init_format.load_trainable_into(snapshot_path(block), actor, critic,
                                                      expected_sha256=rec["sha256"])
    require(int(header["block"]) == int(block), "D4 snapshot block identity mismatch")
    require(int(header["source_seed"]) == int(rec["source_seed"]), "D4 snapshot source lineage mismatch")
    require(header["support_buffer_included"] is False and header["temporal_gate_state_included"] is False,
            "D4 snapshot contains excluded state")
    actor_digest = init_format.trainable_digest(actor)
    critic_digest = init_format.trainable_digest(critic)
    require(actor_digest == rec["actor_trainable_digest"], "D4 actor trainable digest mismatch")
    require(critic_digest == rec["critic_trainable_digest"], "D4 critic trainable digest mismatch")
    att = InitAttestation(condition=condition, block=int(block), snapshot_sha256=actual,
                          snapshot_bytes=int(rec["bytes"]), source_seed=int(rec["source_seed"]),
                          actor_trainable_digest=actor_digest, critic_trainable_digest=critic_digest)
    _event(audit, "snapshot_validated", snapshot_sha256=actual,
           actor_trainable_digest=actor_digest, critic_trainable_digest=critic_digest)
    return PreparedModules(actor=actor, critic=critic, attestation=att, audit=audit)


def create_optimizer(prepared: PreparedModules):
    require(prepared.attestation.validated, "Optimizer requires validated D4 initialization")
    require([x["event"] for x in prepared.audit] == ["validation_started", "snapshot_validated"],
            "Unexpected pre-optimizer audit order")
    opt = torch.optim.Adam(list(prepared.actor.parameters()) + list(prepared.critic.parameters()),
                           lr=HP["lr"], eps=HP["adam_eps"])
    _event(prepared.audit, "optimizer_created")
    return opt


def guarded_optimizer_step(opt, prepared: PreparedModules) -> None:
    names = [x["event"] for x in prepared.audit]
    require(prepared.attestation.validated and "snapshot_validated" in names and "optimizer_created" in names,
            "optimizer.step missing frozen-init preconditions")
    require(names.index("snapshot_validated") < names.index("optimizer_created"),
            "Optimizer created before D4 snapshot validation")
    _event(prepared.audit, "optimizer_step")
    opt.step()


class TrainingEnv:
    def __init__(self, block: int, slot: int, ledger: list[dict], seed_root: int):
        from b1s.execution.instrument import NativeEnv
        self.env = NativeEnv(); self.block = int(block); self.slot = int(slot)
        self.ledger = ledger; self.seed_root = int(seed_root); self.counter = 0
        self.total_steps = 0; self.ret = 0.0; self.length = 0
    def reset(self):
        sd = derive_child_seed(self.seed_root, "slot", self.slot, "episode", self.counter); self.counter += 1
        z, info = self.env.reset(seed=sd); self.ret = 0.0; self.length = 0
        return z, info
    def step(self, action):
        z, r, t, tr, info = self.env.step(action); self.total_steps += 1; self.ret += r; self.length += 1
        if t or tr:
            self.ledger.append(dict(block=self.block, slot=self.slot, episode=self.counter-1,
                seed=self.env.episode_seed, slot_steps=self.total_steps, return_value=self.ret,
                cycles=self.length, displacement=float(info["package_displacement"]),
                fallen_count=int(info["fallen_count"]), package_dropped=bool(info["package_dropped"]),
                terminated=bool(t), truncated=bool(tr)))
        return z, r, t, tr, info
    def close(self): self.env.close()


def gae_returns(rewards, values, dones, last, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32); g = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(len(rewards)-1, -1, -1):
        nt = 1-dones[t].astype(float); nv = last if t == len(rewards)-1 else values[t+1]
        g = rewards[t] + gamma*nv*nt - values[t] + gamma*lam*nt*g; adv[t] = g
    return adv, adv + values


@contextlib.contextmanager
def isolated_rng():
    p, n, t = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try: yield
    finally: random.setstate(p); np.random.set_state(n); torch.set_rng_state(t)


def summarize(rows: Sequence[dict]) -> dict:
    require(bool(rows), "Cannot summarize absent episodes")
    return dict(loss=statistics.mean(float(x["loss"]) for x in rows),
                displacement=statistics.mean(float(x["displacement"]) for x in rows),
                fall_fraction=statistics.mean(int(x["fallen_count"] > 0) for x in rows),
                drop_fraction=statistics.mean(int(bool(x["package_dropped"])) for x in rows),
                cycles=statistics.mean(int(x["cycles"]) for x in rows), episodes=len(rows))


def deterministic_action(actor, z, route_lambda: float, lesion: tuple[int, ...] = ()):
    with torch.no_grad():
        mu, _ = actor(torch.from_numpy(np.asarray(z, dtype=np.float32)[None]),
                      route_lambda=route_lambda, lesion=lesion)
        return mu.tanh()[0].cpu().numpy().astype(np.float32)


def _panel_root(cfg: dict, split: str) -> int:
    return int(cfg[{"diagnostic":"diagnostic_panel_seed", "endpoint":"endpoint_panel_seed",
                    "causal":"causal_panel_seed"}[split]])


def evaluate_actor(actor, cfg: dict, split: str, episodes: int,
                   route_lambda: float, mode: str) -> list[dict]:
    from b1s.execution.instrument import NativeEnv
    require(split in ("diagnostic", "endpoint", "causal"), "Invalid D4 evaluation split")
    cap = {"diagnostic":d4.DIAGNOSTIC_EPISODES, "endpoint":d4.ENDPOINT_EPISODES,
           "causal":d4.CAUSAL_EPISODES}[split]
    require(1 <= int(episodes) <= int(cap), "D4 evaluation count outside cap")
    if split == "causal":
        require(cfg["condition"] in ("LOCAL-S6-LATE", "LOCAL-S6-ALWAYS"),
                "D4 primary causal panel is LATE/ALWAYS only")
    lesion = ()
    if mode == "permanent_all_active_lesion": lesion = tuple(d4.ACTIVE_EDGES)
    elif mode == "inactive_edge_control": lesion = (d4.INACTIVE_CONTROL_EDGE,)
    elif mode not in ("configured", "intact", "sham"): raise RuntimeError("Unknown D4 evaluation mode")
    root = _panel_root(cfg, split); rows = []; env = NativeEnv(); before = model_digest(actor)
    with isolated_rng():
        try:
            for ep in range(int(episodes)):
                sd = derive_child_seed(root, "episode", ep); z, _ = env.reset(seed=sd); ret = 0.0
                trace = hashlib.sha256()
                while True:
                    a = deterministic_action(actor, z, route_lambda, lesion)
                    nz, r, t, tr, audit = env.step(a)
                    trace.update(np.asarray(z, np.float32).tobytes() + a.tobytes() +
                                 np.float64(r).tobytes() + np.asarray(nz, np.float32).tobytes())
                    ret += r; z = nz
                    if t or tr: break
                rows.append(dict(condition=cfg["condition"], mode=mode, block=int(cfg["block"]),
                    episode=ep, seed=sd, route_lambda=float(route_lambda), loss=-ret, return_value=ret,
                    displacement=float(audit["package_displacement"]), cycles=int(env.t),
                    fallen_count=int(audit["fallen_count"]), package_dropped=bool(audit["package_dropped"]),
                    terminated=bool(t), truncated=bool(tr), trace_sha256=trace.hexdigest()))
        finally: env.close()
    require(model_digest(actor) == before, "D4 evaluation altered actor")
    return rows


def evaluate_no_action(cfg: dict, episodes: int = d4.ENDPOINT_EPISODES) -> list[dict]:
    from b1s.execution.instrument import NativeEnv
    require(1 <= int(episodes) <= d4.ENDPOINT_EPISODES, "Witness count outside cap")
    root = int(cfg["endpoint_panel_seed"]); rows = []; env = NativeEnv()
    try:
        for ep in range(int(episodes)):
            sd = derive_child_seed(root, "episode", ep); z, _ = env.reset(seed=sd); ret = 0.0
            trace = hashlib.sha256()
            while True:
                a = np.zeros(12, dtype=np.float32); nz, r, t, tr, audit = env.step(a)
                trace.update(np.asarray(z, np.float32).tobytes() + a.tobytes() +
                             np.float64(r).tobytes() + np.asarray(nz, np.float32).tobytes())
                ret += r; z = nz
                if t or tr: break
            rows.append(dict(condition="no_action", mode="no_action", block=int(cfg["block"]),
                episode=ep, seed=sd, loss=-ret, return_value=ret,
                displacement=float(audit["package_displacement"]), cycles=int(env.t),
                fallen_count=int(audit["fallen_count"]), package_dropped=bool(audit["package_dropped"]),
                terminated=bool(t), truncated=bool(tr), trace_sha256=trace.hexdigest()))
    finally: env.close()
    return rows


def _seed_plan(condition: str, block: int, split: str) -> dict:
    cfg = config_for(condition, block)
    if split == "training":
        env_root = int(cfg["training_environment_seed_root"]); policy = int(cfg["policy_sampling_seed"])
        minibatches = d4.seed("training", d4.paired_seed_identity(block, "ppo-minibatches"))
    else:
        require(split == "qa", "Unknown D4 seed split")
        env_root = qa_seed(block, "environment-stream"); policy = qa_seed(block, "policy-sampling")
        minibatches = qa_seed(block, "ppo-minibatches")
    return dict(environment_seed_root=env_root, policy_sampling_seed=policy, minibatch_seed=minibatches)


def _checkpoint_eval(prepared: PreparedModules, cfg: dict, steps: int, out: Path) -> None:
    lam = d4.diagnostic_lambda(cfg["condition"], steps)
    rows = evaluate_actor(prepared.actor, cfg, "diagnostic", d4.DIAGNOSTIC_EPISODES, lam, "configured")
    write_json(out / f"DIAGNOSTIC_{steps}.json",
               dict(status="D4_DIAGNOSTIC_COMPLETE", native_steps=steps, route_lambda=lam,
                    summary=summarize(rows), episodes=rows, primary_selection_role="NONE"))


def _train(condition: str, block: int, out: Path, budget: int, split: str,
           start_completed_steps: int = 0) -> dict:
    require(split in ("training", "qa"), "Unknown D4 runner split")
    require(HP == EXPECTED_HP and N_ENVS == 8, "Historical PPO profile changed")
    cfg = config_for(condition, block)
    if split == "training":
        execution_guard()
        require(int(budget) == d4.NATIVE_STEPS and int(start_completed_steps) == 0,
                "Scientific D4 budget/start differs from freeze")
    else:
        require(0 < int(budget) <= 2048 and int(budget) % HP["rollout_steps"] == 0,
                "QA D4 microfit budget invalid")
        require(0 <= int(start_completed_steps) <= d4.NATIVE_STEPS-int(budget), "QA schedule start invalid")
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    prepared = validate_and_load_frozen(condition, block); actor = prepared.actor; critic = prepared.critic
    opt = create_optimizer(prepared); seedplan = _seed_plan(condition, block, split)
    torch.manual_seed(seedplan["policy_sampling_seed"]); rng = np.random.default_rng(seedplan["minibatch_seed"])
    ledger = []; curves = []; schedule = []
    envs = [TrainingEnv(block, i, ledger, seedplan["environment_seed_root"]) for i in range(N_ENVS)]
    z = np.stack([e.reset()[0] for e in envs]); batch = HP["rollout_steps"]; horizon = batch // N_ENVS
    t0 = time.monotonic(); updates = 0
    try:
        for offset in range(0, int(budget), batch):
            completed = int(start_completed_steps) + offset
            lam = d4.training_lambda(condition, completed)
            schedule.append(dict(completed_native_steps_before_rollout=completed, route_lambda=lam))
            obs=[]; pre=[]; logps=[]; values=[]; rewards=[]; dones=[]
            for _ in range(horizon):
                with torch.no_grad():
                    mu, ls = actor(torch.from_numpy(z), route_lambda=lam)
                    p = mu + ls.exp()*torch.randn_like(mu); a = p.tanh()
                    lp = log_squashed_gaussian(p, mu, ls); v = critic(torch.from_numpy(z))
                obs.append(z.copy()); pre.append(p.numpy()); logps.append(lp.numpy()); values.append(v.numpy())
                rs=[]; ds=[]; nzs=[]
                for i,e in enumerate(envs):
                    nz,r,t,tr,_ = e.step(a[i].numpy()); done = t or tr
                    rs.append(r); ds.append(done); nzs.append(e.reset()[0] if done else nz)
                rewards.append(rs); dones.append(ds); z = np.stack(nzs)
            with torch.no_grad(): last = critic(torch.from_numpy(z)).numpy()
            adv,target = gae_returns(np.asarray(rewards),np.asarray(values),np.asarray(dones),last,
                                     HP["gamma"],HP["gae_lambda"])
            adv=adv.reshape(-1); adv=(adv-adv.mean())/(adv.std()+1e-8)
            ot=torch.from_numpy(np.asarray(obs).reshape(batch,93)); pt=torch.from_numpy(np.asarray(pre).reshape(batch,12))
            old=torch.from_numpy(np.asarray(logps).reshape(-1)); at=torch.from_numpy(adv); rt=torch.from_numpy(target.reshape(-1))
            als=[]; vls=[]
            for _ in range(HP["epochs"]):
                order=rng.permutation(batch)
                for j in range(0,batch,HP["minibatch"]):
                    ix=order[j:j+HP["minibatch"]]
                    mu,ls=actor(ot[ix],route_lambda=lam); lp=log_squashed_gaussian(pt[ix],mu,ls)
                    ratio=(lp-old[ix]).exp(); al=-torch.min(ratio*at[ix],ratio.clamp(1-HP["clip"],1+HP["clip"])*at[ix]).mean()
                    vl=(critic(ot[ix])-rt[ix]).square().mean(); loss=al+HP["value_coef"]*vl
                    require(bool(torch.isfinite(loss)), "Nonfinite D4 PPO objective")
                    opt.zero_grad(); loss.backward()
                    g=nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),HP["max_grad_norm"])
                    require(bool(torch.isfinite(g)), "Nonfinite D4 gradient")
                    guarded_optimizer_step(opt,prepared); updates+=1; als.append(float(al.detach())); vls.append(float(vl.detach()))
            total=completed+batch
            if total%32768==0 or offset+batch==budget:
                curves.append(dict(native_steps=total,route_lambda=lam,optimizer_updates=updates,
                                   actor_loss=statistics.mean(als),value_loss=statistics.mean(vls),completed_episodes=len(ledger)))
            if split=="training" and total in d4.CHECKPOINTS:
                torch.save(dict(actor_state=actor.state_dict(),critic_state=critic.state_dict(),
                                configuration=cfg,native_steps=total),out/f"checkpoint-{total}.pt")
                _checkpoint_eval(prepared,cfg,total,out)
    finally:
        for e in envs: e.close()
    result=dict(status="D4_SCIENTIFIC_FIT_COMPLETE" if split=="training" else "D4_QA_MICROFIT_COMPLETE",
        condition=condition,block=int(block),split=split,scientific_evidence=bool(split=="training"),
        scientific_training_performed=bool(split=="training"),environment_steps=int(budget),
        start_completed_steps=int(start_completed_steps),final_completed_steps=int(start_completed_steps)+int(budget),
        route_schedule=schedule,initialization=dataclasses.asdict(prepared.attestation),
        initialization_audit=prepared.audit,seed_plan=seedplan,training_curve=curves,
        elapsed_seconds=time.monotonic()-t0,B1E_executed=False,final_seeds_generated=False)
    if split=="qa":
        write_json(out/"QA_RESULT.json",result); return result

    endpoint_lam=d4.endpoint_lambda(condition)
    erows=evaluate_actor(actor,cfg,"endpoint",d4.ENDPOINT_EPISODES,endpoint_lam,"configured")
    write_json(out/"ENDPOINT_CONFIGURED.json",dict(status="D4_ENDPOINT_CONFIGURED_COMPLETE",
               route_lambda=endpoint_lam,summary=summarize(erows),episodes=erows))
    witness=evaluate_no_action(cfg)
    write_json(out/"NO_ACTION_WITNESS.json",dict(status="D4_NO_ACTION_WITNESS_COMPLETE",
               summary=summarize(witness),episodes=witness))
    if condition in ("LOCAL-S6-LATE","LOCAL-S6-ALWAYS"):
        lesion_rows=evaluate_actor(actor,cfg,"endpoint",d4.ENDPOINT_EPISODES,1.0,"permanent_all_active_lesion")
        write_json(out/"ENDPOINT_PERMANENT_LESION.json",
                   dict(status="D4_ENDPOINT_PERMANENT_LESION_COMPLETE",summary=summarize(lesion_rows),episodes=lesion_rows))
        causal={}
        for mode in ("intact","sham","inactive_edge_control","permanent_all_active_lesion"):
            rows=evaluate_actor(actor,cfg,"causal",d4.CAUSAL_EPISODES,1.0,mode)
            causal[mode]=dict(summary=summarize(rows),episodes=rows)
        write_json(out/"CAUSAL.json",dict(status="D4_CAUSAL_PANEL_COMPLETE",regimes=causal))
    torch.save(dict(actor_state=actor.state_dict(),critic_state=critic.state_dict(),configuration=cfg,
                    native_steps=d4.NATIVE_STEPS),out/"model-final.pt")
    write_json(out/"TRAINING_RESOURCES.json",dict(status="D4_TRAINING_COMPLETE",
               initial_snapshot_sha256=prepared.attestation.snapshot_sha256,initialization_audit=prepared.audit,
               seed_plan=seedplan,route_schedule=schedule,training_curve=curves,environment_steps=d4.NATIVE_STEPS))
    write_json(out/"RUN_RESULT.json",result)
    return result


def qa_microfit(condition: str, block: int, out: Path, budget: int = 1024,
                start_completed_steps: int = d4.SWITCH_STEPS-512) -> dict:
    require(not START_REQUEST.exists(), "D4 START_REQUEST must be absent during QA")
    return _train(condition, block, out, budget, "qa", start_completed_steps)


def execution_guard() -> dict:
    require(os.environ.get("GITHUB_REPOSITORY") == d4.REPOSITORY, "Wrong D4 repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/" + RUNNER_BRANCH, "Wrong D4 runner branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Scientific reruns are not authorized")
    require(START_REQUEST.exists(), "D4 START_REQUEST.json is absent")
    req=read_json(START_REQUEST)
    require(req.get("schema") == "B1-SC-D4-START-1.0.0-20260916", "Wrong D4 START schema")
    require(req.get("authorize_b1sc_d4_v1_0") is True, "D4 execution not explicitly authorized")
    require(req.get("authorized_run_attempt") == 1, "Only D4 run attempt 1 may be authorized")
    require(req.get("B1E_executed") is False and req.get("final_seeds_generated") is False,
            "B1-E/final-seed boundary violated")
    current=git("rev-parse","HEAD"); require(current == os.environ.get("GITHUB_SHA"), "Unexpected D4 checkout")
    qualified=str(req.get("qualified_runner_commit", "")); require(bool(qualified), "Qualified D4 runner commit absent from START")
    subprocess.check_call(["git","merge-base","--is-ancestor",qualified,current])
    for rel in ("experiments/b1sc_d4_v1_0/scientific_runner.py",
                "experiments/b1sc_d4_v1_0/RUNNER_CONTRACT.json"):
        require(git("rev-parse",f"{qualified}:{rel}") == git("rev-parse",f"HEAD:{rel}"),
                f"Qualified D4 runner source changed: {rel}")
    verify_all_snapshots()
    return req


def scientific_fit(condition: str, block: int, out: Path) -> dict:
    execution_guard(); runtime()
    return _train(condition, block, out, d4.NATIVE_STEPS, "training", 0)
