#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N = 4
K = 5
BLOCK_LEN = 200
BLOCKS = 10
T = BLOCK_LEN * BLOCKS
MIN_FIT = 28
EXPLORE_VISITS = 46
RIDGE = 0.05
STAY = 0.97
CUE_SIGMA = 0.92
ACTION_PENALTY = 0.025
PLAN_DISCOUNT = 0.72

ACTIONS = np.vstack([
    np.zeros((1, N)),
    np.eye(N) * 0.22,
    -np.eye(N) * 0.22,
])

CUE_CENTERS = np.array([
    [ 1.00,  0.00,  0.35],
    [ 0.31,  0.95, -0.25],
    [-0.81,  0.59,  0.30],
    [-0.81, -0.59, -0.30],
    [ 0.31, -0.95,  0.20],
], dtype=float)

DEV_SEEDS = list(range(2136001, 2136005))
CONF_SEEDS = list(range(2137001, 2137017))
DEV_FAMILY = "sparse_mixed"
CONF_FAMILIES = ["ring_signed", "lowrank_skew"]

def srng(seed, label):
    h = hashlib.sha256(f"R42|{int(seed)}|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))

def mse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.mean(d*d))

def standard_effect(v):
    v = np.asarray(v, float)
    if len(v) < 20:
        return 0.0
    return float(v.mean() / (v.std(ddof=1) + 1e-12))

def spectral_scale(A, limit=0.79):
    rad = float(max(abs(np.linalg.eigvals(A))))
    return A if rad <= limit else A * (limit / rad)

def make_regime_matrix(rng, family):
    diag = rng.uniform(0.43, 0.58, N)
    A = np.diag(diag)
    if family == "sparse_mixed":
        for i in range(N):
            choices = [j for j in range(N) if j != i]
            rng.shuffle(choices)
            for j in choices[:2]:
                A[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.09, 0.16)
    elif family == "ring_signed":
        signs = rng.choice([-1.0, 1.0], size=2*N)
        for i in range(N):
            A[i, (i+1) % N] = signs[2*i] * rng.uniform(0.11, 0.17)
            A[i, (i-1) % N] = signs[2*i+1] * rng.uniform(0.07, 0.13)
    elif family == "lowrank_skew":
        u = rng.normal(0, 1, N); v = rng.normal(0, 1, N)
        low = 0.045 * np.outer(u, v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12)
        M = rng.normal(0, 1, (N, N))
        skew = 0.055 * (M - M.T) / (np.linalg.norm(M - M.T) + 1e-12)
        A = A + low + skew
    else:
        raise ValueError(family)
    return spectral_scale(A)

class World:
    def __init__(self, seed, family):
        self.seed = int(seed)
        self.family = family
        self.t = 0
        rm = srng(seed, f"{family}-matrices")
        rs = srng(seed, f"{family}-schedule")
        rb = srng(seed, f"{family}-body")
        rg = srng(seed, f"{family}-goals")
        rh = srng(seed, f"{family}-nonlinear")
        self.As = np.asarray([make_regime_matrix(rm, family) for _ in range(K)])
        self.Hs = np.asarray([rh.normal(0, 0.28, (N, N)) for _ in range(K)])
        self.bias = rm.normal(0, 0.028, (K, N))
        self.B = np.diag(rb.uniform(0.80, 1.20, N))
        self.schedule = np.r_[rs.permutation(K), rs.permutation(K)]
        self.targets = rg.uniform(-0.52, 0.52, (BLOCKS, N))
        self.x = srng(seed, f"{family}-initial").normal(0, 0.10, N)
        self.proc = srng(seed, f"{family}-process").normal(0, 0.011, (T, N))
        self.cue_noise = srng(seed, f"{family}-cue").normal(0, CUE_SIGMA, (T+1, 3))
        rr = srng(seed, f"{family}-shock")
        mask = rr.random(T) < 0.012
        self.shocks = np.zeros((T, N))
        ix = np.flatnonzero(mask)
        if len(ix):
            self.shocks[ix, rr.integers(N, size=len(ix))] = (
                rr.choice([-1.0, 1.0], size=len(ix)) * rr.uniform(0.16, 0.26, len(ix))
            )

    def block(self):
        return min(BLOCKS-1, self.t // BLOCK_LEN)

    def regime(self):
        return int(self.schedule[self.block()])

    def target(self):
        return self.targets[self.block()].copy()

    def cue(self):
        return CUE_CENTERS[self.regime()] + self.cue_noise[self.t]

    def observation(self):
        return self.x.copy(), self.cue().copy()

    def deterministic_next(self, x, action_idx, regime=None):
        r = self.regime() if regime is None else int(regime)
        x = np.asarray(x, float)
        return (
            self.As[r] @ x
            + self.B @ ACTIONS[int(action_idx)]
            + self.bias[r]
            + 0.025 * np.tanh(self.Hs[r] @ x)
        )

    def true_two_step_cost(self, x, target, a1):
        r = self.regime()
        x1 = self.deterministic_next(x, a1, r)
        c1 = np.mean((x1-target)**2) + ACTION_PENALTY*np.mean(ACTIONS[a1]**2)
        best = 1e99
        for a2 in range(len(ACTIONS)):
            x2 = self.deterministic_next(x1, a2, r)
            c2 = np.mean((x2-target)**2) + ACTION_PENALTY*np.mean(ACTIONS[a2]**2)
            best = min(best, float(c2))
        return float(c1 + PLAN_DISCOUNT*best)

    def step(self, action_idx):
        target = self.target()
        y = self.deterministic_next(self.x, action_idx) + self.proc[self.t] + self.shocks[self.t]
        self.x = np.clip(y, -2.8, 2.8)
        cost = float(np.mean((self.x-target)**2) + ACTION_PENALTY*np.mean(ACTIONS[action_idx]**2))
        self.t += 1
        return cost

class OnlineModel:
    def __init__(self, action_vectors):
        self.action_vectors = np.asarray(action_vectors, float)
        self.p = N + N + 1
        self.xtx = RIDGE * np.eye(self.p)
        self.xty = np.zeros((self.p, N))
        self.n = 0
        self.W = np.zeros((self.p, N))

    def feature(self, x, action_idx):
        return np.r_[np.asarray(x, float), self.action_vectors[int(action_idx)], 1.0]

    def update(self, x, action_idx, y):
        z = self.feature(x, action_idx)
        self.xtx += np.outer(z, z)
        self.xty += np.outer(z, np.asarray(y, float))
        self.n += 1
        if self.n >= MIN_FIT:
            self.W = np.linalg.solve(self.xtx, self.xty)

    @property
    def fitted(self):
        return self.n >= MIN_FIT

    def predict(self, x, action_idx):
        if not self.fitted:
            # coordinate-equivariant initial predictor
            return 0.54*np.asarray(x, float) + 0.90*self.action_vectors[int(action_idx)]
        return self.feature(x, action_idx) @ self.W

    def matrices(self):
        if not self.fitted:
            return 0.54*np.eye(N), 0.90*np.eye(N), np.zeros(N)
        A = self.W[:N].T
        B = self.W[N:2*N].T
        b = self.W[-1].copy()
        return A, B, b

class AdaptiveController:
    def __init__(self, rotation=None):
        self.P = np.eye(N) if rotation is None else np.asarray(rotation, float)
        self.action_vectors = ACTIONS @ self.P.T
        self.models = [OnlineModel(self.action_vectors) for _ in range(K)]
        self.shadow = [OnlineModel(self.action_vectors) for _ in range(K)]
        self.visits = np.zeros(K, dtype=int)
        self.belief = np.ones(K) / K

    def encode(self, x):
        return np.asarray(x, float) @ self.P.T

    def encode_target(self, target):
        return np.asarray(target, float) @ self.P.T

    def cue_likelihood(self, cue):
        d = (CUE_CENTERS - np.asarray(cue, float)[None, :]) / CUE_SIGMA
        logp = -0.5*np.sum(d*d, axis=1)
        p = np.exp(logp - np.max(logp))
        return p / np.sum(p)

    def keys(self, cue):
        like = self.cue_likelihood(cue)
        prior = STAY*self.belief + (1-STAY)/K
        post = prior * like
        post /= np.sum(post)
        self.belief = post
        recurrent_key = int(np.argmax(post))
        current_key = int(np.argmax(like))
        return recurrent_key, current_key

    def model_cost(self, model, x, target, first_action):
        x1 = model.predict(x, first_action)
        c1 = np.mean((x1-target)**2) + ACTION_PENALTY*np.mean(self.action_vectors[first_action]**2)
        best = 1e99
        for a2 in range(len(self.action_vectors)):
            x2 = model.predict(x1, a2)
            c2 = np.mean((x2-target)**2) + ACTION_PENALTY*np.mean(self.action_vectors[a2]**2)
            best = min(best, float(c2))
        return float(c1 + PLAN_DISCOUNT*best)

    def act(self, x_phys, cue, target_phys):
        x = self.encode(x_phys)
        target = self.encode_target(target_phys)
        key, current_key = self.keys(cue)
        self.visits[key] += 1
        m = self.models[key]
        if self.visits[key] <= EXPLORE_VISITS or not m.fitted:
            action = int((self.visits[key]-1) % len(self.action_vectors))
        else:
            vals = [self.model_cost(m, x, target, a) for a in range(len(self.action_vectors))]
            action = int(np.argmin(vals))
        pred = m.predict(x, action)
        return action, key, current_key, x, pred

    def update(self, x_encoded, key, current_key, action, y_phys):
        y = self.encode(y_phys)
        self.models[key].update(x_encoded, action, y)
        self.shadow[current_key].update(x_encoded, action, y)

def orthogonal_matrix(seed, family):
    r = srng(seed, f"{family}-rotation")
    q, _ = np.linalg.qr(r.normal(size=(N, N)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q

def run_adaptive(seed, family, rotated=False):
    world = World(seed, family)
    P = orthogonal_matrix(seed, family) if rotated else None
    agent = AdaptiveController(P)
    costs = []
    actions = []
    stable = []
    d_delta, c_delta, r_delta = [], [], []
    for _ in range(T):
        x_phys, cue = world.observation()
        target = world.target()
        action, key, current_key, x, pred = agent.act(x_phys, cue, target)
        recurrent_model = agent.models[key]
        current_model = agent.shadow[current_key]
        cost = world.step(action)
        y_phys = world.x.copy()
        y = agent.encode(y_phys)

        if recurrent_model.fitted:
            intact_err = mse(pred, y)
            A, B, bias = recurrent_model.matrices()

            # D lesion: remove differentiated coordinates while retaining dimension count.
            xcollapsed = np.repeat(np.mean(x), N)
            pred_d = A @ xcollapsed + B @ agent.action_vectors[action] + bias
            d_delta.append(mse(pred_d, y) - intact_err)

            # R lesion: remove learned cross-coordinate relations.
            pred_r = np.diag(np.diag(A)) @ x + B @ agent.action_vectors[action] + bias
            r_delta.append(mse(pred_r, y) - intact_err)

            if current_model.fitted:
                pred_c = current_model.predict(x, action)
                c_delta.append(mse(pred_c, y) - intact_err)

        agent.update(x, key, current_key, action, y_phys)
        costs.append(cost)
        stable.append(float(np.linalg.norm(y_phys) < 4.0))
        actions.append(action)

    return {
        "mean_cost": float(np.mean(costs)),
        "stable_fraction": float(np.mean(stable)),
        "D_dz": standard_effect(d_delta),
        "C_dz": standard_effect(c_delta),
        "R_dz": standard_effect(r_delta),
        "D_samples": len(d_delta),
        "C_samples": len(c_delta),
        "R_samples": len(r_delta),
        "actions": actions,
    }

def run_random(seed, family):
    world = World(seed, family)
    rr = srng(seed, f"{family}-random-control")
    cs = []
    for _ in range(T):
        a = int(rr.integers(len(ACTIONS)))
        cs.append(world.step(a))
    return float(np.mean(cs))

def run_oracle(seed, family):
    world = World(seed, family)
    cs = []
    for _ in range(T):
        target = world.target()
        vals = [world.true_two_step_cost(world.x, target, a) for a in range(len(ACTIONS))]
        a = int(np.argmin(vals))
        cs.append(world.step(a))
    return float(np.mean(cs))

def evaluate_case(seed, family):
    native = run_adaptive(seed, family, False)
    rotated = run_adaptive(seed, family, True)
    random_cost = run_random(seed, family)
    oracle_cost = run_oracle(seed, family)
    denom = random_cost - oracle_cost

    for q in (native, rotated):
        q["normalized_score"] = float((random_cost-q["mean_cost"])/denom) if denom > 1e-9 else -999.0

    aa = np.asarray(native["actions"], int)
    bb = np.asarray(rotated["actions"], int)
    agreement = float(np.mean(aa == bb))
    score_gap = abs(native["normalized_score"] - rotated["normalized_score"])

    native_pass = bool(
        native["normalized_score"] >= 0.50
        and native["D_dz"] >= 0.20
        and native["C_dz"] >= 0.20
        and native["R_dz"] >= 0.20
        and native["stable_fraction"] >= 0.99
    )
    # Remove long action vectors from nested scientific summary.
    native.pop("actions")
    rotated.pop("actions")
    return {
        "seed": int(seed),
        "family": family,
        "random_cost": random_cost,
        "oracle_cost": oracle_cost,
        "native": native,
        "rotated": rotated,
        "action_agreement": agreement,
        "normalized_score_gap": float(score_gap),
        "native_pass": native_pass,
        "instrument_valid": bool(denom > 1e-9 and native["D_samples"] > 100 and native["C_samples"] > 100 and native["R_samples"] > 100),
    }

def summarize(records):
    out = {}
    for family in sorted(set(x["family"] for x in records)):
        rr = [x for x in records if x["family"] == family]
        out[family] = {
            "n": len(rr),
            "native_pass_count": int(sum(x["native_pass"] for x in rr)),
            "instrument_valid_count": int(sum(x["instrument_valid"] for x in rr)),
            "median_action_agreement": float(np.median([x["action_agreement"] for x in rr])),
            "median_score_gap": float(np.median([x["normalized_score_gap"] for x in rr])),
            "native_medians": {
                k: float(np.median([x["native"][k] for x in rr]))
                for k in ["normalized_score", "D_dz", "C_dz", "R_dz", "stable_fraction", "mean_cost"]
            },
            "rotated_medians": {
                k: float(np.median([x["rotated"][k] for x in rr]))
                for k in ["normalized_score", "stable_fraction", "mean_cost"]
            }
        }
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "confirm"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.mode == "dev":
        records = [evaluate_case(s, DEV_FAMILY) for s in DEV_SEEDS]
    else:
        records = [evaluate_case(s, f) for f in CONF_FAMILIES for s in CONF_SEEDS]

    summary = summarize(records)
    if a.mode == "dev":
        resolution = "R42_DEVELOPMENT_INSTRUMENT_PASS" if all(x["instrument_valid"] for x in records) else "R42_DEVELOPMENT_INSTRUMENT_FAIL"
    else:
        fam_ok = all(
            summary[f]["native_pass_count"] >= 12
            and summary[f]["median_action_agreement"] >= 0.98
            and summary[f]["median_score_gap"] <= 0.02
            for f in CONF_FAMILIES
        )
        resolution = "R42_INDEPENDENT_IMPLEMENTATION_TRANSPORT_PASS_INTERNAL" if fam_ok else "R42_INDEPENDENT_IMPLEMENTATION_TRANSPORT_FAIL"

    result = {
        "campaign": "R42 independent-generator / independent-implementation D+C+R transport",
        "mode": a.mode,
        "resolution": resolution,
        "development_family": DEV_FAMILY,
        "confirmatory_families": CONF_FAMILIES,
        "development_seeds": DEV_SEEDS,
        "confirmatory_seeds": CONF_SEEDS,
        "criteria": {
            "normalized_score_min": 0.50,
            "D_dz_min": 0.20,
            "C_dz_min": 0.20,
            "R_dz_min": 0.20,
            "stable_fraction_min": 0.99,
            "family_pass_required": 12,
            "family_denominator": 16,
            "median_action_agreement_min": 0.98,
            "median_score_gap_max": 0.02,
        },
        "summary": summary,
        "records": records,
        "boundaries": {
            "E6b": "OPEN_SAME_PROGRAM",
            "E7": "OPEN",
            "global_minimality": "OPEN",
            "implementation_privilege": "NOT_SUPPORTED",
            "consciousness": "NOT_ESTABLISHED",
            "core_version": "POLAR Core v1.1 D+C+R with conditional A unchanged",
        }
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))

if __name__ == "__main__":
    main()
