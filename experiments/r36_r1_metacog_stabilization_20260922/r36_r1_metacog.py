#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

N = 3
ACTIONS = np.array([
    [0.0, 0.0, 0.0],
    [0.22, 0.0, 0.0], [-0.22, 0.0, 0.0],
    [0.0, 0.22, 0.0], [0.0, -0.22, 0.0],
    [0.0, 0.0, 0.22], [0.0, 0.0, -0.22],
], float)

BLOCK_LEN = 220
SCHEDULE = [0, 1, 2, 3, 0, 2, 1, 3]
HORIZON = BLOCK_LEN * len(SCHEDULE)
MODEL_WINDOW = 140
MIN_FIT = 20
CUE_EMA = 0.78
ACTION_PENALTY = 0.025
PLAN_DISCOUNT = 0.72
META_NEIGHBORS = 18
META_PLAN_CONF_MIN = 0.30
META_PLAN_ADV_MIN = 1e-6

REGIME_A = np.array([
    [[0.58, 0.18, -0.05], [-0.08, 0.54, 0.16], [0.09, -0.10, 0.50]],
    [[0.52, -0.16, 0.12], [0.14, 0.57, -0.07], [-0.10, 0.13, 0.53]],
    [[0.55, 0.08, 0.17], [-0.14, 0.51, 0.10], [0.11, 0.06, 0.56]],
    [[0.50, -0.11, -0.15], [0.16, 0.55, 0.08], [0.07, -0.14, 0.54]],
], float)
REGIME_TARGET = np.array([
    [0.55, -0.25, 0.20],
    [-0.40, 0.45, -0.20],
    [0.25, 0.35, 0.50],
    [-0.45, -0.30, 0.35],
], float)
CUE_CENTERS = np.array([
    [-0.75, -0.75],
    [-0.75, 0.75],
    [0.75, -0.75],
    [0.75, 0.75],
], float)

def stable_seed(seed, label):
    raw = hashlib.sha256(f"R36R1|{int(seed)}|{label}".encode()).digest()
    return int.from_bytes(raw[:8], "little") & 0xffffffff

def mse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean((a-b)**2))

def ridge(X, Y, lam=0.08):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    return np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ Y)

def cue_key(c):
    c = np.asarray(c, float)
    return (int(c[0] > 0), int(c[1] > 0))

class PersistentEnv:
    def __init__(self, seed, donor=False):
        self.seed = int(seed)
        self.rng = np.random.default_rng(stable_seed(seed, "env-donor" if donor else "env-main"))
        base_gain = np.clip(self.rng.normal(1.0, 0.12, N), 0.75, 1.25)
        self.gain = np.clip(1.95-base_gain, 0.72, 1.28) if donor else base_gain
        self.B = np.diag(self.gain)
        self.x = self.rng.normal(0, 0.12, N)
        self.t = 0
        self.proc_noise = self.rng.normal(0, 0.012, (HORIZON, N))
        self.cue_noise = self.rng.normal(0, 0.72, (HORIZON+1, 2))
        self.shocks = np.zeros((HORIZON, N))
        for t in range(73, HORIZON, 53):
            k = int(self.rng.integers(0, N))
            self.shocks[t, k] = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.28, 0.40)

    def block(self):
        return min(len(SCHEDULE)-1, self.t // BLOCK_LEN)

    def regime(self):
        return SCHEDULE[self.block()]

    def cue(self):
        r = self.regime()
        return CUE_CENTERS[r] + self.cue_noise[min(self.t, HORIZON)]

    def target(self):
        return REGIME_TARGET[self.regime()].copy()

    def obs(self):
        return np.r_[self.x, self.cue()]

    def true_two_step_cost(self, x, a1_idx, a2_idx, target=None):
        r = self.regime()
        A = REGIME_A[r]
        B = self.B
        target = self.target() if target is None else np.asarray(target, float)
        u1 = ACTIONS[a1_idx]; u2 = ACTIONS[a2_idx]
        x1 = A@x + B@u1
        x2 = A@x1 + B@u2
        c1 = np.mean((x1-target)**2) + ACTION_PENALTY*np.mean(u1*u1)
        c2 = np.mean((x2-target)**2) + ACTION_PENALTY*np.mean(u2*u2)
        return float(c1 + PLAN_DISCOUNT*c2)

    def step(self, action_idx):
        r = self.regime()
        A = REGIME_A[r]
        target = REGIME_TARGET[r]
        u = ACTIONS[int(action_idx)]
        shock = self.shocks[self.t]
        source_external = bool(np.linalg.norm(shock) > 0)
        xnext = A@self.x + self.B@u + self.proc_noise[self.t] + shock
        xnext = np.clip(xnext, -2.5, 2.5)
        cost = float(np.mean((xnext-target)**2) + ACTION_PENALTY*np.mean(u*u))
        info = {
            "regime": int(r),
            "block": int(self.block()),
            "target": target.copy(),
            "source_external": source_external,
            "true_A": A.copy(),
            "true_B": self.B.copy(),
            "shock": shock.copy(),
        }
        self.x = xnext
        self.t += 1
        done = self.t >= HORIZON
        return (self.obs() if not done else np.zeros(N+2)), -cost, done, info

class ModelState:
    def __init__(self):
        self.X = []
        self.Y = []
        self.A = np.eye(N)*0.54
        self.B = np.eye(N)*0.90
        self.resid_ema = 0.0030
        self.fitted = False
        self.meta_features = []
        self.meta_errors = []

    def add(self, x, u, y, include=True):
        if include:
            self.X.append(np.r_[x, u].astype(float))
            self.Y.append(np.asarray(y, float).copy())
            self.X = self.X[-MODEL_WINDOW:]
            self.Y = self.Y[-MODEL_WINDOW:]
        if len(self.X) >= MIN_FIT:
            W = ridge(np.asarray(self.X), np.asarray(self.Y), 0.08)
            self.A = W[:N].T
            self.B = W[N:].T
            self.fitted = True

    def predict(self, x, action_idx):
        return self.A@np.asarray(x) + self.B@ACTIONS[int(action_idx)]

    def meta_feature(self, x, action_idx, pred=None):
        x = np.asarray(x, float)
        u = ACTIONS[int(action_idx)]
        pred = self.predict(x, action_idx) if pred is None else np.asarray(pred, float)
        return np.r_[x, 2.0*u, pred]

    def record_error(self, x, action_idx, pred, err):
        self.meta_features.append(self.meta_feature(x, action_idx, pred))
        self.meta_errors.append(float(err))
        self.meta_features = self.meta_features[-MODEL_WINDOW:]
        self.meta_errors = self.meta_errors[-MODEL_WINDOW:]

    def expected_error(self, x, action_idx):
        if len(self.meta_features) < 18:
            return float(self.resid_ema)
        z = self.meta_feature(x, action_idx)
        X = np.asarray(self.meta_features)
        scale = np.std(X, axis=0) + 0.05
        d = np.sum(((X-z[None,:])/scale[None,:])**2, axis=1)
        k = min(META_NEIGHBORS, len(d))
        idx = np.argpartition(d, k-1)[:k]
        w = 1.0/(0.15 + d[idx])
        return float(np.sum(w*np.asarray(self.meta_errors)[idx]) / np.sum(w))

    def confidence(self, x, action_idx):
        expected = self.expected_error(x, action_idx)
        if len(self.meta_errors) < 18:
            return float(1.0/(1.0 + 260.0*max(expected, 0.0)))
        errs = np.asarray(self.meta_errors, float)
        # Online empirical reliability percentile. High confidence means the
        # current action/state is predicted to be in the low-error tail.
        rank = float(np.mean(errs >= expected))
        return float(np.clip(0.05 + 0.90*rank, 0.05, 0.95))

class PersistentAgent:
    def __init__(self):
        self.models = {}
        self.cue_state = np.zeros(2)
        self.clock = 0
        self.memory = []
        self.key_visits = {}
        self.last_pred = None
        self.last_key = None
        self.last_conf = 0.0

    def update_cue(self, cue):
        self.cue_state = CUE_EMA*self.cue_state + (1-CUE_EMA)*np.asarray(cue)
        return cue_key(self.cue_state)

    def model(self, key):
        if key not in self.models:
            self.models[key] = ModelState()
        return self.models[key]

    def confidence(self, key, x, action_idx):
        return self.model(key).confidence(x, action_idx)

    def model_two_step_cost(self, x, target, key, first_action):
        m = self.model(key)
        x1 = m.A@x + m.B@ACTIONS[int(first_action)]
        c1 = np.mean((x1-target)**2) + ACTION_PENALTY*np.mean(ACTIONS[int(first_action)]**2)
        best2 = 1e99
        for a2 in range(len(ACTIONS)):
            x2 = m.A@x1 + m.B@ACTIONS[a2]
            c2 = np.mean((x2-target)**2) + ACTION_PENALTY*np.mean(ACTIONS[a2]**2)
            best2 = min(best2, float(c2))
        return float(c1 + PLAN_DISCOUNT*best2)

    def myopic_action(self, x, target, key):
        m = self.model(key)
        vals = []
        for a in range(len(ACTIONS)):
            xp = m.predict(x, a)
            vals.append(np.mean((xp-target)**2)+ACTION_PENALTY*np.mean(ACTIONS[a]**2))
        return int(np.argmin(vals))

    def plan_action(self, x, target, key, A_override=None, B_override=None):
        m = self.model(key)
        A = m.A if A_override is None else np.asarray(A_override)
        B = m.B if B_override is None else np.asarray(B_override)
        best = (1e99, 0)
        for a1 in range(len(ACTIONS)):
            x1 = A@x + B@ACTIONS[a1]
            c1 = np.mean((x1-target)**2)+ACTION_PENALTY*np.mean(ACTIONS[a1]**2)
            for a2 in range(len(ACTIONS)):
                x2 = A@x1 + B@ACTIONS[a2]
                c2 = np.mean((x2-target)**2)+ACTION_PENALTY*np.mean(ACTIONS[a2]**2)
                val = c1 + PLAN_DISCOUNT*c2
                if val < best[0]:
                    best = (float(val), int(a1))
        return best[1]

    def act(self, obs, target):
        x = np.asarray(obs[:N], float)
        cue = obs[N:N+2]
        key = self.update_cue(cue)
        self.key_visits[key] = self.key_visits.get(key, 0)+1
        visit = self.key_visits[key]
        m = self.model(key)
        if visit <= 42 or not m.fitted:
            action = (visit-1) % len(ACTIONS)
            conf = self.confidence(key, x, action)
            use_plan = False
            predicted_plan_advantage = 0.0
        else:
            a_plan = self.plan_action(x, target, key)
            a_myop = self.myopic_action(x, target, key)
            conf_plan = self.confidence(key, x, a_plan)
            plan_cost = self.model_two_step_cost(x, target, key, a_plan)
            myop_cost = self.model_two_step_cost(x, target, key, a_myop)
            predicted_plan_advantage = float(myop_cost-plan_cost)
            use_plan = bool(conf_plan >= META_PLAN_CONF_MIN and predicted_plan_advantage >= META_PLAN_ADV_MIN)
            action = a_plan if use_plan else a_myop
            conf = self.confidence(key, x, action)
        pred = m.predict(x, action)
        self.last_pred = pred
        self.last_key = key
        self.last_conf = conf
        return action, key, conf, pred, use_plan, predicted_plan_advantage

    def update(self, x, cue, action, y, true_source, key, conf, block, regime):
        m = self.model(key)
        residual = np.asarray(y)-m.predict(x, action)
        source_pred = bool(np.linalg.norm(residual) > 0.17) if m.fitted else False
        # preserve external-source events in episodic memory but exclude detected large shocks from dynamics fitting
        include = not source_pred
        factual_err = float(np.mean(residual*residual))
        m.resid_ema = 0.93*m.resid_ema + 0.07*factual_err
        pred_before = m.predict(x, action)
        m.record_error(x, action, pred_before, factual_err)
        m.add(x, ACTIONS[action], y, include=include)
        self.memory.append({
            "t": int(self.clock),
            "time_code": float(self.clock),
            "key": key,
            "block": int(block),
            "regime": int(regime),
            "x": np.asarray(x).copy(),
            "action": int(action),
            "y": np.asarray(y).copy(),
            "source_pred": bool(source_pred),
            "source_true": bool(true_source),
            "confidence": float(conf),
            "pred_error": float(np.mean(residual*residual)),
        })
        self.clock += 1

def acquire_donor(seed):
    env = PersistentEnv(seed, donor=True)
    agent = PersistentAgent()
    obs = env.obs()
    for t in range(HORIZON):
        x = obs[:N].copy(); cue = obs[N:N+2].copy()
        target = env.target()
        # Force broad excitation so donor B/model is well identified.
        key = agent.update_cue(cue)
        a = t % len(ACTIONS)
        pred = agent.model(key).predict(x, a)
        nxt, rew, done, info = env.step(a)
        pred = agent.model(key).predict(x, a)
        agent.update(x, cue, a, nxt[:N], info["source_external"], key, agent.confidence(key, x, a), info["block"], info["regime"])
        obs = nxt
        if done: break
    return agent

def cold_predict(buffer, x, action):
    if len(buffer) < MIN_FIT:
        A = np.eye(N)*0.54; B = np.eye(N)*0.90
    else:
        X = np.asarray([z[0] for z in buffer[-MODEL_WINDOW:]])
        Y = np.asarray([z[1] for z in buffer[-MODEL_WINDOW:]])
        W = ridge(X, Y, 0.08)
        A = W[:N].T; B = W[N:].T
    return A@x + B@ACTIONS[action]

def balanced_source_accuracy(memory, min_age=80):
    rows = memory[:-min_age] if len(memory) > min_age else []
    if not rows: return 0.0
    pos = [r for r in rows if r["source_true"]]
    neg = [r for r in rows if not r["source_true"]]
    tpr = np.mean([r["source_pred"] for r in pos]) if pos else 0.0
    tnr = np.mean([not r["source_pred"] for r in neg]) if neg else 0.0
    return float(0.5*(tpr+tnr))

def time_order_accuracy(memory, seed):
    rng = np.random.default_rng(stable_seed(seed, "time-query"))
    if len(memory) < 200: return 0.0
    correct = []
    for _ in range(120):
        i = int(rng.integers(0, len(memory)-100))
        j = int(rng.integers(i+60, min(len(memory), i+260)))
        a = memory[i]; b = memory[j]
        correct.append((a["time_code"] < b["time_code"]) == (a["t"] < b["t"]))
    return float(np.mean(correct))

def run_seed(seed):
    donor = acquire_donor(seed + 90000)
    env = PersistentEnv(seed, donor=False)
    agent = PersistentAgent()
    obs = env.obs()

    d_damage=[]; r_damage=[]; c_gain=[]; own_damage=[]
    plan_gain=[]; meta_gain=[]
    reentry_p=[]; reentry_c=[]
    block_cold = []
    confs=[]; errors=[]
    stable=[]
    block_first = {4,5,6,7}

    for t in range(HORIZON):
        x = obs[:N].copy()
        cue = obs[N:N+2].copy()
        target = env.target()
        block = env.block()
        regime = env.regime()

        # recurrent key used by actual agent
        action, key, conf, pred, use_plan, predicted_plan_advantage = agent.act(obs, target)
        m = agent.model(key)
        fitted_before = bool(m.fitted)
        visits_before = int(agent.key_visits[key])

        # current-only context comparator for C
        current_key = cue_key(cue)
        cur_model = agent.model(current_key)
        cur_pred = cur_model.predict(x, action)

        # D and R lesions on same checkpoint
        A_diag = np.diag(np.diag(m.A))
        A_collapse = np.repeat(m.A.mean(axis=1, keepdims=True), N, axis=1)
        pred_R = A_diag@x + m.B@ACTIONS[action]
        pred_D = A_collapse@x + m.B@ACTIONS[action]

        # donor history/model transplant comparator
        donor_m = donor.models.get(key)
        pred_donor = donor_m.predict(x, action) if donor_m is not None else np.eye(N)[0]*0.0 + x*0.54 + ACTIONS[action]*0.90

        # planning and metacognitive counterfactuals
        if m.fitted and agent.key_visits[key] > 55:
            a_plan = agent.plan_action(x, target, key)
            a_myop = agent.myopic_action(x, target, key)
            # optimize the second step for each candidate first action under true dynamics
            def best_true_cost(first):
                return min(env.true_two_step_cost(x, first, a2, target) for a2 in range(len(ACTIONS)))
            plan_gain.append(best_true_cost(a_myop)-best_true_cost(a_plan))
            actual_gate_action = a_plan if use_plan else a_myop
            reverse_gate_action = a_myop if use_plan else a_plan
            meta_gain.append(best_true_cost(reverse_gate_action)-best_true_cost(actual_gate_action))

        # block-local cold comparator for recurrent-regime reentry
        if t % BLOCK_LEN == 0:
            block_cold = []

        nxt, reward, done, info = env.step(action)
        y = nxt[:N].copy()

        # infer source and update same persistent agent
        agent.update(x, cue, action, y, info["source_external"], key, conf, info["block"], info["regime"])
        err_full = mse(pred, y)
        # Confidence is adjudicated only in the learned operating regime.
        # Early exploration/default predictions are not epistemic confidence reports.
        if fitted_before and visits_before > 55 and not info["source_external"]:
            errors.append(err_full); confs.append(conf)
        stable.append(float(np.linalg.norm(y) < 3.0))

        if not info["source_external"] and m.fitted:
            d_damage.append(mse(pred_D,y)-err_full)
            r_damage.append(mse(pred_R,y)-err_full)
            c_gain.append(mse(cur_pred,y)-err_full)
            own_damage.append(mse(pred_donor,y)-err_full)

        if block in block_first and (t % BLOCK_LEN) < 34 and not info["source_external"]:
            coldp = cold_predict(block_cold, x, action)
            reentry_p.append(err_full)
            reentry_c.append(mse(coldp, y))
        if not info["source_external"]:
            block_cold.append((np.r_[x,ACTIONS[action]], y.copy()))

        obs = nxt
        if done: break

    confs=np.asarray(confs); errors=np.asarray(errors)
    if len(confs):
        qlo=np.quantile(confs,.25); qhi=np.quantile(confs,.75)
        low_err=float(np.mean(errors[confs<=qlo])) if np.any(confs<=qlo) else 0.0
        high_err=float(np.mean(errors[confs>=qhi])) if np.any(confs>=qhi) else 0.0
        meta_calibration=low_err-high_err
    else:
        low_err=high_err=meta_calibration=0.0

    return {
        "seed": int(seed),
        "same_agent_steps": int(len(agent.memory)),
        "D_prediction_damage": float(np.mean(d_damage)) if d_damage else 0.0,
        "C_recurrent_context_gain": float(np.mean(c_gain)) if c_gain else 0.0,
        "R_relation_lesion_damage": float(np.mean(r_damage)) if r_damage else 0.0,
        "memory_reentry_gain": float(np.mean(reentry_c)-np.mean(reentry_p)) if reentry_p else 0.0,
        "own_history_transplant_damage": float(np.mean(own_damage)) if own_damage else 0.0,
        "source_balanced_accuracy": balanced_source_accuracy(agent.memory, 80),
        "time_order_accuracy": time_order_accuracy(agent.memory, seed),
        "metacog_calibration_gap": float(meta_calibration),
        "metacog_low_conf_error": float(low_err),
        "metacog_high_conf_error": float(high_err),
        "planning_gain": float(np.mean(plan_gain)) if plan_gain else 0.0,
        "metacog_gate_gain": float(np.mean(meta_gain)) if meta_gain else 0.0,
        "stable_fraction": float(np.mean(stable)),
        "memory_entries": int(len(agent.memory)),
        "models_learned": int(sum(1 for m in agent.models.values() if m.fitted)),
    }

def summarize(records):
    keys=[k for k in records[0] if k not in ("seed",)]
    out={}
    for k in keys:
        vals=[r[k] for r in records]
        if isinstance(vals[0], (int,float,np.integer,np.floating)):
            out[k]=float(np.median(vals))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--mode",choices=["dev","confirm"],required=True)
    a=ap.parse_args()
    lo,hi=map(int,a.seeds.split(":"))
    seeds=list(range(lo,hi+1))
    records=[run_seed(s) for s in seeds]
    out={
        "campaign":"R36-R1 metacognitive stabilization in unified persistent agent",
        "mode":a.mode,
        "seeds":seeds,
        "same_agent_requirement":"single PersistentAgent instance per seed across one 1760-step lifetime",
        "mandatory_core":"D+C+R (POLAR Core v1.1)",
        "coexisting_capacities":["persistent memory","own-history specificity","source attribution","time attribution","metacognitive confidence","two-step counterfactual planning"],
        "records":records,
        "medians":summarize(records),
        "boundaries":{
            "A":"optional/conditional extension, not required for R36 pass",
            "identity":"operational continuity/transplant sensitivity, not phenomenal self",
            "metacognition":"error-confidence calibration and functional planning gate, not introspection",
            "planning":"two-step model-based counterfactual planning in synthetic dynamics",
            "consciousness":"NOT_ESTABLISHED",
            "E6b":"OPEN_EXTERNAL_ONLY",
            "E7":"OPEN_PROSPECTIVE_BIOLOGICAL"
        }
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["medians"],indent=2,sort_keys=True))

if __name__=="__main__":
    main()
