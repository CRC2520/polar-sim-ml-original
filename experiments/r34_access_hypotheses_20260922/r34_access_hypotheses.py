#!/usr/bin/env python3
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "r33_mechanism_emergence_20260921" / "r33_mechanism_emergence.py"
spec = importlib.util.spec_from_file_location("r33base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

Q_TRAIN_EPISODES = 160
Q_EPOCHS = 30
Q_BATCH = 16
Q_INP = 13
Q_ARCH_DEV_H2 = ["CURRENT_MLP", "GRU"]
Q_ARCH_CONFIRM_H2 = ["CURRENT_MLP", "GRU", "LSTM"]
Q_ARCH_DEV_H3 = ["CURRENT_MLP", "RNN", "GRU"]
Q_ARCH_CONFIRM_H3 = ["CURRENT_MLP", "RNN", "GRU", "WINDOW_MLP", "LSTM"]
H1_DEV_ARCHS = ["CURRENT_MLP", "GRU"]
H1_CONFIRM_ARCHS = ["CURRENT_MLP", "GRU", "LSTM"]

H1_DEV_FAMILIES = ["ring"]
H1_CONFIRM_FAMILIES = ["random_dag", "skew"]
Q_DEV_FAMILIES = ["ring"]
Q_CONFIRM_FAMILIES = ["random_dag", "skew"]

SUCCESS_RULE = {
    "stable_min": 0.98,
    "control_gain_min": 0.05,
    "prediction_gain_min": 0.15,
    "other_metric_floor": -0.10,
}
R33_THRESHOLDS = {
    "D": 0.010,
    "C": 0.0010,
    "R": 0.0005,
    "A": 0.0002,
}

def stable_hash_seed(kind):
    table = {
        "CURRENT_MLP": 101,
        "RNN": 211,
        "GRU": 307,
        "WINDOW_MLP": 401,
        "LSTM": 503,
    }
    return table[kind]

def success_r33(q):
    stable = q["stable"] >= SUCCESS_RULE["stable_min"]
    p1 = q["control_gain_vs_current"] >= SUCCESS_RULE["control_gain_min"] and q["prediction_gain_vs_current"] >= SUCCESS_RULE["other_metric_floor"]
    p2 = q["prediction_gain_vs_current"] >= SUCCESS_RULE["prediction_gain_min"] and q["control_gain_vs_current"] >= SUCCESS_RULE["other_metric_floor"]
    return bool(stable and (p1 or p2))

def flags_r33(q):
    return {
        "D": bool(q["D_rank1_damage"] >= R33_THRESHOLDS["D"]),
        "C": bool(q["C_history_damage"] >= R33_THRESHOLDS["C"]),
        "R": bool(q["R_transplant_damage"] >= R33_THRESHOLDS["R"]),
        "A": bool(q["A_history_specificity"] >= R33_THRESHOLDS["A"]),
    }

def h1_eval_seed(seed, mode):
    archs = H1_DEV_ARCHS if mode == "dev" else H1_CONFIRM_ARCHS
    fams = H1_DEV_FAMILIES if mode == "dev" else H1_CONFIRM_FAMILIES
    X, Y = base.make_training_set(seed)
    raw = {}
    for ai, kind in enumerate(archs):
        model = base.train_model(seed + ai * 100 + stable_hash_seed(kind), kind, X, Y)
        raw[kind] = base.evaluate_arch(seed + 909, kind, model, fams)
    cur = raw["CURRENT_MLP"]
    rows = {}
    for kind in archs:
        q = dict(raw[kind])
        denom = max(cur["adapt_cost"] - q["oracle_adapt_cost"], 1e-4)
        q["control_gain_vs_current"] = float((cur["adapt_cost"] - q["adapt_cost"]) / denom)
        q["prediction_gain_vs_current"] = float((cur["drift_mse_rel"] - q["drift_mse_rel"]) / max(cur["drift_mse_rel"], 1e-6))
        q["success"] = success_r33(q)
        q["mechanisms"] = flags_r33(q)
        q["DCR_pass"] = bool(q["mechanisms"]["D"] and q["mechanisms"]["C"] and q["mechanisms"]["R"])
        q["A_fail"] = bool(not q["mechanisms"]["A"])
        rows[kind] = q
    return {"seed": seed, "architectures": rows}

def q_local_target(tokens12):
    obs = np.asarray(tokens12[..., :base.N], np.float32)
    prev_u = np.asarray(tokens12[..., base.N:2 * base.N], np.float32)
    return (0.28 * np.tanh(obs) + 0.04 * prev_u).astype(np.float32)

def q_episode(seed, family, condition, task_variant):
    ep = base.make_episode(seed, family, condition)
    rng = np.random.default_rng(seed + 551)
    cue = rng.integers(0, 2, size=base.SEQ).astype(np.float32)
    # keep both cue values present after burn in every block
    for b in range(base.BLOCKS):
        start = b * base.BLOCK_LEN + base.ADAPT_BURN
        cue[start:start+6] = np.array([0, 1, 0, 1, 0, 1], np.float32)
    x13 = np.concatenate([ep["tokens"], cue[:, None]], axis=1).astype(np.float32)
    local = q_local_target(ep["tokens"])
    if task_variant == "selective":
        target = cue[:, None] * ep["drift"] + (1.0 - cue[:, None]) * local
    elif task_variant == "uniform":
        target = ep["drift"].copy()
    else:
        raise ValueError(task_variant)
    return {
        "tokens": x13,
        "target": target.astype(np.float32),
        "drift": ep["drift"],
        "local": local,
        "cue": cue,
        "family": family,
        "condition": condition,
    }

def q_make_model(kind, seed):
    base.set_seed(seed)
    if kind == "CURRENT_MLP":
        return base.CurrentMLP(inp=Q_INP)
    if kind == "WINDOW_MLP":
        return base.WindowMLP(inp=Q_INP)
    return base.RecurrentAgent(kind, inp=Q_INP)

def q_training_set(seed, task_variant):
    rng = np.random.default_rng(seed + 712)
    eps = []
    for i in range(Q_TRAIN_EPISODES):
        fam = base.TRAIN_FAMILIES[int(rng.integers(0, len(base.TRAIN_FAMILIES)))]
        cond = base.CONDITIONS[int(rng.integers(0, len(base.CONDITIONS)))]
        eps.append(q_episode(seed * 1000 + 100 + i, fam, cond, task_variant))
    X = torch.tensor(np.stack([e["tokens"] for e in eps]))
    Y = torch.tensor(np.stack([e["target"] for e in eps]))
    return X, Y

def q_train(seed, kind, X, Y):
    model = q_make_model(kind, seed + stable_hash_seed(kind))
    opt = torch.optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-5)
    lossfn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed + 77)
    mask = torch.tensor([(t % base.BLOCK_LEN) >= base.ADAPT_BURN for t in range(base.SEQ)], dtype=torch.bool)
    n = X.shape[0]
    model.train()
    for _ in range(Q_EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, Q_BATCH):
            idx = perm[s:s+Q_BATCH]
            pred, _ = model.forward_seq(X[idx])
            loss = lossfn(pred[:, mask, :], Y[idx][:, mask, :])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    return model.eval()

def q_state_from_prefix(model, kind, prefix):
    x = torch.tensor(prefix, dtype=torch.float32)[None, :, :]
    if kind == "CURRENT_MLP":
        return None
    if kind == "WINDOW_MLP":
        return np.asarray(prefix[-(base.WINDOW-1):], np.float32)
    with torch.no_grad():
        _, state = model.core(x)
    return state

def q_predict_from_state(model, kind, state, token):
    tok = torch.tensor(token, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        if kind == "CURRENT_MLP":
            p, _ = model.forward_seq(tok)
            return p[0, 0].numpy()
        if kind == "WINDOW_MLP":
            hist = np.asarray(state, np.float32)
            full = np.vstack([hist, np.asarray(token, np.float32)])
            if len(full) < base.WINDOW:
                pad = np.zeros((base.WINDOW-len(full), full.shape[1]), np.float32)
                full = np.vstack([pad, full])
            full = full[-base.WINDOW:]
            z = model.body(torch.tensor(full.reshape(1, 1, -1)))
            return model.head(z)[0, 0].numpy()
        z, _ = model.core(tok, state)
        return model.head(z)[0, 0].numpy()

def q_predict_reset(model, kind, token):
    tok = torch.tensor(token, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        if kind == "CURRENT_MLP":
            p, _ = model.forward_seq(tok)
            return p[0, 0].numpy()
        if kind == "WINDOW_MLP":
            z = model.body(model.windows(tok, reset=True))
            return model.head(z)[0, 0].numpy()
        z, _ = model.core(tok)
        return model.head(z)[0, 0].numpy()

def q_probe_episode(model, kind, ep, task_variant):
    points = []
    for b in range(base.BLOCKS):
        s = b * base.BLOCK_LEN + base.ADAPT_BURN + 2
        points.extend([s, min(s + 6, (b+1)*base.BLOCK_LEN-2), min(s + 13, (b+1)*base.BLOCK_LEN-2)])
    hd0 = []
    hd1 = []
    err0 = []
    err1 = []
    for t in points:
        prefix = ep["tokens"][:t]
        state = q_state_from_prefix(model, kind, prefix)
        base_token = ep["tokens"][t].copy()
        tok0 = base_token.copy()
        tok1 = base_token.copy()
        tok0[-1] = 0.0
        tok1[-1] = 1.0
        y1 = ep["drift"][t]
        y0 = ep["local"][t] if task_variant == "selective" else ep["drift"][t]
        p0 = q_predict_from_state(model, kind, state, tok0)
        p1 = q_predict_from_state(model, kind, state, tok1)
        r0 = q_predict_reset(model, kind, tok0)
        r1 = q_predict_reset(model, kind, tok1)
        e0 = base.mse(p0, y0)
        e1 = base.mse(p1, y1)
        hd0.append(base.mse(r0, y0) - e0)
        hd1.append(base.mse(r1, y1) - e1)
        err0.append(e0)
        err1.append(e1)
    return {
        "q0_mse": float(np.mean(err0)),
        "q1_mse": float(np.mean(err1)),
        "history_damage_q0": float(np.mean(hd0)),
        "history_damage_q1": float(np.mean(hd1)),
        "A_switch": float(np.mean(hd1) - np.mean(hd0)),
    }

def q_eval_arch(seed, kind, model, families, task_variant):
    vals = []
    for fi, fam in enumerate(families):
        ep = q_episode(seed * 10000 + fi * 1000 + 333, fam, "relational_shift", task_variant)
        vals.append(q_probe_episode(model, kind, ep, task_variant))
    keys = vals[0].keys()
    return {k: float(np.mean([v[k] for v in vals])) for k in keys}

def q_eval_seed(seed, mode, task_variant, archs):
    fams = Q_DEV_FAMILIES if mode == "dev" else Q_CONFIRM_FAMILIES
    X, Y = q_training_set(seed, task_variant)
    raw = {}
    for ai, kind in enumerate(archs):
        model = q_train(seed + ai * 100, kind, X, Y)
        raw[kind] = q_eval_arch(seed + 707, kind, model, fams, task_variant)
    cur = raw["CURRENT_MLP"]
    rows = {}
    for kind in archs:
        q = dict(raw[kind])
        q["q1_gain_vs_current"] = float((cur["q1_mse"] - q["q1_mse"]) / max(cur["q1_mse"], 1e-6))
        q["q0_ratio_vs_current"] = float(q["q0_mse"] / max(cur["q0_mse"], 1e-8))
        rows[kind] = q
    return {"seed": seed, "architectures": rows}

def h2_eval_seed(seed, mode):
    archs = Q_ARCH_DEV_H2 if mode == "dev" else Q_ARCH_CONFIRM_H2
    return q_eval_seed(seed, mode, "selective", archs)

def h3_eval_seed(seed, mode):
    archs = Q_ARCH_DEV_H3 if mode == "dev" else Q_ARCH_CONFIRM_H3
    sel = q_eval_seed(seed, mode, "selective", archs)
    uni = q_eval_seed(seed + 50000, mode, "uniform", archs)
    rows = {}
    for kind in archs:
        a = sel["architectures"][kind]
        b = uni["architectures"][kind]
        rows[kind] = {
            "selective": a,
            "uniform": b,
            "A_interaction": float(a["A_switch"] - b["A_switch"]),
            "selective_q1_gain": a["q1_gain_vs_current"],
            "uniform_q1_gain": b["q1_gain_vs_current"],
            "selective_q0_ratio": a["q0_ratio_vs_current"],
            "uniform_q0_ratio": b["q0_ratio_vs_current"],
        }
    return {"seed": seed, "architectures": rows}

def summarize_h1(records):
    archs = sorted(records[0]["architectures"].keys())
    keys = [
        "D_rank1_damage", "C_history_damage", "R_transplant_damage",
        "A_history_specificity", "control_gain_vs_current",
        "prediction_gain_vs_current", "stable",
    ]
    return {a: {k: float(np.median([r["architectures"][a][k] for r in records])) for k in keys} for a in archs}

def summarize_q(records):
    archs = sorted(records[0]["architectures"].keys())
    keys = ["q0_mse", "q1_mse", "history_damage_q0", "history_damage_q1", "A_switch", "q1_gain_vs_current", "q0_ratio_vs_current"]
    return {a: {k: float(np.median([r["architectures"][a][k] for r in records])) for k in keys} for a in archs}

def summarize_h3(records):
    archs = sorted(records[0]["architectures"].keys())
    out = {}
    for a in archs:
        out[a] = {}
        for part in ["selective", "uniform"]:
            keys = ["q0_mse", "q1_mse", "history_damage_q0", "history_damage_q1", "A_switch", "q1_gain_vs_current", "q0_ratio_vs_current"]
            out[a][part] = {k: float(np.median([r["architectures"][a][part][k] for r in records])) for k in keys}
        for k in ["A_interaction", "selective_q1_gain", "uniform_q1_gain", "selective_q0_ratio", "uniform_q0_ratio"]:
            out[a][k] = float(np.median([r["architectures"][a][k] for r in records]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", choices=["H1", "H2", "H3"], required=True)
    ap.add_argument("--mode", choices=["dev", "confirm"], required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi+1))
    if a.hypothesis == "H1":
        rec = [h1_eval_seed(s, a.mode) for s in seeds]
        summary = summarize_h1(rec)
        archs = H1_DEV_ARCHS if a.mode == "dev" else H1_CONFIRM_ARCHS
    elif a.hypothesis == "H2":
        rec = [h2_eval_seed(s, a.mode) for s in seeds]
        summary = summarize_q(rec)
        archs = Q_ARCH_DEV_H2 if a.mode == "dev" else Q_ARCH_CONFIRM_H2
    else:
        rec = [h3_eval_seed(s, a.mode) for s in seeds]
        summary = summarize_h3(rec)
        archs = Q_ARCH_DEV_H3 if a.mode == "dev" else Q_ARCH_CONFIRM_H3
    out = {
        "campaign": "R34 contextual-access three-hypothesis discrimination",
        "hypothesis": a.hypothesis,
        "mode": a.mode,
        "seeds": seeds,
        "architectures": archs,
        "records": rec,
        "medians": summary,
        "boundaries": {
            "POLAR_Core_v1": "UNCHANGED_D_PLUS_C_PLUS_R_PLUS_A",
            "R33": "UNCHANGED_INCONCLUSIVE_CAPABILITY",
            "E6b": "OPEN_EXTERNAL_ONLY",
            "E7": "OPEN_PROSPECTIVE_BIOLOGICAL",
            "consciousness": "NOT_ESTABLISHED"
        }
    }
    Path(a.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["medians"], indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
