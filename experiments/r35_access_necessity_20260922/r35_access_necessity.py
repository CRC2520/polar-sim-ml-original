#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R34_PATH = ROOT / "r34_access_hypotheses_20260922" / "r34_access_hypotheses.py"
spec = importlib.util.spec_from_file_location("r34base", R34_PATH)
r34 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r34)

DEV_ARCHS = ["CURRENT_MLP", "RNN", "GRU"]
CONFIRM_ARCHS = ["CURRENT_MLP", "RNN", "GRU", "WINDOW_MLP", "LSTM"]
STRICT_HELDOUT_ARCHS = ["WINDOW_MLP", "LSTM"]
DEV_FAMILIES = ["ring"]
CONFIRM_FAMILIES = ["random_dag", "skew"]
TASKS = ["selective", "uniform"]
MODES = ["contextual", "always", "random", "blocked"]

def access_predictions(model, kind, ep, task_variant, seed):
    rng = np.random.default_rng(seed + 8123)
    losses = {m: [] for m in MODES}
    intact_losses = []
    by_query = {m: {0: [], 1: []} for m in MODES}
    hist_norm = []
    access_rate = []

    for t in range(r34.base.SEQ):
        if (t % r34.base.BLOCK_LEN) < r34.base.ADAPT_BURN:
            continue

        prefix = ep["tokens"][:t]
        state = r34.q_state_from_prefix(model, kind, prefix)
        tok = ep["tokens"][t].copy()
        q = int(ep["cue"][t])

        tok0 = tok.copy(); tok0[-1] = 0.0
        tok1 = tok.copy(); tok1[-1] = 1.0

        p_reset0 = r34.q_predict_reset(model, kind, tok0)
        p_reset1 = r34.q_predict_reset(model, kind, tok1)
        p_hist0 = r34.q_predict_from_state(model, kind, state, tok0)
        p_hist1 = r34.q_predict_from_state(model, kind, state, tok1)

        # Representation-independent historical contribution:
        # derive the causal contribution of intact history under the history-demanding q=1 query.
        h = p_hist1 - p_reset1
        hist_norm.append(float(np.mean(h * h)))

        base_pred = p_reset1 if q == 1 else p_reset0
        target = ep["drift"][t] if (task_variant == "uniform" or q == 1) else ep["local"][t]

        random_gate = int(rng.integers(0, 2))
        gates = {
            "contextual": q,
            "always": 1,
            "random": random_gate,
            "blocked": 0,
        }
        access_rate.append(random_gate)

        for mode, gate in gates.items():
            pred = base_pred + float(gate) * h
            e = r34.base.mse(pred, target)
            losses[mode].append(e)
            by_query[mode][q].append(e)

        intact = p_hist1 if q == 1 else p_hist0
        intact_losses.append(r34.base.mse(intact, target))

    out = {
        "losses": {m: float(np.mean(losses[m])) for m in MODES},
        "q0_losses": {m: float(np.mean(by_query[m][0])) for m in MODES},
        "q1_losses": {m: float(np.mean(by_query[m][1])) for m in MODES},
        "intact_loss": float(np.mean(intact_losses)),
        "history_contribution_energy": float(np.mean(hist_norm)),
        "random_access_rate": float(np.mean(access_rate)),
    }
    L = out["losses"]
    out.update({
        "contextual_vs_always": float(L["always"] - L["contextual"]),
        "contextual_vs_random": float(L["random"] - L["contextual"]),
        "contextual_vs_blocked": float(L["blocked"] - L["contextual"]),
        "contextual_ratio_to_always": float(L["contextual"] / max(L["always"], 1e-12)),
        "contextual_ratio_to_blocked": float(L["contextual"] / max(L["blocked"], 1e-12)),
    })
    return out

def eval_task(seed, mode, task_variant, archs):
    fams = DEV_FAMILIES if mode == "dev" else CONFIRM_FAMILIES
    X, Y = r34.q_training_set(seed, task_variant)
    rows = {}
    for ai, kind in enumerate(archs):
        model = r34.q_train(seed + ai * 100, kind, X, Y)
        per_family = {}
        for fi, fam in enumerate(fams):
            ep = r34.q_episode(seed * 10000 + fi * 1000 + 711, fam, "relational_shift", task_variant)
            per_family[fam] = access_predictions(
                model, kind, ep, task_variant,
                seed * 10000 + fi * 1000 + ai * 100 + 911
            )
        keys = [
            "intact_loss", "history_contribution_energy", "random_access_rate",
            "contextual_vs_always", "contextual_vs_random", "contextual_vs_blocked",
            "contextual_ratio_to_always", "contextual_ratio_to_blocked",
        ]
        agg = {k: float(np.mean([per_family[f][k] for f in fams])) for k in keys}
        for access_mode in MODES:
            agg[f"loss_{access_mode}"] = float(np.mean([per_family[f]["losses"][access_mode] for f in fams]))
            agg[f"q0_loss_{access_mode}"] = float(np.mean([per_family[f]["q0_losses"][access_mode] for f in fams]))
            agg[f"q1_loss_{access_mode}"] = float(np.mean([per_family[f]["q1_losses"][access_mode] for f in fams]))
        agg["per_family"] = per_family
        rows[kind] = agg

    cur = rows["CURRENT_MLP"]["loss_contextual"]
    for kind in archs:
        rows[kind]["contextual_gain_vs_current"] = float(
            (cur - rows[kind]["loss_contextual"]) / max(cur, 1e-12)
        )
    return rows

def eval_seed(seed, mode):
    archs = DEV_ARCHS if mode == "dev" else CONFIRM_ARCHS
    selective = eval_task(seed, mode, "selective", archs)
    uniform = eval_task(seed + 50000, mode, "uniform", archs)
    rows = {}
    for kind in archs:
        s = selective[kind]
        u = uniform[kind]
        rows[kind] = {
            "selective": s,
            "uniform": u,
            "access_specificity_interaction": float(
                s["contextual_vs_always"] + (u["loss_contextual"] - u["loss_always"])
            ),
            "selective_contextual_advantage_min": float(min(
                s["contextual_vs_always"],
                s["contextual_vs_random"],
                s["contextual_vs_blocked"],
            )),
            "uniform_always_advantage": float(u["loss_contextual"] - u["loss_always"]),
        }
    return {"seed": seed, "architectures": rows}

def summarize(records):
    archs = sorted(records[0]["architectures"])
    out = {}
    scalar = [
        "access_specificity_interaction",
        "selective_contextual_advantage_min",
        "uniform_always_advantage",
    ]
    taskkeys = [
        "intact_loss", "history_contribution_energy", "random_access_rate",
        "contextual_vs_always", "contextual_vs_random", "contextual_vs_blocked",
        "contextual_ratio_to_always", "contextual_ratio_to_blocked",
        "loss_contextual", "loss_always", "loss_random", "loss_blocked",
        "q0_loss_contextual", "q0_loss_always", "q0_loss_random", "q0_loss_blocked",
        "q1_loss_contextual", "q1_loss_always", "q1_loss_random", "q1_loss_blocked",
        "contextual_gain_vs_current",
    ]
    for a in archs:
        out[a] = {}
        for k in scalar:
            out[a][k] = float(np.median([r["architectures"][a][k] for r in records]))
        for task in TASKS:
            out[a][task] = {
                k: float(np.median([r["architectures"][a][task][k] for r in records]))
                for k in taskkeys
            }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "confirm"], required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split(":"))
    seeds = list(range(lo, hi + 1))
    records = [eval_seed(s, args.mode) for s in seeds]
    out = {
        "campaign": "R35 representation-independent contextual-access necessity",
        "mode": args.mode,
        "development_architectures": DEV_ARCHS,
        "confirmatory_architectures": CONFIRM_ARCHS,
        "strict_heldout_architectures": STRICT_HELDOUT_ARCHS,
        "development_families": DEV_FAMILIES,
        "confirmatory_families": CONFIRM_FAMILIES,
        "access_modes": MODES,
        "seeds": seeds,
        "records": records,
        "medians": summarize(records),
        "boundaries": {
            "POLAR_Core_v1": "UNCHANGED_D_PLUS_C_PLUS_R_PLUS_A",
            "R34": "UNCHANGED_H1_INCONCLUSIVE_H2_SUPPORTED_H3_NOT_SUPPORTED",
            "E6b": "OPEN_EXTERNAL_ONLY",
            "E7": "OPEN_PROSPECTIVE_BIOLOGICAL",
            "global_minimality": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
        }
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["medians"], indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
