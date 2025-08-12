#!/usr/bin/env python
import argparse, os, json, sys, traceback
from copy import deepcopy

# ---------- Imports ----------
try:
    from engine_v1_locked import TensionEngine, EngineConfig
except Exception as e:
    print("[ERROR] Could not import engine_v1_locked:", e)
    traceback.print_exc()
    sys.exit(1)

from report_utils import save_timeseries, save_summary, latex_summary_table, bar_compare, plot_hgi_inc

# ---------- Runner ----------
def run_one(cfg: EngineConfig, out_dir: str, tag: str, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(f"[P7] Running config: {tag}  → {out_dir}")
        print("     cfg:", cfg)

    eng = TensionEngine(cfg)
    ts = eng.run(out_dir=out_dir, print_console=verbose)

    H = ts["HGI"]; I = ts["INC"]
    summary = {
        "Config": tag,
        "HGI_final": float(H[-1]), "INC_final": float(I[-1]),
        "HGI_mean": float(sum(H)/len(H)), "INC_mean": float(sum(I)/len(I)),
        "Interventions": float(ts.get("ethics_interventions_pct", 0.0)),
        "Recovery": ts.get("recovery_steps", "")
    }
    save_timeseries(out_dir, tag, ts)
    save_summary(out_dir, tag, summary)
    plot_hgi_inc(out_dir, tag, ts, f"P7: {tag}")
    print(f"[P7] Done: {tag}")
    return summary

def run_all(args):
    os.makedirs(args.out, exist_ok=True)
    print("[P7] Start run_all →", args)

    base_cfg = EngineConfig(
        N=args.N, K=args.K, steps=args.steps, seed=args.seed,
        learn_M=True, lr_M=args.lr, reg_M=args.reg,
        inc_floor=0.96, inc_gain=0.18, hgi_floor=0.95, hgi_gain=0.10,
        ethics_kappa=1.12, ethics_alpha_min=0.60,
        gamma_cons_start=0.20, gamma_cons_end=0.36, beta_mu=0.07
    )

    rows = []

    if not args.only_no_ethics and not args.only_no_homeostasis:
        rows.append( run_one(base_cfg, os.path.join(args.out, "baseline"), "P7 Baseline", verbose=True) )

    if args.no_ethics or args.only_no_ethics:
        cfg_noe = deepcopy(base_cfg); cfg_noe.no_ethics = True
        rows.append( run_one(cfg_noe, os.path.join(args.out, "no_ethics"), "P7 No Ethics", verbose=True) )

    if args.no_homeostasis or args.only_no_homeostasis:
        cfg_noh = deepcopy(base_cfg); cfg_noh.no_homeostasis = True
        rows.append( run_one(cfg_noh, os.path.join(args.out, "no_homeostasis"), "P7 No Homeostasis", verbose=True) )

    if not rows:
        print("[P7] Nothing was scheduled. Add --no-ethics and/or --no-homeostasis or remove --only-* filters.")
        return

    # Combined table + bars
    latex_summary_table(
        args.out, "p7_ablation",
        caption="Ablation study for learned $M$.",
        label="tab:p7_ablation",
        rows=rows
    )
    bar_compare(args.out, "p7_ablation", rows=rows, title="P7 Ablations — Final HGI/INC")
    with open(os.path.join(args.out, "p7_all_summaries.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    print("[P7] Artifacts written to:", args.out)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--lr", type=float, default=0.0035)
    ap.add_argument("--reg", type=float, default=5e-4)
    ap.add_argument("--out", type=str, default="results_p7_learnedM")

    # toggles
    ap.add_argument("--no-ethics", action="store_true", help="Add a run with ethics disabled")
    ap.add_argument("--no-homeostasis", action="store_true", help="Add a run with homeostasis disabled")

    # filters (optional): run exactly one of the ablations
    ap.add_argument("--only-no-ethics", action="store_true", help="Run ONLY the no-ethics ablation (skip baseline)")
    ap.add_argument("--only-no-homeostasis", action="store_true", help="Run ONLY the no-homeostasis ablation (skip baseline)")

    args = ap.parse_args()
    run_all(args)
