"""P7 corrected learned-M experiment with an explicit, matched stimulus schedule."""
import argparse
import os
from copy import deepcopy
from dataclasses import asdict

from engine_v2_corrected import TensionEngine, EngineConfig
from p6_stimulus_mapping import stim_from_text
from report_utils import export_run, save_summary, latex_summary_table, bar_compare

def run_one(cfg, out_dir, tag, verbose=True, prompt="Libertad", plots=True):
    if not 0 <= cfg.stim_step < cfg.steps:
        raise ValueError("P7 stim_step must address an executed zero-indexed step")
    stimulus = stim_from_text(prompt, cfg.N, cfg.stim_scale)
    eng = TensionEngine(cfg)
    for t in range(cfg.steps):
        eng.step(stimulus=stimulus if t == cfg.stim_step else None)
    control = TensionEngine(cfg)
    control.run(print_console=False)
    metadata = {"experiment": "P7", "config": asdict(cfg), "seed": cfg.seed,
                "stimulus_protocol": {"prompt": prompt, "mapping": "explicit_signed_keyword_v1",
                                      "step_index": cfg.stim_step, "human_step": cfg.stim_step + 1,
                                      "scale": cfg.stim_scale, "duration_steps": 1},
                "paired_control": "same initialization/configuration; external pulse omitted"}
    row = export_run(out_dir, "p7", eng.ts, tag, metadata, control.ts, plots)
    if verbose:
        print(f"[P7] {tag}: HGI={row['HGI_final']:.4f}, INC={row['INC_final']:.4f}, return={row['Recovery_status']}")
    return row

def run_all(args):
    os.makedirs(args.out, exist_ok=True)
    base_cfg = EngineConfig(
        N=args.N, K=args.K, steps=args.steps, seed=args.seed,
        learn_M=True, lr_M=args.lr, reg_M=args.reg,
        stim_step=getattr(args, "stim_step", 8), stim_scale=getattr(args, "stim_scale", .6),
        inc_floor=.96, inc_gain=.18, hgi_floor=.95, hgi_gain=.10,
        ethics_kappa=1.12, ethics_alpha_min=.60,
        gamma_cons_start=.20, gamma_cons_end=.36, beta_mu=.07)
    rows = []
    only_noe, only_noh = args.only_no_ethics, args.only_no_homeostasis
    scheduled = []
    if not only_noe and not only_noh:
        scheduled.append(("baseline", "P7 Learned M", base_cfg))
    if args.no_ethics or only_noe:
        cfg = deepcopy(base_cfg); cfg.no_ethics = True
        scheduled.append(("no_modulation", "P7 No modulation or override", cfg))
    if args.no_homeostasis or only_noh:
        cfg = deepcopy(base_cfg); cfg.no_homeostasis = True
        scheduled.append(("no_homeostasis", "P7 No baseline restoring drive", cfg))
    for folder, tag, cfg in scheduled:
        rows.append(run_one(cfg, os.path.join(args.out, folder), tag,
                            verbose=not getattr(args, "quiet", False),
                            prompt=getattr(args, "prompt", "Libertad"), plots=not getattr(args, "no_plots", False)))
    if not rows:
        raise ValueError("No P7 configurations were scheduled")
    latex_summary_table(args.out, "p7_ablation", "Matched stimulus and seed in learned-M ablations.", "tab:p7_ablation", rows)
    if not getattr(args, "no_plots", False):
        bar_compare(args.out, "p7_ablation", rows, "P7 descriptive dynamics")
    save_summary(args.out, "p7_all", {"rows": rows, "seed": args.seed,
                                    "schedule": {"stim_step_zero_index": base_cfg.stim_step,
                                                 "stim_scale": base_cfg.stim_scale}})
    return rows

def parser():
    ap = argparse.ArgumentParser()
    for name, default in (("N", 160), ("K", 12), ("steps", 24), ("seed", 2025), ("stim-step", 8)):
        ap.add_argument("--" + name, type=int, default=default)
    ap.add_argument("--lr", type=float, default=.0035)
    ap.add_argument("--reg", type=float, default=5e-4)
    ap.add_argument("--stim-scale", type=float, default=.6)
    ap.add_argument("--prompt", default="Libertad")
    ap.add_argument("--out", default="results_corrected/p7_learnedM")
    ap.add_argument("--no-ethics", action="store_true", help="Compatibility alias: ablate modulation and override, not consequence evaluation")
    ap.add_argument("--no-homeostasis", action="store_true")
    ap.add_argument("--only-no-ethics", action="store_true")
    ap.add_argument("--only-no-homeostasis", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap

if __name__ == "__main__":
    run_all(parser().parse_args())
