# p2_baseline.py — P2: Baseline Tension Engine (with CSV/fig/table)
import os, json
from engine_v1_locked import TensionEngine, EngineConfig
from report_utils import save_timeseries, save_summary, plot_hgi_inc, latex_summary_table

def main(out_dir="results_p2_baseline", steps=20, N=120, K=10, seed=2025):
    os.makedirs(out_dir, exist_ok=True)
    cfg = EngineConfig(N=N, K=K, steps=steps, seed=seed, learn_M=False)
    eng = TensionEngine(cfg)
    ts = eng.run(out_dir=out_dir, print_console=True)

    # summary
    H = ts["HGI"]; I = ts["INC"]
    summary = {
        "Config": "P2 Baseline",
        "HGI_final": H[-1], "INC_final": I[-1],
        "HGI_mean": sum(H)/len(H), "INC_mean": sum(I)/len(I),
        "Interventions": ts.get("ethics_interventions_pct", 0.0),
        "Recovery": ts.get("recovery_steps", "")
    }

    save_timeseries(out_dir, "p2_baseline", ts)
    save_summary(out_dir, "p2_baseline", summary)
    plot_hgi_inc(out_dir, "p2_baseline", ts, "P2: Baseline Harmony/Coherence")
    latex_summary_table(
        out_dir, "p2_baseline",
        caption="P2 baseline metrics (control case).",
        label="tab:p2_baseline",
        rows=[summary]
    )
    print("Artifacts written to:", out_dir)

if __name__ == "__main__":
    main()
