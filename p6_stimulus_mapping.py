# p6_stimulus_mapping.py — P6: Stimulus Mapping (NLP) with CSV/fig/table
import os, json, torch
from engine_v1_locked import TensionEngine, EngineConfig, DEVICE
from report_utils import save_timeseries, save_summary, plot_hgi_inc, latex_summary_table

NODES = [
    'Poder vs Vulnerabilidad', 'Placer vs Dolor', 'Integración vs Fragmentación',
    'Control vs Rendición', 'Deseo vs Límite', 'Libertad vs Orden',
    'Preservación vs Transformación', 'Reconocimiento vs Autenticidad'
]

def stim_from_text(text, N, scale=0.5):
    text_l = text.lower()
    # naive keyword map (replicate current behavior)
    idx = []
    for i, name in enumerate(NODES):
        key = name.lower()
        if any(k.strip() in text_l for k in key.split(" vs ")):
            idx.append(i)
    if not idx:
        h = abs(hash(text_l)) % len(NODES)
        idx = [h]
    stim = torch.zeros(N, device=DEVICE)
    for i in idx:
        stim[i::len(NODES)] = scale
    return stim

def main(prompt="Libertad vs Orden", out_dir="results_p6_stimulus",
         steps=20, N=160, K=10, seed=2025, scale=0.5, stim_step=8):
    os.makedirs(out_dir, exist_ok=True)
    cfg = EngineConfig(N=N, K=K, steps=steps, seed=seed, stim_step=stim_step, stim_scale=scale, learn_M=False)
    eng = TensionEngine(cfg, nodes=NODES)

    for t in range(steps):
        stim = None
        if t == stim_step:
            stim = stim_from_text(prompt, eng.N, scale=scale)
        A_next, flags = eng.step(stimulus=stim)
        print(f"Step {t+1}: HGI={eng.ts['HGI'][-1]:.4f}, INC={eng.ts['INC'][-1]:.4f}, "
              f"SAT={eng.ts['SAT'][-1]:.2f}, α={eng.ts['alpha'][-1]:.3f}, guards(H/I)={flags['hgi_guard']}/{flags['inc_guard']}")

    ts = eng.ts
    # simple estimates used for the paper table (you can compute true recovery in engine)
    summary = {
        "Config": f"P6 Stim “{prompt}”",
        "HGI_final": float(ts["HGI"][-1]),
        "INC_final": float(ts["INC"][-1]),
        "HGI_mean": float(sum(ts["HGI"])/len(ts["HGI"])),
        "INC_mean": float(sum(ts["INC"])/len(ts["INC"])),
        "Interventions": ts.get("ethics_interventions_pct", 0.0),
        "Recovery": ts.get("recovery_steps", "")
    }

    save_timeseries(out_dir, "p6_stimulus", ts)
    save_summary(out_dir, "p6_stimulus", summary)
    plot_hgi_inc(out_dir, "p6_stimulus", ts, f"P6: Stimulus Mapping — {prompt}")
    latex_summary_table(
        out_dir, "p6_stimulus",
        caption="Stimulus mapping (NLP) impact on harmony/coherence.",
        label="tab:p6_stimulus",
        rows=[summary]
    )
    print("Artifacts written to:", out_dir)

if __name__ == "__main__":
    main()
