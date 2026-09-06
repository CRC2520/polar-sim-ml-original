# report_utils.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_timeseries(out_dir: str, name: str, ts: dict):
    ensure_dir(out_dir)
    df = pd.DataFrame({
        "step": ts["step"],
        "HGI": ts["HGI"],
        "INC": ts["INC"],
        "SAT": ts.get("SAT", [np.nan]*len(ts["step"])),
        "alpha": ts.get("alpha", [np.nan]*len(ts["step"])),
        "override": ts.get("override", [np.nan]*len(ts["step"])),
    })
    csv_path = os.path.join(out_dir, f"{name}_timeseries.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, f"{name}_timeseries.json"), "w") as f:
        json.dump({"timeseries": ts}, f, indent=2)
    return csv_path

def save_summary(out_dir: str, name: str, summary: dict):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path

def plot_hgi_inc(out_dir: str, name: str, ts: dict, title: str):
    ensure_dir(out_dir)
    x = ts["step"]; H = ts["HGI"]; I = ts["INC"]
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(x, H, label="HGI")
    plt.plot(x, I, label="INC")
    plt.xlabel("Step"); plt.ylabel("Score"); plt.ylim(0.0, 1.01)
    plt.title(title)
    plt.legend()
    out = os.path.join(out_dir, f"{name}_plot.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

def latex_summary_table(out_dir: str, name: str, caption: str, label: str, rows: list):
    """
    rows: list of dicts with keys: 'Config','HGI_mean','INC_mean','HGI_final','INC_final','Interventions','Recovery'
    Any missing keys will be blank.
    """
    ensure_dir(out_dir)
    cols = ["Config", "HGI_mean", "INC_mean", "HGI_final", "INC_final", "Interventions", "Recovery"]
    def cell(d, k): 
        v = d.get(k, "")
        if isinstance(v, float): return f"{v:.3f}"
        return str(v)
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Config & HGI (mean) & INC (mean) & HGI$_{\\text{final}}$ & INC$_{\\text{final}}$ & Interventions (\\%) & Recovery (steps) \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" {} & {} & {} & {} & {} & {} & {} \\\\".format(
            cell(r,"Config"), cell(r,"HGI_mean"), cell(r,"INC_mean"),
            cell(r,"HGI_final"), cell(r,"INC_final"), cell(r,"Interventions"), cell(r,"Recovery")
        ))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    tex = "\n".join(lines)
    path = os.path.join(out_dir, f"{name}_table.tex")
    with open(path, "w") as f:
        f.write(tex)
    return path

def bar_compare(out_dir: str, name: str, rows: list, title: str, metric_keys=("HGI_final","INC_final")):
    """
    rows: list of dicts each with 'Config' + metrics; makes side-by-side bars.
    """
    import numpy as np
    ensure_dir(out_dir)
    labels = [r["Config"] for r in rows]
    vals1 = [r.get(metric_keys[0], np.nan) for r in rows]
    vals2 = [r.get(metric_keys[1], np.nan) for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - w/2, vals1, width=w, label=metric_keys[0])
    plt.bar(x + w/2, vals2, width=w, label=metric_keys[1])
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Score"); plt.ylim(0.0, 1.01)
    plt.title(title); plt.legend()
    out = os.path.join(out_dir, f"{name}_bars.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out
