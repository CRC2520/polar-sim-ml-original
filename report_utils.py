"""Corrected reproducible reporting: missing data is never imputed as zero."""
import csv
import gzip
import shutil
import json
import math
import os
import platform
from dataclasses import asdict, is_dataclass

import numpy as np

REPORT_VERSION = "2.0.1"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if hasattr(value, "detach"):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite report value; represent missing measurements as None")
    return value

def _dump(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(_json_value(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)

def validate_timeseries(ts):
    """List-valued trace columns must align; scalar/dict metadata is separate."""
    if "step" not in ts or not isinstance(ts["step"], (list, tuple)):
        raise ValueError("Trace requires a step list")
    n = len(ts["step"])
    if not n:
        raise ValueError("Cannot report an empty trace")
    if any(b <= a for a, b in zip(ts["step"], ts["step"][1:])):
        raise ValueError("Trace steps must increase strictly")
    columns = {}
    for key, value in ts.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != n:
                raise ValueError(f"Trace column {key!r} has {len(value)} entries, expected {n}")
            columns[key] = _json_value(value)
    return columns

def save_timeseries(out_dir, name, ts, metadata=None):
    ensure_dir(out_dir)
    columns = validate_timeseries(ts)
    trace_meta = {k: _json_value(v) for k, v in ts.items() if k not in columns}
    trace_meta.update(metadata or {})
    trace_meta.setdefault("report_version", REPORT_VERSION)
    trace_meta.setdefault("python_version", platform.python_version())
    path = os.path.join(out_dir, f"{name}_timeseries.csv")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for i in range(len(columns["step"])):
            row = {}
            for key, values in columns.items():
                value = values[i]
                row[key] = json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else ("null" if value is None else value)
            writer.writerow(row)
    _dump(os.path.join(out_dir, f"{name}_timeseries.json"), {"metadata": trace_meta, "timeseries": columns})
    return path

def load_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)

def compress_trace(path):
    """Deterministic gzip archive; compression preserves the complete trace."""
    from pathlib import Path
    path = Path(path)
    destination = path.with_name(path.name + ".gz")
    with open(path, "rb") as source, open(destination, "wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=9) as compressed:
            shutil.copyfileobj(source, compressed)
    path.unlink()
    return destination

def save_summary(out_dir, name, summary):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}_summary.json")
    _dump(path, summary)
    return path

def _percentage(ts, key):
    values = ts.get(key)
    if values is None or not len(values) or any(v is None for v in values):
        return None
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError(f"{key} must contain flags or fractions in [0,1]")
    return float(100 * a.mean())

def recovery_from_traces(ts, control=None, tolerance=0.05, sustained_steps=3):
    """Return to a paired unstimulated trajectory after external input withdrawal.

    Distance: activation RMS over every state dimension. Search starts one step
    after the last nonzero stimulus. Recovery requires sustained_steps successive
    samples below tolerance; delay 1 is the first sample after withdrawal.
    This describes dynamical return, not behavioral recovery or clinical trauma.
    """
    if tolerance <= 0 or sustained_steps < 1:
        raise ValueError("Recovery tolerance and window must be positive")
    result = {"steps": None, "status": "unavailable", "tolerance_rms": tolerance,
              "sustained_steps": sustained_steps, "reference": "matched_unstimulated_control"}
    if "stimulus" not in ts or "A" not in ts:
        return result
    validate_timeseries(ts)
    stimulus = np.asarray(ts["stimulus"], dtype=float)
    if not np.isfinite(stimulus).all():
        raise ValueError("Nonfinite stimulus trace")
    event = np.flatnonzero(np.any(np.abs(stimulus).reshape(len(stimulus), -1) > 1e-12, axis=1))
    if not len(event):
        result["status"] = "not_applicable"
        return result
    result.update(first_stimulus_step=int(ts["step"][event[0]]), last_stimulus_step=int(ts["step"][event[-1]]))
    if control is None or "A" not in control:
        return result
    validate_timeseries(control)
    states, reference = np.asarray(ts["A"], dtype=float), np.asarray(control["A"], dtype=float)
    if states.shape != reference.shape or ts["step"] != control["step"]:
        raise ValueError("Recovery requires matched state shapes and step schedules")
    if not np.isfinite(states).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite state in recovery trace")
    distance = np.sqrt(np.mean((states - reference).reshape(len(states), -1) ** 2, axis=1))
    result["distance_rms"] = distance.tolist()
    if not np.any(distance[event[0]:] > tolerance):
        result["status"] = "no_detectable_departure"
        return result
    first = int(event[-1]) + 1
    for start in range(first, len(distance) - sustained_steps + 1):
        if np.all(distance[start:start + sustained_steps] <= tolerance):
            result.update(steps=int(ts["step"][start] - ts["step"][event[-1]]), status="recovered")
            return result
    result["status"] = "censored"
    return result

def summarize_trace(ts, tag, control=None):
    validate_timeseries(ts)
    summary = {"Config": tag}
    for metric in ("HGI", "INC"):
        values = ts.get(metric)
        if values is None or any(v is None for v in values):
            summary.update({metric + "_final": None, metric + "_mean": None})
        else:
            values = np.asarray(values, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metric {metric}")
            summary.update({metric + "_final": float(values[-1]), metric + "_mean": float(values.mean())})
    for field, key in (("Continuous_modulation_pct", "continuous_modulation"), ("Override_pct", "override"),
                       ("Override_effect_pct", "override_effect"), ("HGI_guard_pct", "hgi_guard"),
                       ("INC_guard_pct", "inc_guard"), ("Clipping_pct", "numeric_clip")):
        summary[field] = _percentage(ts, key)
    if summary["Continuous_modulation_pct"] is None and "alpha" in ts and all(v is not None for v in ts["alpha"]):
        alpha = np.asarray(ts["alpha"], dtype=float)
        if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("alpha must be finite in [0,1]")
        summary["Continuous_modulation_pct"] = float(100 * np.mean(alpha < 1 - 1e-8))
    summary["recovery"] = recovery_from_traces(ts, control)
    summary["Recovery"] = summary["recovery"]["steps"]
    summary["Recovery_status"] = summary["recovery"]["status"]
    return summary

def export_run(out_dir, name, ts, tag, metadata=None, control=None, plots=True):
    row = summarize_trace(ts, tag, control)
    save_timeseries(out_dir, name, ts, metadata)
    if control is not None:
        save_timeseries(out_dir, name + "_control", control, {**(metadata or {}), "role": "paired_unstimulated_control"})
    save_summary(out_dir, name, row)
    latex_summary_table(out_dir, name, "Corrected descriptive dynamics; no consciousness inference.", "tab:" + name, [row])
    if plots:
        plot_hgi_inc(out_dir, name, ts, tag)
    return row

def plot_hgi_inc(out_dir, name, ts, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    validate_timeseries(ts)
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ts["step"], ts["HGI"], label="HGI: graph homogeneity")
    ax.plot(ts["step"], ts["INC"], label="INC: temporal/self-reference similarity")
    ax.set(xlabel="Step (zero indexed)", ylabel="Descriptor", ylim=(0, 1.01), title=title)
    ax.legend(fontsize=8)
    out = os.path.join(out_dir, f"{name}_plot.png")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def _latex(value):
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return "".join({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}.get(c, c) for c in str(value))

def latex_summary_table(out_dir, name, caption, label, rows):
    ensure_dir(out_dir)
    columns = [("Config", "Config"), ("HGI_final", "HGI final"), ("INC_final", "INC final"),
               ("Continuous_modulation_pct", r"Modulation (\%)"), ("Override_pct", r"Override (\%)"),
               ("Recovery", "Return steps"), ("Recovery_status", "Return status")]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", "\\caption{" + caption + "}",
             "\\label{" + label + "}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrl}", r"\toprule",
             " & ".join(c[1] for c in columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(_latex(row.get(c[0])) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path = os.path.join(out_dir, f"{name}_table.tex")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return path

def bar_compare(out_dir, name, rows, title, metric_keys=("HGI_final", "INC_final")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)
    x = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(8, 4.2))
    for offset, key in zip((-.175, .175), metric_keys):
        vals = [np.nan if r.get(key) is None else r[key] for r in rows]
        ax.bar(x + offset, vals, width=.35, label=key)
    ax.set_xticks(x, [r["Config"] for r in rows], rotation=15, ha="right")
    ax.set(ylabel="Descriptor", ylim=(0, 1.01), title=title); ax.legend()
    out = os.path.join(out_dir, f"{name}_bars.png")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out
