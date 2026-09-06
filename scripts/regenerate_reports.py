#!/usr/bin/env python3
"""Regenerate every result from complete saved environment/action/effect traces."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from polar.evaluation import summarize_run, paired_effect


def read_records(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display(x):
    return "NA" if x is None else f"{x:.5f}"


def regenerate(output):
    protocol = json.loads((output/"protocol.json").read_text())
    manifest = json.loads((output/"manifest.json").read_text())
    if hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol checksum mismatch")
    for name, digest in manifest["source_sha256"].items():
        if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Implementation differs from frozen run; use the recorded source revision: "+name)
    for name, digest in manifest["raw_sha256"].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Raw trace checksum mismatch: "+name)
    environments = defaultdict(list)
    for r in read_records(output/"environment.jsonl.gz"):
        environments[(r["split"], r["task"], r["seed"])].append(r)
    expected = {(split, task, seed) for split in ["development", "heldout"]
                for task in protocol["tasks"]
                for seed in (protocol[split+"_seeds"][:2] if manifest["smoke"] else protocol[split+"_seeds"])}
    if set(environments) != expected:
        raise ValueError("Environment trials do not match all frozen protocol seeds/tasks")
    rows = []
    for name in protocol["controllers"]:
        records = defaultdict(list)
        for r in read_records(output/"controllers"/(name+".jsonl.gz")):
            records[(r["split"], r["task"], r["seed"])].append(r)
        if records.keys() != environments.keys():
            raise ValueError("Missing or extra controller trials")
        for key, actions in sorted(records.items()):
            frames = environments[key]
            if [r["t"] for r in actions] != list(range(protocol["steps"])) or [f["t"] for f in frames] != list(range(protocol["steps"])):
                raise ValueError("Incomplete, duplicate, or misordered trace")
            rows.append(dict(split=key[0], task=key[1], seed=key[2], controller=name,
                             mean_compute_us=float(np.mean([r["compute_ns"] for r in actions]))/1000.,
                             **summarize_run(frames, actions, protocol["recovery"]["regret_max"], protocol["recovery"]["sustained_steps"], protocol["success_rule"])))
    generated = output/"generated"
    generated.mkdir(exist_ok=True)
    write_csv(generated/"per_run.csv", rows)
    (generated/"per_run.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    aggregate = []
    metrics = ["mean_regret", "tracking_rmse", "mean_cost", "mean_compute_us", "mean_recovery_steps", "recovery_fraction", "recall_regret", "reacquisition_regret", "effect_discrimination", "mean_node_homogeneity", "mean_temporal_action_cosine", "external_pass", "hard_violations"]
    for split in ["development", "heldout"]:
        for name in protocol["controllers"]:
            rr = [r for r in rows if r["split"]==split and r["controller"]==name]
            a = dict(split=split, controller=name, runs=len(rr))
            for m in metrics:
                values = [float(r[m]) for r in rr if r[m] is not None]
                a[m] = float(np.mean(values)) if values else None
                if m in {"mean_recovery_steps", "recall_regret"}:
                    a[m+"_observed_runs"] = len(values)
            a["recovery_censored"] = sum(r["recovery_censored"] for r in rr)
            a["recovery_events"] = sum(r["recovery_events"] for r in rr)
            aggregate.append(a)
    write_csv(generated/"aggregate.csv", aggregate)
    effects = []
    seed_means = {}
    for name in protocol["controllers"]:
        byseed = defaultdict(list)
        for r in rows:
            if r["split"] == "heldout" and r["controller"] == name:
                byseed[r["seed"]].append(r["mean_regret"])
        if any(len(v)!=len(protocol["tasks"]) for v in byseed.values()):
            raise ValueError("Incomplete paired-seed task set")
        seed_means[name] = dict((s,float(np.mean(v))) for s,v in byseed.items())
    for name in protocol["controllers"]:
        if name == "dual_pole":
            continue
        seeds = sorted(seed_means["dual_pole"])
        if set(seeds) != set(seed_means[name]):
            raise ValueError("Unpaired seeds")
        result = paired_effect([seed_means["dual_pole"][s] for s in seeds],
                               [seed_means[name][s] for s in seeds], protocol["bootstrap_seed"], protocol["bootstrap_draws"])
        result["multiplicity"] = "single_prespecified_primary" if name == "recurrent" else "exploratory_unadjusted"
        effects.append(dict(reference="dual_pole", comparator=name, metric="mean_regret", **result))
    write_csv(generated/"paired_effects.csv", effects)
    heldout = [a for a in aggregate if a["split"]=="heldout"]
    primary = next(e for e in effects if e["comparator"]=="recurrent")
    title = "SMOKE ONLY: not scientific evidence\n\n" if manifest["smoke"] else ""
    lines = [title+"# Frozen synthetic benchmark", "", f"Protocol SHA-256: `{manifest['protocol_sha256']}`", "",
             "All tables regenerated from checksummed per-step traces. Negative differences favor dual_pole. Missing values are NA, not zero.", "",
             "| Controller | Mean regret | RMSE | Recall regret | External pass fraction |", "|---|---:|---:|---:|---:|"]
    for a in heldout:
        lines.append(f"| {a['controller']} | {display(a['mean_regret'])} | {display(a['tracking_rmse'])} | {display(a['recall_regret'])} | {display(a['external_pass'])} |")
    lines += ["", "| Contrast (dual_pole minus comparator) | Regret difference | 95% CI | Paired dz |", "|---|---:|---|---:|"]
    for e in effects:
        lines.append(f"| {e['comparator']} | {display(e['mean_difference'])} | [{display(e['ci95_low'])}, {display(e['ci95_high'])}] | {display(e['paired_dz'])} |")
    lines += ["", f"Primary paired mean-regret difference, dual_pole minus recurrent: {display(primary['mean_difference'])}; 95% seed-bootstrap interval [{display(primary['ci95_low'])}, {display(primary['ci95_high'])}]; paired dz {display(primary['paired_dz'])}; n={primary['n']} paired seeds.",
              "", "Signed-intensity is an exact coordinate control, not an independent competing theory. Recurrent shares state size, observations, memory, capability estimation, workspace, constraints, and one update per step; its projected-gradient rule differs from inverse-planning. Any difference is a rule comparison and cannot be attributed uniquely to polarity.",
              "", "All other contrasts are exploratory unadjusted intervals, without confirmatory significance claims. Recovery means exclude censored events and must be read with recovery_fraction and censor counts. Heldout tasks share the same generator family; this is not broad out-of-distribution generalization.",
              "", "These results test functional mechanisms in an engineered synthetic controller. They do not establish consciousness, sentience, ASI, or universality of polarity."]
    (generated/"results.md").write_text("\n".join(lines)+"\n")
    tex = ["\\begin{table}[htbp]", "\\centering\\small", "\\begin{tabular}{lrrrr}", "\\hline", "Controller & Regret & RMSE & Recall regret & Pass rate " + chr(92)*2, "\\hline"]
    for a in heldout:
        tex.append(a["controller"].replace("_", "\\_")+" & "+" & ".join(display(a[m]) for m in ["mean_regret", "tracking_rmse", "recall_regret", "external_pass"])+r" \\")
    caption_prefix = "SMOKE ONLY, not paper evidence. " if manifest["smoke"] else ""
    tex += ["\\hline", "\\end{tabular}", "\\caption{"+caption_prefix+f"Heldout synthetic tasks; each controller uses the same {primary['n']} paired seeds and {len(protocol['tasks'])} task suites. "+"Pass rate is the fraction of seed--task runs satisfying every external criterion. NA denotes unavailable measurement.}", "\\end{table}",
            f"The prespecified paired difference in mean regret (dual-pole minus recurrent) is {display(primary['mean_difference'])}, with a 95\\% seed-bootstrap interval [{display(primary['ci95_low'])}, {display(primary['ci95_high'])}] and paired $d_z={display(primary['paired_dz'])}$. ",
            "The signed-intensity comparator is a coordinate-isomorphism control. The recurrent comparator changes the update rule while retaining auxiliary mechanisms; this does not identify an advantage caused uniquely by polarity. Other contrasts are exploratory, without multiplicity-adjusted significance claims. No result is a test of phenomenal consciousness."]
    tex += ["\\begin{table}[htbp]", "\\centering\\small", "\\begin{tabular}{lrrr}", "\\hline",
            "Comparator & Difference & 95\\% interval & Paired $d_z$ "+chr(92)*2, "\\hline"]
    for e in effects:
        tex.append(e["comparator"].replace("_", "\\_")+" & "+display(e["mean_difference"])+" & ["+display(e["ci95_low"])+", "+display(e["ci95_high"])+"] & "+display(e["paired_dz"])+" "+chr(92)*2)
    tex += ["\\hline", "\\end{tabular}", "\\caption{Paired mean-regret differences, dual-pole minus comparator. Negative favors dual-pole. Recurrent is the sole prespecified primary contrast; others are exploratory unadjusted intervals. NA $d_z$ indicates zero-variance paired differences.}", "\\end{table}"]
    full = next(a for a in heldout if a["controller"]=="dual_pole")
    tex.append(f"For the full controller, {full['recovery_censored']} of {full['recovery_events']} phase changes are right censored for recovery; the fraction recovered is {display(full['recovery_fraction'])}. Conditional recovery duration must not be interpreted without this censoring rate.")
    (generated/"results.tex").write_text("\n".join(tex)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [e["comparator"] for e in effects]
    y = np.arange(len(effects))
    point = np.array([e["mean_difference"] for e in effects])
    ax.errorbar(point, y, xerr=np.array([point-np.array([e["ci95_low"] for e in effects]), np.array([e["ci95_high"] for e in effects])-point]), fmt="o", color="#245887", capsize=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean regret difference: dual_pole − comparator (95% paired seed bootstrap)")
    ax.set_title("Synthetic controller comparisons; negative favors dual_pole")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(generated/"evaluation_summary.pdf")
    fig.savefig(generated/"evaluation_summary.png", dpi=170)
    plt.close(fig)
    print(f"Regenerated {len(rows)} runs in {generated}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    regenerate(parser.parse_args().output)
