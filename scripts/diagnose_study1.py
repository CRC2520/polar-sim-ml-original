#!/usr/bin/env python3
"""Exploratory paired diagnosis of Study 1; never a new held-out evaluation.

Runs immutable v2.1 sources under explicit diagnostic interventions. Recreate with
python scripts/diagnose_study1.py --output results_study2/diagnosis
Existing results are protected unless --overwrite is explicitly requested.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from polar.baselines import make_controller
from polar.evaluation import paired_effect, step_metrics, summarize_run
from polar.tasks import make_trial, observation, jsonable

VARIANTS = ["dual_pole", "recurrent", "no_self_model", "reset_capacity_recall", "reset_gain_only_recall",
            "freeze_capacity_recall", "reset_freeze_capacity_recall",
            "unit_confidence_recall", "zero_noise", "gate_small_action_updates"]
DEFINITIONS = {
    "dual_pole": "Unmodified Study 1 controller and environment.",
    "recurrent": "Study 1 matched projected-gradient recurrence.",
    "no_self_model": "Study 1 capability-model ablation throughout the trial.",
    "reset_capacity_recall": "At t=48 lesion estimated gain to one and reset counts/error; subsequent learning retained.",
    "reset_gain_only_recall": "At t=48 replace estimated gain by one, preserving uncertainty/count/error state and subsequent learning.",
    "freeze_capacity_recall": "Suppress capability feedback updates during t=48..63; preserve estimates acquired before recall.",
    "reset_freeze_capacity_recall": "At t=48 reset capability state; suppress updates only during t=48..63.",
    "unit_confidence_recall": "For t=48..63 set recall confidence to one for stored channels; preserve stored values, all other states and environment.",
    "zero_noise": "Set exogenous observation noise to zero for the whole trial; preserve targets, phases and controller.",
    "gate_small_action_updates": "Throughout the trial suppress gain/count/error updates when own action <0.02 (10 original noise SD); fixed diagnostic, not an optimized estimator.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failure_reasons(summary, rule):
    """Report individual failed criteria; never coerce an absent value to zero."""
    criteria = [("mean_regret", "mean_regret_max", "max"),
                ("hard_violations", "hard_violations_max", "max"),
                ("effect_discrimination", "effect_discrimination_min", "min"),
                ("recovery_fraction", "recovery_fraction_min", "min"),
                ("recall_regret", "recall_regret_max", "optional_max")]
    failures = []
    for metric, threshold, direction in criteria:
        value = summary.get(metric)
        if value is None and direction == "optional_max":
            continue
        failed = value is None or (value < rule[threshold] if direction == "min" else value > rule[threshold])
        if failed:
            failures.append({"metric": metric, "value": value, "threshold": rule[threshold],
                             "direction": direction, "excess": None if value is None else value-rule[threshold]})
    return failures


def gate_capability_update(model, action, effect, threshold=.02):
    """Causal intervention: withhold only weak-excitation estimator updates."""
    capability = model.self_model
    before = {key: getattr(capability, key).copy() for key in ["gain_hat", "counts", "error_ema"]}
    result = model.learn({"effect": effect})
    suppressed = action < threshold
    for key, value in before.items():
        getattr(capability, key)[suppressed] = value[suppressed]
    result["diagnostic_suppressed_channels"] = int(np.count_nonzero(suppressed & (action > 1e-6)))
    result["gain_after"] = capability.gain_hat.tolist()
    result["counts_after"] = capability.counts.tolist()
    return result


def finite_mean(values):
    values = np.asarray(values, float)
    return float(values.mean()) if values.size else None


def execute(seed, variant, protocol):
    trial = make_trial("switching_memory", seed, "heldout", protocol["agents"], protocol["types"], protocol["steps"])
    model = make_controller(variant if variant in ["dual_pole", "recurrent", "no_self_model"] else "dual_pole", seed)
    base_recall = model.memory.recall
    if variant == "unit_confidence_recall":
        def recall(cue):
            value, confidence, trace = base_recall(cue)
            if 49 <= model.memory.clock <= 64 and trace["found"]:
                confidence = (np.asarray(trace["counts"]) > 0).astype(float)
                trace["diagnostic_confidence_override"] = True
            return value, confidence, trace
        model.memory.recall = recall
    records, diagnostics, channel_details = [], [], []
    for frame in trial.steps:
        t = frame["t"]
        if variant == "zero_noise":
            frame["noise"] = np.zeros_like(frame["noise"])
        if t == 48 and variant in ["reset_capacity_recall", "reset_freeze_capacity_recall"]:
            model.self_model.lesion(gain=1.)
        if t == 48 and variant == "reset_gain_only_recall":
            model.self_model.gain_hat.fill(1.)
        action = model.act(observation(frame))
        trace = model.last_trace
        gain_before = model.self_model.gain_hat.copy()
        effect = np.clip(action * frame["gains"] + frame["noise"], 0., 1.)
        measured = np.ones_like(action)
        valid = action > 1e-6
        np.divide(effect, action, out=measured, where=valid)
        high = frame["target"] > .4
        small = valid & (action < .02)
        remembered = np.asarray(trace["memory_read"]["value"])
        confidence = np.asarray(trace["memory_read"]["confidence"])
        stats = step_metrics(frame, action, effect)
        row = dict(seed=seed, variant=variant, t=t, phase=frame["phase"], recall=frame["recall"],
                   **stats, gain_high_mean=finite_mean(gain_before[high]),
                   gain_high_max=float(np.max(gain_before[high])),
                   planning_gain_high_mean=finite_mean(np.asarray(trace["planning_gain"])[high]),
                   uncertainty_high_mean=finite_mean(model.self_model.uncertainty[high]),
                   confidence_high_mean=finite_mean(confidence[high]),
                   stored_target_mse=float(np.mean((remembered-frame["target"])**2)),
                   effective_target_mse=float(np.mean((np.asarray(trace["effective_target"])-frame["target"])**2)),
                   action_high_mean=finite_mean(action[high]),
                   small_action_updates=int(small.sum()),
                   low_ratio_clips=int((valid & (measured < .1)).sum()),
                   high_ratio_clips=int((valid & (measured > 4.)).sum()),
                   effect_lower_clips=int((action*frame["gains"]+frame["noise"] < 0.).sum()),
                   effect_upper_clips=int((action*frame["gains"]+frame["noise"] > 1.).sum()),
                   action_clipped_channels=trace["stability"]["clipped_channels"],
                   min_resource_scale=min(trace["constraints"]["resource_scale"]))
        if seed in [1012, 1020] and t in [47, 48, 49, 63]:
            for index in np.ndindex(action.shape):
                channel_details.append(dict(seed=seed, variant=variant, t=t, agent=index[0],
                    axis=index[1], pole=index[2], target=float(frame["target"][index]),
                    action=float(action[index]), effect=float(effect[index]),
                    gain_before=float(gain_before[index]), planning_gain=float(model._planning_gain[index]),
                    gain_count=int(model.self_model.counts[index]),
                    uncertainty=float(model.self_model.uncertainty[index]),
                    memory_value=float(remembered[index]), memory_confidence=float(confidence[index]),
                    raw_effect_action_ratio=float(measured[index]) if valid[index] else None,
                    noise=float(frame["noise"][index])))
        if frame["recall"] and variant in ["freeze_capacity_recall", "reset_freeze_capacity_recall"]:
            model.skip_feedback("Exploratory intervention: suppress capability updates during masked recall")
        elif variant == "gate_small_action_updates":
            gate_capability_update(model, action, effect)
        else:
            model.learn({"effect": effect})
        diagnostics.append(row)
        records.append({"action": action, "effect": effect})
        model.traces.clear()
    summary = {"seed": seed, "variant": variant, **summarize_run(trial.steps, records,
        protocol["recovery"]["regret_max"], protocol["recovery"]["sustained_steps"], protocol["success_rule"])}
    return summary, diagnostics, channel_details


def write_csv(path, rows, compressed=False):
    if not rows:
        raise ValueError("Cannot write empty diagnosis table")
    raw = io.StringIO(newline="")
    writer = csv.DictWriter(raw, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    data = raw.getvalue().encode()
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if compressed else data)


def run(output, overwrite=False):
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError("Diagnosis already exists: use a new directory or explicit --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "results_corrected/contextual"
    protocol = json.loads((previous/"protocol.json").read_text())
    old_manifest = json.loads((previous/"manifest.json").read_text())
    for relative, digest in old_manifest["source_sha256"].items():
        if relative.startswith("polar/") and sha(ROOT/relative) != digest:
            raise ValueError(f"Frozen Study 1 implementation changed: {relative}")
    archived = json.loads((previous/"run_summaries.json").read_text())
    failures = [{**r, "failed_criteria": failure_reasons(r, protocol["success_rule"])}
                for r in archived if r["split"] == "heldout" and r["controller"] == "dual_pole" and not r["external_pass"]]
    summaries, diagnostics, channels = [], [], []
    max_replication_error = 0.
    for seed in protocol["heldout_seeds"]:
        for variant in VARIANTS:
            summary, steps, details = execute(seed, variant, protocol)
            if variant in ["dual_pole", "recurrent", "no_self_model"]:
                old = next(r for r in archived if r["split"] == "heldout" and r["task"] == "switching_memory" and
                           r["seed"] == seed and r["controller"] == variant)
                for key in ["mean_regret", "recall_regret", "reacquisition_regret", "tracking_rmse"]:
                    error = abs(summary[key]-old[key])
                    max_replication_error = max(max_replication_error, error)
                    if error > 1e-12:
                        raise ValueError(f"Source replay failed: {seed}/{variant}/{key}: {error}")
                if summary["external_pass"] != old["external_pass"]:
                    raise ValueError("Pass/fail replay mismatch")
            summaries.append(summary)
            diagnostics.extend(steps)
            channels.extend(details)
        print(f"Diagnosed prior Study 1 seed {seed}", flush=True)
    reference = [r["recall_regret"] for r in summaries if r["variant"] == "dual_pole"]
    aggregates = []
    for variant in VARIANTS:
        rows = [r for r in summaries if r["variant"] == variant]
        effect = paired_effect([r["recall_regret"] for r in rows], reference, seed=20260906, bootstrap=2000)
        aggregates.append(dict(variant=variant, n=len(rows),
            mean_regret=float(np.mean([r["mean_regret"] for r in rows])),
            recall_regret=float(np.mean([r["recall_regret"] for r in rows])),
            reacquisition_regret=float(np.mean([r["reacquisition_regret"] for r in rows])),
            pass_count=sum(r["external_pass"] for r in rows), **{k: v for k, v in effect.items() if k != "n"}))
    previous_tradeoff = []
    for task in protocol["tasks"]:
        for name in ["dual_pole", "recurrent", "no_self_model"]:
            rows = [r for r in archived if r["split"] == "heldout" and r["task"] == task and r["controller"] == name]
            previous_tradeoff.append(dict(task=task, controller=name, n=len(rows),
                mean_regret=float(np.mean([r["mean_regret"] for r in rows])), pass_count=sum(r["external_pass"] for r in rows)))
    (output/"failure_cases.json").write_text(json.dumps(failures, indent=2, allow_nan=False)+"\n")
    (output/"summaries.json").write_text(json.dumps(summaries, indent=2, allow_nan=False)+"\n")
    write_csv(output/"summaries.csv", summaries)
    write_csv(output/"paired_recall_effects.csv", aggregates)
    write_csv(output/"previous_task_tradeoff.csv", previous_tradeoff)
    write_csv(output/"step_diagnostics.csv.gz", diagnostics, compressed=True)
    write_csv(output/"failure_channel_details.csv.gz", channels, compressed=True)
    tex = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{lrrr}", r"\toprule",
           r"Exploratory intervention & Recall regret & Difference & Passes \\", r"\midrule"]
    labels = {"dual_pole": "Original dual pole", "recurrent": "Matched recurrent", "no_self_model": "No capability model",
              "reset_capacity_recall": "Reset capability at recall", "reset_gain_only_recall": "Reset gain only at recall",
              "freeze_capacity_recall": "Freeze capability during recall",
              "reset_freeze_capacity_recall": "Reset and freeze capability", "unit_confidence_recall": "Unit recall confidence",
              "zero_noise": "Remove observation noise", "gate_small_action_updates": "Gate weak-action updates"}
    for row in aggregates:
        tex.append(f"{labels[row['variant']]} & {row['recall_regret']:.6f} & {row['mean_difference']:+.6f} & {row['pass_count']}/30 " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{Exploratory paired diagnosis on the 30 previously observed Study 1 memory seeds. Difference is intervention minus original recall regret; lower is better. These interventions are diagnostic instruments, not prospectively validated improvements.}",
        r"\label{tab:study1-diagnosis}", r"\end{table}"])
    (output/"diagnosis.tex").write_text("\n".join(tex)+"\n")
    manifest = dict(schema="study1-exploratory-diagnosis-1.0", scope="Previously observed seeds; post hoc exploratory mechanism diagnosis only",
        python=platform.python_version(), numpy=np.__version__, task="switching_memory", seeds=protocol["heldout_seeds"],
        variants=DEFINITIONS, diagnostic_small_action_threshold=.02, completed_runs=len(summaries),
        baseline_replays=3*len(protocol["heldout_seeds"]), max_baseline_summary_replay_error=max_replication_error,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*sorted((ROOT/"polar").glob("*.py")), Path(__file__)]},
        prior_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [previous/"manifest.json", previous/"protocol.json", previous/"run_summaries.json",
            previous/"environment.jsonl.gz", previous/"controllers/dual_pole.jsonl.gz",previous/"controllers/recurrent.jsonl.gz",previous/"controllers/no_self_model.jsonl.gz"]},
        output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"},
        uncertainty="2000 paired seed bootstrap samples; descriptive unadjusted 95% intervals, all contrasts exploratory",
        data_contract="Complete per-step scalar diagnostics; channel-level records at t=47,48,49,63 for failures. Full trajectories reproducible from frozen source, saved seeds, intervention definitions and original environment. Empty CSV values mean undefined, not zero.")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return aggregates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results_study2/diagnosis")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.output, args.overwrite)
