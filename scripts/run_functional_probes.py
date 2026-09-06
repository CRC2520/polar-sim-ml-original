#!/usr/bin/env python3
"""Small causal audits of existing functions. No consciousness classification.

Run once, then regenerate all statistics from the complete archived JSON:
  python scripts/run_functional_probes.py
  python scripts/run_functional_probes.py --regenerate
"""
import argparse
import copy
import gzip
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from polar import ContextualPolarModel, ModelConfig
from polar.workspace import ResourceWorkspace

PROTOCOL = ROOT / "docs/consciousness_protocol.json"


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(target, **extra):
    return {"target": np.asarray(target), "budget": float(np.size(target)), **extra}


def cycle(model, obs, effect=None):
    action = model.act(obs)
    model.learn({"effect": action if effect is None else effect(action)})
    return action


def run_seed(seed, protocol):
    rng = np.random.default_rng(seed)
    cfg = ModelConfig(agents=protocol["agents"], types=protocol["types"], seed=seed,
                      step_size=protocol["step_size"])
    shape = (cfg.agents, cfg.types, 2)
    chosen = rng.integers(0, 2, shape[:-1])
    high = rng.uniform(.65, .85, shape[:-1])
    target = np.full(shape, .05)
    np.put_along_axis(target, chosen[..., None], high[..., None], axis=-1)
    trained = ContextualPolarModel(cfg)
    for _ in range(protocol["memory_learning_steps"]):
        cycle(trained, observation(target, cue="learned"))
    memory = {"training": copy.deepcopy(trained.traces), "target": target.tolist(), "conditions": {}}
    for name in protocol["assays"]["cue_only_memory"]["conditions"]:
        model = copy.deepcopy(trained)
        model.q = np.zeros(shape)
        if name == "erase":
            model.memory.erase("learned")
        if name == "edit_opposite":
            model.memory.edit("learned", target[..., ::-1])
        cue = "unknown" if name == "unknown_cue" else "learned"
        before = model.snapshot()
        cycle(model, observation(np.zeros(shape), observed=np.zeros(shape, bool), cue=cue))
        memory["conditions"][name] = {"before": before, "trace": model.last_trace}

    # Clamp q and set horizon zero to replay precisely the same own actions.
    # The intended effect source is external ground truth, not an input the
    # current capability estimator can use to distinguish self from another.
    own_gain = rng.uniform(.55, .75, shape)
    foreign_gain = 2 - own_gain
    intact, contaminated = ContextualPolarModel(cfg), ContextualPolarModel(cfg)
    replay_actions = rng.uniform(.15, .35, (protocol["capability_learning_steps"], *shape))
    for replay in replay_actions:
        for model, gain in ((intact, own_gain), (contaminated, foreign_gain)):
            model.q = replay
            obs = observation(np.zeros(shape), observed=np.zeros(shape, bool), horizon=0.)
            cycle(model, obs, lambda action, gain=gain: action * gain)
    reset = copy.deepcopy(intact)
    reset.self_model.lesion(gain=1.)
    capability = {"own_gain": own_gain.tolist(), "foreign_gain": foreign_gain.tolist(),
                  "replay_actions": replay_actions.tolist(),
                  "training": {"intact": copy.deepcopy(intact.traces),
                               "foreign_feedback": copy.deepcopy(contaminated.traces)},
                  "conditions": {}}
    # Fixed action identifies prediction differences independently of planner differences.
    for name, model in (("intact", intact), ("reset", reset), ("foreign_feedback", contaminated)):
        replay_probe = copy.deepcopy(model)
        replay_probe.q = np.full(shape, .25)
        cycle(replay_probe, observation(np.zeros(shape), observed=np.zeros(shape, bool), horizon=0.),
              lambda action: action * own_gain)
        model.q = np.zeros(shape)
        before = model.snapshot()
        desired = np.full(shape, .3)
        cycle(model, observation(desired), lambda action: action * own_gain)
        capability["conditions"][name] = {"before": before,
            "prediction_probe": replay_probe.last_trace,
            "decision_target": desired.tolist(), "decision_probe": model.last_trace}

    proposals = rng.uniform(.05, .6, shape)
    permuted = np.roll(proposals.reshape(cfg.agents, -1), 3, axis=1).reshape(shape)
    allocator = ResourceWorkspace(cfg.agents, seed)
    allocator_results = {}
    for name, proposed in (("original", proposals), ("content_permuted", permuted)):
        allocations, trace = allocator.allocate(proposed, np.ones(shape), np.ones(shape), 3., np.ones(shape))
        allocator_results[name] = {"proposal": proposed.tolist(), "trace": trace}
    return {"seed": seed, "memory": memory, "capability": capability, "allocator": allocator_results}


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def summarize_seed(record):
    metrics = {"seed": record["seed"]}
    mem = record["memory"]
    for name, result in mem["conditions"].items():
        metrics[f"memory_{name}_target_mse"] = mse(result["trace"]["action"], mem["target"])
    intact = np.asarray(mem["conditions"]["intact"]["trace"]["action"])
    edited = np.asarray(mem["conditions"]["edit_opposite"]["trace"]["action"])
    metrics["memory_edit_flip_max_abs_error"] = float(np.max(np.abs(edited - intact[..., ::-1])))
    for name, result in record["capability"]["conditions"].items():
        probe = result["prediction_probe"]
        metrics[f"self_{name}_prediction_mse"] = mse(probe["effect_predicted"], probe["feedback"]["effect"])
        metrics[f"self_{name}_uncertainty"] = float(np.mean(probe["self_model"]["uncertainty"]))
        metrics[f"self_{name}_decision_mse"] = mse(result["decision_probe"]["feedback"]["effect"], result["decision_target"])
    alloc = record["allocator"]
    metrics["allocator_permutation_max_abs_difference"] = float(np.max(np.abs(
        np.asarray(alloc["original"]["trace"]["allocations"]) -
        np.asarray(alloc["content_permuted"]["trace"]["allocations"]))))
    metrics["allocator_content_mean_abs_difference"] = float(np.mean(np.abs(
        np.asarray(alloc["original"]["proposal"]) - np.asarray(alloc["content_permuted"]["proposal"]))))
    metrics["memory_erasure_mse_increase"] = metrics["memory_erase_target_mse"] - metrics["memory_intact_target_mse"]
    metrics["self_reset_prediction_mse_increase"] = metrics["self_reset_prediction_mse"] - metrics["self_intact_prediction_mse"]
    metrics["self_foreign_prediction_mse_increase"] = metrics["self_foreign_feedback_prediction_mse"] - metrics["self_intact_prediction_mse"]
    return metrics


def regenerate(out):
    with gzip.open(out / "traces.json.gz", "rt") as handle:
        artifact = json.load(handle)
    protocol = artifact["protocol"]
    records = [summarize_seed(record) for record in artifact["records"]]
    if [r["seed"] for r in records] != protocol["seeds"]:
        raise ValueError("incomplete or reordered functional probe seed set")
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    bootstrap = rng.integers(0, len(records), (protocol["bootstrap_draws"], len(records)))
    summary = {"status": "functional_diagnostic_only", "seeds": protocol["seeds"],
               "n_seeds": len(records), "interval": "descriptive_seed_bootstrap_95_percent_unadjusted", "metrics": {}}
    for key in records[0]:
        if key == "seed":
            continue
        values = np.asarray([r[key] for r in records])
        means = values[bootstrap].mean(axis=1)
        summary["metrics"][key] = {"mean": float(values.mean()),
            "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)),
            "min": float(values.min()), "max": float(values.max())}
    for filename, obj in (("per_seed.json", records), ("summary.json", summary)):
        (out / filename).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    labels = [
        ("Memory: intact recall MSE", "memory_intact_target_mse"),
        ("Memory: erased recall MSE", "memory_erase_target_mse"),
        ("Memory: erasure MSE increase", "memory_erasure_mse_increase"),
        ("Prediction: intact MSE", "self_intact_prediction_mse"),
        ("Prediction: reset MSE", "self_reset_prediction_mse"),
        ("Prediction: foreign feedback MSE", "self_foreign_feedback_prediction_mse"),
        ("Prediction: foreign feedback MSE increase", "self_foreign_prediction_mse_increase"),
        ("Allocator: content-permutation allocation difference", "allocator_permutation_max_abs_difference")]
    lines = ["# Functional mechanism audit", "", "Exploratory synthetic diagnostics, not a consciousness assessment.", "",
             f"{len(records)} paired seeds; all intervals are descriptive 95% seed bootstrap intervals, unadjusted.", "",
             "| Metric | Mean | Interval |", "|---|---:|---:|"]
    tex = [r"\begin{tabular}{p{0.56\linewidth}rr}", r"\toprule", r"Functional diagnostic & Mean & 95\% interval \\", r"\midrule"]
    for label, key in labels:
        m = summary["metrics"][key]
        lines.append(f"| {label} | {m['mean']:.6f} | [{m['ci_low']:.6f}, {m['ci_high']:.6f}] |")
        tex.append(f"{label} & {m['mean']:.6f} & [{m['ci_low']:.6f}, {m['ci_high']:.6f}] " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    lines.extend(["", "Memory edits reverse recalled channel orientation; erasure and an unknown cue isolate cue-addressed storage from residual action state. The capability estimator learns an action/effect gain and can improve prediction; deliberately supplying another source's effects corrupts it. The model does not infer effect provenance. The resource allocator responds to aggregate weighted demand, not to broadcast content identity.", "", "The uncertainty variable is a count/residual heuristic, not calibrated probability. No global-content access test, phenomenal-experience test, neural validation, or self/other attribution mechanism is claimed.", "", "Regenerate: `python scripts/run_functional_probes.py --regenerate`. Full inputs, outputs, replay schedules, model states and interventions are in `traces.json.gz`; software and source hashes are in `manifest.json`."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "functional_results.tex").write_text("\n".join(tex) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_study2/functional")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.regenerate:
        protocol = json.loads(PROTOCOL.read_text())
        artifact = {"protocol": protocol, "records": [run_seed(seed, protocol) for seed in protocol["seeds"]]}
        data = json.dumps(artifact, default=json_value, separators=(",", ":"), allow_nan=False).encode()
        (args.out / "traces.json.gz").write_bytes(gzip.compress(data, mtime=0))
    summary = regenerate(args.out)
    if not args.regenerate:
        sources = [PROTOCOL, Path(__file__), *(ROOT / "polar").glob("*.py")]
        manifest = {"python": platform.python_version(), "numpy": np.__version__,
                    "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sorted(sources)},
                    "artifact_sha256": {path.name: digest(path) for path in sorted(args.out.iterdir()) if path.is_file() and path.name != "manifest.json"},
                    "run_command": "python scripts/run_functional_probes.py",
                    "regeneration_command": "python scripts/run_functional_probes.py --regenerate",
                    "status": "exploratory_functional_audit_not_consciousness_validation"}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"n_seeds": summary["n_seeds"], "output": str(args.out), "status": summary["status"]}))


if __name__ == "__main__":
    main()
