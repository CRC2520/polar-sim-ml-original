"""P2 matched-state history transplantation and fixed-policy transfer.

Only bridge-v2 is evaluated; legacy results remain unmodified.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import tempfile
import platform
import time
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "polar-bridge-p2-v1"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def paired_interval(a, b, *, confidence=0.95, draws=10000, bootstrap_seed=914270):
    """Resample whole paired seed effects, never individual agents/time steps."""
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if aa.shape != bb.shape or aa.ndim != 1 or len(aa) < 2:
        raise ValueError("paired finite one-dimensional arrays with >=2 seeds required")
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        raise ValueError("missing/non-finite outcomes cannot be interpreted as zero")
    delta = aa - bb
    rng = np.random.default_rng(bootstrap_seed)
    boots = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    q = (1.0 - confidence) / 2.0
    return {"n_seeds": len(delta), "mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
            "difference_a_minus_b": float(delta.mean()), "ci_level": confidence,
            "ci": np.quantile(boots, [q, 1-q]).tolist(),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "inference": "descriptive paired seed bootstrap; no multiplicity-adjusted confirmation"}


def save_gzip_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    # mtime=0 makes the artifact deterministic for replay/integrity checks.
    path.write_bytes(gzip.compress(payload, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()

from .engine import BridgeConfig, BridgeEngine

FINAL_SEEDS = tuple(range(920001, 920031))
PILOT_SEEDS = (919001, 919002, 919003)
TRAIN_EPISODES = 6
TRAIN_STEPS = 240
PROBE_STEPS = 240
CONTROLLERS = ("full", "generic_shared_target", "coordinate_equivalent")
METRICS = ("alive_fraction", "resource_fraction", "restraint_behavior", "action_mean", "viable_groups")
HISTORY_TREATMENTS = {
    "baseline": (), "sham": (),
    "q_only": ("qt", "qv", "qg"), "counts_only": ("visits",),
    "controller": ("qt", "qv", "qg", "visits"),
    "ecology": ("resource_history",), "lineage": ("lineage",),
    "controller_ecology": ("qt", "qv", "qg", "visits", "resource_history"),
    "all_history": ("qt", "qv", "qg", "visits", "resource_history", "lineage"),
}
COMPACT_FIELDS = ("action", "resource_after", "alive_after")


def p2_config(**changes):
    # Population matches the bridge-v2 study. Probe horizon is deliberately bounded.
    return replace(BridgeConfig(), generations=TRAIN_EPISODES, tail_generations=2,
                   steps=TRAIN_STEPS, **changes)


def manifest():
    return {"schema": SCHEMA_VERSION, "inference": "exploratory descriptive paired-seed study",
            "final_seeds": list(FINAL_SEEDS), "pilot_seeds": list(PILOT_SEEDS),
            "config": asdict(p2_config()), "train_episodes": TRAIN_EPISODES,
            "train_steps": TRAIN_STEPS, "probe_steps": PROBE_STEPS,
            "history_ecologies": {"low": {"growth_mean": .12, "growth_amplitude": .035},
                                  "high": {"growth_mean": .22, "growth_amplitude": .035}},
            "history_treatments": HISTORY_TREATMENTS,
            "matching": {"resource_fraction": .5, "vitality": 5., "alive": True,
                         "initial_composition": .5, "future_generation_key": 100},
            "transfer_controllers": CONTROLLERS,
            "transfer_targets": {"source": {}, "low_growth": {"growth_mean": .12, "growth_amplitude": .035},
                                 "high_metabolism": {"metabolism": .42},
                                 "resource_shock": {"initial_resource_fraction": .35}},
            "transfer_modes": {"zero_shot": False, "adaptive": True},
            "transfer_memory": ["trained", "untrained_reset"],
            "primary_metric": "alive_fraction", "metrics": METRICS,
            "uncertainty": "paired seed bootstrap, 10000 draws, 95%; no confirmatory multiplicity claim",
            "trace_schema": {"saved_fields": COMPACT_FIELDS, "initial_state": "JSON recipes + content-addressed exact NPZ arrays",
                             "randomness": "keyed seed/generation/stream tapes; no hidden RNG",
                             "Q_trajectory": "reconstructed from initial state, policy and deterministic replay"},
            "consciousness_claim": False}


def outcome_metrics(outcome, config):
    result = {key: np.asarray(outcome[key], dtype=float) for key in METRICS if key != "viable_groups"}
    result["viable_groups"] = ((outcome["group_alive"] >= config.viable_alive) &
                               (outcome["group_resource"] >= config.viable_resource)).mean(axis=1)
    if any(not np.all(np.isfinite(x)) for x in result.values()):
        raise ValueError("P2 outcome contains missing/non-finite measurements")
    return result


def compact_trace(raw_path: Path, *, initial_snapshot, final_state_hash, output_path=None):
    """Keep exact observed trajectories; state dynamics can be fully replayed.

    Tables are not redundantly stored at every transition. This is explicitly a
    compact replayable trace, not a claim to contain every intermediate tensor.
    """
    with np.load(raw_path, allow_pickle=False) as raw:
        kept = {key: raw[key].copy() for key in COMPACT_FIELDS}
        for key in ("action",):
            kept[key] = kept[key].astype(np.uint8)
        meta = json.loads(str(raw["metadata"]))
        meta.update(schema=SCHEMA_VERSION, initial_snapshot=initial_snapshot,
                    final_state_hash=final_state_hash, stored_fields=COMPACT_FIELDS)
        kept["metadata"] = np.array(json.dumps(meta, sort_keys=True))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **kept)
    payload = buffer.getvalue()
    with np.load(io.BytesIO(payload), allow_pickle=False) as check:
        for key in COMPACT_FIELDS:
            if not np.array_equal(check[key], kept[key]):
                raise IOError("NPZ integrity failed before atomic publication")
    target = raw_path if output_path is None else Path(output_path)
    staging = target.with_suffix(".npz.atomic")
    staging.write_bytes(payload)
    staging.replace(target)
    return hashlib.sha256(payload).hexdigest()


def save_shared_snapshot(engine, path, generation):
    """Deduplicate exact arrays across conditions; recipes retain complete state."""
    path = Path(path)
    shared = path.parent.parent / "state_arrays"
    shared.mkdir(parents=True, exist_ok=True)
    refs = {}
    for key, value in engine.state.items():
        a = np.ascontiguousarray(value)
        header = json.dumps({"dtype": a.dtype.str, "shape": a.shape}, sort_keys=True).encode()
        digest = hashlib.sha256(header + a.tobytes()).hexdigest()
        target = shared / f"{digest}.npz"
        if not target.exists():
            buffer = io.BytesIO()
            np.savez_compressed(buffer, value=a)
            staging = target.with_suffix(".npz.atomic")
            staging.write_bytes(buffer.getvalue())
            staging.replace(target)
        refs[key] = str(target.relative_to(path.parent.parent))
    meta = dict(config=asdict(engine.config), seeds=engine.seeds.tolist(),
                initial_fraction=engine.initial_fraction, study=engine.study,
                variant=engine.variant, generation=generation, state_arrays=refs)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def load_shared_snapshot(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    engine = BridgeEngine(BridgeConfig(**meta["config"]), meta["seeds"],
                          meta["initial_fraction"], meta["study"], meta["variant"])
    for key, relative in meta["state_arrays"].items():
        with np.load(path.parent.parent / relative, allow_pickle=False) as stored:
            engine.state[key] = stored["value"].copy()
    return engine, meta


def perform_episode(engine, directory, name, *, generation, learn, steps, reset_physical=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / f"{name}_initial.json"
    trace = directory / f"{name}_trace.npz"
    if reset_physical:
        engine.reset_physical()
    save_shared_snapshot(engine, snapshot, generation)
    with tempfile.TemporaryDirectory(prefix="polar_p2_raw_") as raw_dir:
        raw = Path(raw_dir) / "trace.npz"
        result = engine.episode(generation, reset_physical=False, steps=steps, learn=learn, trace_path=raw, trace_mode="full")
        state_hash = canonical_hash(engine.state)
        trace_hash = compact_trace(raw, initial_snapshot=snapshot.name, final_state_hash=state_hash, output_path=trace)
    return {"name": name, "generation": generation, "learn": learn, "steps": steps,
            "initial_snapshot": str(snapshot.relative_to(directory.parent)),
            "trace": str(trace.relative_to(directory.parent)), "trace_sha256": trace_hash,
            "final_state_sha256": state_hash, "metrics": jsonable(outcome_metrics(result, engine.config))}


def train(config, seeds, controller, directory, prefix):
    engine = BridgeEngine(config, seeds, .5, study="C", variant=controller)
    records = []
    for generation in range(TRAIN_EPISODES):
        record = perform_episode(engine, directory, f"{prefix}_train_{generation:02d}",
                                 generation=generation, learn=True, steps=TRAIN_STEPS, reset_physical=True)
        # No copying during these assays: isolate acquired local-state history.
        records.append(record)
    return engine, records


def matched_history(engine, config=None):
    result = engine.clone()
    result.config = p2_config() if config is None else config
    result.state["resource"][:] = .5 * result.capacity
    result.state["vitality"][:] = result.config.initial_vitality
    result.state["alive"][:] = True
    result.last_episode = None
    return result


def transplant(recipient, donor, keys):
    if not np.array_equal(recipient.seeds, donor.seeds):
        raise ValueError("A transplant requires paired identical future seed keys")
    for name in ("resource", "vitality", "alive"):
        if not np.array_equal(recipient.state[name], donor.state[name]):
            raise ValueError(f"Present state is not matched: {name}")
    result = recipient.clone()
    for key in keys:
        if key not in ("qt", "qv", "qg", "visits", "resource_history", "lineage"):
            raise ValueError(f"Not an independently swappable historical field: {key}")
        result.state[key] = donor.state[key].copy()
    return result


def trajectory_equal(path_a, path_b):
    with np.load(path_a, allow_pickle=False) as aa, np.load(path_b, allow_pickle=False) as bb:
        return all(np.array_equal(aa[k], bb[k], equal_nan=True) for k in COMPACT_FIELDS)


def run_history(seeds, root):
    directory = Path(root) / "history"
    histories, training = {}, []
    for label, mean in (("low", .12), ("high", .22)):
        engine, records = train(p2_config(growth_mean=mean, growth_amplitude=.035), seeds,
                                "full", directory, label)
        histories[label] = matched_history(engine)
        training.extend(records)
    # Genealogical labels are inert after initialization in C. Permute one donor's
    # labels within groups, preserving exact composition, for a nontrivial negative control.
    histories["high"].state["lineage"] = np.roll(histories["high"].state["lineage"], 1, axis=-1)
    records = []
    for recipient_label, donor_label in (("low", "high"), ("high", "low")):
        for treatment, keys in HISTORY_TREATMENTS.items():
            engine = transplant(histories[recipient_label], histories[donor_label], keys)
            record = perform_episode(engine, directory, f"{recipient_label}_{treatment}",
                                     generation=100, learn=True, steps=PROBE_STEPS)
            record.update(recipient_history=recipient_label, donor_history=donor_label,
                          treatment=treatment, fields_swapped=list(keys))
            records.append(record)
    checks = []
    for recipient, donor in (("low", "high"), ("high", "low")):
        for control, expected in (("sham", f"{recipient}_baseline"),
                                  ("lineage", f"{recipient}_baseline"),
                                  ("all_history", f"{donor}_baseline"),
                                  ("controller_ecology", f"{donor}_baseline")):
            equal = trajectory_equal(directory / f"{recipient}_{control}_trace.npz", directory / f"{expected}_trace.npz")
            checks.append({"case": f"{recipient}_{control}", "expected": expected, "exact_trace_match": equal})
    if not all(check["exact_trace_match"] for check in checks):
        raise AssertionError("History sham, inert-lineage or donor-rescue invariant failed")
    return {"training": training, "records": records, "checks": checks,
            "lineage_note": "high-history labels rolled within groups after training, preserving composition; negative control only",
            "transmission": "disabled during training and probes to isolate local learned-state and ecological memory"}


def transfer_start(trained, config, memory):
    engine = trained.clone()
    engine.config = config
    engine.reset_physical()
    engine.last_episode = None
    if memory == "untrained_reset":
        fresh = BridgeEngine(config, engine.seeds, .5, study="C", variant=engine.variant)
        for key in ("qt", "qv", "qg", "visits"):
            engine.state[key] = fresh.state[key].copy()
    return engine


def run_transfer(seeds, root):
    directory = Path(root) / "transfer"
    training, records = [], []
    targets = manifest()["transfer_targets"]
    for controller in CONTROLLERS:
        trained, source_records = train(p2_config(), seeds, controller, directory, controller)
        training.extend(source_records)
        for target, changes in targets.items():
            config = p2_config(**changes)
            for memory in ("trained", "untrained_reset"):
                for mode, learn in (("zero_shot", False), ("adaptive", True)):
                    engine = transfer_start(trained, config, memory)
                    record = perform_episode(engine, directory, f"{controller}_{target}_{memory}_{mode}",
                                             generation=200, learn=learn, steps=PROBE_STEPS)
                    record.update(controller=controller, target=target, memory=memory, mode=mode)
                    records.append(record)
    return {"training": training, "records": records,
            "training_scope": "six 240-step local-learning episodes; no transmission; same source for all variants",
            "zero_shot_scope": "Q and visitation counts remain frozen for all 240 target steps"}


def summarize_history(history):
    indexed = {(r["recipient_history"], r["treatment"]): r for r in history["records"]}
    contrasts = []
    for recipient in ("low", "high"):
        for treatment in HISTORY_TREATMENTS:
            if treatment == "baseline":
                continue
            for metric in METRICS:
                contrast = paired_interval(indexed[recipient, treatment]["metrics"][metric],
                                           indexed[recipient, "baseline"]["metrics"][metric])
                contrasts.append(dict(recipient_history=recipient, treatment=treatment, metric=metric, **contrast))
    interaction = []
    for recipient in ("low", "high"):
        for metric in METRICS:
            values = {k: np.asarray(indexed[recipient, k]["metrics"][metric])
                      for k in ("baseline", "controller", "ecology", "controller_ecology")}
            effect = values["controller_ecology"] - values["controller"] - values["ecology"] + values["baseline"]
            interaction.append(dict(recipient_history=recipient, metric=metric,
                                    **paired_interval(effect, np.zeros_like(effect))))
    # The first-decision effect is separately extracted from exact saved actions.
    return {"contrasts": contrasts, "controller_by_ecological_history_interaction": interaction,
            "checks": history["checks"], "claim": "bounded history dependence; not proof of hysteresis"}


def summarize_transfer(transfer):
    index = {(r["controller"], r["target"], r["memory"], r["mode"]): r for r in transfer["records"]}
    comparisons = []
    for target in manifest()["transfer_targets"]:
        for mode in ("zero_shot", "adaptive"):
            for a, b, kind in (("full", "generic_shared_target", "polar_minus_generic"),
                               ("full", "coordinate_equivalent", "equivalent_coordinates")):
                for metric in METRICS:
                    comparisons.append(dict(target=target, mode=mode, comparison=kind, metric=metric,
                        **paired_interval(index[a, target, "trained", mode]["metrics"][metric],
                                          index[b, target, "trained", mode]["metrics"][metric])))
            for controller in CONTROLLERS:
                for metric in METRICS:
                    comparisons.append(dict(target=target, mode=mode, controller=controller,
                        comparison="trained_minus_untrained_reset", metric=metric,
                        **paired_interval(index[controller, target, "trained", mode]["metrics"][metric],
                                          index[controller, target, "untrained_reset", mode]["metrics"][metric])))
    return {"comparisons": comparisons,
            "claim": "transfer to declared target distributions only; no universal/general consciousness claim"}


def first_decision_summary(history, root):
    first = {}
    for record in history["records"]:
        with np.load(Path(root) / record["trace"], allow_pickle=False) as trace:
            first[record["recipient_history"], record["treatment"]] = trace["action"][:, 0].mean(axis=(1, 2))
    result = []
    for recipient in ("low", "high"):
        for treatment in HISTORY_TREATMENTS:
            if treatment != "baseline":
                result.append(dict(recipient_history=recipient, treatment=treatment,
                    **paired_interval(first[recipient, treatment], first[recipient, "baseline"])))
    return result


def metrics_from_trace(root, record):
    """Regenerate all reported endpoints directly from compact exact traces."""
    root = Path(root)
    engine, _ = load_shared_snapshot(root / record["initial_snapshot"])
    with np.load(root / record["trace"], allow_pickle=False) as data:
        alive_after = data["alive_after"].astype(bool)
        alive_before = np.concatenate((engine.state["alive"][:, None], alive_after[:, :-1]), axis=1)
        actions = data["action"]
        resource = data["resource_after"]
    denominator = alive_before.sum(axis=(1, 2, 3))
    if np.any(denominator == 0):
        raise ValueError("Undefined action statistics in fully ineligible trace")
    group_alive = alive_after.mean(axis=(1, 3))
    group_resource = resource.mean(axis=1) / engine.capacity
    return {"alive_fraction": alive_after.mean(axis=(1, 2, 3)),
            "resource_fraction": resource.mean(axis=(1, 2)) / engine.capacity,
            "restraint_behavior": ((actions <= 1) & alive_before).sum(axis=(1, 2, 3)) / denominator,
            "action_mean": actions.sum(axis=(1, 2, 3)) / denominator,
            "viable_groups": ((group_alive >= engine.config.viable_alive) &
                              (group_resource >= engine.config.viable_resource)).mean(axis=1)}


def verify_all_endpoints(root, results):
    records = (results["history"]["training"] + results["history"]["records"] +
               results["transfer"]["training"] + results["transfer"]["records"])
    max_error = 0.
    for record in records:
        measured = metrics_from_trace(root, record)
        for key in METRICS:
            error = float(np.max(np.abs(np.asarray(record["metrics"][key]) - measured[key])))
            max_error = max(max_error, error)
            if error > 1e-12:
                raise AssertionError(f"Stored endpoint does not match trace: {record['name']} {key}")
    return {"cases": len(records), "all_metrics_regenerated_from_traces": True,
            "absolute_tolerance": 1e-12, "maximum_absolute_error": max_error}


def replay_case(root, record):
    """Reload saved state; compare every stored transition and complete end state."""
    root = Path(root)
    engine, meta = load_shared_snapshot(root / record["initial_snapshot"])
    temporary = root / "replay_temporary.npz"
    outcome = engine.episode(record["generation"], reset_physical=False, steps=record["steps"],
                            learn=record["learn"], trace_path=temporary, trace_mode="full")
    with np.load(temporary, allow_pickle=False) as replay, np.load(root / record["trace"], allow_pickle=False) as stored:
        exact = all(np.array_equal(replay[k], stored[k], equal_nan=True) for k in COMPACT_FIELDS)
    temporary.unlink()
    state_equal = canonical_hash(engine.state) == record["final_state_sha256"]
    metrics_equal = canonical_hash(outcome_metrics(outcome, engine.config)) == canonical_hash(record["metrics"])
    return {"name": record["name"], "exact_trace": exact, "exact_final_state": state_equal, "exact_metrics": metrics_equal}


def run_replay(root, results):
    # Fixed case selection, established before results: history baseline/transplants,
    # all controller variants, both transfer-learning modes, first training episode.
    selected = [r for r in results["history"]["records"] if r["treatment"] in ("baseline", "q_only", "ecology", "all_history")]
    selected += [r for r in results["transfer"]["records"]
                 if r["target"] == "low_growth" and r["memory"] == "trained"]
    selected += [results["history"]["training"][0], results["transfer"]["training"][0]]
    endpoint_check = verify_all_endpoints(root, results)
    checks = [replay_case(root, record) for record in selected]
    if not all(c["exact_trace"] and c["exact_final_state"] and c["exact_metrics"] for c in checks):
        raise AssertionError("Deterministic reconstruction failed")
    return {"scope": "within-project computational reproduction; not an external laboratory replication",
            "checks": checks, "all_exact": True, "endpoint_reconstruction": endpoint_check}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final", "replay", "manifest"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration-commit", default=None)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "manifest":
        target = args.output / "P2_IMPLEMENTATION_MANIFEST.json"
        target.write_text(json.dumps(jsonable(manifest()), indent=2, sort_keys=True) + "\n")
        print(target)
        return
    if args.phase == "replay":
        with gzip.open(args.output / "P2_RESULTS.json.gz", "rt") as handle:
            results = json.load(handle)
        replay = run_replay(args.output, results)
        (args.output / "P2_REPLAY.json").write_text(json.dumps(replay, indent=2) + "\n")
        print(json.dumps(replay))
        return
    if args.phase == "final":
        if args.registration_commit is None or len(args.registration_commit) != 40 or any(c not in "0123456789abcdef" for c in args.registration_commit):
            parser.error("final execution requires the public pre-outcome registration commit SHA")
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    started = time.perf_counter()
    results = {"phase": args.phase, "started_utc": datetime.now(timezone.utc).isoformat(),
               "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "manifest": manifest(), "seeds": list(seeds),
               "registration_commit": args.registration_commit,
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("engine.py"))}}
    print(f"Starting P2 {args.phase}, {len(seeds)} paired seeds", flush=True)
    results["history"] = run_history(seeds, args.output)
    print("History interventions completed", flush=True)
    results["transfer"] = run_transfer(seeds, args.output)
    print("Transfer probes completed", flush=True)
    results["summary"] = {"history": summarize_history(results["history"]),
                          "transfer": summarize_transfer(results["transfer"])}
    results["summary"]["history"]["first_decision_action_contrasts"] = first_decision_summary(results["history"], args.output)
    results["simulation_finished_utc"] = datetime.now(timezone.utc).isoformat()
    results["simulation_runtime_seconds"] = time.perf_counter() - started
    save_gzip_json(args.output / "P2_RESULTS.json.gz", results)
    (args.output / "P2_SUMMARY.json").write_text(json.dumps(jsonable(results["summary"]), indent=2, sort_keys=True) + "\n")
    replay = run_replay(args.output, results)
    (args.output / "P2_REPLAY.json").write_text(json.dumps(replay, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "all_replays_exact": replay["all_exact"],
                      "case_count": len(results["history"]["records"]) + len(results["transfer"]["records"])}))


if __name__ == "__main__":
    main()
