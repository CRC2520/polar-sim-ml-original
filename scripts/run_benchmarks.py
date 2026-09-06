#!/usr/bin/env python3
"""Execute frozen protocol once: python scripts/run_benchmarks.py --output results/contextual_v2_2"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from polar.baselines import make_controller, intervene
from polar.tasks import make_trial, observation, jsonable
from polar.evaluation import summarize_run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(stream, value):
    stream.write(json.dumps(jsonable(value), separators=(",", ":"), allow_nan=False) + "\n")


def run(output, protocol_path, smoke=False):
    protocol = json.loads(protocol_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; preserve previous runs and use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "controllers").mkdir()
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    manifest = dict(protocol_sha256=sha(protocol_path), smoke=smoke,
                    python=platform.python_version(), numpy=np.__version__, started_unix=time.time(),
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for group in ["polar/*.py", "scripts/run_benchmarks.py", "scripts/regenerate_reports.py"] for p in sorted(ROOT.glob(group))},
                    trace_schema="environment joined to controller trace by (split,task,seed,t); full precision JSON numbers",
                    tuning="none", model_config=asdict(make_controller("dual_pole", 0).config))
    # Persist protocol AND implementation hashes before first simulation.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    summaries = []
    started = time.monotonic()
    with ExitStack() as stack:
        env_file = stack.enter_context(gzip.open(output/"environment.jsonl.gz", "wt", compresslevel=9))
        streams = {name: stack.enter_context(gzip.open(output/"controllers"/(name+".jsonl.gz"), "wt", compresslevel=9)) for name in protocol["controllers"]}
        for split in ["development", "heldout"]:
            seeds = protocol[split+"_seeds"][:2] if smoke else protocol[split+"_seeds"]
            for task in protocol["tasks"]:
                for seed in seeds:
                    trial = make_trial(task, seed, split, protocol["agents"], protocol["types"], protocol["steps"])
                    key = dict(split=split, task=task, seed=seed)
                    for frame in trial.steps:
                        write_record(env_file, {**key, **frame})
                    for name in protocol["controllers"]:
                        model = make_controller(name, seed, protocol["agents"], protocol["types"])
                        records = []
                        for frame in trial.steps:
                            compute_start = time.perf_counter_ns()
                            event = intervene(model, name, frame, task, seed)
                            action = model.act(observation(frame))
                            effect = np.clip(action * frame["gains"] + frame["noise"], 0., 1.)
                            model.learn({"effect": effect})
                            compute_ns = time.perf_counter_ns()-compute_start
                            trace = model.last_trace
                            record = dict(**key, t=frame["t"], action=action, effect=effect, intervention=event, compute_ns=compute_ns,
                                          mechanisms={k: trace[k] for k in ["workspace", "stability", "constraints", "memory_write"] if k in trace})
                            # q is exactly action; targets, masks and resource rules live in
                            # the shared environment stream. Snapshot latent mechanisms
                            # at every phase end and immediately around targeted lesions.
                            if hasattr(model, "snapshot") and (frame["t"] % 16 in [0, 15] or event):
                                record["state_snapshot"] = model.snapshot()
                            write_record(streams[name], record)
                            records.append(dict(action=action, effect=effect))
                            # Full per-step internal snapshots are reconstructible from
                            # saved inputs/actions/effects and would otherwise duplicate
                            # the same memory contents dozens of times.
                            if hasattr(model, "traces"):
                                model.traces.clear()
                        summary = {**key, "controller": name, **summarize_run(trial.steps, records,
                                   protocol["recovery"]["regret_max"], protocol["recovery"]["sustained_steps"], protocol["success_rule"])}
                        summaries.append(summary)
                print(f"Completed {split}/{task}: {len(seeds)} paired seeds", flush=True)
    (output/"run_summaries.json").write_text(json.dumps(summaries, indent=2, allow_nan=False)+"\n")
    manifest.update(elapsed_seconds=time.monotonic()-started, completed_unix=time.time(),
                    raw_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*.jsonl.gz"))},
                    completed_runs=len(summaries))
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    from regenerate_reports import regenerate
    regenerate(output)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"docs/evaluation_protocol.json")
    parser.add_argument("--smoke", action="store_true", help="Separate verification artifact only: two seeds per split; never paper evidence")
    args = parser.parse_args()
    run(args.output, args.protocol, args.smoke)
