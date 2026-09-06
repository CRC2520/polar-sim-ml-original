#!/usr/bin/env python3
"""Run pilot or prospectively registered final Study 2, preserving raw traces."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import platform
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study2.controllers import ControllerConfig, CoupledController, project_action
from study2.environments import EnvironmentConfig, build_environment, observation, feedback, transition_details
from study2.evaluation import plain, read_json, write_json, sha256, source_hashes, regenerate, pack_artifacts


def run_trial(environment, controller_name, seed):
    controller = None if controller_name in ("zero", "hold") else CoupledController(ControllerConfig(mode=controller_name, seed=seed))
    state = np.asarray(environment["initial_state"], float)
    records = []
    for frame in environment["frames"]:
        obs = observation(frame, state)
        start = time.perf_counter()
        if controller is None:
            proposal = np.zeros(8) if controller_name == "zero" else np.full(8, .3)
            action = project_action(proposal, np.asarray(obs["costs"]), obs["budget"], np.asarray(obs["allowed"], bool))
        else:
            # Strict boolean arrays at API boundary, including JSON-origin masks.
            obs["allowed"] = np.asarray(obs["allowed"], bool)
            action = controller.act(obs)
        action_seconds = time.perf_counter()-start
        details = transition_details(frame, state, action)
        start = time.perf_counter()
        fb = feedback(frame, details["state"], details["transition_valid"])
        if controller is not None:
            controller.learn(fb)
        learning_seconds = time.perf_counter()-start
        record = dict(step=frame["step"], action=action, state_before=state,
                      state_after=details["state"], transition_valid=details["transition_valid"],
                      action_seconds=action_seconds, learning_seconds=learning_seconds,
                      mechanism=controller.last_trace if controller is not None else dict(observation=obs, feedback=fb, fixed_action_level=0. if controller_name == "zero" else .3))
        records.append(plain(record))
        state = details["state"]
    return dict(schema="polar-study2-controller-trace-1", controller=controller_name, seed=seed,
                family=environment["config"]["family"], regime=environment["config"]["regime"],
                config=asdict(controller.config) if controller is not None else dict(mode=controller_name, action_level=0. if controller_name == "zero" else .3),
                profile=controller.profile if controller is not None else dict(estimator_coefficients=0, planner_steps=0, scope="fixed negative control with shared action constraints"),
                records=records)


def make_freeze(path, pilot_manifest_path, reviewed_by):
    if not reviewed_by.strip():
        raise ValueError("A named prospective review is required before freezing")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Freeze already exists; do not silently replace prospective registration")
    pilot_manifest_path = Path(pilot_manifest_path)
    manifest = read_json(pilot_manifest_path)
    current_hashes = source_hashes(ROOT)
    if manifest["split"] != "pilot" or manifest["source_sha256"] != current_hashes:
        raise ValueError("Freeze requires pilot produced by the exact current code and protocol")
    report = regenerate(pilot_manifest_path, pilot_manifest_path.parent/"generated")
    frozen = dict(schema="polar-study2-freeze-1", created_utc=datetime.now(timezone.utc).isoformat(),
                  reviewed_by=reviewed_by, source_sha256=current_hashes,
                  protocol_sha256=sha256(ROOT/"docs/study2_protocol.json"),
                  pilot_manifest_sha256=sha256(pilot_manifest_path), sample_size=report["sample_size"],
                  source_registration="Root will commit these sources and this freeze to public GitHub before the first final run.",
                  final_first_generation=None, modifications_after_final="No tuning, threshold changes, sample expansion or optional stopping allowed.")
    write_json(path, frozen)
    return frozen


def run(split, output, freeze_path=None, registration_sha=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output already contains data. Runs never overwrite pilot or final evidence.")
    protocol = read_json(ROOT/"docs/study2_protocol.json")
    hashes = source_hashes(ROOT)
    frozen = None
    if split == "final":
        if not registration_sha or not re.fullmatch(r"[0-9a-f]{40}", registration_sha):
            raise ValueError("Final mode requires the prior public registration commit SHA (40 hex characters)")
        if freeze_path is None:
            raise ValueError("Final mode requires reviewed frozen protocol/code")
        frozen = read_json(freeze_path)
        if not frozen.get("reviewed_by") or frozen["source_sha256"] != hashes or frozen["protocol_sha256"] != sha256(ROOT/"docs/study2_protocol.json"):
            raise ValueError("Current source/protocol differs from reviewed freeze")
        seeds = frozen["sample_size"]["final_seeds"]
        if set(seeds) & set(protocol["pilot_seeds"]) or len(seeds) != frozen["sample_size"]["chosen_n"]:
            raise ValueError("Invalid frozen final seed allocation")
    elif split == "pilot":
        seeds = protocol["pilot_seeds"]
    else:
        raise ValueError("Unknown split")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest = dict(schema="polar-study2-manifest-1", split=split, seeds=seeds, protocol=protocol,
                    source_sha256=hashes, registration_sha=registration_sha, freeze=frozen,
                    started_utc=started, python=platform.python_version(), numpy=np.__version__,
                    platform=platform.platform(), machine=platform.machine(),
                    thread_policy="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 recommended and used for recorded runs",
                    trials=[])
    write_json(output/"RUN_STARTED.json", {k:v for k,v in manifest.items() if k != "trials"})
    total = time.perf_counter()
    for i, seed in enumerate(seeds):
        for family in protocol["families"]:
            for regime in protocol["regimes"]:
                environment = build_environment(EnvironmentConfig(family, regime, seed, steps=protocol["steps"], phase_length=protocol["phase_length"]))
                stem = f"{seed}_{family}_{regime}"
                environment_rel = f"environments/{stem}.json.gz"
                write_json(output/environment_rel, environment)
                environment_sha = sha256(output/environment_rel)
                # Rotate run order by seed/cell to reduce systematic timing order.
                names = protocol["controllers"]
                offset = (seed + protocol["families"].index(family) + protocol["regimes"].index(regime)) % len(names)
                order = names[offset:] + names[:offset]
                for controller in order:
                    trace = run_trial(environment, controller, seed)
                    trace_rel = f"traces/{controller}/{stem}.json.gz"
                    write_json(output/trace_rel, trace)
                    manifest["trials"].append(dict(seed=seed, family=family, regime=regime, controller=controller,
                                                  environment_path=environment_rel, environment_sha256=environment_sha,
                                                  trace_path=trace_rel, trace_sha256=sha256(output/trace_rel)))
        print(f"{split}: {i+1}/{len(seeds)} seeds complete; elapsed {time.perf_counter()-total:.1f}s", flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["run_wall_seconds"] = time.perf_counter()-total
    logical_paths = [item[key] for item in manifest["trials"] for key in ("environment_path", "trace_path")]
    manifest.update(pack_artifacts(output, logical_paths))
    write_json(output/"manifest.json", manifest)
    result = regenerate(output/"manifest.json", output/"generated")
    print(f"Regenerated {len(manifest['trials'])} complete trials: {result['decision']}", flush=True)
    if split == "pilot":
        print(f"Pilot-derived sample size: {result['sample_size']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze", type=Path, default=ROOT/"docs/STUDY2_FREEZE.json")
    parser.add_argument("--registration-sha")
    parser.add_argument("--make-freeze", type=Path)
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--reviewed-by", default="")
    args = parser.parse_args()
    if args.make_freeze:
        if args.pilot_manifest is None:
            parser.error("--make-freeze requires --pilot-manifest")
        frozen = make_freeze(args.make_freeze, args.pilot_manifest, args.reviewed_by)
        print(f"Frozen {frozen['sample_size']['chosen_n']} final seeds; root must now publish this registration before running final.")
    else:
        run(args.split, args.out or ROOT/f"results_study2/coupling/{args.split}", args.freeze, args.registration_sha)


if __name__ == "__main__":
    main()
