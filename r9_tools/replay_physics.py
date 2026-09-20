"""Replay already saved R9 actions through the recorded physical environment.

No agent, runner, policy, fitting, action selection or new seed generation is
imported here. This audit is intentionally outside the frozen R9 source tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r8_completion.io import atomic_json, sha256, verify_npz
from r9_completion import environments


ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seed_directory(campaign, seed):
    campaign = Path(campaign).resolve()
    candidates = {campaign / f"seed_{seed}", campaign / "data" / f"seed_{seed}"}
    if campaign.name == f"seed_{seed}":
        candidates.add(campaign)
    found = [path for path in candidates if (path / "COMPLETE.json").is_file()]
    _require(len(found) == 1, "Require exactly one already completed, saved seed directory")
    return campaign, found[0]


def _source_proof(campaign, seed_directory, manifest):
    active = Path(environments.__file__).resolve()
    active_hash = sha256(active)
    phase = manifest["metadata"]["phase"]
    if phase == "pilot":
        search = [campaign, seed_directory.parent, seed_directory.parent.parent]
        records = {p / "SOURCE_RECORD.json" for p in search if (p / "SOURCE_RECORD.json").is_file()}
        _require(len(records) == 1, "Pilot replay needs its preserved SOURCE_RECORD.json")
        record_path = records.pop()
        record = json.loads(record_path.read_text())
        expected = record["sources"]["environments.py"]
        snapshot = record_path.parent / "SOURCE_SNAPSHOT" / "environments.py"
        _require(snapshot.is_file() and sha256(snapshot) == expected, "Pilot environment snapshot mismatch")
        proof = {"kind": "pilot_source_snapshot", "record": str(record_path),
                 "record_sha256": sha256(record_path), "snapshot_sha256": sha256(snapshot)}
    elif phase == "final":
        record_path = ROOT / "r9_completion" / "FREEZE_R9.json"
        record = json.loads(record_path.read_text())
        expected = record["source_sha256"]["r9_completion/environments.py"]
        _require(manifest["metadata"]["source"]["freeze_sha256"] == sha256(record_path),
                 "Saved final manifest and prospective freeze differ")
        _require(manifest["metadata"]["seed"] in record["final_seeds"], "Saved seed absent from freeze")
        proof = {"kind": "final_source_freeze", "record": str(record_path),
                 "record_sha256": sha256(record_path)}
    else:
        raise ValueError("Unrecognized saved phase")
    _require(active_hash == expected, "Current physical source differs from the source used for saved data")
    return dict(proof, active_environment_sha256=active_hash, phase=phase)


def _compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape, "Saved and replayed field shapes differ")
    _require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite physical replay field")
    return {"exact": bool(np.array_equal(actual, expected)), "shape": list(actual.shape),
            "max_absolute_difference": float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0.))}


def replay_one(trace_path, metric_path, metadata, seed_directory):
    verified = verify_npz(trace_path)
    with np.load(trace_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in
                  ("action", "observation", "next_observation", "reward", "alive", "reserve", "service", "constraint")}
        if "target" in stored:
            arrays["target"] = stored["target"].copy()
    actions = arrays["action"]
    _require(actions.shape == (environments.HORIZON,) and actions.dtype.kind in "iu", "Need all saved integer actions")
    _require(np.isin(actions, np.arange(4)).all(), "Saved action outside 0..3")
    _require(metadata["measured_steps"] == len(actions), "Saved metadata horizon differs")
    environment = environments.ScalarEnvironment(metadata["domain"])
    observation = environment.reset(metadata["environment_seed"])
    tape_matches = environment.tape_digest() == metadata["environment_tape_sha256"]
    generated = {name: [] for name in arrays if name != "action"}
    terminations = []
    for action in actions:
        before = observation.copy()
        observation, reward, terminated, info = environment.step(int(action))
        factual = {"observation": before, "next_observation": observation,
                   "reward": reward, "alive": info["alive"], "reserve": info["reserve"],
                   "service": info["service"], "constraint": info["constraint_violation"],
                   "target": np.r_[observation, reward]}
        for name in generated:
            generated[name].append(factual[name])
        terminations.append(terminated)
    checks = {name: _compare(values, arrays[name]) for name, values in generated.items()}
    expected_termination = [False] * (len(actions) - 1) + [True]
    termination_matches = terminations == expected_termination
    saved_metrics = json.loads(metric_path.read_text())["metrics"]
    endpoint_checks = {}
    for metric, field in (("reward_mean", "reward"), ("alive_fraction", "alive"),
                          ("reserve_mean", "reserve"), ("service_mean", "service"),
                          ("constraint_fraction", "constraint")):
        endpoint_checks[metric] = _compare(float(np.mean(generated[field])), saved_metrics[metric])
    return {"rollout": str(trace_path.parent.relative_to(seed_directory)), "steps": len(actions),
            "domain": metadata["domain"], "variant": metadata["variant"], "episode": metadata["episode"],
            "environment_seed_from_saved_metadata": metadata["environment_seed"],
            "trace_sha256": verified["sha256"], "metrics_sha256": sha256(metric_path),
            "tape_sha256_matches": tape_matches, "fixed_horizon_termination_matches": termination_matches,
            "field_checks": checks, "endpoint_checks": endpoint_checks,
            "exact": bool(tape_matches and termination_matches and
                          all(value["exact"] for value in (*checks.values(), *endpoint_checks.values())))}


def audit_saved(campaign, seed, output, *, section="evaluation", domains=None, variants=None):
    _require(not Path(output).exists(), "Refusing to overwrite an audit report")
    campaign, seed_directory = _seed_directory(campaign, seed)
    manifest_path = seed_directory / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("complete") is True and manifest["metadata"]["seed"] == seed,
             "Saved manifest is incomplete or belongs to a different seed")
    proof = _source_proof(campaign, seed_directory, manifest)
    files = {record["path"]: record for record in manifest["files"]}
    search = seed_directory if section == "all" else seed_directory / section
    selected = []
    for metric_path in sorted(search.rglob("metrics.json")):
        metadata = json.loads(metric_path.read_text())["metadata"]
        if domains and metadata["domain"] not in domains:
            continue
        if variants and metadata["variant"] not in variants:
            continue
        _require(metadata["seed"] == seed, "Rollout metadata seed differs from requested saved seed")
        if proof["phase"] == "pilot":
            _require(metadata["domain"] in ("ecology_train", "ecology_delay9"),
                     "Transfer-family replay is forbidden for pre-freeze pilot evidence")
        trace_path = metric_path.parent / "trace.npz"
        for path in (metric_path, trace_path):
            relative = str(path.relative_to(seed_directory))
            _require(relative in files and sha256(path) == files[relative]["sha256"],
                     f"Saved manifest checksum mismatch: {relative}")
        selected.append((trace_path, metric_path, metadata))
    _require(bool(selected), "No already saved rollout matches the requested filters")
    rows = [replay_one(*values, seed_directory) for values in selected]
    _require(sha256(environments.__file__) == proof["active_environment_sha256"],
             "Physical source changed while auditing")
    result = {"schema": "r9-saved-action-physical-replay-v1", "seed": seed,
              "input_campaign": str(campaign), "seed_directory": str(seed_directory),
              "complete_manifest_sha256": sha256(manifest_path), "source_proof": proof,
              "audit_script_sha256": sha256(__file__), "section": section,
              "requested_domains": domains, "requested_variants": variants,
              "rollouts": rows, "rollouts_replayed": len(rows),
              "saved_transitions_replayed": sum(row["steps"] for row in rows),
              "all_exact": all(row["exact"] for row in rows),
              "scope": "Physical replay of saved actions only; no model or policy imported, no fitting, no action selection, no new seeds."}
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section", choices=("evaluation", "training", "calibration", "selection", "all"), default="evaluation")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    result = audit_saved(args.input, args.seed, args.output, section=args.section,
                         domains=args.domains, variants=args.variants)
    print(json.dumps({key: result[key] for key in ("seed", "rollouts_replayed", "saved_transitions_replayed", "all_exact")}))
    if not result["all_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
