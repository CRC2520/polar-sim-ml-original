"""Independent reconstruction/audit of R8 population artifacts.

This utility is outside the prospective scientific source tree. By default it
only reads artifacts; optional replay requires an explicit flag and, for final
seeds, the intact prospective freeze. It never changes original run artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r8_completion.io import atomic_json, atomic_npz, atomic_text, write_manifest

RULES = ("success", "conformity")
ARMS = ("legacy_full", "shared_successor", "normalized_independent",
        "normalized_common", "global_common", "state_common",
        "shuffled_common", "qv_policy_off")
METRICS = ("survival_time_fraction", "resource_time_fraction")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def finite_mean(values, axis=None):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    numerator = np.where(finite, values, 0.).sum(axis=axis)
    return np.divide(numerator, count, out=np.full(np.shape(count), np.nan), where=count > 0)


def compare(expected, actual, context, atol=1e-12):
    """Recursively compare reports without hiding missing keys or observations."""
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{context}: keys differ: {set(expected) ^ set(actual)}")
        for key in expected:
            compare(expected[key], actual[key], f"{context}/{key}", atol)
    elif expected is None or isinstance(expected, (str, bool)):
        if expected != actual:
            raise AssertionError(f"{context}: {expected!r} != {actual!r}")
    else:
        if np.asarray(expected).dtype.kind in "OUS" or np.asarray(actual).dtype.kind in "OUS":
            np.testing.assert_array_equal(expected, actual, err_msg=context)
        else:
            np.testing.assert_allclose(expected, actual, rtol=0, atol=atol, equal_nan=True, err_msg=context)


def bootstrap_ci(difference):
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": None, "n_seeds": 0}
    # Use the prospectively declared resampling stream, not the author's helper.
    draws = np.random.default_rng(937003).integers(0, len(values), (2000, len(values)))
    return {"mean": float(values.mean()), "ci95": np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
            "n_seeds": len(values)}


class Audit:
    def __init__(self):
        self.manifests = {}
        self.cells = {}
        self.files = 0
        self.npz_members = 0

    def manifest(self, folder):
        folder = Path(folder).resolve()
        if folder in self.manifests:
            return self.manifests[folder]
        path = folder / "COMPLETE.json"
        manifest = read_json(path)
        if manifest.get("complete") is not True:
            raise AssertionError(f"Incomplete manifest: {path}")
        seen = set()
        for record in manifest["files"]:
            artifact = (folder / record["path"]).resolve()
            if artifact in seen or not artifact.is_relative_to(folder):
                raise AssertionError(f"Invalid/duplicate manifest path: {artifact}")
            seen.add(artifact)
            if artifact.stat().st_size != record["bytes"] or digest(artifact) != record["sha256"]:
                raise AssertionError(f"SHA256/size mismatch: {artifact}")
            if artifact.suffix == ".npz":
                with zipfile.ZipFile(artifact) as archive:
                    if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
                        raise AssertionError(f"CRC or duplicate members: {artifact}")
                with np.load(artifact, allow_pickle=False) as data:
                    shapes = {}
                    for key in data.files:
                        value = data[key]
                        if value.dtype.hasobject:
                            raise AssertionError(f"Object dtype: {artifact}/{key}")
                        shapes[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                        self.npz_members += 1
                    compare(record.get("arrays", shapes), shapes, f"NPZ descriptor:{artifact}")
            self.files += 1
        self.manifests[folder] = {"manifest_sha256": digest(path), "files": len(seen)}
        return self.manifests[folder]

    def cell(self, folder):
        folder = Path(folder).resolve()
        if folder in self.cells:
            return self.cells[folder]
        self.manifest(folder)
        protocol = read_json(folder / "PROTOCOL.json")
        h, tx = load(folder / "outcomes.npz"), load(folder / "transmissions.npz")
        saved, config = load(folder / "endpoints.npz"), protocol["config"]
        n, generations = len(protocol["seeds"]), config["generations"]
        for collection in (h, tx):
            for key, value in collection.items():
                if value.shape[:2] != (n, generations):
                    raise AssertionError(f"Missing seed/generation: {folder}/{key}")
        tail = slice(-config["tail_generations"], None)
        def mean_tail(values):
            return finite_mean(values[:, tail], axis=tuple(range(1, values.ndim)))
        classes = tx["lineage_before"].astype(bool) if protocol["study"] in ("causal_A", "threshold") else h["restraint_agent"] >= .5
        def gap(values):
            return finite_mean(np.where(classes, values, np.nan), -1) - finite_mean(np.where(~classes, values, np.nan), -1)
        alive, resource = mean_tail(h["alive_fraction"]), mean_tail(h["resource_fraction"])
        reconstructed = {
            "survival_time_fraction": alive, "resource_time_fraction": resource,
            "terminal_survival_fraction": mean_tail(h["terminal_survival"]),
            "terminal_resource_fraction": mean_tail(h["terminal_resource"]),
            "restraint_behavior": mean_tail(h["restraint_behavior"]),
            "restrained_mode_fraction": mean_tail((h["restraint_agent"] >= .5).astype(float)),
            "lineage_fraction": mean_tail(h["lineage_fraction"]),
            "raw_survival_gap": mean_tail(gap(tx["raw_fitness"])),
            "copy_score_gap": mean_tail(gap(tx["copy_score"])),
            "real_income_rate_gap": mean_tail(gap(h["income_agent"])),
            "real_income_cumulative_gap": mean_tail(gap(h["income_cumulative"])),
            "survival_time_gap": mean_tail(gap(h["alive_agent_time"] * config["steps"])),
            "viable": ((alive >= config["viable_alive"]) & (resource >= config["viable_resource"])).astype(float),
        }
        if "public_resource_fraction" in h:
            reconstructed["observed_resource_time_fraction"] = mean_tail(h["public_resource_fraction"])
        compare(saved, reconstructed, f"Reconstructed endpoints:{folder}")
        for name in METRICS:
            if not np.isfinite(reconstructed[name]).all():
                raise AssertionError(f"Missing primary endpoint: {folder}/{name}")
        metrics_path = folder / "conditional_metrics.npz"
        if metrics_path.exists():
            raw_gap = gap(tx["raw_fitness"])
            conditional = {"raw_survival_gap": raw_gap, "copy_score_gap": gap(tx["copy_score"]),
                "real_income_rate_gap": gap(h["income_agent"]), "real_income_cumulative_gap": gap(h["income_cumulative"]),
                "survival_time_gap": gap(h["alive_agent_time"] * config["steps"]),
                "conditional_gap_observed": np.isfinite(raw_gap)}
            compare(load(metrics_path), conditional, f"Conditional outcomes:{folder}")
        result = {"protocol": protocol, "endpoints": reconstructed, "path": folder}
        self.cells[folder] = result
        return result


def audit_q(audit, folder, phase):
    folder = Path(folder)
    regimes = ("id",) if phase == "pilot" else ("id", "higher_metabolism", "lower_regeneration")
    saved = read_json(folder / f"Q_{phase.upper()}_SUMMARY.json")
    expected_seeds = list(range(940015, 940021)) if phase == "pilot" else list(range(941001, 941031))
    compare(expected_seeds, saved["seeds"], "Q summary seeds")
    values, descriptives = {}, {}
    for rule in RULES:
        for arm in ARMS:
            cells = [audit.cell(folder / f"{rule}__{regime}__{arm}") for regime in regimes]
            for cell, regime in zip(cells, regimes):
                p = cell["protocol"]
                compare(expected_seeds, p["seeds"], f"Q/{rule}/{regime}/{arm}/seeds")
                if (p["rule"], p["regime"], p["arm"]) != (rule, regime, arm):
                    raise AssertionError("Q condition mismatch")
            values[rule, arm] = {metric: np.stack([cell["endpoints"][metric] for cell in cells]).mean(0) for metric in METRICS}
            descriptives[f"{rule}/{arm}"] = {
                "regime_means": {regime: {metric: float(cell["endpoints"][metric].mean()) for metric in METRICS}
                                 for cell, regime in zip(cells, regimes)},
                "across_regime_means": {metric: float(values[rule, arm][metric].mean()) for metric in METRICS}}
    contrasts, factorial, components, wins = {}, {}, {}, {}
    per_rule_q1, per_rule_q2 = [], []
    for rule in RULES:
        for arm in ARMS[1:]:
            contrasts[f"{rule}/{arm}-minus-legacy_full"] = {m: bootstrap_ci(values[rule, arm][m] - values[rule, "legacy_full"][m]) for m in METRICS}
        for control, label in (("global_common", "global"), ("shuffled_common", "shuffled")):
            contrasts[f"{rule}/state-minus-{label}"] = {m: bootstrap_ci(values[rule, "state_common"][m] - values[rule, control][m]) for m in METRICS}
        for metric in METRICS:
            a, b, c, d = [values[rule, arm][metric] for arm in ARMS[:4]]
            for label, effect in (("normalization_given_independent", c-a), ("normalization_given_common", d-b),
                ("common_successor_given_raw", b-a), ("common_successor_given_normalized", d-c),
                ("normalization_x_successor_interaction", d-c-b+a)):
                factorial[f"{rule}/{label}/{metric}"] = bootstrap_ci(effect)
        da = values[rule, "normalized_common"][METRICS[0]] - values[rule, "legacy_full"][METRICS[0]]
        sa = values[rule, "state_common"][METRICS[0]] - values[rule, "global_common"][METRICS[0]]
        sr = values[rule, "state_common"][METRICS[1]] - values[rule, "global_common"][METRICS[1]]
        components[rule] = {"Q1_delta_alive": da.tolist(), "Q2_delta_alive": sa.tolist(), "Q2_delta_resource": sr.tolist()}
        per_rule_q1.append(da >= .01)
        per_rule_q2.append((sr >= .01) & (sa >= -.005))
    wins = {"Q1": np.logical_and.reduce(per_rule_q1).tolist(), "Q2": np.logical_and.reduce(per_rule_q2).tolist()}
    compare(saved["primary_wins_by_seed"], wins, "Q primary wins", 0)
    compare(saved["descriptives"], descriptives, "Q descriptive regime means")
    compare(saved["primary_components_by_seed"], components, "Q primary components")
    compare(saved["paired_descriptive_ci"], contrasts, "Q descriptive contrasts")
    compare(saved["factorial_secondary_descriptive_ci"], factorial, "Q factorial contrasts")
    compare(saved["primary_win_counts"], {k: sum(v) for k, v in wins.items()}, "Q counts", 0)
    return {"seeds": expected_seeds, "cells": len(RULES)*len(ARMS)*len(regimes), "primary_wins_by_seed": wins,
            "primary_win_counts": {k: sum(v) for k, v in wins.items()}, "factorial_contrasts": factorial,
            "state_minus_shuffled": {rule: contrasts[f"{rule}/state-minus-shuffled"] for rule in RULES},
            "all_reconstructed_from_raw": True}


def equal_artifacts(left, right, filename, ignore=()):
    a, b = load(Path(left) / filename), load(Path(right) / filename)
    if set(a) != set(b):
        raise AssertionError(f"Different members in {left}, {right}: {filename}")
    for key in set(a) - set(ignore):
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"Exact control: {left}/{right}/{filename}/{key}")
    return len(a) - len(ignore)


def audit_a(audit, folder, phase):
    folder = Path(folder)
    saved = read_json(folder / f"CAUSAL_A_{phase.upper()}_SUMMARY.json")
    expected_seeds = list(range(940015, 940018)) if phase == "pilot" else list(range(941001, 941031))
    compare(expected_seeds, saved["seeds"], "A summary seeds")
    accumulated, by_fraction, controls, exact_arrays = {}, {}, {}, 0
    metrics = (*METRICS, "raw_survival_gap", "copy_score_gap", "real_income_rate_gap")
    for p in (.35, .5, .65):
        prefix = f"success__p{round(p*100):02d}"
        reference = folder / f"{prefix}__endogenous__s0p0"
        data = {}
        for mode in ("endogenous", "fixed_donors"):
            for score in (0, 1):
                for pool in (0, 1):
                    path = folder / f"{prefix}__{mode}__s{score}p{pool}"
                    cell = audit.cell(path)
                    compare(expected_seeds, cell["protocol"]["seeds"], f"A cell seeds:{path}")
                    data[mode, score, pool] = cell["endpoints"]
                    if mode == "fixed_donors":
                        exact_arrays += equal_artifacts(reference, path, "donor_maps.npz")
            for metric in metrics:
                a, b, c, d = [data[mode, s, q][metric] for s, q in ((0,0),(1,0),(0,1),(1,1))]
                for effect, difference in (("score_at_pool0", b-a), ("score_at_pool1", d-c),
                    ("pool_at_score0", c-a), ("pool_at_score1", d-b), ("interaction", d-b-c+a)):
                    name = f"{mode}/{effect}/{metric}"
                    accumulated.setdefault(name, []).append(difference)
                    by_fraction[f"p{p:.2f}/{name}"] = bootstrap_ci(difference)
        exact_arrays += equal_artifacts(reference, folder / f"{prefix}__fixed_donors__s0p0", "outcomes.npz")
        for pool in (0, 1):
            left = folder / f"{prefix}__fixed_donors__s0p{pool}"
            right = folder / f"{prefix}__fixed_donors__s1p{pool}"
            for artifact in ("outcomes.npz", "final_state.npz", "donor_maps.npz"):
                exact_arrays += equal_artifacts(left, right, artifact)
            exact_arrays += equal_artifacts(left, right, "transmissions.npz", ignore=("copy_score",))
            controls[f"fixed_donors/p{p:.2f}/pool{pool}"] = True
        left = folder / f"conformity__p{round(p*100):02d}__endogenous__s0p0"
        right = folder / f"conformity__p{round(p*100):02d}__endogenous__s1p0"
        for path in (left, right):
            cell = audit.cell(path)
            compare(expected_seeds, cell["protocol"]["seeds"], f"A conformity seeds:{path}")
        for artifact in ("outcomes.npz", "final_state.npz", "donor_maps.npz"):
            exact_arrays += equal_artifacts(left, right, artifact)
        controls[f"conformity/p{p:.2f}/score_negative"] = True
    contrasts = {k: bootstrap_ci(finite_mean(np.stack(v, axis=1), 1)) for k, v in accumulated.items()}
    compare(saved["contrasts_averaged_over_composition"], contrasts, "A factorial contrasts")
    compare(saved["by_fraction"], by_fraction, "A per-composition contrasts")
    compare(saved["negative_controls_exact"], controls, "A negative controls", 0)
    return {"cells": 30, "seeds": expected_seeds, "exact_control_array_comparisons": exact_arrays,
            "negative_controls_exact": controls, "factorial_contrasts": contrasts, "all_reconstructed_from_raw": True}


def point_crossing(grid, probabilities):
    above = np.asarray(probabilities) >= .5
    if np.any(above[:-1] & ~above[1:]):
        return "no_unique_crossing", None, None
    if above.all():
        return "at_or_below_support", None, None
    if not above.any():
        return "above_support_or_absent", None, None
    right = int(np.flatnonzero(above)[0])
    left = right - 1
    value = grid[left] + (.5 - probabilities[left]) * (grid[right] - grid[left]) / (probabilities[right] - probabilities[left])
    return "within_support", float(value), [float(grid[left]), float(grid[right])]


def independent_threshold(grid, binary):
    grid, binary = np.asarray(grid), np.asarray(binary, dtype=float)
    n, m = binary.shape
    probs, counts = binary.mean(0), binary.sum(0).astype(int)
    def intervals(alpha):
        result = []
        for k in counts:
            result.append([0. if k == 0 else float(beta.ppf(alpha/2, k, n-k+1)),
                           1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))])
        return np.asarray(result)
    marginal, simultaneous = intervals(.05), intervals(.05/m)
    status, estimate, crossing = point_crossing(grid, probs)
    low, high = np.flatnonzero(simultaneous[:,1] < .5), np.flatnonzero(simultaneous[:,0] > .5)
    reversal = bool(len(high) and len(low) and high.min() < low.max())
    bracket, bracket_status = None, "insufficient_confident_endpoints"
    if reversal:
        status, estimate, crossing = "no_unique_crossing", None, None
        bracket_status = "no_unique_crossing"
    elif len(low) and len(high) and low.max() < high.min():
        bracket = [float(grid[low.max()]), float(grid[high.min()])]
        bracket_status = "ordered_confidence_bracket"
    categories = dict.fromkeys(("at_or_below_support", "above_support_or_absent", "no_unique_crossing", "within_support"), 0)
    interior = []
    rng = np.random.default_rng(946991)
    for _ in range(2000):
        sampled = binary[rng.integers(n, size=n)].mean(0)
        category, value, _ = point_crossing(grid, sampled)
        categories[category] += 1
        if category == "within_support":
            interior.append(value)
    fraction = len(interior) / 2000
    return {"status": status, "estimate": estimate, "crossing_interval": crossing,
        "grid": grid.tolist(), "n_seeds": n, "successes": counts.tolist(), "probability": probs.tolist(),
        "marginal_ci95": marginal.tolist(), "simultaneous_ci95": simultaneous.tolist(),
        "interval_method": "Clopper-Pearson exact binomial; simultaneous 95% with Bonferroni over all grid points",
        "confidence_bracket": bracket, "confidence_bracket_status": bracket_status, "confident_reversal": reversal,
        "bootstrap": {"replicates": 2000, "seed": 946991, "paired_across_grid": True,
            "status_counts": categories, "valid_fraction": fraction,
            "conditional_in_support_ci95": np.quantile(interior, [.025,.975]).tolist() if interior else None},
        "reliable_identification": bool(status == "within_support" and bracket is not None and fraction >= .8)}


def audit_threshold(audit, folder, phase, selection_path=None):
    folder = Path(folder)
    summary = read_json(folder / f"THRESHOLD_{phase.upper()}_SUMMARY.json")
    expected_seeds = list(range(940031, 940036)) if phase == "pilot" else list(range(941001, 941031))
    selection = read_json(selection_path) if selection_path else summary.get("frozen_pilot_selection")
    if phase == "final":
        if selection is None:
            raise ValueError("Final threshold audit requires frozen selection")
        compare(selection, summary["frozen_pilot_selection"], "Threshold frozen selection")
        expected_profiles = [(selection["pressure"], rule) for rule in RULES]
        grid = selection["grid"]
    else:
        expected_profiles = [(pressure, rule) for pressure in (.3,.45,.6) for rule in RULES]
        grid = [i/10 for i in range(11)]
    indexed = {(r["metabolism"], r["rule"]): r for r in summary["records"]}
    if set(indexed) != set(expected_profiles) or len(indexed) != len(summary["records"]):
        raise AssertionError("Threshold profile coverage differs")
    results = []
    for pressure, rule in expected_profiles:
        cells = [audit.cell(folder / f"m{round(pressure*100):02d}__{rule}__p{round(p*100):02d}") for p in grid]
        for cell, p in zip(cells, grid):
            protocol = cell["protocol"]
            compare(expected_seeds, protocol["seeds"], "Threshold cell seeds")
            compare([pressure,p], [protocol["metabolism"], protocol["initial_fraction"]], "Threshold cell condition")
        binary = np.stack([cell["endpoints"]["viable"] for cell in cells], 1)
        row = indexed[pressure, rule]
        compare(binary, row["viable"], "Threshold raw viability", 0)
        compare(grid, row["grid"], "Threshold grid", 0)
        result = independent_threshold(grid, binary)
        compare(row["threshold"], result, f"Threshold independent profile:{pressure}/{rule}")
        results.append({"pressure": pressure, "rule": rule, "threshold": result})
    return {"profiles": results, "seeds": expected_seeds, "cells": len(grid)*len(expected_profiles),
            "selection": selection, "all_reconstructed_from_raw": True}


def audit_calibration(audit, folder):
    folder = Path(folder)
    palette, frozen = read_json(folder / "CANDIDATE_PALETTE.json"), read_json(folder / "CALIBRATION_FROZEN.json")
    models = palette["models"]
    if len(models) != 10 or frozen["candidate_budget_per_family"] != 10:
        raise AssertionError("Expected ten candidate settings per family")
    reuse_path = folder / "FACTUAL_DATA_REUSE.json"
    training_path = Path(read_json(reuse_path)["path"]) if reuse_path.exists() else folder / "calibration_training/factual_uniform_probes.npz"
    if not training_path.is_absolute():
        candidates = {(base / training_path).resolve() for base in (ROOT, ROOT.parent, Path.cwd())}
        matching = [path for path in candidates if path.is_file() and digest(path) == frozen["training_artifact_sha256"]]
        if len(matching) != 1:
            raise AssertionError(f"Cannot uniquely resolve factual training bytes: {training_path}")
        training_path = matching[0]
    audit.manifest(training_path.parent)
    compare(frozen["training_artifact_sha256"], digest(training_path), "Factual training artifact SHA256")
    scores = {}
    for family in ("global_common", "state_common"):
        for name, model in models.items():
            objectives = []
            for rule in RULES:
                cell = audit.cell(folder / "calibration_validation" / f"{name}__{family}__{rule}")
                compare(list(range(940011,940015)), cell["protocol"]["seeds"], "Calibration validation seeds")
                compare(model, cell["protocol"]["calibration"], "Candidate model identity")
                ep = cell["endpoints"]
                objectives.extend((.5*ep[METRICS[0]] + .5*ep["observed_resource_time_fraction"]).tolist())
            scores[f"{family}/{name}"] = float(np.mean(objectives))
    compare(frozen["validation_scores"], scores, "Calibration observed validation scores")
    selected = {}
    for family in ("global_common", "state_common"):
        selected[family] = min(models, key=lambda name: (-round(scores[f"{family}/{name}"],12),
            0 if models[name]["candidate_type"] == "constant_palette" else 1, name))
        label = "global" if family == "global_common" else "state"
        compare(frozen[f"selected_candidate_{label}"], selected[family], "Calibration selected candidate")
    compare(frozen["global_weights"], models[selected["global_common"]]["global_weights"], "Selected global weights")
    compare(frozen["state_weights"], models[selected["state_common"]]["state_weights"], "Selected state weights")
    compare(frozen["shuffled_state_weights"], models[selected["state_common"]]["shuffled_state_weights"], "Selected shuffled weights")
    return {"cells": 40, "candidate_budget_each": 10, "selected": selected, "validation_scores": scores,
            "selection_uses_public_resource": True, "all_reconstructed_from_raw": True}


def replay_cell(audit, folder, output, fixed_maps=None):
    from collective.bridge_v2.engine import BridgeConfig
    from r8_completion.population import RegulatedBridge, FactorialBridge, _history_run
    folder, output = Path(folder), Path(output)
    cell = audit.cell(folder)
    p = cell["protocol"]
    seed = p["seeds"][0]
    config = BridgeConfig(**p["config"])
    if p["study"] == "Q":
        engine = RegulatedBridge(config, [seed], p["initial_fraction"], p["arm"], p["calibration"])
    else:
        engine = FactorialBridge(config, [seed], p["initial_fraction"], p["score_equal"], p["pool"])
    history, transmissions, maps, endpoints, conditional = _history_run(engine, fixed_maps)
    replayed = {"outcomes.npz": history, "transmissions.npz": transmissions,
                "endpoints.npz": endpoints, "conditional_metrics.npz": conditional,
                "final_state.npz": engine.state}
    if maps:
        replayed["donor_maps.npz"] = maps
    checks, artifacts, omitted = 0, [], []
    for name, result in replayed.items():
        original = load(folder / name)
        for key, value in original.items():
            if key == "full_transition_sha256":
                omitted.append(f"{name}/{key}: hashes all batch seeds, not the single replayed seed")
                continue
            np.testing.assert_array_equal(value[:1], result[key], err_msg=f"Replay exact:{folder}/{name}/{key}")
            checks += 1
        artifacts.append(atomic_npz(output / name, result))
    record = {"cell": str(folder), "seed": seed, "exact_array_comparisons": checks,
              "omitted_batch_only_hash": omitted, "new_independent_replicate": False}
    write_manifest(output / "COMPLETE.json", artifacts, record)
    return record


def replays(audit, q_folder, a_folder, output, phase, final_authorized):
    if phase == "final":
        if not final_authorized:
            raise ValueError("Final replay requires explicit --authorized-final-replay")
        from r8_completion.provenance import verify
        verify()
    results = []
    if q_folder:
        for rule in RULES:
            for arm in ("legacy_full", "normalized_common", "global_common", "state_common"):
                name = f"{rule}__id__{arm}"
                results.append(replay_cell(audit, Path(q_folder)/name, Path(output)/"replay_Q"/name))
    if a_folder:
        prefix = "success__p35"
        baseline = Path(a_folder)/f"{prefix}__endogenous__s0p0"
        results.append(replay_cell(audit, baseline, Path(output)/"replay_A"/baseline.name))
        maps = {k:v[:1] for k,v in load(baseline/"donor_maps.npz").items()}
        for score in (0,1):
            for pool in (0,1):
                name = f"{prefix}__fixed_donors__s{score}p{pool}"
                results.append(replay_cell(audit, Path(a_folder)/name, Path(output)/"replay_A"/name, maps))
    if phase == "final":
        verify()
    return results


def verify_final_inputs(audit, threshold_selection=None):
    frozen = read_json(ROOT / "r8_completion/FREEZE_R8.json")
    source = frozen["source_sha256"]
    def artifact(suffix):
        names = [name for name in source if name.endswith(suffix)]
        if len(names) != 1:
            raise AssertionError(f"Expected one frozen input: {suffix}")
        return read_json(ROOT / names[0])
    calibration = artifact("CALIBRATION_FROZEN.json")
    if threshold_selection is not None:
        compare(artifact("THRESHOLD_SELECTION.json"), threshold_selection, "Actual frozen threshold selection")
    checked = 0
    for cell in audit.cells.values():
        protocol = cell["protocol"]
        if protocol["seeds"][0] != 941001:
            continue
        compare(source["r8_completion/population.py"], protocol["source_sha256"]["population.py"], "Final population source")
        compare(source["collective/bridge_v2/engine.py"], protocol["source_sha256"]["inherited_engine.py"], "Final inherited engine source")
        if protocol["study"] == "Q":
            compare(calibration, protocol["calibration"], "Actual frozen Q calibration")
        checked += 1
    return {"verified_final_protocols": checked, "calibration_and_selection_match_freeze": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot","final"), default="pilot")
    parser.add_argument("--q")
    parser.add_argument("--a")
    parser.add_argument("--threshold")
    parser.add_argument("--selection")
    parser.add_argument("--calibration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--authorized-final-replay", action="store_true")
    args = parser.parse_args()
    if not any((args.q,args.a,args.threshold,args.calibration)):
        raise ValueError("Provide at least one completed campaign")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    audit = Audit()
    report = {"schema":"r8-independent-population-audit-v1", "phase":args.phase,
              "audit_script_sha256":digest(__file__), "original_artifacts_modified":False}
    if args.phase == "final":
        from r8_completion.provenance import verify
        report["prospective_freeze_before"] = verify()
    if args.calibration:
        report["calibration"] = audit_calibration(audit,args.calibration)
    if args.q:
        report["Q"] = audit_q(audit,args.q,args.phase)
    if args.a:
        report["A"] = audit_a(audit,args.a,args.phase)
    if args.threshold:
        report["threshold"] = audit_threshold(audit,args.threshold,args.phase,args.selection)
    if args.phase == "final":
        report["frozen_inputs"] = verify_final_inputs(audit, report.get("threshold", {}).get("selection"))
    if args.replay:
        report["replays"] = replays(audit,args.q,args.a,output,args.phase,args.authorized_final_replay)
    report["integrity"] = {"complete_manifests":len(audit.manifests), "sha256_files":audit.files,
                           "crc_and_safe_load_npz_members":audit.npz_members, "all_pass":True,
                           "manifests":{str(k):v for k,v in audit.manifests.items()}}
    if args.phase == "final":
        report["prospective_freeze_after"] = verify()
    report["all_pass"] = True
    artifact = atomic_json(output/"POPULATION_AUDIT.json",report)
    source_artifact = atomic_text(output/"audit_population_source.py",Path(__file__).read_text())
    write_manifest(output/"COMPLETE.json",[artifact,source_artifact],{"audit_script_sha256":digest(__file__), "phase":args.phase})
    print(json.dumps({"output":str(output),"all_pass":True,"cells":len(audit.cells),"files":audit.files,
                      "npz_members":audit.npz_members,"replays":len(report.get("replays",[]))}),flush=True)


if __name__ == "__main__":
    main()
