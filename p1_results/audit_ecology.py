"""Post-freeze independent reconstruction of P1.E primary measures from raw files.

This audit intentionally does not call ecology.calibrate/summarize or import its
feature/policy routines. It does not modify frozen sources or final results.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(r, v, a, generic=False):
    r, v, a = np.broadcast_arrays(r, v, a)
    if generic:
        powers = [p for p in itertools.product(range(4), repeat=3) if sum(p) <= 3]
        return np.stack([np.power(r, i)*np.power(v, j)*np.power(a/3, k) for i,j,k in powers], -1)
    output = np.zeros(r.shape + (20,))
    for action in range(4):
        ix = (a == action)
        for offset, values in enumerate([np.ones_like(r), r, r*r, v, r*v]):
            output[..., action*5+offset] = ix * values
    return output


def bootstrap(values):
    values = np.asarray(values)
    rng = np.random.default_rng(990017)
    boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "bootstrap95": np.quantile(boot, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "MANIFEST.json").read_text())["files"]
    checks = {"all_artifact_hashes": all(digest(out/k) == value for k,value in manifest.items())}
    run = json.loads((out / "RUN.json").read_text())
    seeds = run["seeds"]
    assert seeds == list(range(933001, 933031))
    capacity = run["config_id"]["capacity_per_agent"] * run["config_id"]["agents"]
    arms = ("aligned", "no_head", "shifted_credit", "generic", "immediate")
    raw_rows, native_rows = [], []
    max_prediction_error, max_scalar_error, max_coef_error = 0., 0., 0.
    frozen_q = True
    shifted_contracts = True
    for seed in seeds:
        folder = out / f"seed_{seed}"
        stored = json.loads((folder / "result.json").read_text())
        with np.load(folder / "learned_models.npz", allow_pickle=False) as data:
            coef = {k: data[k].copy() for k in data.files}
        with np.load(folder / "training_factual_trials.npz", allow_pickle=False) as train:
            r = train["public_resource"][:, 0] / capacity
            v = train["own_initial_vitality"] / run["config_id"]["max_vitality"]
            a = train["own_action"]
            y = train["public_resource"][:, 1:].mean(axis=1)
            shifted = np.roll(y.reshape(128, 32), 1, axis=1).ravel()
            shifted_contracts &= np.array_equal(shifted, train["target_shifted_credit"])
            shifted_contracts &= np.all(r.reshape(128,32) == r.reshape(128,32)[:,:1])
            shifted_contracts &= np.all(v.reshape(128,32) == v.reshape(128,32)[:,:1])
            for model in ("aligned", "shifted_credit", "generic", "immediate"):
                x = matrix(r, v, a, generic=model=="generic")
                target = shifted if model=="shifted_credit" else train["public_resource"][:,1] if model=="immediate" else y
                fitted = np.linalg.solve(x.T @ x + .01*np.eye(20), x.T @ target)
                max_coef_error = max(max_coef_error, float(np.max(np.abs(fitted - coef[f"coef_{model}"]))))
        for domain in ("id", "ood"):
            with np.load(folder / f"holdout_{domain}_counterfactual_diagnostic.npz", allow_pickle=False) as data:
                r = data["public_resource"][::4,0] / capacity
                v = data["own_initial_vitality"][::4] / run["config_id"]["max_vitality"]
                physical = data["true_resource_DIAGNOSTIC"][:,1:].mean(axis=1).reshape(-1,4)
                truth = physical[:,:1] - physical
                immediate = data["true_resource_DIAGNOSTIC"][:,1].reshape(-1,4)
                immediate_cost = immediate[:,:1] - immediate
            metrics = {"seed": seed, "domain": domain,
                       "zero_cost_or_action_free": float(np.mean(truth[:,1:]**2)),
                       "physical_immediate_oracle_DIAGNOSTIC": float(np.mean((immediate_cost[:,1:] - truth[:,1:])**2))}
            with np.load(folder / f"holdout_{domain}_predictions.npz", allow_pickle=False) as expected:
                max_prediction_error = max(max_prediction_error, float(np.max(np.abs(expected["true_cost"]-truth))))
                for model in ("aligned", "shifted_credit", "generic", "immediate"):
                    predictions = matrix(r[:,None],v[:,None],np.arange(4),generic=model=="generic") @ coef[f"coef_{model}"]
                    costs = predictions[:,:1] - predictions
                    max_prediction_error = max(max_prediction_error, float(np.max(np.abs(costs-expected[model]))))
                    metrics[model] = float(np.mean((costs[:,1:]-truth[:,1:])**2))
            raw_rows.append(metrics)
            for name in ("zero_cost_or_action_free", "physical_immediate_oracle_DIAGNOSTIC", "aligned", "shifted_credit", "generic", "immediate"):
                max_scalar_error = max(max_scalar_error, abs(metrics[name]-stored["calibration"][domain][name]["causal_cost_mse"]))
            for arm in arms:
                for episode in range(2):
                    trace_path = folder / f"native_{domain}_{arm}_{episode:02d}.npz"
                    with np.load(trace_path, allow_pickle=False) as trace:
                        alive = float(trace["alive_after"].mean())
                        resource = float(trace["resource_after"].mean() / capacity)
                        eligible = trace["alive_before"].sum()
                        action = float(trace["action"].sum() / eligible)
                        for name in ("qt", "qv", "qg", "visits"):
                            frozen_q &= np.array_equal(trace[f"initial_{name}"],trace[f"final_{name}"])
                        metrics_native = {"alive_fraction": alive, "resource_fraction": resource,
                                          "action_mean": action, "joint_designer_score": .5*(alive+resource)}
                    for k,value in metrics_native.items():
                        max_scalar_error = max(max_scalar_error, abs(value-stored["native"][domain][arm][episode][k]))
                    native_rows.append({"seed":seed,"domain":domain,"arm":arm,"episode":episode,**metrics_native})
    checks.update({"credit_assignment_reconstructed": bool(shifted_contracts),
                   "all_native_q_and_visits_frozen": bool(frozen_q),
                   "fit_coefficients_reconstructed": max_coef_error < 1e-9,
                   "causal_predictions_reconstructed": max_prediction_error < 1e-10,
                   "primary_raw_metrics_reconstructed": max_scalar_error < 1e-12})
    summary, summary_error = json.loads((out / "SUMMARY.json").read_text()), 0.
    mechanism, policy = {}, {}
    for domain in ("id","ood"):
        mechanism[domain], policy[domain] = {}, {}
        rows = [row for row in raw_rows if row["domain"]==domain]
        for comparator in ("zero_cost_or_action_free","shifted_credit","generic","immediate","physical_immediate_oracle_DIAGNOSTIC"):
            difference = [row[comparator]-row["aligned"] for row in rows]
            reconstructed = bootstrap(difference)
            key = f"mse_gain_aligned_vs_{comparator}"
            mechanism[domain][key] = reconstructed
            for stat in ("mean","bootstrap95"):
                summary_error = max(summary_error,float(np.max(np.abs(np.asarray(reconstructed[stat])-summary["mechanism"][domain][key][stat]))))
        for comparator in ("no_head","shifted_credit","generic","immediate"):
            policy[domain][comparator] = {}
            for metric in ("alive_fraction","resource_fraction","action_mean","joint_designer_score"):
                difference = []
                for seed in seeds:
                    aligned = [row[metric] for row in native_rows if row["seed"]==seed and row["domain"]==domain and row["arm"]=="aligned"]
                    control = [row[metric] for row in native_rows if row["seed"]==seed and row["domain"]==domain and row["arm"]==comparator]
                    difference.append(float(np.mean(aligned)-np.mean(control)))
                reconstructed = bootstrap(difference)
                policy[domain][comparator][metric] = reconstructed
                for stat in ("mean","bootstrap95"):
                    summary_error = max(summary_error,float(np.max(np.abs(np.asarray(reconstructed[stat])-summary["native"][domain][f"aligned_minus_{comparator}"][metric][stat]))))
    checks["seed_level_summary_and_intervals_reconstructed"] = summary_error < 1e-12
    report = {"all_checks_pass": all(checks.values()), "checks": checks,
              "counts": {"seeds":len(seeds),"training_factual_trials":len(seeds)*4096,
                         "holdout_diagnostic_trajectories":len(seeds)*1024,
                         "native_episodes_reconstructed":len(native_rows),"hashed_artifacts":len(manifest)},
              "maximum_error": {"coefficients":max_coef_error,"predictions":max_prediction_error,
                                "raw_scalar_metrics":max_scalar_error,"summary_and_intervals":summary_error},
              "mechanism_reconstructed":mechanism,"policy_reconstructed":policy,
              "script_sha256":digest(Path(__file__)),
              "limitation":"Same-runtime independent formula reconstruction, not external independent replication"}
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"all_checks_pass":report["all_checks_pass"],"checks":checks,"counts":report["counts"],"maximum_error":report["maximum_error"]},indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
