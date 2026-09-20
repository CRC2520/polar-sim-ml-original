"""Post-run reconstruction of the frozen six-claim R8 confirmation family.

This reporting utility cannot change source, endpoints, margins, or seed lists.
Independent raw-data auditors supply a separate check on the recorded metrics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r8_completion.io import atomic_json, sha256
from r8_completion.confirmation import confirmation_family, mean_interval, conservative_power
from r8_completion.provenance import verify
from r8_completion.population import RULES, Q_REGIMES, Q_ARMS
from r8_completion.integrated import PROTOCOL, VARIANTS


def load_json(path):
    return json.loads(Path(path).read_text())


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
            "by_seed": values.tolist()}


def read_seed(path):
    complete = load_json(path.parent / "COMPLETE.json")
    if complete.get("complete") is not True:
        raise ValueError(f"Incomplete seed: {path}")
    record = next(row for row in complete["files"] if row["path"] == path.name)
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Changed result: {path}")
    return load_json(path)


def compile_results(output):
    source_status = verify()
    population_root = ROOT / "r8_results/Q_final"
    qdata, q_descriptive = {}, {}
    for rule in RULES:
        for arm in Q_ARMS:
            regimes = []
            for regime in Q_REGIMES:
                directory = population_root / f"{rule}__{regime}__{arm}"
                complete = load_json(directory / "COMPLETE.json")
                endpoint = directory / "endpoints.npz"
                expected = next(row for row in complete["files"] if row["path"] == endpoint.name)
                if sha256(endpoint) != expected["sha256"]:
                    raise ValueError(f"Changed endpoint: {endpoint}")
                protocol = load_json(directory / "PROTOCOL.json")
                if protocol["seeds"] != source_status["final_seeds"]["population"]:
                    raise ValueError("Population seed ordering differs from frozen design")
                with np.load(endpoint, allow_pickle=False) as arrays:
                    regimes.append({key: arrays[key].copy() for key in
                                    ("survival_time_fraction", "resource_time_fraction")})
                q_descriptive[f"{rule}/{regime}/{arm}"] = {
                    key: describe(value) for key, value in regimes[-1].items()}
            qdata[(rule, arm)] = {
                key: np.stack([row[key] for row in regimes]).mean(axis=0)
                for key in regimes[0]}

    q_components, q1, q2 = {}, [], []
    for rule in RULES:
        alive = qdata[(rule, "normalized_common")]["survival_time_fraction"] - qdata[(rule, "legacy_full")]["survival_time_fraction"]
        ds = qdata[(rule, "state_common")]["survival_time_fraction"] - qdata[(rule, "global_common")]["survival_time_fraction"]
        dr = qdata[(rule, "state_common")]["resource_time_fraction"] - qdata[(rule, "global_common")]["resource_time_fraction"]
        q1.append(alive >= .01)
        q2.append((dr >= .01) & (ds >= -.005))
        q_components[rule] = {"Q1_delta_alive": alive.tolist(), "Q2_delta_alive": ds.tolist(),
                              "Q2_delta_resource": dr.tolist(),
                              "Q1_mean_ci": mean_interval(alive),
                              "Q2_alive_mean_ci": mean_interval(ds),
                              "Q2_resource_mean_ci": mean_interval(dr)}

    paths = sorted((ROOT / "r8_results/integrated_final").glob("shard_*/seed_*/result.json"))
    results = sorted([read_seed(path) for path in paths], key=lambda row: row["seed"])
    seeds = [row["seed"] for row in results]
    if seeds != source_status["final_seeds"]["integrated"]:
        raise ValueError(f"Require exactly the 30 frozen integrated seeds, found {seeds}")
    for row in results:
        if set(row["native"]) != set(PROTOCOL["domains"]):
            raise ValueError("Missing native domain")
        for domain in PROTOCOL["domains"]:
            if set(row["native"][domain]) != set(VARIANTS):
                raise ValueError("Missing native variant")

    success = {"Q1": np.logical_and.reduce(q1), "Q2": np.logical_and.reduce(q2)}
    for claim in ("E_credit", "E_utility", "T_utility", "I_generic"):
        success[claim] = np.ones(30, dtype=bool)
    integrated_components, native, prediction, contrasts = {}, {}, {}, {}
    for domain in PROTOCOL["domains"]:
        aligned = np.array([row["prediction"][domain]["aligned"]["mse"] for row in results])
        wrong = np.array([row["prediction"][domain]["miscredit"]["mse"] for row in results])
        credit_wins = wrong-aligned >= np.maximum(.1*wrong, .0001)
        success["E_credit"] &= credit_wins
        integrated_components[domain] = {"E_credit_aligned_mse": aligned.tolist(),
            "E_credit_miscredit_mse": wrong.tolist(), "E_credit_success": credit_wins.tolist()}
        prediction[domain] = {"aligned": describe(aligned), "miscredit": describe(wrong),
                             "miscredit_minus_aligned_ci": mean_interval(wrong-aligned)}
        for name in ("aligned", "miscredit"):
            matrix = np.array([row["prediction"][domain][name]["mse_by_horizon_variable"] for row in results])
            prediction[domain][name]["mse_by_horizon_variable_mean"] = matrix.mean(axis=0).tolist()
            prediction[domain][name]["mse_by_horizon_variable_by_seed"] = matrix.tolist()
        native[domain], contrasts[domain] = {}, {}
        arrays = {}
        metrics = ("alive_fraction", "resource_fraction", "public_resource_fraction", "action_mean",
                   "guard_restriction_fraction", "guard_violation_fraction", "uniform_probe_fraction")
        for variant in VARIANTS:
            arrays[variant] = {metric: np.array([row["native"][domain][variant][metric] for row in results])
                               for metric in metrics}
            native[domain][variant] = {metric: describe(values) for metric, values in arrays[variant].items()}
            if variant != "full":
                contrasts[domain]["full-minus-"+variant] = {
                    metric: mean_interval(arrays["full"][metric]-arrays[variant][metric])
                    for metric in ("alive_fraction", "resource_fraction")}
        for claim, comparator, alive_margin, resource_margin in (
            ("E_utility", "noEcology", -.01, .02),
            ("T_utility", "noNetwork", -.005, .01),
            ("I_generic", "generic_joint", .01, -.02)):
            da = arrays["full"]["alive_fraction"] - arrays[comparator]["alive_fraction"]
            dr = arrays["full"]["resource_fraction"] - arrays[comparator]["resource_fraction"]
            wins = (da >= alive_margin) & (dr >= resource_margin)
            success[claim] &= wins
            integrated_components[domain][claim] = {"delta_alive": da.tolist(), "delta_resource": dr.tolist(),
                                                     "success": wins.tolist()}

    confirmation = confirmation_family(success)
    confirmation.update(source_commit=load_json(ROOT / "r8_results/FINAL_EXECUTION_AUTHORIZATION.json")["source_commit"],
                        freeze_sha256=source_status["freeze_sha256"],
                        seed_lists=source_status["final_seeds"], Q_components=q_components,
                        integrated_components=integrated_components, conservative_power=conservative_power())
    selections = {variant: [row["selection"][variant]["selected"] for row in results]
                  for variant in ("full", "generic_joint")}
    diagnostics = {"selections_by_seed": selections,
                   "full_network_gate_zero": sum(row[1] == 0 for row in selections["full"]),
                   "full_ecology_weight_zero": sum(row[0] == 0 for row in selections["full"]),
                   "coordinate_actions_equal_all": all(row["coordinate_action_equal"] for row in results),
                   "checkpoint_replays_exact_all": all(row["replay_exact"] for row in results),
                   "network_ranks_last_fit": [row["fit_records"][-1]["network_diagnostics"]["joint_standardized_feature_rank"] for row in results],
                   "near_constant_tension_last_fit": [row["fit_records"][-1]["network_diagnostics"]["constant_or_near_constant_tension"] for row in results],
                   "credit_elapsed_quantiles_last_fit": [row["fit_records"][-1]["credit_elapsed_steps_quantiles"] for row in results],
                   "credit_changed_fraction_last_fit": [row["fit_records"][-1]["miscredit_changed_fraction"] for row in results]}
    output = Path(output)
    atomic_json(output / "R8_CONFIRMATION.json", confirmation)
    atomic_json(output / "R8_DESCRIPTIVES.json", {"integrated_seeds": seeds, "native": native,
        "contrasts": contrasts, "prediction": prediction, "diagnostics": diagnostics,
        "population_by_regime": q_descriptive,
        "limitations": "Mean intervals are descriptive. Six tests concern seed robustness. Domains and historical implementations are not pooled."})
    verify()
    return {"claims": [{k: row[k] for k in ("claim", "wins", "n", "p_holm", "supported_at_family_alpha_0_05")}
                        for row in confirmation["rows"]], "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_results(args.output), indent=2))
