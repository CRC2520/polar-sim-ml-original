"""External measurement independent of controller-internal homogeneity scores."""
import numpy as np


def weighted_mse(effect, target, weights):
    return float(np.sum(weights * (effect - target)**2) / np.sum(weights))


def step_metrics(frame, action, effect):
    action, effect = np.asarray(action, float), np.asarray(effect, float)
    target = np.asarray(frame["target"], float)
    weights = np.asarray(frame["weights"], float)
    if action.shape != target.shape or effect.shape != target.shape:
        raise ValueError("Action/effect/target dimensions differ")
    if not np.isfinite(action).all() or not np.isfinite(effect).all():
        raise ValueError("Non-finite action or observed effect")
    oracle_effect = np.clip(np.asarray(frame["oracle_action"]) * np.asarray(frame["gains"]) + np.asarray(frame["noise"]), 0., 1.)
    mse = weighted_mse(effect, target, weights)
    oracle_mse = weighted_mse(oracle_effect, target, weights)
    total_cost = float(np.sum(action * np.asarray(frame["costs"])))
    violation = bool((action < -1e-10).any() or (action > 1+1e-10).any() or
                     total_cost > frame["budget"] + 1e-8 or
                     np.any(np.abs(action[~np.asarray(frame["allowed"], bool)]) > 1e-10))
    # Descriptors explicitly reward homogeneity and temporal resemblance only.
    net = action[..., 0] - action[..., 1]
    homogeneity = float(1 - np.mean((net - np.mean(net))**2) / 4.)
    return dict(mse=mse, oracle_mse=oracle_mse, regret=mse-oracle_mse,
                cost=total_cost, hard_violation=violation, node_homogeneity=homogeneity)


def sustained_recovery(regrets, start, end, threshold=.015, sustained=3):
    """First three-success run; None means right censored, never reported as 0."""
    for t in range(start, end - sustained + 1):
        if np.all(np.asarray(regrets[t:t+sustained]) <= threshold):
            return t-start
    return None


def summarize_run(frames, records, threshold=.015, sustained=3, success_rule=None):
    if len(frames) != len(records) or not frames:
        raise ValueError("Complete non-empty traces required")
    metrics = [step_metrics(f, r["action"], r["effect"]) for f, r in zip(frames, records)]
    regrets = np.array([m["regret"] for m in metrics])
    mse = np.array([m["mse"] for m in metrics])
    switches = [i for i, f in enumerate(frames) if f["switch"]]
    recoveries = [sustained_recovery(regrets, s, switches[j+1] if j+1 < len(switches) else len(frames), threshold, sustained)
                  for j, s in enumerate(switches)]
    observed = [r for r in recoveries if r is not None]
    recall = [i for i, f in enumerate(frames) if f["recall"]]
    reacq = [i for i, f in enumerate(frames) if f["reacquisition"]]
    # Desired-effect polarity discrimination penalizes inactivity and uniformity.
    desired = np.concatenate([np.asarray(f["target"]).ravel() for f in frames])
    actual = np.concatenate([np.asarray(r["effect"]).ravel() for r in records])
    high, low = desired > .4, desired < .05
    discrimination = float(np.mean(actual[high])-np.mean(actual[low])) if high.any() and low.any() else None
    # Correlation is descriptive, not sufficient; constant sequences have no value.
    correlation = float(np.corrcoef(desired, actual)[0, 1]) if np.std(actual)>1e-12 and np.std(desired)>1e-12 else None
    violations = sum(m["hard_violation"] for m in metrics)
    temporal_cosines = []
    for previous, current in zip(records, records[1:]):
        a, b = np.asarray(previous["action"]).ravel(), np.asarray(current["action"]).ravel()
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        if denominator > 1e-12:
            temporal_cosines.append(float(np.dot(a, b)/denominator))
    recall_regret = float(np.mean(regrets[recall])) if recall else None
    recovery_fraction = len(observed) / len(recoveries) if recoveries else None
    rule = success_rule or dict(mean_regret_max=.015, hard_violations_max=0,
                               effect_discrimination_min=.15, recovery_fraction_min=.6, recall_regret_max=.015)
    passed = bool(np.mean(regrets) <= rule["mean_regret_max"] and violations <= rule["hard_violations_max"] and
                  discrimination is not None and discrimination >= rule["effect_discrimination_min"] and
                  recovery_fraction is not None and recovery_fraction >= rule["recovery_fraction_min"] and
                  (recall_regret is None or recall_regret <= rule["recall_regret_max"]))
    return dict(tracking_rmse=float(np.sqrt(np.mean(mse))), mean_regret=float(np.mean(regrets)),
                mean_cost=float(np.mean([m["cost"] for m in metrics])),
                hard_violations=int(violations), mean_recovery_steps=float(np.mean(observed)) if observed else None,
                recovery_censored=int(len(recoveries)-len(observed)), recovery_events=len(recoveries),
                recovery_fraction=recovery_fraction, recall_regret=recall_regret,
                reacquisition_regret=float(np.mean(regrets[reacq])) if reacq else None,
                effect_discrimination=discrimination, effect_target_correlation=correlation,
                mean_node_homogeneity=float(np.mean([m["node_homogeneity"] for m in metrics])),
                mean_temporal_action_cosine=float(np.mean(temporal_cosines)) if temporal_cosines else None,
                external_pass=passed)


def paired_effect(reference, comparator, seed=20260906, bootstrap=2000):
    """Paired reference-minus-comparator effect with seed-level percentile CI.

    Positive regret differences favor comparator. dz undefined at zero variance.
    """
    a, b = np.asarray(reference, float), np.asarray(comparator, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require >=2 complete, finite, paired seed-level values")
    d = a-b
    draws = np.random.default_rng(seed).integers(0, len(d), (bootstrap, len(d)))
    means = np.mean(d[draws], axis=1)
    sd = float(np.std(d, ddof=1))
    return dict(n=len(d), mean_difference=float(np.mean(d)),
                ci95_low=float(np.quantile(means, .025)), ci95_high=float(np.quantile(means, .975)),
                paired_dz=float(np.mean(d)/sd) if sd>1e-14 else None,
                dz_undefined=sd<=1e-14, multiplicity="exploratory_unadjusted")
