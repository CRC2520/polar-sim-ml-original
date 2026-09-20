"""Pure, support-respecting population-threshold summaries for R8.

Rows of ``success`` represent independent seeds; columns represent the same
seeds at ordered population fractions.  Bootstrap resampling therefore draws
whole rows.  No simulator or experiment I/O is imported by this module.

Clopper--Pearson intervals are exact binomial intervals under independent seeds.
The simultaneous version uses Bonferroni adjustment over all grid points and
does not require independence across columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta


GRID_DEV = [float(index / 10) for index in range(11)]
PILOT_PRESSURES = (0.30, 0.45, 0.60)
THRESHOLD_STATUSES = (
    "at_or_below_support",
    "above_support_or_absent",
    "no_unique_crossing",
    "within_support",
)


def viable_seed(alive: Any, resource: Any) -> bool | list:
    """Apply both inclusive viability criteria, returning JSON-safe booleans.

    Inputs must have matching shapes and finite fractions in [0, 1].  Missing
    observations must be resolved upstream instead of being counted as failures.
    """
    alive_array = np.asarray(alive, dtype=float)
    resource_array = np.asarray(resource, dtype=float)
    if alive_array.shape != resource_array.shape:
        raise ValueError("alive and resource must have identical shapes")
    for name, values in (("alive", alive_array), ("resource", resource_array)):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} must contain finite fractions in [0, 1]")
    return ((alive_array >= 0.8) & (resource_array >= 0.2)).tolist()


def _validated_inputs(grid: Any, success: Any) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.asarray(grid, dtype=float)
    if fractions.ndim != 1 or len(fractions) < 2:
        raise ValueError("grid must be a one-dimensional sequence of at least two points")
    if not np.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("grid must contain finite fractions in [0, 1]")
    if not (np.diff(fractions) > 0).all():
        raise ValueError("grid must be strictly increasing")
    outcomes = np.asarray(success, dtype=float)
    if outcomes.ndim != 2 or outcomes.shape[0] < 1 or outcomes.shape[1] != len(fractions):
        raise ValueError("success must have shape (n_seeds >= 1, len(grid))")
    if not np.isfinite(outcomes).all() or not np.isin(outcomes, [0, 1]).all():
        raise ValueError("success must contain only binary, finite seed outcomes")
    return fractions, outcomes


def _point_threshold(fractions: np.ndarray, probabilities: np.ndarray) -> dict:
    """Identify one crossing of the >= .5 indicator, without smoothing."""
    high = probabilities >= 0.5
    changes = np.diff(high.astype(np.int8))
    if (changes < 0).any():
        return {"status": "no_unique_crossing", "estimate": None, "crossing_interval": None}
    if high.all():
        return {"status": "at_or_below_support", "estimate": None, "crossing_interval": None}
    if not high.any():
        return {"status": "above_support_or_absent", "estimate": None, "crossing_interval": None}
    crossings = np.flatnonzero(changes > 0)
    if len(crossings) != 1:
        return {"status": "no_unique_crossing", "estimate": None, "crossing_interval": None}
    left = int(crossings[0])
    right = left + 1
    estimate = fractions[left] + (0.5 - probabilities[left]) * (
        fractions[right] - fractions[left]
    ) / (probabilities[right] - probabilities[left])
    return {
        "status": "within_support",
        "estimate": float(estimate),
        "crossing_interval": [float(fractions[left]), float(fractions[right])],
    }


def _clopper_pearson(successes: np.ndarray, n: int, alpha: float) -> np.ndarray:
    counts = np.asarray(successes, dtype=int)
    lower = np.zeros(len(counts), dtype=float)
    upper = np.ones(len(counts), dtype=float)
    positive = counts > 0
    imperfect = counts < n
    lower[positive] = beta.ppf(alpha / 2.0, counts[positive], n - counts[positive] + 1)
    upper[imperfect] = beta.ppf(1.0 - alpha / 2.0, counts[imperfect] + 1, n - counts[imperfect])
    return np.column_stack((lower, upper))


def estimate_threshold(
    grid: Any, success: Any, bootstrap: int = 2000, seed: int = 946991
) -> dict:
    """Summarize an observed .5 viability crossing and its uncertainty.

    Values between adjacent fractions are interpolated only when those adjacent
    observations cross .5.  A decrease on one side of .5 is allowed; a high-to-low
    reversal is not.  No estimate is supplied for a crossing outside support.

    A confidence bracket requires both a simultaneously confident low point and
    a simultaneously confident high point in the appropriate order.  Reliable
    identification additionally requires >= 80% of paired bootstrap replicates
    to have one observed crossing.  The bootstrap interval is explicitly
    conditional on those in-support replicates, with all other statuses counted.
    """
    fractions, outcomes = _validated_inputs(grid, success)
    if isinstance(bootstrap, (bool, np.bool_)) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    bootstrap, seed = int(bootstrap), int(seed)
    n = int(outcomes.shape[0])
    probabilities = outcomes.mean(axis=0)
    point = _point_threshold(fractions, probabilities)
    successes = outcomes.sum(axis=0).astype(int)
    marginal = _clopper_pearson(successes, n, alpha=0.05)
    simultaneous = _clopper_pearson(successes, n, alpha=0.05 / len(fractions))
    confident_low = np.flatnonzero(simultaneous[:, 1] < 0.5)
    confident_high = np.flatnonzero(simultaneous[:, 0] > 0.5)
    confidence_bracket = None
    confidence_status = "insufficient_confident_endpoints"
    confident_reversal = bool(
        len(confident_high) and len(confident_low) and confident_high.min() < confident_low.max()
    )
    if confident_reversal:
        point = {"status": "no_unique_crossing", "estimate": None, "crossing_interval": None}
        confidence_status = "no_unique_crossing"
    elif len(confident_low) and len(confident_high):
        left, right = int(confident_low.max()), int(confident_high.min())
        if left < right:
            confidence_bracket = [float(fractions[left]), float(fractions[right])]
            confidence_status = "ordered_confidence_bracket"

    counts = {status: 0 for status in THRESHOLD_STATUSES}
    in_support = []
    generator = np.random.default_rng(seed)
    # Each draw preserves the within-seed dependence over the complete grid.
    for _ in range(bootstrap):
        rows = generator.integers(0, n, size=n)
        resampled = _point_threshold(fractions, outcomes[rows].mean(axis=0))
        counts[resampled["status"]] += 1
        if resampled["status"] == "within_support":
            in_support.append(resampled["estimate"])
    valid_fraction = len(in_support) / bootstrap if bootstrap else None
    conditional_ci = np.quantile(in_support, [0.025, 0.975]).tolist() if in_support else None
    return {
        "status": point["status"],
        "estimate": point["estimate"],
        "crossing_interval": point["crossing_interval"],
        "grid": fractions.tolist(),
        "n_seeds": n,
        "successes": successes.tolist(),
        "probability": probabilities.tolist(),
        "marginal_ci95": marginal.tolist(),
        "simultaneous_ci95": simultaneous.tolist(),
        "interval_method": "Clopper-Pearson exact binomial; simultaneous 95% with Bonferroni over all grid points",
        "confidence_bracket": confidence_bracket,
        "confidence_bracket_status": confidence_status,
        "confident_reversal": confident_reversal,
        "bootstrap": {
            "replicates": bootstrap,
            "seed": seed,
            "paired_across_grid": True,
            "status_counts": counts,
            "valid_fraction": valid_fraction,
            "conditional_in_support_ci95": conditional_ci,
        },
        "reliable_identification": bool(
            point["status"] == "within_support"
            and confidence_bracket is not None
            and valid_fraction is not None
            and valid_fraction >= 0.8
        ),
    }


def choose_pressure_and_grid(records: list[dict]) -> dict:
    """Select only from the six declared pilot cells, retaining an audit trail.

    Each record contains metabolism, rule, seeds, grid, and binary ``viable``
    shaped (seed, fraction).  All cells must use the same ordered seeds and
    GRID_DEV.  Eligibility uses the prospectively specified observed endpoint
    proportions and crossing topology, not confidence or favorable bootstrap
    draws.  This function does not run an experiment or access final seeds.
    """
    if not isinstance(records, (list, tuple)) or len(records) != 6:
        raise ValueError("pilot selection requires exactly two rules by three pressures")
    cells: dict[tuple[float, str], dict] = {}
    rules = set()
    reference_seeds = None
    for record in records:
        pressure = float(record["metabolism"])
        matches = [item for item in PILOT_PRESSURES if abs(pressure - item) < 1e-12]
        if len(matches) != 1:
            raise ValueError("pilot pressures must be 0.30, 0.45, and 0.60")
        pressure = matches[0]
        rule = record["rule"]
        if not isinstance(rule, str) or not rule:
            raise ValueError("rule must be a nonempty string")
        rules.add(rule)
        fractions, outcomes = _validated_inputs(record["grid"], record["viable"])
        if len(fractions) != len(GRID_DEV) or not np.allclose(fractions, GRID_DEV, rtol=0, atol=1e-12):
            raise ValueError("every pilot cell must use GRID_DEV")
        seeds = np.asarray(record["seeds"])
        if seeds.ndim != 1 or len(seeds) != len(outcomes):
            raise ValueError("seeds must identify every row of viable")
        if not np.issubdtype(seeds.dtype, np.integer) or len(np.unique(seeds)) != len(seeds):
            raise ValueError("seeds must be unique integers")
        if reference_seeds is None:
            reference_seeds = seeds.copy()
        elif not np.array_equal(seeds, reference_seeds):
            raise ValueError("pilot cells must use the same seeds in the same row order")
        key = (pressure, rule)
        if key in cells:
            raise ValueError("duplicate pilot pressure/rule cell")
        cells[key] = estimate_threshold(fractions, outcomes, bootstrap=0)
    if len(rules) != 2 or any((pressure, rule) not in cells for pressure in PILOT_PRESSURES for rule in rules):
        raise ValueError("pilot selection requires a complete two-rule by three-pressure design")

    candidates = []
    for pressure in PILOT_PRESSURES:
        summaries = {rule: cells[pressure, rule] for rule in sorted(rules)}
        endpoints_valid = all(
            summary["probability"][0] <= 0.2 and summary["probability"][-1] >= 0.8
            for summary in summaries.values()
        )
        crossings_valid = all(summary["status"] == "within_support" for summary in summaries.values())
        eligible = endpoints_valid and crossings_valid
        estimates = {rule: summary["estimate"] for rule, summary in summaries.items()}
        mean_estimate = float(np.mean(list(estimates.values()))) if crossings_valid else None
        candidates.append({
            "pressure": pressure,
            "eligible": eligible,
            "endpoint_criterion_met": endpoints_valid,
            "unique_crossings": crossings_valid,
            "thresholds": estimates,
            "mean_threshold": mean_estimate,
            "distance_from_half": abs(mean_estimate - 0.5) if mean_estimate is not None else None,
            "by_rule": summaries,
        })
    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    common = {
        "selection_basis": "pilot only; both endpoint criteria and one observed crossing per rule",
        "pilot_seeds": reference_seeds.astype(int).tolist(),
        "candidates": candidates,
    }
    if not eligible_candidates:
        return {
            **common,
            "status": "no_supported_pilot_crossing",
            "pressure": 0.3,
            "grid": GRID_DEV.copy(),
            "grid_reason": "diagnostic_full_support_no_eligible_pilot",
            "center": None,
            "selected_thresholds": None,
            "threshold_spread": None,
            "final_diagnostic_only": True,
        }
    # Round the score only to make mathematically equal distances tie reliably.
    selected = min(eligible_candidates, key=lambda candidate: (round(candidate["distance_from_half"], 12), candidate["pressure"]))
    values = list(selected["thresholds"].values())
    spread = float(max(values) - min(values))
    # Half-way rounding is specified as upward; binary float noise is tolerated.
    center_ticks = int(np.floor(selected["mean_threshold"] * 20 + 0.5 + 1e-12))
    center = center_ticks / 20
    requested_local = [(center_ticks + offset) / 20 for offset in range(-3, 4)]
    supported_local = [value for value in requested_local if 0 <= value <= 1]
    if spread <= 0.3 + 1e-12:
        final_grid = sorted(set([0.0, 1.0, *supported_local]))
        grid_reason = "local_refinement_with_support_endpoints"
    else:
        final_grid = GRID_DEV.copy()
        grid_reason = "full_support_rule_thresholds_differ_by_more_than_0.30"
    return {
        **common,
        "status": "selected",
        "pressure": selected["pressure"],
        "grid": final_grid,
        "grid_reason": grid_reason,
        "center": center,
        "selected_thresholds": selected["thresholds"],
        "threshold_spread": spread,
        "local_points_requested": requested_local,
        "local_points_in_support": supported_local,
        "final_diagnostic_only": False,
    }
