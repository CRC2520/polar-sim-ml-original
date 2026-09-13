"""Analytical frozen precision design; this module never generates any seed."""
from math import ceil, isfinite, log, sqrt
from b1.contracts import ContractError, load_contracts


def expanded_registry(contracts=None):
    rules = (contracts or load_contracts()).rules
    registry = rules["contrast_registry"]
    endpoints = {}
    for pilot in registry["pilots"]:
        for template in registry["endpoint_templates"]:
            name = pilot + "." + template["id"]
            if name in endpoints:
                raise ContractError("Duplicate endpoint in frozen registry")
            endpoints[name] = dict(template, pilot=pilot, full_id=name,
                support=registry["support_overrides"].get(name, template["support"]))
    if len(endpoints) != registry["endpoint_count"] or len(endpoints) != rules["thresholds"]["family_size"]:
        raise ContractError("Registry size, endpoint count and family size differ")
    if not set(registry["support_overrides"]).issubset(endpoints):
        raise ContractError("Unregistered support override")
    return endpoints


def precision_plan(contracts=None):
    contracts = contracts or load_contracts()
    thresholds = contracts.rules["thresholds"]
    alpha, family_size = thresholds["alpha"], thresholds["family_size"]
    if not 0 < alpha < 1 or type(family_size) is not int or family_size < 1:
        raise ContractError("Invalid familywise precision inputs")
    endpoints = expanded_registry(contracts)
    calculations = {}
    for name, spec in endpoints.items():
        lower, upper = spec["support"]
        family = thresholds[spec["margin_family"]]
        h_star = family["target_half_width"]
        if not (all(isfinite(x) for x in (lower, upper, h_star)) and
                lower < upper and h_star > 0 and
                h_star == family["epsilon"] / 2 and
                family["epsilon"] == family["delta"] / 2):
            raise ContractError(f"Inconsistent precision requirements: {name}")
        width = upper - lower
        n = ceil(width * width * log(2 * family_size / alpha) / (2 * h_star * h_star))
        calculations[name] = {"support": [lower, upper], "range": width,
            "margin_family": spec["margin_family"], "target_half_width": h_star,
            "required_N": n}
    common_n = max(item["required_N"] for item in calculations.values())
    return {"method": contracts.rules["sampling_and_intervals"]["interval_method"],
        "alpha": alpha, "family_size": family_size, "endpoint_count": len(endpoints),
        "B1E_required_N": common_n, "per_endpoint": calculations,
        "power_claim": "none_fixed_sample_precision_design",
        "final_seeds_generated": False, "B1E_executed": False,
        "feasibility": "requires_development_only_resource_assessment",
        "infeasible_action": "block_B1_E_no_cap_and_proceed"}


def bounded_interval(values, support, *, contracts=None):
    """Descriptive development interval under the same stated bounded-mean formula.

    A development interval is never a confirmatory result and is not substituted
    for the future fixed N or preregistration.
    """
    contracts = contracts or load_contracts()
    values = tuple(values)
    lower, upper = support
    if not values or not (isfinite(lower) and isfinite(upper) and lower < upper):
        raise ValueError("A nonempty bundle sample and finite nondegenerate support are required")
    if any(isinstance(x, bool) or not isfinite(x) or not lower <= x <= upper for x in values):
        raise ValueError("A bundle observation is nonfinite or outside its registered support")
    thresholds = contracts.rules["thresholds"]
    mean = sum(values) / len(values)
    h = (upper - lower) * sqrt(log(2 * thresholds["family_size"] / thresholds["alpha"]) / (2 * len(values)))
    return {"mean": mean, "ci_lower": max(lower, mean - h),
        "ci_upper": min(upper, mean + h), "N": len(values),
        "support": [lower, upper], "half_width_unclipped": h,
        "family_size": thresholds["family_size"], "development_only": True,
        "confirmatory": False, "reusable_as_final": False}
