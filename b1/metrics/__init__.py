"""Metrics with denominators and limits imported from the frozen registry."""
from fractions import Fraction
from b1.contracts import load_contracts


def primary_loss(pilot_id, failed_jobs, *, controller_failure=False, contracts=None):
    contracts = contracts or load_contracts()
    metric = contracts.rules["metric_contracts"][f"{pilot_id}.primary_loss"]
    denominator = metric["denominator"]
    if type(failed_jobs) is not int or not 0 <= failed_jobs <= denominator:
        raise ValueError("failed_jobs must be an integer within the frozen denominator")
    if type(controller_failure) is not bool:
        raise ValueError("controller_failure must be boolean")
    return Fraction(denominator if controller_failure else failed_jobs, denominator)


def guardrails_from_counts(counts, *, contracts=None):
    contracts = contracts or load_contracts()
    limits = contracts.rules["thresholds"]["guardrails"]
    failures, missing = [], []
    for limit_name, limit in limits.items():
        name = limit_name.removeprefix("max_")
        value = counts.get(name)
        if type(value) is not int or value < 0:
            missing.append(name)
        elif value > limit:
            failures.append(name)
    return {"guardrails_pass": not failures and not missing,
            "violations": failures, "missing_or_invalid": missing}


def paired_difference(first, second):
    """Pairing is by bundle upstream; no epoch may become an independent sample."""
    for value in (first, second):
        if isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError("Both loss values must be bounded failure fractions")
    return first - second
