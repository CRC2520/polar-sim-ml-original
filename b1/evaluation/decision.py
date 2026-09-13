"""Fail-closed interpreter of frozen predicates and development-only exporter.

The predicate engine is exercised with explicitly synthetic rule fixtures. Real
development records never obtain confirmatory verdicts, even when their numbers
would satisfy a frozen confirmatory predicate. B1-E execution is unavailable.
"""
from enum import Enum
from math import isfinite

from b1.contracts import load_contracts
from b1.metrics import guardrails_from_counts
from .precision import bounded_interval, expanded_registry, precision_plan


class Truth(Enum):
    FALSE = "false"
    TRUE = "true"
    UNKNOWN = "unknown"


class DecisionError(ValueError):
    pass


_MISSING = object()


def resolve(record, path):
    current = record
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return _MISSING if current is None else current


def _numeric(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


class PredicateEvaluator:
    def __init__(self, contracts=None):
        self.contracts = contracts or load_contracts()
        self.rules = self.contracts.rules

    def evaluate(self, node, record):
        if not isinstance(node, dict):
            raise DecisionError("A predicate must be a structured mapping")
        composites = set(node) & {"all", "any", "not"}
        if composites:
            if len(node) != 1:
                raise DecisionError("A composite predicate has exactly one node")
            operation = next(iter(composites))
            if operation == "not":
                result = self.evaluate(node[operation], record)
                return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE,
                        Truth.UNKNOWN: Truth.UNKNOWN}[result]
            if not isinstance(node[operation], list) or not node[operation]:
                raise DecisionError("all/any require a nonempty predicate list")
            results = [self.evaluate(child, record) for child in node[operation]]
            if operation == "all":
                return (Truth.FALSE if Truth.FALSE in results else
                        Truth.UNKNOWN if Truth.UNKNOWN in results else Truth.TRUE)
            return (Truth.TRUE if Truth.TRUE in results else
                    Truth.UNKNOWN if Truth.UNKNOWN in results else Truth.FALSE)
        if not set(node).issubset(self.rules["predicate_language"]["leaf_fields"]):
            raise DecisionError("Unknown leaf field")
        op, path = node.get("op"), node.get("field")
        if op not in self.rules["predicate_language"]["allowed_operators"] or not isinstance(path, str):
            raise DecisionError("Invalid predicate operator or field")
        actual = resolve(record, path)
        if op == "present":
            if "value" in node or "threshold_ref" in node:
                raise DecisionError("present has no comparison operand")
            return Truth.FALSE if actual is _MISSING else Truth.TRUE
        if ("value" in node) == ("threshold_ref" in node):
            raise DecisionError("Exactly one literal or threshold operand is required")
        target = (resolve(self.rules["thresholds"], node["threshold_ref"])
                  if "threshold_ref" in node else node["value"])
        if actual is _MISSING or target is _MISSING or target is None:
            return Truth.UNKNOWN
        if op in {"gt", "gte", "lt", "lte"}:
            if not _numeric(actual) or not _numeric(target):
                return Truth.UNKNOWN
            match = {"gt": actual > target, "gte": actual >= target,
                     "lt": actual < target, "lte": actual <= target}[op]
        elif op in {"eq", "ne"}:
            # Python's True == 1 is inappropriate for typed decision predicates.
            if type(actual) is not type(target) and not (_numeric(actual) and _numeric(target)):
                return Truth.UNKNOWN
            equal = type(actual) is type(target) and actual == target
            if _numeric(actual) and _numeric(target):
                equal = actual == target
            match = equal if op == "eq" else not equal
        elif op in {"in", "not_in"}:
            if not isinstance(target, list):
                raise DecisionError("Membership comparison requires a list")
            equal = any(type(actual) is type(item) and actual == item for item in target)
            match = equal if op == "in" else not equal
        else:
            raise DecisionError("Unhandled predicate operator")
        return Truth.TRUE if match else Truth.FALSE


def evaluate_rule(rule_id, record, *, data_class, contracts=None):
    """Test a frozen rule or report its default for development input.

    `synthetic-rule-test` is a unit-test fixture classification, never evidence.
    All B1-E / final classifications are rejected, not silently relabeled.
    """
    if data_class not in {"synthetic-rule-test", "B1-D"}:
        raise DecisionError("B1-E evaluation is unavailable in this development implementation")
    engine = PredicateEvaluator(contracts)
    if rule_id not in engine.rules["decisions"]:
        raise DecisionError(f"Unknown frozen decision: {rule_id}")
    rule = engine.rules["decisions"][rule_id]
    missing = [path for path in rule["required_fields"] if resolve(record, path) is _MISSING]
    result = {"rule_id": rule_id, "status": rule["default_status"],
              "missing_fields": missing, "record_complete": not missing,
              "aggregate_complete": not missing, "data_class": data_class,
              "confirmatory": False, "reusable_as_final": False}
    if missing or data_class == "B1-D":
        return result
    for index, case in enumerate(rule["cases"]):
        if engine.evaluate(case["when"], record) is Truth.TRUE:
            result.update(status=case["status"], matched_case=index)
            break
    return result


def _checks_pass(checks):
    return (isinstance(checks, dict) and bool(checks) and
            all(isinstance(check, dict) and "observed" in check and "expected" in check and
                check["observed"] is not None and check["expected"] is not None and
                _typed_equal(check["observed"], check["expected"]) for check in checks.values()))


def _typed_equal(first, second):
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_typed_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and type(first) is type(second):
        return len(first) == len(second) and all(_typed_equal(a, b) for a, b in zip(first, second))
    return first == second


def derive_audit(instrumentation, contracts=None):
    """Derive guard flags from counts and paired raw instrument checks.

    Precomputed guardrails_pass/common_admissible/eligibility fields are ignored.
    An axis requires left and right snapshots and exact equality. Approximate
    arithmetic checks belong in explicit certified-tolerance instrument checks.
    """
    contracts = contracts or load_contracts()
    rules = contracts.rules
    guardrails = guardrails_from_counts(instrumentation.get("guardrail_counts", {}), contracts=contracts)
    axes = {}
    for name in rules["six_invariant_axes"]:
        item = instrumentation.get("invariants", {}).get(name, {})
        complete = (isinstance(item, dict) and "left" in item and "right" in item and
                    item["left"] is not None and item["right"] is not None)
        axes[name] = {"complete": complete,
                      "equal": complete and _typed_equal(item["left"], item["right"])}
    changed = instrumentation.get("manipulation_changed_axes")
    if not isinstance(changed, list) or not all(isinstance(x, str) for x in changed):
        changed = ["invalid_manipulation_description"]
    # A declared route contrast changing any additional resource is joint.
    joint = bool(changed)
    c6_pass = _checks_pass(instrumentation.get("c6_checks"))
    mediator = instrumentation.get("mediator_checks", {})
    mediator_pass = (set(mediator) >= {"delivery", "timing", "instrumentation"}
                     and _checks_pass(mediator))
    return {**guardrails, "invariant_axes": axes,
            "invariants_pass": all(x["equal"] for x in axes.values()),
            "joint_manipulation": joint, "changed_axes": changed,
            "sensitivity_pass": _checks_pass(instrumentation.get("sensitivity_checks")),
            "C6_conjugacy_pass": c6_pass,
            "mediator_contract_verified": mediator_pass,
            "blocked_reasons": list(instrumentation.get("blocked_reasons", []))}


def evaluate_development(pilot_id, raw_bundle_endpoints, instrumentation=None, *, contracts=None):
    """Export descriptive endpoint summaries without dropping missing endpoints.

    Input is {endpoint_id: [{bundle_index: int, value: float}, ...]}. IDs may
    include the pilot prefix. Bundles must be unique, within the B1-D ranges,
    and may not mix tuning with calibration inside one endpoint summary.
    Missing or blocked endpoints receive no imputed observation or effect.
    """
    contracts = contracts or load_contracts()
    rules = contracts.rules
    if pilot_id not in rules["scope"]["pilots"]:
        raise DecisionError(f"Unregistered pilot: {pilot_id}")
    instrumentation = instrumentation or {}
    registry = expanded_registry(contracts)
    canonical_ids = {name for name, spec in registry.items() if spec["pilot"] == pilot_id}
    ranges = rules["calibration"]["development_bundle_indices"]
    tuning_start, tuning_end = ranges["tuning"]
    calibration_start, calibration_end = ranges["blinded_calibration"]
    normalized = {}
    for name, rows in raw_bundle_endpoints.items():
        full_id = name if name.startswith(pilot_id + ".") else pilot_id + "." + name
        if full_id not in canonical_ids or full_id in normalized:
            raise DecisionError("Unknown or duplicate endpoint")
        normalized[full_id] = rows
    summaries = {}
    for name in sorted(canonical_ids):
        rows = normalized.get(name, [])
        if not rows:
            summaries[name] = {"status": "incomplete", "N": 0,
                               "missing_required_outcome": True,
                               "support": registry[name]["support"]}
            continue
        indices, values = [], []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"bundle_index", "value"}:
                raise DecisionError("Each endpoint row requires only bundle_index and value")
            index = row["bundle_index"]
            if (type(index) is not int or index in indices or
                    not (tuning_start <= index <= tuning_end or calibration_start <= index <= calibration_end)):
                raise DecisionError("Invalid or duplicate development bundle index")
            indices.append(index)
            values.append(row["value"])
        if min(indices) <= tuning_end and max(indices) >= calibration_start:
            raise DecisionError("Tuning and blinded calibration cannot be pooled")
        try:
            interval = bounded_interval(values, registry[name]["support"], contracts=contracts)
        except (ValueError, TypeError) as exc:
            summaries[name] = {"status": "invalid", "N": len(rows), "reason": str(exc)}
            continue
        summaries[name] = {**interval, "status": "descriptive_only",
                           "bundle_indices": sorted(indices)}
    audit = derive_audit(instrumentation, contracts)
    role = rules["pilot_roles"][pilot_id]
    complete = all(x["status"] == "descriptive_only" for x in summaries.values())
    common = (complete and audit["guardrails_pass"] and audit["sensitivity_pass"] and
              audit["invariants_pass"] and audit["C6_conjugacy_pass"] and
              not audit["joint_manipulation"] and not audit["blocked_reasons"])
    statuses = {name: {"status": rules["decisions"][name]["default_status"],
                       "reason": "development_data_cannot_support_confirmatory_verdict"}
                for name in rules["decisions"]}
    for name, flag in (("practical_utility", "positive_utility_eligible"),
                       ("pairing_specificity", "positive_pairing_specificity_eligible")):
        if role[flag] is False:
            statuses[name] = {"status": "not_eligible", "reason": role["reason"]}
    return {"pilot": pilot_id, "role": role["role"], "data_class": "B1-D",
            "development_only": True, "confirmatory": False, "reusable_as_final": False,
            "endpoint_count_registered_family": len(registry), "endpoints": summaries,
            "aggregate_complete": complete, "development_instrument_admissible": common,
            "audit": audit, "eligibility": {key: value for key, value in role.items() if key.endswith("eligible")},
            "hypotheses": statuses, "B1E_required_N": precision_plan(contracts)["B1E_required_N"],
            "architecture_comparison_blocked": bool(audit["blocked_reasons"])}


def evaluate_synthetic_pilot(pilot_id, scalar_records, instrumentation, *, contracts=None):
    """Exercise the complete ordered decision graph on artificial rule fixtures.

    Aggregate statuses and positive eligibility are always recomputed. This API
    is for instrument unit tests; output is explicitly synthetic and non-evidence.
    It cannot accept a development or confirmatory data classification.
    """
    contracts = contracts or load_contracts()
    rules, engine = contracts.rules, PredicateEvaluator(contracts)
    registry = expanded_registry(contracts)
    required_n = precision_plan(contracts)["B1E_required_N"]
    role = rules["pilot_roles"][pilot_id]
    audit = derive_audit(instrumentation, contracts)
    results = {}

    def scalar(endpoint, rule_id, extra=None):
        raw = scalar_records.get(endpoint, {})
        record = {key: raw[key] for key in ("mean", "ci_lower", "ci_upper", "N") if key in raw}
        support = registry[pilot_id + "." + endpoint]["support"]
        numeric_complete = all(key in record and _numeric(record[key]) for key in
                               ("mean", "ci_lower", "ci_upper", "N"))
        interval_ok = bool(numeric_complete and type(record["N"]) is int and
            record["N"] == required_n and
            support[0] <= record["ci_lower"] <= record["mean"] <= record["ci_upper"] <= support[1])
        common_inputs = {"record_complete": numeric_complete,
            "freeze_valid": True, "collection_complete": raw.get("N") == required_n,
            "intervals_valid": interval_ok, "guardrails_pass": audit["guardrails_pass"],
            "sensitivity_pass": audit["sensitivity_pass"] and audit["C6_conjugacy_pass"],
            "invariants_pass": audit["invariants_pass"], "joint_manipulation": audit["joint_manipulation"]}
        record["common_admissible"] = (engine.evaluate(rules["guards"]["common_admissible"], common_inputs)
                                       is Truth.TRUE and not audit["blocked_reasons"])
        record["mediator_contract_verified"] = audit["mediator_contract_verified"]
        if extra:
            record.update(extra)
        result = evaluate_rule(rule_id, record, data_class="synthetic-rule-test", contracts=contracts)
        result["admissible"] = record["common_admissible"]
        results[endpoint] = result
        return result

    first = scalar("mechanism.kappa1", "mechanism_effect")
    second = scalar("mechanism.kappa2", "neutral_context_effect")
    mechanism = evaluate_rule("mechanism_contract", {
        "aggregate_complete": all(x["record_complete"] and x["admissible"] for x in (first, second)),
        "kappa1": first, "kappa2": second}, data_class="synthetic-rule-test", contracts=contracts)
    comparisons = {name: scalar("utility." + name, "practical_utility_comparison")
                   for name in rules["required_pairing_comparators"]}
    for endpoint in ("acute_route.kappa1", "acute_route.kappa2"):
        scalar(endpoint, "acute_route_effect")
    aggregates = {"mechanism_contract": mechanism}
    for decision, required, eligibility in (
        ("practical_utility", rules["required_utility_comparators"], "positive_utility_eligible"),
        ("pairing_specificity", rules["required_pairing_comparators"], "positive_pairing_specificity_eligible")):
        complete = all(comparisons[name]["record_complete"] and comparisons[name]["admissible"] for name in required)
        aggregates[decision] = evaluate_rule(decision, {
            "aggregate_complete": complete, eligibility: role[eligibility],
            "all_required_comparisons_admissible": complete, "comparators": comparisons},
            data_class="synthetic-rule-test", contracts=contracts)
    scalar("context.interaction", "context_modulation", {"kappa1": first, "kappa2": second})
    aggregates["generalization"] = evaluate_rule("generalization", {}, data_class="synthetic-rule-test", contracts=contracts)
    return {"data_class": "synthetic-rule-test", "synthetic_fixture": True,
            "scientific_evidence": False, "confirmatory": False,
            "scalar_decisions": results, "aggregate_decisions": aggregates, "audit": audit}
