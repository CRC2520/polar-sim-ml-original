"""Machine-readable matching records with conservative claim restrictions.

Equality of recorded allocations is not certification of an unmeasured axis.
Pending runtime or worst-case certification is explicit and prevents a PASS.
"""
from copy import deepcopy

from .core import Limits

AXES = ("available_information", "restrictions_and_feasibility",
        "learning_and_training_resources", "memory",
        "response_and_planning_dynamics", "decision_compute")
EXTRA = ("action_space", "communication_width", "planning_horizon", "adapter",
         "training_data", "hyperparameter_search", "latency")


def profile(comparator, *, limits=None, training_data_hash=None,
            training_operations=None, config_manifest_hash=None,
            resource_certified=False):
    limits = limits or Limits()
    larger = comparator == "C5"
    return {
        "comparator": comparator,
        "available_information": {"raw_cells": [0, 1, 2], "hidden_future": False,
                                  "public_context": True, "history": "complete task-provided public records"},
        "restrictions_and_feasibility": "same task final gate, permissions, operations and resources",
        "learning_and_training_resources": {"operation_allowance": training_operations,
                                            "history_hash": training_data_hash,
                                            "selection": "frozen finite configuration family"},
        "memory": {"scalar_cap": limits.memory_scalars * (2 if larger else 1),
                   "raw_snapshots": 2 if larger else 1, "reset": "each episode"},
        "response_and_planning_dynamics": {"cadence": "one decision/epoch",
                                           "future_horizon": 1, "task_delay": "unchanged"},
        "decision_compute": {"operation_cap": limits.decision_operations * (2 if larger else 1),
                             "unit": "declared useful-operation proxy", "certified": resource_certified},
        "action_space": "same A/B requests and pilot-specific authorized targets",
        "communication_width": {"slots": 6 if larger else 3,
                                "payload_schema": "canonical pilot typed route; source IDs retained"},
        "planning_horizon": 1, "adapter": "same task physical admission/execution adapter",
        "training_data": training_data_hash,
        "hyperparameter_search": {"slots": 1 if comparator == "C4" else 8,
                                  "manifest_hash": config_manifest_hash,
                                  "fixed_analytical_exception": comparator == "C4"},
        "latency": limits.latency_seconds,
    }


def audit_contrast(left, right, *, contrast_kind="competitive_training",
                   expected_differences=(), gamma_manipulated=False,
                   intended_source_intervention=False):
    if contrast_kind not in {"competitive_training", "acute_route", "source_intervention",
                             "context", "algebraic_control"}:
        raise ValueError("Unknown contrast kind")
    if intended_source_intervention and contrast_kind != "source_intervention":
        raise ValueError("Operational treatment exception only belongs to a source intervention")
    expected = set(expected_differences)
    all_fields = AXES + EXTRA
    if expected - set(all_fields):
        raise ValueError("Unknown expected invariant field")
    items = {}
    for key in all_fields:
        same = left.get(key) == right.get(key)
        # C4 has a genuine fixed-controller exception; it cannot receive a
        # fabricated eight-candidate search to make the metadata look equal.
        fixed_exception = key == "hyperparameter_search" and "C4" in {
            left.get("comparator"), right.get("comparator")}
        items[key] = {"left": deepcopy(left.get(key)), "right": deepcopy(right.get(key)),
                      "equal": same, "unequal_axis": not same,
                      "expected_difference": key in expected,
                      "fixed_controller_exception": fixed_exception}
    pending = []
    for name, p in (("left", left), ("right", right)):
        if p.get("training_data") is None:
            pending.append(name + ":training_data_provenance")
        if p.get("learning_and_training_resources", {}).get("operation_allowance") is None:
            pending.append(name + ":training_operation_allowance")
        if p.get("hyperparameter_search", {}).get("manifest_hash") is None:
            pending.append(name + ":configuration_freeze")
        if p.get("latency") is None:
            pending.append(name + ":latency_calibration")
        if not p.get("decision_compute", {}).get("certified", False):
            pending.append(name + ":smallest_common_compute_certification")
    changed = [k for k in all_fields if not items[k]["equal"]]
    unexpected = [k for k in changed if k not in expected and not items[k]["fixed_controller_exception"]]
    causal_axes = {"available_information", "restrictions_and_feasibility", "memory",
                   "decision_compute", "action_space", "communication_width", "adapter"}
    # Labeling an information loss 'expected' does not rescue selective Gamma
    # attribution. Genuine source doses/mediators are separately scoped.
    joint = bool(gamma_manipulated and set(changed) & causal_axes and not intended_source_intervention)
    descriptive = "C5" in {left.get("comparator"), right.get("comparator")}
    matched = not unexpected and not pending and not joint and not descriptive
    return {"contrast_kind": contrast_kind, "axes": items,
            "unequal_axis": bool(changed), "changed_fields": changed,
            "unexpected_differences": unexpected, "pending_certification": pending,
            "joint_manipulation": joint, "expected_source_intervention": intended_source_intervention,
            "matching_passed": matched,
            "gamma_specific_verdict_allowed": matched and not intended_source_intervention,
            "claim_limit": ("descriptive_capacity_frontier" if descriptive else
                            "joint_manipulation_no_gamma_specific_verdict" if joint else
                            "pending_invariant_certification" if pending else
                            "unmatched_no_specificity" if unexpected else
                            "operational_source_and_mediator_only" if intended_source_intervention else
                            "matched_within_registered_scope")}


def mean_both_c2_cycles(loss_by_cycle):
    if set(loss_by_cycle) != {1, 2}:
        raise ValueError("Both fixed C2 cycles must be retained")
    return (loss_by_cycle[1] + loss_by_cycle[2]) / 2

