#!/usr/bin/env python3
"""Post-run adjudication of a disclosed frozen-code/protocol discrepancy.

The frozen evaluator and its output are never changed. This separate interpreter
requires that *only* the aligned regime meets practical benefit before choosing
the conditional-prior pivot. It uses existing numeric results, not a new test.

Reproduce: python scripts/adjudicate_study2.py
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT = ROOT / "results_study2/coupling/final/generated/decision.json"
PROTOCOL_INPUT = ROOT / "docs/study2_protocol.json"
MANIFEST_INPUT = ROOT / "results_study2/coupling/final/manifest.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be an explicit finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def upper(estimate, name):
    interval = estimate["ci95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{name} requires a two-sided interval")
    low, high = (number(value, f"{name}.ci95") for value in interval)
    if low > high:
        raise ValueError(f"{name} interval is reversed")
    return high


def adjudicate(raw, protocol):
    """Return interpretation without mutating input or recomputing statistics."""
    if raw["split"] != "final":
        raise ValueError("Post-run adjudication requires final results, not pilot output")
    if protocol["schema"] != "polar-study2-protocol-1.0" or \
       "If only aligned paired regimes meet practical benefit" not in protocol["decision_rule"]:
        raise ValueError("Unsupported protocol wording; an explicit new interpretation is required")
    if protocol["regimes"] != ["paired", "diagonal", "misaligned"]:
        raise ValueError("Unsupported regime mapping")
    delta = number(protocol["minimum_practical_improvement"], "minimum_practical_improvement", nonnegative=True)
    limits = protocol["guardrails"]
    guards = raw["guards"]
    guard_fields = {
        "failures": "failure_rate_noninferiority_margin",
        "action_cost": "action_cost_ratio_max",
        "return_loss": "return_loss_difference_max",
        "stockout": "inventory_stockout_difference_max",
        "overflow": "inventory_overflow_difference_max",
    }
    guard_pass = {name: upper(guards[name], name) <= number(limits[threshold], threshold)
                  for name, threshold in guard_fields.items()}
    guard_pass["hard_constraints"] = number(raw["aggregates"]["paired"]["hard_violations"], "hard_violations", nonnegative=True) <= number(limits["hard_violations_max"], "hard_violations_max", nonnegative=True)
    guard_pass["runtime"] = number(raw["runtime_ratio"], "runtime_ratio", nonnegative=True) <= number(limits["runtime_ratio_max"], "runtime_ratio_max", nonnegative=True)
    negative_failures = {name: number(raw["negative_control_failures"][name], f"negative_control_failures.{name}", nonnegative=True)
                         for name in ("zero", "hold")}
    negative_valid = all(value > 0 for value in negative_failures.values())
    equiv_valid = number(raw["max_signed_action_difference"], "max_signed_action_difference", nonnegative=True) <= number(protocol["signed_equivalence_tolerance"], "signed_equivalence_tolerance", nonnegative=True)
    practical_primary = upper(raw["primary"], "primary") < -delta
    practical_by_regime = {name: upper(raw["regimes"][name], f"regimes.{name}") < -delta
                           for name in protocol["regimes"]}
    shuffled_benefit = upper(raw["comparisons"]["shuffled"], "paired_minus_shuffled") < 0
    dense_noninferiority = upper(raw["dense_transfer"], "dense_transfer") <= number(protocol["dense_transfer_noninferiority_margin"], "dense_transfer_noninferiority_margin", nonnegative=True)
    broader_criteria = practical_primary and all(guard_pass.values()) and shuffled_benefit and dense_noninferiority
    aligned_only = practical_by_regime["paired"] and not any(
        practical_by_regime[name] for name in ("diagonal", "misaligned"))
    if not negative_valid or not equiv_valid:
        decision = "evaluator_or_equivalence_invalid"
        reason = "Negative controls or coordinate-equivalence validity failed; no positive decision is allowed."
    elif broader_criteria:
        decision = "continue_bounded_contextual_coupling_research"
        reason = "The primary practical threshold, every guardrail, shuffled comparison and dense-transfer requirement all pass."
    elif aligned_only:
        decision = "pivot_to_conditional_structural_prior"
        reason = "Broader criteria fail and practical benefit meets the threshold only in the aligned paired regime."
    else:
        decision = "suspend_exclusive_polar_advantage_claim"
        reason = "Broader criteria fail and the aligned-only condition is false; the protocol's otherwise clause applies."
    return {
        "raw_decision": raw["decision"],
        "raw_proposed_final_rule_outcome": raw["proposed_final_rule_outcome"],
        "adjudicated_decision": decision,
        "label_changed": raw["decision"] != decision,
        "reason": reason,
        "literal_rule_interpretation": "aligned_only := practical(paired) AND NOT practical(diagonal) AND NOT practical(misaligned); practical := upper95CI < -minimum_practical_improvement",
        "evaluated_clauses": {
            "negative_controls_valid": negative_valid,
            "signed_equivalence_valid": equiv_valid,
            "practical_primary_benefit": practical_primary,
            "guard_pass": guard_pass,
            "paired_beats_shuffled": shuffled_benefit,
            "dense_transfer_noninferiority": dense_noninferiority,
            "broader_continue_criteria": broader_criteria,
            "practical_benefit_by_regime": practical_by_regime,
            "only_aligned_regime_has_practical_benefit": aligned_only,
        },
        "numeric_evidence": {
            "minimum_practical_improvement": delta,
            "primary": raw["primary"],
            "regimes": raw["regimes"],
            "paired_minus_shuffled": raw["comparisons"]["shuffled"],
            "dense_transfer": raw["dense_transfer"],
            "guards": guards,
            "runtime_ratio": raw["runtime_ratio"],
            "negative_control_failures": raw["negative_control_failures"],
            "max_signed_action_difference": raw["max_signed_action_difference"],
        },
        "sample_size_qualification": raw["sample_size"],
        "scope": "Post-run interpretation correction using existing statistics; not a new prospectively registered confirmatory test. Numeric results, raw decisions, frozen code and protocol are preserved."
    }


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_INPUT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_INPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_INPUT)
    parser.add_argument("--out", type=Path, default=ROOT / "results_study2/adjudication.json")
    parser.add_argument("--adjudicated-at", help="Explicit ISO timestamp; otherwise preserve the existing matching-input adjudication timestamp")
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest["protocol"] != protocol:
        raise ValueError("Protocol differs from the final run's frozen protocol")
    if sha256(args.protocol) != manifest["source_sha256"]["docs/study2_protocol.json"]:
        raise ValueError("Protocol bytes do not match the recorded freeze hash")
    if sha256(args.manifest) != raw["manifest_sha256"]:
        raise ValueError("Raw decision is not derived from the supplied final manifest")
    inputs = {relative(path): sha256(path) for path in (args.input, args.protocol, args.manifest)}
    result = adjudicate(raw, protocol)
    timestamp = args.adjudicated_at
    if timestamp is None and args.out.exists():
        previous = json.loads(args.out.read_text())
        if previous.get("input_sha256") == inputs:
            timestamp = previous["adjudicated_at_utc"]
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Adjudication timestamp must include the UTC time zone")
    artifact = {
        "schema": "polar-study2-postrun-adjudication-1.0",
        "adjudicated_at_utc": timestamp,
        "post_run_interpretation_correction": True,
        "new_confirmatory_test": False,
        "numeric_results_modified": False,
        "frozen_protocol_rule": protocol["decision_rule"],
        "discrepancy": "Frozen study2/evaluation.py selects pivot whenever aligned practical benefit is present after broader criteria fail; the frozen protocol says ONLY aligned regimes. The code omitted the exclusion of practical benefit in diagonal and misaligned regimes.",
        "input_sha256": inputs,
        "adjudicator_source_sha256": sha256(Path(__file__)),
        "reproduction_command": "python scripts/adjudicate_study2.py",
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    clauses = result["evaluated_clauses"]
    lines = ["# Study 2: disclosed post-run decision adjudication", "", f"Adjudicated at {timestamp}.", "",
             f"Frozen evaluator output: **{result['raw_decision']}**.", "",
             f"Literal-protocol adjudication: **{result['adjudicated_decision']}**.", "",
             artifact["discrepancy"], "", result["reason"], "",
             "| Regime | Upper 95% interval bound | Practical benefit |", "|---|---:|---|"]
    for name in protocol["regimes"]:
        lines.append(f"| {name} | {raw['regimes'][name]['ci95'][1]:.9f} | {clauses['practical_benefit_by_regime'][name]} |")
    lines.extend(["", f"The practical-benefit rule is upper interval bound < {-protocol['minimum_practical_improvement']:.6f}. Broader continuation criteria: {clauses['broader_continue_criteria']}. Aligned-only condition: {clauses['only_aligned_regime_has_practical_benefit']}.", "",
                  "No numeric result was changed. The raw decision and frozen evaluator remain available with their original label so that regeneration reproduces the historical output. This document corrects the subsequent interpretation transparently; it is not a newly registered hypothesis test. The publication's decision should use the adjudicated label and disclose the discrepancy.", "",
                  "This decision concerns the exclusive polar-advantage claim. It does not negate the measured benefit of including learned coupling relative to its lesion or prohibit a new, separately specified study of useful generic coupling.", "",
                  "Reproduce with `python scripts/adjudicate_study2.py`. The JSON records exact input hashes, the frozen rule, every decision predicate, unchanged supporting estimates, sample-size qualification, and adjudicator source hash."])
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    # Inputs must still be byte-identical: the script is an additive interpretation.
    if any(sha256(ROOT / path if not Path(path).is_absolute() else Path(path)) != expected for path, expected in inputs.items()):
        raise RuntimeError("An input changed during adjudication")
    print(json.dumps({"raw_decision": result["raw_decision"], "adjudicated_decision": result["adjudicated_decision"], "output": str(args.out)}))


if __name__ == "__main__":
    main()
