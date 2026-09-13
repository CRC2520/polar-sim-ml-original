"""Reproduce prospective planning using archived DEVELOPMENT observations only."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from .method import (ALPHA, FAMILY_SIZE, VARIANCE_FLOOR, fixed_n,
                     planning_second_moment_upper, planning_variance, radius, v1_n)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FLAGS = {"development_only": True, "confirmatory": False, "reusable_as_final": False,
         "final_seeds_generated": False, "B1E_executed": False}
V1_COMMIT = "c27235deb1688c105bfe96597dfd233108307895"


def source(path):
    return {"repository": "CRC2520/polar-sim-ml-original", "commit": V1_COMMIT,
            "path": "b1/" + path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def calibration_audit():
    base = ROOT / "development_results/calibration"
    identities = json.loads((base / "IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json").read_text())["identities"]
    rows = [json.loads(line) for line in (base / "masked_trials.jsonl").read_text().splitlines()]
    selected = []
    keys = set()
    for row in rows:
        if row["pilot"] != "P7":
            continue
        identity = identities[row["arm_id"]]
        if identity["pilot"] != "P7":
            raise ValueError("Invalid identity/pilot pairing")
        key = (row["bundle_index"], identity["comparator"], identity["cycle"],
               identity["acute_context"], identity["acute_lesion"])
        if key in keys:
            raise ValueError("Duplicate calibration identity")
        keys.add(key)
        selected.append((key, Fraction(*row["loss_exact"])))
        if not row["development_only"] or row["confirmatory"] or row["reusable_as_final"]:
            raise ValueError("Unexpected evidence flags")
    expected = set()
    for bundle in range(32, 64):
        for comparator in ("C0", "C1", "C2", "C3", "C4", "C5", "C6"):
            for cycle in ((1, 2) if comparator == "C2" else (1,)):
                expected.add((bundle, comparator, cycle, None, False))
        for context in ("kappa1", "kappa2"):
            for lesion in (False, True):
                expected.add((bundle, "C1", 1, context, lesion))
    if keys != expected:
        raise ValueError("Incomplete or unexpected 32-by-12 P7 calibration schedule")
    lookup = dict(selected)
    differences = {k: [] for k in ("C1_minus_C0", "C1_minus_C2", "C1_minus_C3", "C1_minus_C4",
                                  "acute_route.kappa1", "acute_route.kappa2")}
    for b in range(32, 64):
        for c in ("C0", "C2", "C3", "C4"):
            cycles = (1, 2) if c == "C2" else (1,)
            other = sum((lookup[(b, c, k, None, False)] for k in cycles), Fraction()) / len(cycles)
            differences["C1_minus_" + c].append(lookup[(b, "C1", 1, None, False)] - other)
        for context in ("kappa1", "kappa2"):
            differences["acute_route." + context].append(lookup[(b, "C1", 1, context, False)]
                                                        - lookup[(b, "C1", 1, context, True)])
    previous = json.loads((ROOT / "B1D_RESULTS.json").read_text())["paired_development_differences"]["P7"]
    for key, values in differences.items():
        if [float(x) for x in values] != previous[key]["bundle_values"]:
            raise ValueError("Exact archived trial differences do not match v1 report")
    counts = [sum(x != 0 for x in values) for values in differences.values()]
    return {"pilot": "P7", "bundle_indices": [32, 63], "independent_bundle_count": 32,
            "trial_count": len(selected), "exact_rational_reduction": True,
            "C2_cycles": "equal mean of two cycles before one bundle contrast",
            "endpoints": list(differences), "nonzero_counts": counts,
            "matches_v1_report": True, "source_artifacts": [source(p) for p in (
                "B1D_RESULTS.json", "development_results/calibration/masked_trials.jsonl",
                "development_results/calibration/IDENTITY_MAP_AFTER_TECHNICAL_CLOSE.json",
                "SELECTED_CONFIG.json", "TUNING_LEDGER.json", "CALIBRATION_TECHNICAL.json")], **FLAGS}


def design():
    audit = calibration_audit()
    q = planning_second_moment_upper(audit["nonzero_counts"])
    n = fixed_n(q)
    sim = json.loads((HERE / "SIMULATION_RESULTS.json").read_text())
    return {
        "schema_version": "1.1", "statistical_design_version": "B1-E-protocol-v2-candidate",
        "status": "PROSPECTIVE_METHOD_VALIDATED", "change_type": "statistical_design_only",
        "N_v1": v1_n(), "N_v2": n,
        "immutable_v1_method": "two-sided Hoeffding plus union bound over27 endpoints",
        "selected_method": "fixed-sample two-sided empirical Bernstein on paired bundle means; simultaneous9-endpoint Bonferroni coverage plus fixed positive gatekeeping",
        "alpha": ALPHA, "interval_family_size": FAMILY_SIZE,
        "interval_formula": "h=sqrt(2*s_unbiased_squared*log(4*9/0.05)/N)+7*R*log(4*9/0.05)/(3*(N-1));CI=clip(mean+/-h,known_support)",
        "assumptions": ["Independent identically distributed seed bundles after all task/controller/config freezes",
                        "Within-bundle dependence is retained; one scalar paired difference per endpoint per bundle",
                        "Known bounded support; no missing-outcome deletion, seed replacement, or outcome-based stopping",
                        "Finite fixed endpoint family; arbitrary dependence across its endpoints is allowed"],
        "eligibility_provenance": [source("contracts/frozen_b0/B1_DECISION_RULES_v1.yaml"),
                                   source("contracts/frozen_b0/B1_PROTOCOL_DRAFT_v1.md")],
        "endpoint_roles": {
            "P5": {"role": "engineering_ceiling_negative_control", "positive_utility_eligible": False,
                   "positive_pairing_specificity_eligible": False, "endpoints_retained": 9,
                   "future_bundle_count": 32, "population_confirmatory_claims": False},
            "P6": {"role": "information_redundancy_expected_Bayes_optimum_negative_control",
                   "positive_utility_eligible": False, "positive_pairing_specificity_eligible": False,
                   "endpoints_retained": 9, "future_bundle_count": 32, "population_confirmatory_claims": False},
            "P7": {"role": "operational_coordination_candidate", "H_mechanism_eligible": True,
                   "H_utility_eligible": True, "H_pairing_eligible": True, "H_context_eligible": True,
                   "endpoints_retained": 9, "future_bundle_count": n,
                   "eligibility_does_not_override_causal_admissibility": True}},
        "P7_interval_endpoints": ["mechanism.kappa1", "mechanism.kappa2", "context.interaction",
                                  "utility.C0", "utility.C2", "utility.C3", "utility.C4",
                                  "acute_route.kappa1", "acute_route.kappa2"],
        "margin_contract": {
            "utility": {"delta": 1 / 64, "epsilon": 1 / 128, "target_half_width": 1 / 256},
            "mechanism": {"delta": 0.5, "epsilon": 0.25, "target_half_width": 0.125},
            "context": {"delta": 0.5, "epsilon": 0.25, "target_half_width": 0.125},
            "changed": False, "all_endpoint_supports_preserved": True},
        "gatekeeping": {
            "order": ["instrument_and_admissible_mechanism", "practical_utility", "pairing_specificity", "context_moderation"],
            "utility_requires": ["C0 improved", "C4 improved"],
            "pairing_requires": ["C0 improved", "C2 improved", "C3 improved", "C4 improved"],
            "all_data_collected_before_testing": True, "fixed_sequence_alpha_recycling": False,
            "reason": "Simultaneous intervals protect improved/equivalent/worsened assertions even when positive gates fail. Gates only restrict favorable claims; no hidden alpha transfer or optional sampling.",
            "FWER_proof": "Each two-sided interval misses with probability<=alpha/9; union gives any miss<=alpha. Every false scalar declaration implies its interval missed the true mean. Composite all-of requirements and gate restrictions cannot enlarge that event.",
            "joint_manipulation": "Source/sham remains joint_manipulation=true; frozen causal guards apply. No source/sham result can establish Gamma-selective necessity. Acute route endpoints are separate and cannot rescue failed utility or pairing gates.",
            "current_joint_fixture_consequence": "With the current joint source/sham fixtures, source_causal_admissible=false. Therefore this hierarchy cannot issue any aggregate positive H_mechanism, H_utility, H_pairing, or H_context claim, regardless of scalar outcome signs. This is an explicit scientific claim limitation, not a technical defect to remove by retuning. Correct outputs remain not_supported/inconclusive as applicable; scalar improved/equivalent/worsened/inconclusive reports remain separate. A future causal redesign would require a separate prospective protocol and cannot silently relax the frozen guard.",
            "negative_and_equivalence_reporting": "Preserved regardless of positive gate progression; distinguish corrected scalar verdict from gated aggregate positive claim"},
        "planning": {
            "uses_only_development_data": True, "development_audit": audit,
            "second_moment_bound": q, "variance_floor": VARIANCE_FLOOR,
            "zero_event_bound_formula": "q=max(1/64,1-(0.05/6)^(1/32));E[D^2]<=Pr(D!=0)",
            "development_family_bound_failure_probability": 0.05,
            "future_precision_family_failure_probability_conditional_on_second_moment_bound": 0.05,
            "unconditional_design_and_future_precision_lower_bound": 0.90,
            "future_variance_bound_formula": "s^2<=N/(N-1)*min(1,q+sqrt(log(6/0.05)/(2*N)))",
            "planned_variance_upper_at_N": planning_variance(n, q),
            "planned_half_width_at_N": radius(n, planning_variance(n, q)),
            "planned_half_width_at_N_minus_one": radius(n - 1, planning_variance(n - 1, q)),
            "sample_selection_rule": "Smallest integer N>=2 meeting all preserved target half-widths under declared planning event; exact binary search, no cap",
            "worst_case_second_moment_fallback_N": fixed_n(1.0),
            "variance_is_not_known": True, "final_CI_uses_planning_bound_or_floor": False,
            "precision_if_distribution_shifts": "Not guaranteed; sample variance remains in CI; precision failure is inconclusive, never extend or lower margins",
            "transport_condition": "Same P7 task, selected policies and bundle law; resource-envelope certification must verify unchanged actions/outputs. Any behavior change voids this planning derivation before final freeze.",
            "power_target": "Precision design, not uniform power guarantee. Conditional on planning moment bounds, all six narrow-margin CIs meet h<=epsilon/2 with probability>=0.95. Joint CI coverage>=0.95 gives classification probability>=0.90 when true means have >2h separation from decision boundary (and analogous strict equivalence interior).",
            "no_outcome_based_extension": True},
        "future_schedule": {
            "P7": {"bundles": n, "episodes_per_bundle": 12, "source_diagnostic_events_per_bundle": 24,
                   "ordinary_episodes": ["C0", "C1", "C2.cycle1", "C2.cycle2", "C3", "C4", "C5", "C6"],
                   "acute_episodes": ["kappa1.intact", "kappa1.bypass", "kappa2.intact", "kappa2.bypass"]},
            "P5": {"bundles": 32, "episodes_per_bundle": 13, "source_diagnostic_events_per_bundle": 24,
                   "extra_episode": "C4_FULL_REFILL_CEILING_WITNESS"},
            "P6": {"bundles": 32, "episodes_per_bundle": 12, "source_diagnostic_events_per_bundle": 24},
            "engineering_schedule_rationale": "Retain the frozen32-bundle development block as prospective engineering replication of a32-epoch task; it supplies no powered population superiority claim. All18 P5/P6 endpoint summaries remain explicit descriptive checks.",
            "C6": "Algebraic state/event/behavior check, not an additional statistical endpoint",
            "C5": "Descriptive double-capacity comparator, not an additional confirmatory endpoint"},
        "candidate_comparison": [
            {"method": "A fixed-sample empirical Bernstein", "decision": "SELECTED",
             "reason": "Finite-sample bounded-mean validity; adapts interval width to paired variance without treating development variance as known"},
            {"method": "B paired Student-t/normal CI", "decision": "NOT_SELECTED",
             "reason": "Exact t requires Gaussian iid differences; bounded/discrete/rare-tail laws do not supply this. CLT is asymptotic, with no uniform finite-N guarantee here"},
            {"method": "C paired permutation/randomization", "decision": "NOT_SELECTED_AS_GENERAL_MEAN_METHOD",
             "reason": "Shared seeds do not establish sign exchangeability or a randomized treatment allocation; sharp-null randomization inference cannot silently test the heterogeneous weak mean/margin null"},
            {"method": "D exact/binomial-type", "decision": "SELECTED_ONLY_FOR_PLANNING_NONZERO_INDICATOR",
             "reason": "192 jobs within a bundle are dependent, so loss fraction is not Binomial(192,p). Exact zero-event binomial inference across32 independent bundles validly bounds nonzero probability, not individual job success"},
            {"method": "E equivalence/noninferiority", "decision": "EQUIVALENCE_BY_SIMULTANEOUS_CI_CONTAINMENT",
             "reason": "Preserve epsilon=1/128 and both strict boundaries; failure to reject zero is never equivalence. Noninferiority alone cannot establish positive utility or pairing necessity"},
            {"method": "F hierarchical multiple testing", "decision": "SIMULTANEOUS_INTERVALS_PLUS_FIXED_GATEKEEPING",
             "reason": "Holm/closed testing/fixed-sequence recycling can be valid with specified elementary hypotheses; selected Bonferroni intervals also protect negative/equivalence classifications after a failed positive gate. No dependence assumptions across endpoints"}],
        "simulation": {"path": "b1/v1_1/statistics/SIMULATION_RESULTS.json",
                       "sha256": hashlib.sha256((HERE / "SIMULATION_RESULTS.json").read_bytes()).hexdigest(),
                       "scenario_cells": sim["scenario_cells"], "repetitions_per_scenario": sim["repetitions_per_scenario"],
                       "minimum_family_coverage": sim["minimum_family_coverage"],
                       "maximum_family_false_declaration_rate": sim["maximum_family_false_declaration_rate"],
                       "model_executions": 0, "statistical_validation_is_model_evidence": False},
        "sources": [
            {"id": "MP2009", "title": "Maurer and Pontil, Empirical Bernstein Bounds and Sample Variance Penalization, Theorem4", "url": "https://arxiv.org/pdf/0907.3740"},
            {"id": "EM2007", "title": "Edwards and Madsen, Constructing multiple test procedures for partially ordered hypothesis sets", "url": "https://doi.org/10.1002/sim.2905"},
            {"id": "S1987", "title": "Schuirmann, A comparison of the two one-sided tests procedure and the power approach", "url": "https://doi.org/10.1007/BF01068419"},
            {"id": "CR2013", "title": "Chung and Romano, Exact and asymptotically robust permutation tests", "url": "https://arxiv.org/abs/1304.5939"}],
        "ready_for_b1e_confirmatory_run": False,
        "remaining_requirements": ["Complete resource and immutability certificates", "External preregistration", "Final code/config/retention freeze", "Independent final seed custody", "Explicit resource and execution authorization"],
        **FLAGS}


def main():
    output = ROOT / "v1_1/B1E_STATISTICAL_REDESIGN.json"
    if output.exists():
        raise FileExistsError("Refusing to replace prospective design silently")
    value = design()
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "N_v1": value["N_v1"], "N_v2": value["N_v2"]}))


if __name__ == "__main__":
    main()
