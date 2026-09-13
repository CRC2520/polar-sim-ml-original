"""Generate audit tables from retained machine-readable development exports.

This exporter does not instantiate tasks, choose configurations or generate seeds.
"""
import csv
from hashlib import sha256
import json
from pathlib import Path

from b1.contracts import load_contracts
from b1.controllers.core import Limits
from b1.controllers.invariants import profile, audit_contrast
from b1.evaluation.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(root=ROOT):
    root = Path(root)
    results = read(root / "B1D_RESULTS.json")
    config = read(root / "B1D_CONFIG.json")
    selected = read(root / "SELECTED_CONFIG.json")["configurations"]
    technical = results["calibration_technical"]
    gate = read(root / "INSTRUMENT_GATE.json")
    contracts = load_contracts()
    rules = contracts.rules
    precision = read(root / "B1E_RESOURCE_PROJECTION.json")
    limits = Limits(**config["resources"]["limits"])
    # This is the permitted technical reporting rule, not certification of the
    # smallest worst-input cap or a claim that unmeasured hardware is affordable.
    maximum = max(results["tuning"]["max_observed_decision_seconds"],
                  technical["max_observed_decision_seconds"])
    latency = 2 * maximum
    limits = Limits(limits.memory_scalars, limits.decision_operations, 1, latency)
    config_hash = sha256((root / "PRE_TUNING_FREEZE.json").read_bytes()).hexdigest()
    histories_hash = sha256((root / "development_results/tuning/bundles.json").read_bytes()).hexdigest()
    training_allowance = len(config["tuning_bundles"]) * rules["thresholds"]["epoch_count"] * limits.decision_operations
    comparators, invariants, evidence = [], [], []
    detailed_invariants = []
    for pilot in rules["scope"]["pilots"]:
        for comparator, name in rules["comparators"].items():
            ordinary = [row for row in results["calibration_summaries"]
                        if row["pilot"] == pilot and row["comparator"] == comparator and row["acute_context"] is None]
            comparators.append({"pilot": pilot, "comparator": comparator, "name": name,
                "role": "algebraic_control" if comparator == "C6" else "descriptive_capacity" if comparator == "C5" else "architecture_comparison",
                "implementation": "blocked_source_conflict" if pilot == "P5" and comparator == "C4" else "implemented",
                "development_execution": "instrument_only_architecture_blocked" if pilot == "P5" else "executed_nonconfirmatory",
                "selected_configuration": selected.get(pilot, {}).get(comparator, "not_selected"),
                "cycles": "both_fixed_cycles_mean" if comparator == "C2" else "not_applicable",
                "failed_calibration_attempts": sum(row["failed_attempts"] for row in ordinary),
                "positive_utility_eligible": rules["pilot_roles"][pilot]["positive_utility_eligible"],
                "positive_pairing_eligible": rules["pilot_roles"][pilot]["positive_pairing_specificity_eligible"],
                "resource_certified": False, "confirmatory": False})
        def arm(comparator):
            return profile(comparator, limits=limits, training_data_hash=histories_hash,
                           training_operations=training_allowance * (2 if comparator == "C5" else 1),
                           config_manifest_hash=config_hash, resource_certified=False)
        for comparator in ("C0", "C2", "C3", "C4", "C5", "C6"):
            kind = "algebraic_control" if comparator == "C6" else "competitive_training"
            expected = ("memory", "decision_compute", "communication_width", "learning_and_training_resources") if comparator == "C5" else ()
            audit = audit_contrast(arm("C1"), arm(comparator), contrast_kind=kind,
                                   expected_differences=expected, gamma_manipulated=comparator != "C6")
            audit.update(pilot=pilot, contrast_id=f"{pilot}.C1_vs_{comparator}",
                         architecture_execution_blocked=pilot == "P5")
            detailed_invariants.append(audit)
        for context in ("kappa1", "kappa2"):
            audit = audit_contrast(arm("C1"), arm("C1"), contrast_kind="acute_route", gamma_manipulated=True)
            audit.update(pilot=pilot, contrast_id=f"{pilot}.acute_route.{context}",
                         architecture_execution_blocked=pilot == "P5")
            detailed_invariants.append(audit)
            source = audit_contrast(arm("C1"), arm("C1"), contrast_kind="source_intervention",
                                    intended_source_intervention=True)
            source.update(pilot=pilot, contrast_id=f"{pilot}.source.{context}",
                          joint_manipulation=True, gamma_specific_verdict_allowed=False,
                          claim_limit="source_operation_and_its_mediator_are_jointly_changed_not_a_selective_internal_route_lesion")
            detailed_invariants.append(source)
    for audit in detailed_invariants:
        for axis, values in audit["axes"].items():
            invariants.append({"contrast_id": audit["contrast_id"], "pilot": audit["pilot"],
                "axis": axis, "left": json.dumps(values["left"], sort_keys=True),
                "right": json.dumps(values["right"], sort_keys=True), "equal": values["equal"],
                "unequal_axis": values["unequal_axis"], "expected_difference": values["expected_difference"],
                "pending_certification": ";".join(audit["pending_certification"]),
                "joint_manipulation": audit["joint_manipulation"],
                "gamma_specific_verdict_allowed": audit["gamma_specific_verdict_allowed"],
                "claim_limit": audit["claim_limit"]})
    csv_write(root / "B1D_COMPARATOR_MATRIX.csv", comparators)
    csv_write(root / "B1D_INVARIANT_MATRIX.csv", invariants)
    witnesses = read(root / "development_results/calibration/P5_ceiling_witnesses.json")["rows"]
    c = contracts.pilot("P5")["task_contract"]["constants"]
    horizon = rules["thresholds"]["epoch_count"]
    ceiling_pass = len(witnesses) == 32 * 3 * 2 and all(
        row["completed_jobs"] == c["production_capacity_per_epoch"] * (horizon - int(row["initial_reserve"] == 0))
        for row in witnesses)
    transcript = (root / "development_results/instrumentation/unit-tests.txt").read_text()
    bound_pass = gate["unit_tests_exit_code"] == 0 and "test_expected_feedback_bound_exhaustive_task_law" in transcript
    pilot_status = {}
    for pilot in rules["scope"]["pilots"]:
        instrument_pass = gate["status"] == "PASS" and results["mechanism_diagnostics"][pilot]["task_law_pass"]
        ordinary = [row for row in results["calibration_summaries"] if row["pilot"] == pilot and row["acute_context"] is None]
        pilot_status[pilot] = {
            "instrument_status": "PASS" if instrument_pass else "FAIL",
            "mechanism_development_status": "imposed_task_law_reproduced" if instrument_pass else "instrument_blocked",
            "comparator_status": "blocked_canonical_C4_conflict" if pilot == "P5" else
                                  "functional_with_uncertified_resource_matching" if len(ordinary) == 8 and not any(r["failed_attempts"] for r in ordinary) else "execution_failures_see_retained_trials",
            "utility_eligibility": rules["pilot_roles"][pilot]["positive_utility_eligible"],
            "pairing_eligibility": rules["pilot_roles"][pilot]["positive_pairing_specificity_eligible"],
            "no_O3_O4_promotion": True}
    pilot_status["P5"].update(conventional_optimum_reproduced=ceiling_pass,
                              ceiling_status="both_explicitly_unselected_witnesses_reproduce_bound" if ceiling_pass else "FAIL",
                              unique_frozen_C4_implementation_selected=False)
    pilot_status["P6"].update(expected_bound_reproduced=bound_pass,
                              expected_optimal_primary_loss=rules["pilot_roles"]["P6"]["expected_optimal_primary_loss"],
                              bound_evidence="Exhaustive initial-mapping/flip finite-law instrument test; not closeness of a sampled mean",
                              per_bundle_bound_claim=False)
    audit = {"status": results["status"], "pilot_status": pilot_status,
             "C6": {"instrument_full_state_conjugacy_tests": gate["unit_tests_exit_code"] == 0,
                    "calibration_exact_equivalence": technical["exact_coordinate_invariance_pass"],
                    "absolute_tolerance": 0, "relative_tolerance": 0, "numeric_format": "exact rational"},
             "invariants": detailed_invariants, "six_invariant_axes": rules["six_invariant_axes"],
             "smallest_common_resource_cap_certified": False,
             "development_deadline_seconds_reporting_only": latency,
             "deadline_derivation": "2 times maximum observed tuning/calibration decision duration; not a worst-input certificate",
             "actual_guardrails": technical["guardrails"],
             "source_interventions_jointly_change_inputs_or_information": True,
             "actual_acute_interventions_remove_raw_information": False,
             "final_seeds_generated": False, "B1E_executed": False,
             "scientific_code_changed": True, "scientific_code_scope": "new isolated b1 namespace",
             "historical_code_changed": False, "historical_results_changed": False,
             "historical_verification": "773-blob source-tree inventory, new-file publication allowlist and full Git CI verification",
             "B1E_required_N": precision["precision_plan"]["B1E_required_N"],
             "b1e_precision_feasible": False, "ready_for_b1e_freeze": False,
             "ready_for_b1e_confirmatory_run": False,
             "unresolved_blockers": results["unresolved_blockers"],
             "development_only": True, "confirmatory": False, "reusable_as_final": False}
    write_json(root / "B1D_AUDIT.json", audit)
    for pilot, record in pilot_status.items():
        for name, value in record.items():
            evidence.append({"pilot": pilot, "item": name, "value": json.dumps(value, ensure_ascii=False),
                             "source_path": "b1/B1D_AUDIT.json", "source_pointer": f"/pilot_status/{pilot}/{name}",
                             "development_only": True, "confirmatory": False})
    for row in results["calibration_summaries"]:
        evidence.append({"pilot": row["pilot"], "item": f"development_loss.{row['comparator']}.cycle{row['cycle']}.{row['acute_context']}.{row['acute_lesion']}",
                         "value": json.dumps({"mean": row["mean_development_loss"], "N": row["N"]}),
                         "source_path": "b1/B1D_RESULTS.json", "source_pointer": "/calibration_summaries",
                         "development_only": True, "confirmatory": False})
    csv_write(root / "B1D_EVIDENCE_EXPORT.csv", evidence)
    traceability = [{"requirement": name, "source": "B1 user prompt", "implementation": target,
                     "evidence": reference, "disposition": disposition} for name, target, reference, disposition in [
        ("immutable B0 imports", "b1/contracts/loader.py", "b1/B1D_CONTRACT_IMPORT.json", "implemented_hash_checked"),
        ("instrument before comparisons", "b1/run.py:instrument", "b1/INSTRUMENT_GATE.json", gate["status"]),
        ("activation/admission/execution/effect", "b1/tasks/core.py", "b1/development_results/calibration/traces/INDEX.json", "logged_separately"),
        ("P5 unique conventional comparator", "b1/controllers/core.py", "b1/B1D_AUDIT.json", "BLOCKED_source_conflict"),
        ("P6 expected bound", "b1/tests/test_tasks.py", "b1/B1D_AUDIT.json", "finite_law_test"),
        ("P7 distinct versioned writes", "b1/tasks/p7.py", "b1/development_results/instrumentation/unit-tests.txt", "instrument_tested"),
        ("C6 exact conjugacy", "b1/controllers/c6.py", "b1/CALIBRATION_TECHNICAL.json", "PASS" if technical["exact_coordinate_invariance_pass"] else "FAIL"),
        ("six invariant axes", "b1/controllers/invariants.py", "b1/B1D_INVARIANT_MATRIX.csv", "BLOCKED_resource_certificate"),
        ("guardrails from actual execution facts", "b1/tasks/core.py:audit_execution", "b1/CALIBRATION_TECHNICAL.json", "computed_not_assumed"),
        ("separate tuning and calibration", "b1/seeds/policy.py", "b1/TUNING_FREEZE.json", "content_verified_order"),
        ("machine readable evaluator", "b1/evaluation/decision.py", "b1/B1D_RESULTS.json", "development_only_no_confirmatory_verdict"),
        ("prospective precision", "b1/evaluation/precision.py", "b1/B1E_RESOURCE_PROJECTION.json", "N_unchanged_no_final_collection"),
        ("historical preservation", "b1/validate_b1.py", "b1/manifests/HISTORICAL_BASELINE.json", "full_git_CI_required"),
        ("future confirmatory gate", "b1/run.py", "b1/B1D_AUDIT.json", "BLOCKED_no_final_seed_or_runner")]]
    csv_write(root / "B1D_TRACEABILITY_EXPORT.csv", traceability)
    return {"status": audit["status"], "comparator_rows": len(comparators),
            "invariant_rows": len(invariants), "evidence_rows": len(evidence),
            "traceability_rows": len(traceability), "new_experiments_run_by_exporter": False}


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
