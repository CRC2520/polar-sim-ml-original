#!/usr/bin/env python3
import argparse, json, statistics, hashlib
from pathlib import Path

def median(xs): return float(statistics.median(xs))

def load(p): return json.loads(Path(p).read_text())

def all_true(d): return all(bool(v) for v in d.values())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    r=load(a.results); f=load(a.freeze)
    sha=hashlib.sha256(Path(a.source).read_bytes()).hexdigest()
    if sha!=f["experiment_sha256"]:
        raise SystemExit(f"source hash mismatch: {sha} != {f['experiment_sha256']}")
    if r["seeds"]!=f["confirmatory_seeds"] or r["mode"]!="confirm":
        raise SystemExit("confirmatory seed/mode mismatch")

    A=r["R26A_records"]; B=r["R26B_records"]; C=r["R26C_records"]

    ma={k:median([x["metrics"][k] for x in A]) for k in [
        "full_control_mse","generic_gap","noD_damage","noR_damage","memory_advantage",
        "access_advantage","access_promotion","relation_score","full_memory_accuracy",
        "full_query_report_accuracy"]}
    ta=f["r26a"]["median"]
    achecks={
        "full_control_mse":ma["full_control_mse"]<=ta["full_control_mse_max"],
        "generic_equivalence":ma["generic_gap"]<=ta["generic_gap_max"],
        "D_damage":ma["noD_damage"]>=ta["noD_damage_min"],
        "R_damage":ma["noR_damage"]>=ta["noR_damage_min"],
        "C_memory":ma["memory_advantage"]>=ta["memory_advantage_min"],
        "A_access":ma["access_advantage"]>=ta["access_advantage_min"],
        "A_promotion":ma["access_promotion"]>=ta["access_promotion_min"],
        "relation_reidentification":ma["relation_score"]>=ta["relation_score_min"],
        "memory_accuracy":ma["full_memory_accuracy"]>=ta["full_memory_accuracy_min"],
        "query_report_accuracy":ma["full_query_report_accuracy"]>=ta["full_query_report_accuracy_min"],
    }
    ga=f["r26a"]["seed_guard"]
    aguard=[]
    for x in A:
        m=x["metrics"]
        aguard.append(
            m["full_control_mse"]<=ga["full_control_mse_max"] and
            m["generic_gap"]<=ga["generic_gap_max"] and
            m["noD_damage"]>=ga["noD_damage_min"] and
            m["noR_damage"]>=ga["noR_damage_min"] and
            m["memory_advantage"]>=ga["memory_advantage_min"] and
            m["access_advantage"]>=ga["access_advantage_min"] and
            m["relation_score"]>=ga["relation_score_min"] and
            m["full_memory_accuracy"]>=ga["full_memory_accuracy_min"] and
            m["full_query_report_accuracy"]>=ga["full_query_report_accuracy_min"]
        )
    r26a_pass=all_true(achecks) and sum(aguard)>=f["r26a"]["seed_guard_required"]

    mb={k:median([x["metrics"][k] for x in B]) for k in [
        "generic_gap","null_offdiag_norm","null_relation_damage","null_coactivation_damage"]}
    tb=f["r26b"]["median"]
    bchecks={
        "generic_equivalence":mb["generic_gap"]<=tb["generic_gap_max"],
        "null_offdiag":mb["null_offdiag_norm"]<=tb["null_offdiag_norm_max"],
        "null_relation_damage":abs(mb["null_relation_damage"])<=tb["abs_null_relation_damage_max"],
        "null_coactivation_damage":abs(mb["null_coactivation_damage"])<=tb["abs_null_coactivation_damage_max"],
    }
    gb=f["r26b"]["seed_guard"]; bguard=[]
    for x in B:
        m=x["metrics"]
        bguard.append(
            m["generic_gap"]<=gb["generic_gap_max"] and
            m["null_offdiag_norm"]<=gb["null_offdiag_norm_max"] and
            abs(m["null_relation_damage"])<=gb["abs_null_relation_damage_max"] and
            abs(m["null_coactivation_damage"])<=gb["abs_null_coactivation_damage_max"]
        )
    r26b_pass=all_true(bchecks) and sum(bguard)>=f["r26b"]["seed_guard_required"]

    mc={k:median([x[k] for x in C]) for k in [
        "core_generic_ratio","core_factorized_ratio","generic_factorized_ratio",
        "core_frozen_ratio","core_noC_ratio","core_oracle_ratio","core_stable","generic_stable"]}
    to=f["r26c"]["external_operation"]
    ochecks={
        "core_stable":mc["core_stable"]>=to["core_stable_min"],
        "generic_stable":mc["generic_stable"]>=to["generic_stable_min"],
        "generic_equivalence_low":mc["core_generic_ratio"]>=to["core_generic_ratio_min"],
        "generic_equivalence_high":mc["core_generic_ratio"]<=to["core_generic_ratio_max"],
        "recurrent_state_utility":mc["core_noC_ratio"]<=to["core_noC_ratio_max"],
        "frozen_noninferiority":mc["core_frozen_ratio"]<=to["core_frozen_ratio_max"],
    }
    go=to["seed_guard"]; oguard=[]
    for x in C:
        oguard.append(
            x["core_stable"]>=go["core_stable_min"] and
            x["generic_stable"]>=go["generic_stable_min"] and
            x["core_generic_ratio"]>=go["core_generic_ratio_min"] and
            x["core_generic_ratio"]<=go["core_generic_ratio_max"] and
            x["core_noC_ratio"]<=go["core_noC_ratio_max"] and
            x["core_frozen_ratio"]<=go["core_frozen_ratio_max"]
        )
    external_operation_pass=all_true(ochecks) and sum(oguard)>=to["seed_guard_required"]

    tr=f["r26c"]["external_relational_advantage"]
    rchecks={
        "core_beats_factorized":mc["core_factorized_ratio"]<=tr["core_factorized_ratio_max"],
        "generic_beats_factorized":mc["generic_factorized_ratio"]<=tr["generic_factorized_ratio_max"],
    }
    rguard=[
        x["core_factorized_ratio"]<=tr["seed_guard_ratio_max"] and
        x["generic_factorized_ratio"]<=tr["seed_guard_ratio_max"]
        for x in C
    ]
    external_relational_pass=all_true(rchecks) and sum(rguard)>=tr["seed_guard_required"]
    strong_transfer_pass=external_operation_pass and external_relational_pass

    out={
        "campaign":r["campaign"],
        "source_sha256":sha,
        "seeds":r["seeds"],
        "R26A":{
            "medians":ma,"checks":achecks,"seed_guard_pass_count":sum(aguard),
            "pass":r26a_pass
        },
        "R26B":{
            "medians":mb,"checks":bchecks,"seed_guard_pass_count":sum(bguard),
            "pass":r26b_pass
        },
        "R26C":{
            "medians":mc,
            "external_operation_checks":ochecks,
            "external_operation_seed_guard_pass_count":sum(oguard),
            "external_operation_pass":external_operation_pass,
            "external_relational_checks":rchecks,
            "external_relational_seed_guard_pass_count":sum(rguard),
            "external_relational_advantage_pass":external_relational_pass,
            "strong_H_TRANSFER_pass":strong_transfer_pass,
            "oracle_ratio_status":"descriptive_only_privileged_reference"
        },
        "global_interpretation":{
            "core_class_single_agent_supported":r26a_pass,
            "contingent_polarity_null_supported":r26b_pass,
            "external_operation_supported":external_operation_pass,
            "strong_external_relational_transfer_supported":strong_transfer_pass,
            "E6b":"OPEN_EXTERNAL_ONLY",
            "E7":"OPEN_PROSPECTIVE_BIOLOGICAL"
        }
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["global_interpretation"],indent=2,sort_keys=True))
if __name__=="__main__":
    main()
