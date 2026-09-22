#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def med(rec,key):
    return float(np.median([float(x[key]) for x in rec]))

def between(x,lo,hi):
    return float(lo) <= float(x) <= float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()

    r=json.loads(Path(a.results).read_text())
    f=json.loads(Path(a.freeze).read_text())
    source_bytes=Path(a.source).read_bytes()
    header=("blob "+str(len(source_bytes))).encode()+bytes([0])
    git_blob=hashlib.sha1(header+source_bytes).hexdigest()

    assert f["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert git_blob==f["source_git_blob_sha"], (git_blob,f["source_git_blob_sha"])
    assert r["mode"]=="confirm"
    assert r["seeds"]==f["confirmatory_seeds"]
    assert r["confirmatory_families"]==f["confirmatory_families"]
    assert r["strict_heldout_families"]==f["strict_heldout_families"]
    rec=r["records"]
    assert len(rec)==12

    c=f["median_criteria"]
    m={k:med(rec,k) for k in c}
    checks={
      "relational_damage":m["delta_R_relational_shift"]>=c["delta_R_relational_shift"]["min"],
      "specificity":m["specificity_margin"]>=c["specificity_margin"]["min"],
      "diagonal_null":m["delta_R_diagonal"]<=c["delta_R_diagonal"]["max"],
      "gain_null":m["delta_R_gain_shift"]<=c["delta_R_gain_shift"]["max"],
      "static_null":m["delta_R_static_relational"]<=c["delta_R_static_relational"]["max"],
      "generic_low":m["core_generic_ratio_rel"]>=c["core_generic_ratio_rel"]["min"],
      "generic_high":m["core_generic_ratio_rel"]<=c["core_generic_ratio_rel"]["max"],
      "factorized":m["core_factorized_ratio_rel"]<=c["core_factorized_ratio_rel"]["max"],
      "frozen":m["core_frozen_ratio_rel"]<=c["core_frozen_ratio_rel"]["max"],
      "recovery":m["relation_recovery_gain"]>=c["relation_recovery_gain"]["min"],
      "wrong_damage":m["wrong_relation_damage"]>=c["wrong_relation_damage"]["min"],
      "rescue":m["correct_relation_rescue"]>=c["correct_relation_rescue"]["min"],
      "stable":m["core_stable"]>=c["core_stable"]["min"],
      "heldout_damage":m["heldout_delta_R_relational_shift"]>=c["heldout_delta_R_relational_shift"]["min"],
      "heldout_specificity":m["heldout_specificity_margin"]>=c["heldout_specificity_margin"]["min"],
      "heldout_generic_low":m["heldout_core_generic_ratio_rel"]>=c["heldout_core_generic_ratio_rel"]["min"],
      "heldout_generic_high":m["heldout_core_generic_ratio_rel"]<=c["heldout_core_generic_ratio_rel"]["max"],
      "heldout_factorized":m["heldout_core_factorized_ratio_rel"]<=c["heldout_core_factorized_ratio_rel"]["max"],
      "heldout_recovery":m["heldout_relation_recovery_gain"]>=c["heldout_relation_recovery_gain"]["min"]
    }

    sg=f["seed_guard"]
    seed_guard=[]
    heldout_guard=[]
    for x in rec:
        seed_guard.append(
          x["delta_R_relational_shift"]>=sg["delta_R_relational_shift_min"] and
          x["specificity_margin"]>=sg["specificity_margin_min"] and
          between(x["core_generic_ratio_rel"],sg["core_generic_ratio_min"],sg["core_generic_ratio_max"]) and
          x["relation_recovery_gain"]>=sg["relation_recovery_gain_min"] and
          x["wrong_relation_damage"]>=sg["wrong_relation_damage_min"] and
          x["core_stable"]>=sg["core_stable_min"]
        )
        heldout_guard.append(
          x["heldout_delta_R_relational_shift"]>=sg["heldout_delta_R_relational_shift_min"] and
          x["heldout_specificity_margin"]>=sg["heldout_specificity_margin_min"] and
          between(x["heldout_core_generic_ratio_rel"],sg["heldout_core_generic_ratio_min"],sg["heldout_core_generic_ratio_max"])
        )

    mech_keys=["relational_damage","specificity","diagonal_null","gain_null","static_null",
               "generic_low","generic_high","factorized","frozen","recovery",
               "wrong_damage","rescue","stable"]
    hold_keys=["heldout_damage","heldout_specificity","heldout_generic_low",
               "heldout_generic_high","heldout_factorized","heldout_recovery"]

    mechanism_pass=all(checks[k] for k in mech_keys) and sum(seed_guard)>=f["seed_guard_required"]
    heldout_pass=all(checks[k] for k in hold_keys) and sum(heldout_guard)>=f["heldout_seed_guard_required"]
    strong=mechanism_pass and heldout_pass

    if strong:
        resolution="R32_STRONG_RELATIONAL_TRANSFER_PASS_INTERNAL_HELDOUT"
    elif mechanism_pass:
        resolution="R32_RELATIONAL_MECHANISM_PASS_HELDOUT_TRANSFER_FAIL"
    else:
        resolution="R32_FAIL"

    out={
      "campaign":f["campaign"],
      "source_git_blob_sha":git_blob,
      "confirmatory_seeds":f["confirmatory_seeds"],
      "confirmatory_families":f["confirmatory_families"],
      "strict_heldout_families":f["strict_heldout_families"],
      "medians":m,
      "median_checks":checks,
      "seed_guard_count":int(sum(seed_guard)),
      "heldout_seed_guard_count":int(sum(heldout_guard)),
      "mechanism_pass":bool(mechanism_pass),
      "heldout_transfer_pass":bool(heldout_pass),
      "strong_transfer_pass":bool(strong),
      "resolution":resolution,
      "boundaries":f["fixed_boundaries"]
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
