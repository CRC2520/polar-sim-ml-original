#!/usr/bin/env python3
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

def median(records,key):
    return float(np.median([float(r[key]) for r in records]))

def between(x,lo,hi):
    return float(lo) <= float(x) <= float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    results=json.loads(Path(args.results).read_text())
    freeze=json.loads(Path(args.freeze).read_text())
    source_sha=hashlib.sha256(Path(args.source).read_bytes()).hexdigest()

    assert freeze["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert source_sha==freeze["source_sha256"], (source_sha,freeze["source_sha256"])
    expected=freeze["confirmatory_seeds"]
    assert results["seeds"]==expected, (results["seeds"],expected)
    rec=results["records"]
    assert len(rec)==12

    keys=[
      "adaptive_stable","core_adaptive_ratio","core_rl_postshift_ratio","core_rl_ratio",
      "core_stable","rl_adaptive_ratio","rl_frozen_ratio","rl_gain_scale_mean",
      "rl_nonzero_fraction","rl_stable"
    ]
    m={k:median(rec,k) for k in keys}

    bv=freeze["baseline_validity"]
    bvm=bv["median"]
    baseline_checks={
      "rl_stable":m["rl_stable"]>=bvm["rl_stable_min"],
      "rl_frozen_ratio":m["rl_frozen_ratio"]<=bvm["rl_frozen_ratio_max"],
      "rl_adaptive_ratio":m["rl_adaptive_ratio"]<=bvm["rl_adaptive_ratio_max"],
      "rl_nonzero_fraction":m["rl_nonzero_fraction"]>=bvm["rl_nonzero_fraction_min"],
    }
    bsg=bv["seed_guard"]
    baseline_guard=sum(
      r["rl_stable"]>=bsg["rl_stable_min"] and
      r["rl_frozen_ratio"]<=bsg["rl_frozen_ratio_max"] and
      r["rl_adaptive_ratio"]<=bsg["rl_adaptive_ratio_max"] and
      r["rl_nonzero_fraction"]>=bsg["rl_nonzero_fraction_min"]
      for r in rec
    )
    baseline_valid=all(baseline_checks.values()) and baseline_guard>=bv["seed_guard_required"]

    ni=freeze["core_noninferiority"]
    nim=ni["median"]
    noninf_checks={
      "core_stable":m["core_stable"]>=nim["core_stable_min"],
      "core_rl_ratio":m["core_rl_ratio"]<=nim["core_rl_ratio_max"],
      "core_rl_postshift_ratio":m["core_rl_postshift_ratio"]<=nim["core_rl_postshift_ratio_max"],
      "core_adaptive_low":m["core_adaptive_ratio"]>=nim["core_adaptive_ratio_min"],
      "core_adaptive_high":m["core_adaptive_ratio"]<=nim["core_adaptive_ratio_max"],
    }
    nsg=ni["seed_guard"]
    noninf_guard=sum(
      r["core_stable"]>=nsg["core_stable_min"] and
      r["core_rl_ratio"]<=nsg["core_rl_ratio_max"] and
      r["core_rl_postshift_ratio"]<=nsg["core_rl_postshift_ratio_max"] and
      between(r["core_adaptive_ratio"],nsg["core_adaptive_ratio_min"],nsg["core_adaptive_ratio_max"])
      for r in rec
    )
    noninf=baseline_valid and all(noninf_checks.values()) and noninf_guard>=ni["seed_guard_required"]

    eq=freeze["practical_equivalence"]
    eqm=eq["median"]
    equivalence_checks={
      "core_rl_ratio_band":between(m["core_rl_ratio"],eqm["core_rl_ratio_min"],eqm["core_rl_ratio_max"]),
      "core_rl_postshift_band":between(m["core_rl_postshift_ratio"],eqm["core_rl_postshift_ratio_min"],eqm["core_rl_postshift_ratio_max"]),
    }
    esg=eq["seed_guard"]
    equivalence_guard=sum(
      between(r["core_rl_ratio"],esg["core_rl_ratio_min"],esg["core_rl_ratio_max"]) and
      between(r["core_rl_postshift_ratio"],esg["core_rl_postshift_ratio_min"],esg["core_rl_postshift_ratio_max"])
      for r in rec
    )
    equivalent=baseline_valid and all(equivalence_checks.values()) and equivalence_guard>=eq["seed_guard_required"]

    ca=freeze["exclusive_core_advantage"]
    core_adv_guard=sum(r["core_rl_ratio"]<=ca["seed_guard_core_rl_ratio_max"] for r in rec)
    core_adv=(
      baseline_valid and
      m["core_rl_ratio"]<=ca["median_core_rl_ratio_max"] and
      core_adv_guard>=ca["seed_guard_required"]
    )

    ra=freeze["exclusive_rl_advantage"]
    rl_adv_guard=sum(r["core_rl_ratio"]>=ra["seed_guard_core_rl_ratio_min"] for r in rec)
    rl_adv=(
      baseline_valid and
      m["core_rl_ratio"]>=ra["median_core_rl_ratio_min"] and
      rl_adv_guard>=ra["seed_guard_required"]
    )

    if not baseline_valid:
        resolution="MODEL_FREE_BASELINE_INVALID_NO_CORE_VS_RL_INFERENCE"
    elif core_adv:
        resolution="CORE_EXCLUSIVE_ADVANTAGE_WITHIN_R31"
    elif rl_adv:
        resolution="MODEL_FREE_EXCLUSIVE_ADVANTAGE_WITHIN_R31"
    elif equivalent:
        resolution="PRACTICAL_EQUIVALENCE_CORE_MODEL_FREE_RL"
    elif noninf:
        resolution="CORE_NONINFERIOR_BUT_NOT_EQUIVALENT"
    else:
        resolution="CORE_NONINFERIORITY_FAIL"

    out={
      "campaign":freeze["campaign"],
      "source_sha256":source_sha,
      "confirmatory_seeds":expected,
      "medians":m,
      "baseline_validity_checks":baseline_checks,
      "baseline_validity_seed_guard_count":int(baseline_guard),
      "baseline_valid":bool(baseline_valid),
      "core_noninferiority_checks":noninf_checks,
      "core_noninferiority_seed_guard_count":int(noninf_guard),
      "core_noninferiority_pass":bool(noninf),
      "practical_equivalence_checks":equivalence_checks,
      "practical_equivalence_seed_guard_count":int(equivalence_guard),
      "practical_equivalence_pass":bool(equivalent),
      "exclusive_core_advantage_seed_guard_count":int(core_adv_guard),
      "exclusive_core_advantage_pass":bool(core_adv),
      "exclusive_model_free_advantage_seed_guard_count":int(rl_adv_guard),
      "exclusive_model_free_advantage_pass":bool(rl_adv),
      "resolution":resolution,
      "boundaries":freeze["fixed_boundaries"]
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
