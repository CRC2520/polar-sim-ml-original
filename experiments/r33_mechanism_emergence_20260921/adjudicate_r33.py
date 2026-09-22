#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def median_arch(records,arch,key):
    return float(np.median([float(r["architectures"][arch][key]) for r in records]))

def success(q,rule):
    stable=q["stable"]>=rule["stable_min"]
    path_control=(q["control_gain_vs_current"]>=rule["control_gain_min"] and
                  q["prediction_gain_vs_current"]>=rule["other_metric_floor"])
    path_prediction=(q["prediction_gain_vs_current"]>=rule["prediction_gain_min"] and
                     q["control_gain_vs_current"]>=rule["other_metric_floor"])
    return bool(stable and (path_control or path_prediction))

def mechanism_flags(q,t):
    return {
      "D": q["D_rank1_damage"]>=t["D_rank1_damage_min"],
      "C": q["C_history_damage"]>=t["C_history_damage_min"],
      "R": q["R_transplant_damage"]>=t["R_transplant_damage_min"],
      "A": q["A_history_specificity"]>=t["A_history_specificity_min"]
    }

def ranks(a):
    a=np.asarray(a,float)
    order=np.argsort(a)
    r=np.empty(len(a),float); r[order]=np.arange(len(a),dtype=float)
    return r

def spearman(x,y):
    if len(x)<3: return 0.0
    rx=ranks(x); ry=ranks(y)
    if np.std(rx)<1e-12 or np.std(ry)<1e-12: return 0.0
    return float(np.corrcoef(rx,ry)[0,1])

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
    assert r["confirmatory_eval_families"]==f["confirmatory_eval_families"]
    assert r["confirmatory_architectures"]==f["confirmatory_architectures"]
    assert r["strict_heldout_architectures"]==f["strict_heldout_architectures"]

    rec=r["records"]
    archs=f["confirmatory_architectures"]
    keys=[
      "D_rank1_damage","C_history_damage","R_transplant_damage","R_decode_r2",
      "A_history_specificity","adapt_cost","oracle_adapt_cost","stable",
      "control_gain_vs_current","prediction_gain_vs_current","performance_score"
    ]
    med={a:{k:median_arch(rec,a,k) for k in keys} for a in archs}

    sr=f["success_rule"]; mt=f["mechanism_thresholds"]
    arch_eval={}
    for arch in archs:
        s=success(med[arch],sr)
        flags=mechanism_flags(med[arch],mt)
        arch_eval[arch]={
          "success":s,
          "mechanisms":flags,
          "mechanism_count":int(sum(flags.values())),
          "DCRA_conjunction":bool(all(flags.values()))
        }

    current_negative=(
      not arch_eval["CURRENT_MLP"]["success"] and
      med["CURRENT_MLP"]["C_history_damage"]<=f["negative_control"]["C_max"] and
      med["CURRENT_MLP"]["R_transplant_damage"]<=f["negative_control"]["R_max"] and
      med["CURRENT_MLP"]["A_history_specificity"]<=f["negative_control"]["A_max"]
    )

    noncurrent=[a for a in archs if a!="CURRENT_MLP"]
    successful=[a for a in noncurrent if arch_eval[a]["success"]]
    successful_conj=[a for a in successful if arch_eval[a]["DCRA_conjunction"]]
    heldout=f["strict_heldout_architectures"]
    heldout_success=[a for a in heldout if arch_eval[a]["success"]]
    heldout_success_conj=[a for a in heldout_success if arch_eval[a]["DCRA_conjunction"]]

    conj_rate=(len(successful_conj)/len(successful)) if successful else 0.0
    heldout_conj_rate=(len(heldout_success_conj)/len(heldout_success)) if heldout_success else 0.0
    severe_counterexamples=[a for a in successful if arch_eval[a]["mechanism_count"]<=2]

    # Secondary seed-level relationship; not a primary gate.
    perf=[]; mech=[]
    for rr in rec:
        for arch in noncurrent:
            q=rr["architectures"][arch]
            fl=mechanism_flags(q,mt)
            perf.append(float(q["performance_score"]))
            mech.append(float(sum(fl.values())))
    perf_mech_spearman=spearman(perf,mech)

    enough_capability=len(successful)>=f["minimum_successful_architectures"]
    heldout_anchor=len(heldout_success_conj)>=f["minimum_successful_heldout_architectures"]
    all_successful_converge=(len(successful)>0 and len(successful_conj)==len(successful))

    if arch_eval["CURRENT_MLP"]["success"] or severe_counterexamples:
        resolution="R33_ORGANIZATIONAL_COUNTEREXAMPLE"
    elif enough_capability and all_successful_converge and heldout_anchor and current_negative:
        resolution="R33_MECHANISM_EMERGENCE_STRONG_PASS"
    elif enough_capability and conj_rate>=f["partial_conjunction_rate_min"] and not severe_counterexamples and current_negative:
        resolution="R33_MECHANISM_EMERGENCE_PARTIAL"
    else:
        resolution="R33_INCONCLUSIVE_CAPABILITY"

    out={
      "campaign":f["campaign"],
      "source_git_blob_sha":git_blob,
      "confirmatory_seeds":f["confirmatory_seeds"],
      "architecture_medians":med,
      "architecture_adjudication":arch_eval,
      "successful_architectures":successful,
      "successful_conjunction_architectures":successful_conj,
      "conjunction_rate_among_successful":conj_rate,
      "heldout_successful_architectures":heldout_success,
      "heldout_successful_conjunction_architectures":heldout_success_conj,
      "heldout_conjunction_rate":heldout_conj_rate,
      "severe_counterexamples":severe_counterexamples,
      "current_negative_control_pass":bool(current_negative),
      "performance_mechanism_spearman_secondary":perf_mech_spearman,
      "resolution":resolution,
      "strong_pass":resolution=="R33_MECHANISM_EMERGENCE_STRONG_PASS",
      "boundaries":f["fixed_boundaries"]
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
