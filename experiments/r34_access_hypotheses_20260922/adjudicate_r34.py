#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def med(records, arch, key, section=None):
    vals=[]
    for r in records:
        q=r["architectures"][arch]
        if section is not None:
            q=q[section]
        vals.append(float(q[key]))
    return float(np.median(vals))

def source_blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()

def r33_success(q, rule):
    stable=q["stable"]>=rule["stable_min"]
    p1=q["control_gain_vs_current"]>=rule["control_gain_min"] and q["prediction_gain_vs_current"]>=rule["other_metric_floor"]
    p2=q["prediction_gain_vs_current"]>=rule["prediction_gain_min"] and q["control_gain_vs_current"]>=rule["other_metric_floor"]
    return bool(stable and (p1 or p2))

def r33_flags(q,t):
    return {
      "D":q["D_rank1_damage"]>=t["D_min"],
      "C":q["C_history_damage"]>=t["C_min"],
      "R":q["R_transplant_damage"]>=t["R_min"],
      "A":q["A_history_specificity"]>=t["A_min"],
    }

def adjud_h1(r,f):
    rec=r["records"]; archs=f["architectures"]
    keys=["D_rank1_damage","C_history_damage","R_transplant_damage","A_history_specificity",
          "control_gain_vs_current","prediction_gain_vs_current","stable"]
    m={a:{k:med(rec,a,k) for k in keys} for a in archs}
    ae={}
    for a in archs:
        s=r33_success(m[a],f["success_rule"])
        fl=r33_flags(m[a],f["mechanism_thresholds"])
        ae[a]={"success":s,"mechanisms":fl,"DCR":bool(fl["D"] and fl["C"] and fl["R"]),"A_fail":not fl["A"]}
    current_ok=not ae["CURRENT_MLP"]["success"]
    noncurrent=[a for a in archs if a!="CURRENT_MLP"]
    counts={}
    for a in noncurrent:
        n=0
        for rr in rec:
            q=rr["architectures"][a]
            s=r33_success(q,f["success_rule"])
            fl=r33_flags(q,f["mechanism_thresholds"])
            if s and fl["D"] and fl["C"] and fl["R"] and not fl["A"]:
                n+=1
        counts[a]=n
    supported=[a for a in noncurrent if ae[a]["success"] and ae[a]["DCR"] and ae[a]["A_fail"] and counts[a]>=f["seed_guard_required"]]
    successful=[a for a in noncurrent if ae[a]["success"]]
    necessity=(len(successful)>=f["minimum_successful_architectures"] and all(ae[a]["mechanisms"]["A"] for a in successful) and current_ok)
    if supported and current_ok:
        resolution="R34_H1_A_NOT_NECESSARY_SUPPORTED_UNDER_R33_TASK"
    elif necessity:
        resolution="R34_H1_A_NECESSITY_SUPPORTED_UNDER_R33_TASK"
    else:
        resolution="R34_H1_INCONCLUSIVE"
    return {"architecture_medians":m,"architecture_adjudication":ae,"A_fail_DCR_success_seed_counts":counts,
            "supporting_architectures":supported,"resolution":resolution}

def q_cap(q,rule):
    return bool(q["q1_gain_vs_current"]>=rule["q1_gain_min"] and q["q0_ratio_vs_current"]<=rule["q0_ratio_max"])

def adjud_h2(r,f):
    rec=r["records"]; archs=f["architectures"]
    keys=["q0_mse","q1_mse","history_damage_q0","history_damage_q1","A_switch","q1_gain_vs_current","q0_ratio_vs_current"]
    m={a:{k:med(rec,a,k) for k in keys} for a in archs}
    ae={}
    counts={}
    for a in archs:
        cap=q_cap(m[a],f["capability"])
        apass=m[a]["A_switch"]>=f["A_switch_min"]
        hist=m[a]["history_damage_q1"]>=f["history_q1_damage_min"]
        ae[a]={"capable":cap,"A_switch_pass":apass,"history_q1_pass":hist,"direct_A":bool(apass and hist)}
        if a!="CURRENT_MLP":
            n=0
            for rr in rec:
                q=rr["architectures"][a]
                if q_cap(q,f["capability"]) and q["A_switch"]>=f["seed_guard"]["A_switch_min"] and q["history_damage_q1"]>=f["seed_guard"]["history_q1_damage_min"]:
                    n+=1
            counts[a]=n
    current_ok=abs(m["CURRENT_MLP"]["A_switch"])<=f["current_A_abs_max"]
    targets=[a for a in archs if a!="CURRENT_MLP"]
    supported=[a for a in targets if ae[a]["capable"] and ae[a]["direct_A"] and counts[a]>=f["seed_guard_required"]]
    capable=[a for a in targets if ae[a]["capable"]]
    if len(supported)==len(targets) and current_ok:
        resolution="R34_H2_BROADER_A_CAPACITY_SUPPORTED"
    elif supported and current_ok:
        resolution="R34_H2_PARTIAL_BROADER_A_CAPACITY"
    elif len(capable)==len(targets) and not supported:
        resolution="R34_H2_BROADER_A_NOT_SUPPORTED"
    else:
        resolution="R34_H2_INCONCLUSIVE_CAPABILITY"
    return {"architecture_medians":m,"architecture_adjudication":ae,"seed_guard_counts":counts,
            "supporting_architectures":supported,"resolution":resolution}

def h3_cap(row,rule):
    return bool(row["q1_gain_vs_current"]>=rule["q1_gain_min"] and row["q0_ratio_vs_current"]<=rule["q0_ratio_max"])

def adjud_h3(r,f):
    rec=r["records"]; archs=f["architectures"]
    m={}
    ae={}
    counts={}
    for a in archs:
        m[a]={
          "selective":{k:med(rec,a,k,"selective") for k in ["q0_mse","q1_mse","history_damage_q0","history_damage_q1","A_switch","q1_gain_vs_current","q0_ratio_vs_current"]},
          "uniform":{k:med(rec,a,k,"uniform") for k in ["q0_mse","q1_mse","history_damage_q0","history_damage_q1","A_switch","q1_gain_vs_current","q0_ratio_vs_current"]},
          "A_interaction":float(np.median([rr["architectures"][a]["A_interaction"] for rr in rec]))
        }
        if a=="CURRENT_MLP":
            continue
        cap_sel=h3_cap(m[a]["selective"],f["capability_selective"])
        cap_uni=h3_cap(m[a]["uniform"],f["capability_uniform"])
        pattern=(cap_sel and cap_uni and
                 m[a]["selective"]["A_switch"]>=f["selective_A_min"] and
                 abs(m[a]["uniform"]["A_switch"])<=f["uniform_A_abs_max"] and
                 m[a]["A_interaction"]>=f["interaction_min"])
        ae[a]={"capable_selective":cap_sel,"capable_uniform":cap_uni,"contingency_pattern":bool(pattern)}
        n=0
        for rr in rec:
            q=rr["architectures"][a]
            if (h3_cap(q["selective"],f["capability_selective"]) and
                h3_cap(q["uniform"],f["capability_uniform"]) and
                q["selective"]["A_switch"]>=f["seed_guard"]["selective_A_min"] and
                abs(q["uniform"]["A_switch"])<=f["seed_guard"]["uniform_A_abs_max"] and
                q["A_interaction"]>=f["seed_guard"]["interaction_min"]):
                n+=1
        counts[a]=n
    noncurrent=[a for a in archs if a!="CURRENT_MLP"]
    pattern_arch=[a for a in noncurrent if ae[a]["contingency_pattern"] and counts[a]>=f["seed_guard_required"]]
    heldout=[a for a in f["strict_heldout_architectures"] if a in pattern_arch]
    capable_both=[a for a in noncurrent if ae[a]["capable_selective"] and ae[a]["capable_uniform"]]
    current_ok=abs(m["CURRENT_MLP"]["A_interaction"])<=f["current_interaction_abs_max"]
    if len(pattern_arch)>=f["minimum_pattern_architectures"] and len(heldout)>=f["minimum_heldout_pattern_architectures"] and current_ok:
        resolution="R34_H3_A_TASK_CONTINGENCY_SUPPORTED"
    elif len(capable_both)>=f["minimum_capable_architectures"] and len(pattern_arch)<f["minimum_pattern_architectures"]:
        resolution="R34_H3_A_TASK_CONTINGENCY_NOT_SUPPORTED"
    else:
        resolution="R34_H3_INCONCLUSIVE_CAPABILITY"
    return {"architecture_medians":m,"architecture_adjudication":ae,"seed_guard_counts":counts,
            "pattern_architectures":pattern_arch,"heldout_pattern_architectures":heldout,
            "resolution":resolution}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    r=json.loads(Path(a.results).read_text())
    f=json.loads(Path(a.freeze).read_text())
    blob=source_blob(a.source)
    assert f["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert blob==f["source_git_blob_sha"],(blob,f["source_git_blob_sha"])
    assert r["hypothesis"]==f["hypothesis"]
    assert r["mode"]=="confirm"
    assert r["seeds"]==f["confirmatory_seeds"]
    assert r["architectures"]==f["architectures"]
    if f["hypothesis"]=="H1":
        out=adjud_h1(r,f)
    elif f["hypothesis"]=="H2":
        out=adjud_h2(r,f)
    else:
        out=adjud_h3(r,f)
    out.update({
      "campaign":f["campaign"],
      "hypothesis":f["hypothesis"],
      "source_git_blob_sha":blob,
      "confirmatory_seeds":f["confirmatory_seeds"],
      "boundaries":f["fixed_boundaries"]
    })
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
