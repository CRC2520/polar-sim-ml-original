from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .experiments import experiment1,experiment2,experiment3,experiment4
from .config_confirm_v2 import *

ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=(
    "p2_critical/experiments.py",
    "p2_critical/config_confirm_v2.py",
    "p2_critical/PREREG_P2_CRITICAL_V2.md",
    "p2_critical/confirm.py",
    "p2_critical/audit.py",
)

def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def json_default(o):
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)

def e1_check(r):
    p,g=r["polar"],r["generic"]
    common=(p["true_overlap"]>=E1_REL_OVERLAP_MIN and
            g["true_overlap"]>=E1_REL_OVERLAP_MIN and
            r["random_advantage_polar"]>=E1_RANDOM_ADVANTAGE and
            r["random_advantage_generic"]>=E1_RANDOM_ADVANTAGE)
    specific=(g["mse"]-p["mse"]>=E1_SPECIFIC_ADVANTAGE)
    equiv=(abs(p["mse"]-g["mse"])<=E1_EQUIV_MARGIN)
    return bool(common and (specific or equiv)), ("POLAR_SPECIFIC" if specific else
        "EQUIVALENT_WHEN_GENERIC_RECONSTRUCTS_RELATIONS" if equiv and common else "UNRESOLVED")

def e2_check(r):
    a=r["adaptive"]
    return bool(a["pre_overlap"]>=E2_OVERLAP_MIN and
                a["post_overlap"]>=E2_OVERLAP_MIN and a["changed"] and
                r["advantage_vs_frozen"]>=E2_REWARD_ADVANTAGE and
                r["lesion_damage"]>=E2_LESION_DAMAGE)

def e3_check(r):
    f=r["full"];p=r["polar_lesion"];g=r["gwt_lesion"];h=r["hot_lesion"]
    unique=(g["primary_damage"]>p["primary_damage"]+.05 and
            p["workspace_damage"]>h["workspace_damage"]+.08 and
            h["meta_damage"]>0)
    return bool(f["primary_accuracy"]>=E3_FULL_PRIMARY_MIN and
                f["workspace_recall"]>=E3_FULL_WORKSPACE_MIN and
                f["source_ba"]>=E3_FULL_SOURCE_MIN and
                p["primary_damage"]<=E3_PRIMARY_POLAR_MAX and
                p["workspace_damage"]>=E3_POLAR_WORKSPACE_DAMAGE and
                p["meta_damage"]>=E3_POLAR_META_DAMAGE and
                p["source_damage"]>=E3_POLAR_SOURCE_DAMAGE and
                g["primary_damage"]>=E3_GWT_PRIMARY_DAMAGE and
                h["primary_damage"]<=E3_HOT_PRIMARY_MAX and
                h["workspace_damage"]<=E3_HOT_WORKSPACE_MAX and
                h["meta_damage"]>=E3_HOT_META_DAMAGE and unique)

def e4_check(r):
    return all(v["oracle"]["alive_fraction"]>=E4_ORACLE_ALIVE_MIN and
               v["agent"]["alive_fraction"]>=E4_ALIVE_MIN and
               v["return_ratio"]>=E4_RETURN_RATIO_MIN
               for v in r["tasks"].values())

def run_seed(seed):
    raw=dict(E1=experiment1(seed),E2=experiment2(seed),E3=experiment3(seed),E4=experiment4(seed))
    e1,res=e1_check(raw["E1"])
    checks=dict(E1=e1,E2=e2_check(raw["E2"]),E3=e3_check(raw["E3"]),E4=e4_check(raw["E4"]))
    raw["E1"]["confirm_resolution"]=res
    return dict(seed=int(seed),checks=checks,pass_all=all(checks.values()),metrics=raw)

def median(rows,path):
    vals=[]
    for row in rows:
        v=row["metrics"]
        for p in path:v=v[p]
        vals.append(float(v))
    return float(np.median(vals))

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[run_seed(s) for s in FINAL_SEEDS]
    for row in rows:
        (out/f"seed_{row['seed']}.json").write_text(json.dumps(row,indent=2,sort_keys=True,default=json_default))
    counts={k:sum(r["checks"][k] for r in rows) for k in ("E1","E2","E3","E4")}
    verdicts={k:("PASS" if v>=GLOBAL_REQUIRED else "FAIL") for k,v in counts.items()}
    res_counts={}
    for r in rows:
        x=r["metrics"]["E1"]["confirm_resolution"];res_counts[x]=res_counts.get(x,0)+1
    medians={
      "E1_polar_minus_generic":median(rows,("E1","polar_minus_generic")),
      "E1_polar_overlap":median(rows,("E1","polar","true_overlap")),
      "E1_generic_overlap":median(rows,("E1","generic","true_overlap")),
      "E2_pre_overlap":median(rows,("E2","adaptive","pre_overlap")),
      "E2_post_overlap":median(rows,("E2","adaptive","post_overlap")),
      "E2_advantage_vs_frozen":median(rows,("E2","advantage_vs_frozen")),
      "E2_lesion_damage":median(rows,("E2","lesion_damage")),
      "E3_primary":median(rows,("E3","full","primary_accuracy")),
      "E3_workspace":median(rows,("E3","full","workspace_recall")),
      "E3_source":median(rows,("E3","full","source_ba")),
      "E3_polar_workspace_damage":median(rows,("E3","polar_lesion","workspace_damage")),
      "E3_polar_meta_damage":median(rows,("E3","polar_lesion","meta_damage")),
      "E3_polar_source_damage":median(rows,("E3","polar_lesion","source_damage")),
      "E3_gwt_primary_damage":median(rows,("E3","gwt_lesion","primary_damage")),
      "E3_hot_meta_damage":median(rows,("E3","hot_lesion","meta_damage")),
    }
    for task in ("logistic_harvest","thermal_rc","queue_service"):
        medians[f"E4_{task}_ratio"]=median(rows,("E4","tasks",task,"return_ratio"))
        medians[f"E4_{task}_alive"]=median(rows,("E4","tasks",task,"agent","alive_fraction"))
    result={
      "protocol":"PREREG_P2_CRITICAL_V2.md",
      "seeds":list(FINAL_SEEDS),"required":GLOBAL_REQUIRED,
      "source_sha256":{p:sha(p) for p in SOURCE_FILES},
      "counts":counts,"verdicts":verdicts,
      "e1_resolution_counts":res_counts,
      "all_four_resolved":all(v=="PASS" for v in verdicts.values()),
      "medians":medians,"records":rows,
      "interpretation_boundary":{
        "E3":"functional theory-discriminating lesion signatures only; not phenomenal consciousness",
        "E4":"standard-equation OOD validation only; not independent-team replication"
      }
    }
    text=json.dumps(result,indent=2,sort_keys=True,default=json_default)+"\n"
    (out/"RESULTS_P2_CRITICAL.json").write_text(text)
    report=["# POLAR P2-Critical confirmatory results","",
            f"Seeds: {FINAL_SEEDS[0]}–{FINAL_SEEDS[-1]}; global criterion: >={GLOBAL_REQUIRED}/12.","",
            "| Gap | Pass seeds | Verdict |","|---|---:|---|"]
    names={"E1":"POLAR vs GENERIC isomorphic","E2":"Integrated relational discovery",
           "E3":"Theory-discriminating lesions","E4":"Standard-equation OOD"}
    for k in ("E1","E2","E3","E4"):report.append(f"| {names[k]} | {counts[k]}/12 | **{verdicts[k]}** |")
    report+=["",f"All four operational gaps resolved: **{result['all_four_resolved']}**.",
             "",f"E1 resolution counts: {res_counts}.","",
             "E3 does not establish phenomenal consciousness. E4 does not constitute independent-team replication."]
    (out/"REPORT_P2_CRITICAL.md").write_text("\n".join(report)+"\n")
    print(text)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True)
    main(ap.parse_args().output)
