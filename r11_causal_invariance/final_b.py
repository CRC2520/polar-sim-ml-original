"""R11b confirmatory runner. Refuses execution without a matching frozen source map."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from r10_discriminating.experiments import _train_agent
from .config import GLOBAL_REQUIRED
from .experiments import experiment2,experiment3,PILOT_CONFIGS
from .r11b import FINAL_B_SEEDS,feasibility_audit,experiment4b,experiment5b

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_freeze():
    p=HERE/"FREEZE_R11B.json"
    if not p.exists(): raise RuntimeError("R11b confirmation is not frozen/authorized")
    f=json.loads(p.read_text())
    if tuple(f["final_seeds"])!=FINAL_B_SEEDS:
        raise RuntimeError("Final seed set mismatch")
    for rel,expected in f["sha256"].items():
        actual=sha256(ROOT/rel)
        if actual!=expected: raise RuntimeError(f"Frozen source mismatch: {rel}")
    allowed={x["name"]:x for x in PILOT_CONFIGS}
    cfg=f["selected_config"]
    if cfg["name"] not in allowed or cfg!=allowed[cfg["name"]]:
        raise RuntimeError("Selected config is outside the preregistered pilot grid")
    return f

def median(records,path):
    vals=[]
    for r in records:
        v=r
        for key in path: v=v[key]
        vals.append(float(v))
    return float(np.median(vals))

def run(output):
    freeze=load_freeze()
    audit=feasibility_audit()
    if not audit["pass_audit"]: raise RuntimeError("Transfer feasibility audit no longer passes")
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    cfg=freeze["selected_config"]; records=[]
    for seed in FINAL_B_SEEDS:
        base=_train_agent(seed)
        rec=dict(seed=seed,E2=experiment2(base,seed,cfg),E3=experiment3(seed),
                 E4=experiment4b(base,seed,cfg),E5=experiment5b(base,seed,cfg))
        records.append(rec)
        (out/f"seed_{seed}.json").write_text(json.dumps(rec,indent=2))
    counts={k:sum(bool(r[k]["pass_seed"]) for r in records) for k in ("E2","E3","E4","E5")}
    verdicts={k:("PASS" if v>=GLOBAL_REQUIRED else "FAIL") for k,v in counts.items()}
    medians=dict(
      E2_relational_reward_diff=median(records,("E2","domains","relational","reward_diff")),
      E2_delayed_reward_diff=median(records,("E2","domains","relational_delay","reward_diff")),
      E3_balanced_accuracy=median(records,("E3","balanced_accuracy")),
      E3_mse_improvement=median(records,("E3","improvement")),
      E4_reservoir_return=median(records,("E4","domains","oscillatory_reservoir_v2","r11","return_mean")),
      E4_maintenance_return=median(records,("E4","domains","maintenance_queue_v2","r11","return_mean")),
      E4_reservoir_alive=median(records,("E4","domains","oscillatory_reservoir_v2","r11","alive_fraction")),
      E4_maintenance_alive=median(records,("E4","domains","maintenance_queue_v2","r11","alive_fraction")),
      E5_self_world=median(records,("E5","self_world")),
      E5_source=median(records,("E5","source")),
      E5_time=median(records,("E5","time")),
      E5_auc=median(records,("E5","auc")),
      E5_counterfactual=median(records,("E5","counterfactual")),
      E5_memory_drop=median(records,("E5","memory_drop")),
      E5_content_drop=median(records,("E5","content_drop")),
      E5_cross_drop=median(records,("E5","cross_drop")))
    result=dict(version="R11b",freeze=freeze,feasibility=audit,n=len(records),
                required=GLOBAL_REQUIRED,counts=counts,verdicts=verdicts,
                medians=medians,records=records)
    (out/"RESULTS_R11B.json").write_text(json.dumps(result,indent=2))
    lines=["# POLAR R11b — confirmatory four-gap campaign","",
      f"Final seeds: {len(records)}; required: {GLOBAL_REQUIRED}/{len(records)}.","",
      "| Experiment | Complete seed passes | Verdict |","|---|---:|---|"]
    for k in ("E2","E3","E4","E5"):
        lines.append(f"| {k} | {counts[k]}/{len(records)} | **{verdicts[k]}** |")
    lines+=["","## Medians",""]
    for k,v in medians.items(): lines.append(f"- {k}: **{v:.6f}**")
    lines+=["","## Boundary",
      "R11b tests bounded computational mechanisms in synthetic environments.",
      "E5 is a functional precursor benchmark and does not measure phenomenal consciousness.",
      "R10 and R11a adverse results remain unchanged."]
    (out/"REPORT_R11B.md").write_text("\n".join(lines)+"\n")
    h=hashlib.sha256()
    for p in sorted(out.glob("*.json"))+sorted(out.glob("*.md")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    (out/"SHA256.txt").write_text(h.hexdigest()+"  R11B_OUTPUT_SET\n")
    return result

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True)
    r=run(ap.parse_args().output)
    print(json.dumps({"counts":r["counts"],"verdicts":r["verdicts"],"medians":r["medians"]},indent=2))
