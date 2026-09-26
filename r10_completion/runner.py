"""R10 pilot/final runner. Final mode is authorized only after a frozen source manifest exists."""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
from .config import VERSION,PILOT_SEEDS,FINAL_SEEDS,GLOBAL_REQUIRED
from . import exp1_causal_state as e1
from . import exp2_isomorphic as e2
from . import exp3_gate as e3
from . import exp4_transfer as e4
from . import exp5_integrated as e5
from .core import jsonable, bootstrap_median, bootstrap_mean

ROOT=Path(__file__).resolve().parents[1]

def _write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(jsonable(obj),indent=2,sort_keys=True,allow_nan=False)+"\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()

def run_seed(seed):
    return dict(seed=int(seed),E1=e1.run(seed),E2=e2.run(seed),E3=e3.run(seed),
                E4=e4.run(seed),E5=e5.run(seed))

def _metric(rows,path):
    vals=[]
    for r in rows:
        x=r
        for p in path: x=x[p]
        vals.append(float(x))
    return vals

def aggregate(rows,phase):
    counts={k:int(sum(bool(r[k]['pass_strong']) for r in rows)) for k in ("E1","E2","E3","E4","E5")}
    required=GLOBAL_REQUIRED if phase=="final" else None
    verdict={k:(counts[k]>=GLOBAL_REQUIRED if phase=="final" else None) for k in counts}
    e2_exact=int(sum(r["E2"]["pass_exact"] for r in rows))
    e2_spec=int(sum(r["E2"]["pass_specificity"] for r in rows))
    summary=dict(
        version=VERSION,phase=phase,n=len(rows),counts=counts,required=required,verdict=verdict,
        joint_all_five=(all(verdict.values()) if phase=="final" else None),
        E2_exact_count=e2_exact,E2_specificity_count=e2_spec,
        selected_dimension_median=float(np.median(_metric(rows,["E1","selected_dimension"]))),
        E1_reward_loss_median=float(np.median(_metric(rows,["E1","confirmatory","reward_loss"]))),
        E2_structural_advantage_median=float(np.median(_metric(rows,["E2","matched_structural","return_advantage"]))),
        E3_gate_balanced_accuracy_median=float(np.median(_metric(rows,["E3","confirmatory","gate_balanced_accuracy"]))),
        E3_advantage_best_constant_median=float(np.median(_metric(rows,["E3","confirmatory","advantage_best_constant"]))),
        E4_queue_return_median=float(np.median(_metric(rows,["E4","heldout","queue","polar_return"]))),
        E4_inertia_return_median=float(np.median(_metric(rows,["E4","heldout","inertia","polar_return"]))),
        E5_primary_median=float(np.median(_metric(rows,["E5","intact","primary_balanced_accuracy"]))),
        E5_meta_gain_median=float(np.median(_metric(rows,["E5","intact","meta_brier_improvement"]))),
        E5_signature_count_median=float(np.median(_metric(rows,["E5","signature_count"]))),
    )
    if phase=="final":
        for name,path in {
            "E1_reward_loss":["E1","confirmatory","reward_loss"],
            "E2_advantage":["E2","matched_structural","return_advantage"],
            "E3_gate_accuracy":["E3","confirmatory","gate_balanced_accuracy"],
            "E3_advantage":["E3","confirmatory","advantage_best_constant"],
            "E4_queue_return":["E4","heldout","queue","polar_return"],
            "E4_inertia_return":["E4","heldout","inertia","polar_return"],
            "E5_meta_gain":["E5","intact","meta_brier_improvement"],
        }.items():
            vals=_metric(rows,path)
            summary[name+"_bootstrap95_median"]=bootstrap_median(vals,seed=99000+len(name))
    return summary

def verify_final_authorization():
    freeze=ROOT/"r10_completion/FREEZE_R10.json"
    auth=ROOT/"r10_completion/FINAL_EXECUTION_AUTHORIZATION.json"
    if not freeze.exists() or not auth.exists():
        raise RuntimeError("Final execution requires frozen source and explicit authorization")
    f=json.loads(freeze.read_text()); a=json.loads(auth.read_text())
    if not a.get("authorized") or a.get("freeze_sha256")!=f.get("freeze_sha256"):
        raise RuntimeError("Final authorization does not match frozen source")
    from .freeze import verify
    verify(f)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("pilot","final"),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    seeds=PILOT_SEEDS if args.phase=="pilot" else FINAL_SEEDS
    if args.phase=="final": verify_final_authorization()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        row=run_seed(seed); rows.append(row)
        _write(args.output/f"seed_{seed}.json",row)
        print(json.dumps({"seed":seed,"passes":{k:row[k]["pass_strong"] for k in ("E1","E2","E3","E4","E5")}},sort_keys=True),flush=True)
    agg=aggregate(rows,args.phase)
    _write(args.output/"R10_RESULTS.json",dict(aggregate=agg,per_seed=rows))
    print(json.dumps(agg,indent=2,sort_keys=True))
if __name__=="__main__": main()
