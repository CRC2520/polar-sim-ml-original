from __future__ import annotations
import argparse, json
from pathlib import Path
from .config_confirm_v2 import FINAL_SEEDS,GLOBAL_REQUIRED

def main(run1,run2,out):
    run1=Path(run1);run2=Path(run2);out=Path(out)
    a=json.loads((run1/"RESULTS_P2_CRITICAL.json").read_text())
    b=json.loads((run2/"RESULTS_P2_CRITICAL.json").read_text())
    checks=[]
    def check(cond,name):
        checks.append({"name":name,"pass":bool(cond)})
        if not cond: raise ValueError(name)
    check(a==b,"full confirmatory result exact reproduction")
    check(a["seeds"]==list(FINAL_SEEDS),"exact confirmatory seed set")
    check(a["required"]==GLOBAL_REQUIRED,"global threshold")
    check(set(a["counts"])=={"E1","E2","E3","E4"},"four gap endpoints")
    for k,v in a["counts"].items():
        check(a["verdicts"][k]==("PASS" if v>=GLOBAL_REQUIRED else "FAIL"),f"reconstruct {k} verdict")
    check(a["all_four_resolved"]==all(v=="PASS" for v in a["verdicts"].values()),"reconstruct global verdict")
    rec={k:sum(bool(r["checks"][k]) for r in a["records"]) for k in ("E1","E2","E3","E4")}
    check(rec==a["counts"],"reconstruct counts from seed records")
    result={"status":"PASS","checks_passed":len(checks),"checks_total":len(checks),
            "exact_reproduction":12,"checks":checks,
            "source_sha256":a["source_sha256"]}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run1",required=True);ap.add_argument("--run2",required=True);ap.add_argument("--output",required=True)
    x=ap.parse_args();main(x.run1,x.run2,x.output)
