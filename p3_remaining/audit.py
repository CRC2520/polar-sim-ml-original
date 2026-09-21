from __future__ import annotations
import argparse,json
from pathlib import Path
from .config_confirm import FINAL_SEEDS,GLOBAL_REQUIRED

def main(run1,run2,out):
    a=json.loads((Path(run1)/"RESULTS_P3.json").read_text())
    b=json.loads((Path(run2)/"RESULTS_P3.json").read_text())
    checks=[]
    def check(cond,name):
        checks.append({"name":name,"pass":bool(cond)})
        if not cond: raise ValueError(name)
    check(a==b,"full confirmatory result exact reproduction")
    check(a["seeds"]==list(FINAL_SEEDS),"exact confirmatory seed set")
    check(a["required"]==GLOBAL_REQUIRED,"global threshold")
    check(set(a["counts"])=={"A","B","C","D"},"four P3 internal endpoints")
    for k in ("A","B","C","D"):
        check(a["verdicts"][k]==("PASS" if a["counts"][k]>=GLOBAL_REQUIRED else "FAIL"),f"reconstruct {k} verdict")
    rec={k:sum(bool(r["checks"][k]) for r in a["records"]) for k in ("A","B","C","D")}
    check(rec==a["counts"],"reconstruct counts from seed records")
    check(a["all_four_internal_gaps_resolved"]==all(v=="PASS" for v in a["verdicts"].values()),"reconstruct global verdict")
    result={"status":"PASS","checks_total":len(checks),"checks_passed":len(checks),
            "exact_reproduction":12,"checks":checks,"source_sha256":a["source_sha256"]}
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run1",required=True);ap.add_argument("--run2",required=True);ap.add_argument("--output",required=True)
    x=ap.parse_args();main(x.run1,x.run2,x.output)
