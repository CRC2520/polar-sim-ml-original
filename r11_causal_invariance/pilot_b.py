from __future__ import annotations
import argparse,json
from pathlib import Path
from .r11b import run_pilot_b
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True)
    r=run_pilot_b(ap.parse_args().output)
    print(json.dumps({"selected":r["selected"],"feasibility":r["feasibility"]["pass_audit"],
      "ranking":[{"name":x["config"]["name"],"valid":x["valid"],"score":x["median_score"]} for x in r["ranking"]]},indent=2))
