"""QA entry point for B1-SC-D3 design only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from experiments.b1sc_d3_v1_0.implementation import qa_tests

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    report=qa_tests()
    (out/"D3_DESIGN_QA.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
