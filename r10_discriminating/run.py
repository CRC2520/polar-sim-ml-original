from __future__ import annotations
import argparse, json
from pathlib import Path
from .experiments import run_all

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    summary=run_all(args.output)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
