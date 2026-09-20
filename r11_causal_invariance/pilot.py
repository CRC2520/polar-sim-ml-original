from __future__ import annotations
import argparse, json
from pathlib import Path
from .experiments import run_pilot

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    r=run_pilot(args.output)
    print(json.dumps({'selected':r['selected'],'ranking':[{'name':x['config']['name'],'valid':x['valid'],'score':x['median_score']} for x in r['ranking']]},indent=2))
if __name__=='__main__': main()
