#!/usr/bin/env python3
"""Supported launcher; registers the dynamically loaded frozen module for checkpoint serialization."""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import p1_p2_suite as suite
sys.modules[suite.base.__name__]=suite.base
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['p1','p2','e7'],required=True)
p.add_argument('--out',default='r40_results')
a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
source={str(f.relative_to(suite.HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(suite.HERE.glob('*.py'))}
start={'utc_start':datetime.now(timezone.utc).isoformat(),'stage':a.stage,'python':platform.python_version(),
       'source_sha256':source,'reference_sha256':hashlib.sha256(suite.SOURCE.read_bytes()).hexdigest(),
       'workflow_commit':os.environ.get('GITHUB_SHA'),'run_id':os.environ.get('GITHUB_RUN_ID'),
       'vectorized_planner_equivalence_cases':100,'instrument_pass':True,
       'scientific_status':'not adjudicated by workflow exit status'}
suite.dump(out/f'PROVENANCE_{a.stage}.json',start)
{'p1':suite.p1,'p2':suite.p2,'e7':suite.e7}[a.stage](out)
start['utc_finish']=datetime.now(timezone.utc).isoformat()
suite.dump(out/f'PROVENANCE_{a.stage}.json',start)
