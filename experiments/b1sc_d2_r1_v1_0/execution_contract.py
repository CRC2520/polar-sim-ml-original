"""Constants and checks for D2-R1 byte-frozen execution readiness."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
SCHEMA='B1-SC-D2-R1-1.0.0-20260915'
EXEC_BRANCH='research/b1sc-d2-r1-v1.0-execution-readiness-20260915'
CANONICAL_RUN_ID=35007729637
CANONICAL_ARTIFACT_ID=10412775720
CANONICAL_ARTIFACT_NAME='b1sc-d2-r1-init-candidates'
CANONICAL_BUNDLE_SHA256='0f3a2ac9fe3a29cbf2764257527735be000e487db1462dc3c555ef4bb34580a4'
INIT_FREEZE=ROOT/'INIT_FREEZE.json'
START_REQUEST=ROOT/'START_REQUEST.json'
def manifest():
    m=json.loads(INIT_FREEZE.read_text(encoding='utf-8'))
    ca=m['canonical_artifact']
    if (ca['run_id'],ca['artifact_id'],ca['name'])!=(CANONICAL_RUN_ID,CANONICAL_ARTIFACT_ID,CANONICAL_ARTIFACT_NAME): raise RuntimeError('canonical artifact changed')
    if m['bundle_sha256']!=CANONICAL_BUNDLE_SHA256 or not m['same_snapshot_for_on_off'] or not m['seed_reconstruction_for_training_forbidden']: raise RuntimeError('frozen initialization contract changed')
    return m
