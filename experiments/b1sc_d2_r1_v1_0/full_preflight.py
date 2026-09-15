"""Publish full D2-R1 execution-readiness freeze; never runs science."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from experiments.b1sc_d2_r1_v1_0 import execution_contract as contract
from experiments.b1sc_d2_v1_0 import implementation as d2
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
SOURCES=(
'experiments/b1sc_d2_r1_v1_0/INIT_FREEZE.json',
'experiments/b1sc_d2_r1_v1_0/FIT_REGISTRY.json',
'experiments/b1sc_d2_r1_v1_0/PIPELINE_CONTRACT.json',
'experiments/b1sc_d2_r1_v1_0/execution_contract.py',
'experiments/b1sc_d2_r1_v1_0/init_format.py',
'experiments/b1sc_d2_r1_v1_0/initialization.py',
'experiments/b1sc_d2_r1_v1_0/scientific_runner.py',
'experiments/b1sc_d2_r1_v1_0/full_execution.py',
'experiments/b1sc_d2_r1_v1_0/runner.py',
'experiments/b1sc_d2_r1_v1_0/runner_tests.py',
'experiments/b1sc_d2_r1_v1_0/full_pipeline_tests.py',
'experiments/b1sc_d2_r1_v1_0/full_preflight.py',
'.github/workflows/b1sc-d2-r1-v1-0-scientific.yml',
'.github/workflows/b1sc-d2-r1-full-qa.yml',
'experiments/b1sc_d2_r1_v1_0/frozen_init/PERSISTED_MANIFEST.json',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-0.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-1.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-2.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-3.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-4.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-5.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-6.bin',
'experiments/b1sc_d2_r1_v1_0/frozen_init/block-7.bin',
)
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source-commit',required=True);p.add_argument('--qa-run-id',required=True,type=int);a=p.parse_args()
 contract.manifest();assert not (ROOT/'START_REQUEST.json').exists()
 for rel in SOURCES:assert (REPO/rel).is_file(),rel
 wf=(REPO/'.github/workflows/b1sc-d2-r1-v1-0-scientific.yml').read_text(encoding='utf-8')
 assert 'workflow_dispatch' not in wf and 'START_REQUEST.json' in wf and 'paths:' in wf
 for token in ('authorize:','fit:','training_aggregate:','online:','online_aggregate:','final:'):assert token in wf
 assert 'max-parallel: 4' in wf and 'contents: read' in wf and 'd2r1-final-review' in wf
 hashes={rel:sha(REPO/rel) for rel in SOURCES}
 freeze={'schema':contract.SCHEMA,'status':'D2_R1_FULL_EXECUTION_PIPELINE_QUALIFIED_NOT_AUTHORIZED','branch':'research/b1sc-d2-r1-v1.0-run2-repair-20260915','qualified_source_commit':a.source_commit,'qa_run_id':a.qa_run_id,'canonical_init_generation_run_id':contract.CANONICAL_RUN_ID,'canonical_init_artifact_id':contract.CANONICAL_ARTIFACT_ID,'canonical_init_bundle_sha256':contract.CANONICAL_BUNDLE_SHA256,'planned_fits':d2.FITS,'blocks':d2.BLOCKS,'native_steps_per_fit':d2.NATIVE_STEPS,'checkpoints':list(d2.CHECKPOINTS),'diagnostic_episodes':d2.DIAGNOSTIC_EPISODES,'endpoint_episodes':d2.ENDPOINT_EPISODES,'causal_episodes':d2.CAUSAL_EPISODES,'minimum_blocks':d2.MIN_BLOCKS,'pipeline_stages':['authorize','fit','training_aggregate','online','online_aggregate','final'],'snapshot_consumption_mandatory':True,'persistent_init_in_repository':True,'canonical_artifact_dependency_at_runtime':False,'same_snapshot_for_on_off':True,'seed_reconstruction_for_training_forbidden':True,'optimizer_requires_validated_init':True,'first_optimizer_step_requires_validated_init':True,'training_gate':'aggregate_and_at_least_6_of_8','online_gate':'aggregate_and_at_least_6_of_8_and_controls','window_9_40_secondary_only':True,'does_not_rewrite_d2':True,'scientific_workflow_present':True,'scientific_training_performed':False,'scientific_evaluation_performed':False,'scientific_results_exist':False,'start_request_present':False,'source_hashes':hashes,'B1E_disposition':'ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION','ready_for_b1e_protocol_design':False,'ready_for_b1e_freeze':False,'ready_for_b1e_confirmatory_run':False,'B1E_executed':False,'final_seeds_generated':False,'H_CAT':'NOT_EVALUABLE','H_TRANSFER':'NOT_EVALUATED'}
 status=dict(freeze,status='D2_R1_FULL_PREFLIGHT_PASS_NOT_EXECUTED')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'FULL_EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
 (out/'FULL_PREFLIGHT_STATUS.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','source_count':len(SOURCES),'training_performed':False,'evaluation_performed':False,'start_request_present':False},sort_keys=True))
if __name__=='__main__':main()
