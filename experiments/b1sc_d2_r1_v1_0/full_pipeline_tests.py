"""QA-only qualification tests for the full D2-R1 pipeline."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from experiments.b1sc_d2_r1_v1_0 import full_execution as f
from experiments.b1sc_d2_r1_v1_0 import execution_contract as contract
from experiments.b1sc_d2_v1_0 import implementation as d2

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
WF=REPO/'.github/workflows/b1sc-d2-r1-v1-0-scientific.yml'


def rule_tests():
    wf=WF.read_text(encoding='utf-8');src=(ROOT/'scientific_runner.py').read_text(encoding='utf-8');pipe=(ROOT/'full_execution.py').read_text(encoding='utf-8')
    assert 'workflow_dispatch' not in wf
    assert 'experiments/b1sc_d2_r1_v1_0/START_REQUEST.json' in wf
    assert 'research/b1sc-d2-r1-v1.0-run2-repair-20260915' in wf
    for token in ('authorize:','fit:','training_aggregate:','online:','online_aggregate:','final:'):assert token in wf,token
    assert 'max-parallel: 4' in wf and 'd2r1-final-review' in wf
    assert not (ROOT/'START_REQUEST.json').exists()
    for step in d2.CHECKPOINTS:assert f'model-{{steps}}.pt' in src or 'model-{steps}.pt' in src
    assert 'model-final.pt' in src and 'DIAGNOSTIC_' in src and 'ENDPOINT.json' in src and 'COMPLETE.json' in src
    for fn in ('aggregate_training','online_block','aggregate_online','final_aggregate'):assert f'def {fn}' in pipe
    assert 'H_TRAIN_D2_R1' in pipe and 'H_ONLINE_D2_R1' in pipe and 'does_not_rewrite_d2=True' in pipe
    m=contract.manifest();assert len(m['blocks'])==8 and m['same_snapshot_for_on_off'] and m['seed_reconstruction_for_training_forbidden']
    return {'status':'PASS','rule_tests':16,'scientific_training_performed':False,'scientific_evaluation_performed':False,'start_request_present':False}


def runtime_tests(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tests=[]
    def ok(name,cond):
        assert cond,name;tests.append(name)
    a=f.qa_training_gate([1,1,1,1,1,1,0,0],True);ok('training_6of8_pass',a['H_TRAIN_D2'] is True)
    a=f.qa_training_gate([1,1,1,1,1,0,0,0],True);ok('training_5of8_fail',a['H_TRAIN_D2'] is False)
    a=f.qa_training_gate([1]*8,False);ok('training_aggregate_required',a['H_TRAIN_D2'] is False)
    a=f.qa_online_gate([1,1,1,1,1,1,0,0],True,True);ok('online_6of8_pass',a['H_ONLINE_D2'] is True)
    a=f.qa_online_gate([1]*8,True,False);ok('online_controls_required',a['H_ONLINE_D2'] is False)
    a=f.qa_online_gate([1,1,1,1,1,0,0,0],True,True);ok('online_5of8_fail',a['H_ONLINE_D2'] is False)
    matrix=f.qa_adjudication_matrix();ok('adjudication_scaffold',matrix[(True,False)]=='TRAINING_SCAFFOLD_SUPPORTED');ok('adjudication_both',matrix[(True,True)]=='TRAINING_AND_ONLINE_DEPENDENCE_SUPPORTED');ok('adjudication_online',matrix[(False,True)]=='ONLINE_DEPENDENCE_ONLY_SUPPORTED');ok('adjudication_none',matrix[(False,False)]=='NO_REPRODUCIBLE_S6_MECHANISM_UNDER_D2')
    for b in range(8):ok(f'snapshot_{b}',f.expected_snapshot(b)['block']==b)
    payload={'status':'PASS','runtime_tests':len(tests),'tests':tests,'synthetic_fixtures_only':True,'scientific_training_performed':False,'scientific_evaluation_performed':False,'scientific_results_generated':False}
    (out/'FULL_PIPELINE_TESTS.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('rules','runtime'));p.add_argument('--out',default='/tmp/d2r1-full-tests');a=p.parse_args()
    print(json.dumps(rule_tests() if a.mode=='rules' else runtime_tests(a.out),sort_keys=True))
if __name__=='__main__':main()
