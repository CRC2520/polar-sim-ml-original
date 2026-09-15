"""Qualification tests for the byte-frozen D2-R1 scientific runner."""
from __future__ import annotations
import argparse, ast, dataclasses, json, os, shutil, tempfile
from pathlib import Path
from experiments.b1sc_d2_r1_v1_0 import scientific_runner as r
from experiments.b1sc_d2_r1_v1_0 import execution_contract as contract

ROOT=Path(__file__).resolve().parent
# QA trigger marker: workflow existed before this no-op source change.


def expect_fail(fn, contains:str):
    try:fn()
    except Exception as e:
        assert contains in str(e),(contains,str(e));return
    raise AssertionError('expected failure')


def rule_tests()->dict:
    src=(ROOT/'scientific_runner.py').read_text(encoding='utf-8')
    tree=ast.parse(src)
    funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert 'validate_and_load_frozen' in funcs and 'create_optimizer' in funcs and 'guarded_optimizer_step' in funcs and '_train' in funcs
    adam_calls=[];step_calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr=='Adam':adam_calls.append(node.lineno)
            if node.func.attr=='step':step_calls.append(node.lineno)
    assert len(adam_calls)==1 and len(step_calls)==1
    assert funcs['create_optimizer'].lineno < adam_calls[0] <= funcs['create_optimizer'].end_lineno
    assert funcs['guarded_optimizer_step'].lineno < step_calls[0] <= funcs['guarded_optimizer_step'].end_lineno
    train=ast.get_source_segment(src,funcs['_train'])
    assert train.index('validate_and_load_frozen') < train.index('create_optimizer') < train.index('guarded_optimizer_step')
    assert 'torch.manual_seed(int(cfg["initial_seed"]))' not in src
    assert 'START_REQUEST.is_file()' in src
    assert not (ROOT/'START_REQUEST.json').exists()
    reg=json.loads((ROOT/'FIT_REGISTRY.json').read_text())
    assert reg['fits']==16 and reg['same_snapshot_for_on_off'] and reg['seed_reconstruction_for_training_forbidden']
    c=contract.manifest();assert c['same_snapshot_for_on_off'] and c['seed_reconstruction_for_training_forbidden']
    return {'status':'PASS','rule_tests':10,'optimizer_constructor_sites':adam_calls,'optimizer_step_sites':step_calls,'start_request_present':False}


def runtime_tests(init_dir:str,out_root:str)->dict:
    os.environ['D2R1_INIT_DIR']=init_dir
    r.runtime();results=[]
    def ok(name,fn):fn();results.append(name)
    on=r.validate_and_load_frozen('S6-ON',0);off=r.validate_and_load_frozen('S6-OFF-TRAIN',0)
    ok('same_snapshot_on_off',lambda: (_ for _ in ()).throw(AssertionError()) if not (on.attestation.snapshot_sha256==off.attestation.snapshot_sha256 and on.attestation.actor_digest==off.attestation.actor_digest and on.attestation.critic_digest==off.attestation.critic_digest) else None)
    invalid_att=dataclasses.replace(on.attestation,validated=False);invalid=r.PreparedModules(on.actor,on.critic,invalid_att,list(on.audit))
    ok('optimizer_rejects_unvalidated_token',lambda:expect_fail(lambda:r.create_optimizer(invalid),'validated frozen initialization'))
    tmp=Path(tempfile.mkdtemp(prefix='d2r1-corrupt-'))
    try:
        shutil.copytree(init_dir,tmp/'init',dirs_exist_ok=True);p=tmp/'init'/'block-0.bin';data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
        old=os.environ['D2R1_INIT_DIR'];os.environ['D2R1_INIT_DIR']=str(tmp/'init')
        ok('corrupted_snapshot_rejected',lambda:expect_fail(lambda:r.validate_and_load_frozen('S6-ON',0),'byte hash mismatch'))
        os.environ['D2R1_INIT_DIR']=old
    finally:shutil.rmtree(tmp,ignore_errors=True)
    root=Path(out_root);root.mkdir(parents=True,exist_ok=True)
    a=r.qa_microfit('S6-ON',0,init_dir,str(root/'on'),512);b=r.qa_microfit('S6-OFF-TRAIN',0,init_dir,str(root/'off'),512)
    ok('qa_microfits_not_scientific',lambda: (_ for _ in ()).throw(AssertionError()) if a['scientific_evidence'] or b['scientific_evidence'] else None)
    ok('microfit_same_frozen_initialization',lambda: (_ for _ in ()).throw(AssertionError()) if a['initialization']['snapshot_sha256']!=b['initialization']['snapshot_sha256'] or a['initial_actor_digest']!=b['initial_actor_digest'] or a['initial_critic_digest']!=b['initial_critic_digest'] else None)
    def order_check():
        for x in (a,b):
            names=[e['event'] for e in x['initialization_audit']]
            assert names[:4]==['validation_started','snapshot_validated','optimizer_created','optimizer_step']
            assert names.index('snapshot_validated')<names.index('optimizer_created')<names.index('optimizer_step')
    ok('first_optimizer_step_after_validation',order_check)
    ok('off_message_path_frozen',lambda: (_ for _ in ()).throw(AssertionError()) if b['initial_message_A_digest']!=b['final_message_A_digest'] else None)
    old_env={k:os.environ.get(k) for k in ('GITHUB_REPOSITORY','GITHUB_REF','GITHUB_RUN_ATTEMPT','GITHUB_RUN_NUMBER')}
    os.environ['GITHUB_REPOSITORY']='CRC2520/polar-sim-ml-original';os.environ['GITHUB_REF']='refs/heads/'+r.RUNNER_BRANCH;os.environ['GITHUB_RUN_ATTEMPT']='1';os.environ['GITHUB_RUN_NUMBER']='1'
    ok('scientific_guard_blocks_without_start_request',lambda:expect_fail(r.scientific_guard,'START_REQUEST.json is absent'))
    for k,v in old_env.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v
    payload={'status':'PASS','runtime_tests':len(results),'tests':results,'qa_microfit_performed':True,'qa_microfit_scientific_evidence':False,'scientific_training_performed':False,'scientific_evaluation_performed':False,'on':a,'off':b}
    (root/'RUNTIME_TESTS.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('rules','runtime'));p.add_argument('--init-dir');p.add_argument('--out',default='/tmp/d2r1-runner-tests');a=p.parse_args()
    if a.mode=='rules':print(json.dumps(rule_tests(),sort_keys=True))
    else:
        if not a.init_dir:raise SystemExit('--init-dir required')
        print(json.dumps(runtime_tests(a.init_dir,a.out),sort_keys=True))
if __name__=='__main__':main()
