"""Qualification tests for repository-persisted byte-frozen D2-R1 runner."""
from __future__ import annotations
import argparse,ast,dataclasses,json,os
from pathlib import Path
from experiments.b1sc_d2_r1_v1_0 import scientific_runner as r
from experiments.b1sc_d2_r1_v1_0 import initialization as init
ROOT=Path(__file__).resolve().parent
def expect_fail(fn,contains):
    try: fn()
    except Exception as e:
        assert contains in str(e),(contains,str(e)); return
    raise AssertionError('expected failure')
def rule_tests():
    src=(ROOT/'scientific_runner.py').read_text(); tree=ast.parse(src); funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}; adam=[]; steps=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr=='Adam': adam.append(node.lineno)
            if node.func.attr=='step' and isinstance(node.func.value,ast.Name) and node.func.value.id=='opt': steps.append(node.lineno)
    assert len(adam)==1 and len(steps)==1
    train=ast.get_source_segment(src,funcs['_train']); assert train.index('validate_and_load_frozen')<train.index('create_optimizer')<train.index('guarded_optimizer_step')
    ins=(ROOT/'initialization.py').read_text(); assert 'D2R1_INIT_DIR' not in ins and 'PERSISTED_ROOT' in ins and 'repository-versioned' in ins
    pm=init.persisted_manifest(); assert pm['status']=='BYTE_FOR_BYTE_COPY_OF_CANONICAL_ARTIFACT' and pm['regenerated'] is False and len(pm['blocks'])==8
    assert not (ROOT/'START_REQUEST.json').exists()
    return {'status':'PASS','rule_tests':10,'persistent_repository_bytes':True,'start_request_present':False}
def runtime_tests(_ignored,out_root):
    r.runtime(); results=[]
    def ok(name,fn): fn(); results.append(name)
    on=r.validate_and_load_frozen('S6-ON',0); off=r.validate_and_load_frozen('S6-OFF-TRAIN',0)
    ok('same_persisted_snapshot_on_off',lambda: (_ for _ in ()).throw(AssertionError()) if not(on.attestation.snapshot_sha256==off.attestation.snapshot_sha256 and on.attestation.actor_digest==off.attestation.actor_digest and on.attestation.critic_digest==off.attestation.critic_digest) else None)
    def all_hashes():
        for b in range(8):
            e,p=init.entry(b); q=init.artifact_dir()/f'block-{b}.bin'; assert q.is_file() and init.sha256_file(q)==e['sha256']==p['sha256']
    ok('all_8_repository_hashes_exact',all_hashes)
    invalid=r.PreparedModules(on.actor,on.critic,dataclasses.replace(on.attestation,validated=False),list(on.audit)); ok('optimizer_rejects_unvalidated',lambda:expect_fail(lambda:r.create_optimizer(invalid),'validated frozen initialization'))
    root=Path(out_root); root.mkdir(parents=True,exist_ok=True); a=r.qa_microfit('S6-ON',0,'repository-persisted',str(root/'on'),512); b=r.qa_microfit('S6-OFF-TRAIN',0,'repository-persisted',str(root/'off'),512)
    ok('qa_not_scientific',lambda: (_ for _ in ()).throw(AssertionError()) if a['scientific_evidence'] or b['scientific_evidence'] else None)
    ok('microfit_same_init',lambda: (_ for _ in ()).throw(AssertionError()) if a['initial_actor_digest']!=b['initial_actor_digest'] or a['initial_critic_digest']!=b['initial_critic_digest'] else None)
    ok('first_step_after_validation',lambda: [(_ for _ in ()).throw(AssertionError()) for x in (a,b) if [e['event'] for e in x['initialization_audit']][:4]!=['validation_started','snapshot_validated','optimizer_created','optimizer_step']])
    ok('off_message_path_frozen',lambda: (_ for _ in ()).throw(AssertionError()) if b['initial_message_A_digest']!=b['final_message_A_digest'] else None)
    old={k:os.environ.get(k) for k in ('GITHUB_REPOSITORY','GITHUB_REF','GITHUB_RUN_ATTEMPT','GITHUB_RUN_NUMBER')}; os.environ['GITHUB_REPOSITORY']='CRC2520/polar-sim-ml-original'; os.environ['GITHUB_REF']='refs/heads/'+r.RUNNER_BRANCH; os.environ['GITHUB_RUN_ATTEMPT']='1'; os.environ['GITHUB_RUN_NUMBER']='1'; ok('guard_blocks_without_start',lambda:expect_fail(r.scientific_guard,'START_REQUEST.json is absent'))
    for k,v in old.items(): os.environ.pop(k,None) if v is None else os.environ.__setitem__(k,v)
    payload={'status':'PASS','runtime_tests':len(results),'tests':results,'persistent_repository_bytes':True,'qa_microfit_performed':True,'scientific_training_performed':False,'scientific_evaluation_performed':False}; (root/'RUNTIME_TESTS.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'
'); return payload
def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('rules','runtime')); p.add_argument('--init-dir',default='repository-persisted'); p.add_argument('--out',default='/tmp/d2r1-tests'); a=p.parse_args(); print(json.dumps(rule_tests() if a.mode=='rules' else runtime_tests(a.init_dir,a.out),sort_keys=True))
if __name__=='__main__': main()
