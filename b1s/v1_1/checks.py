"""QA-only optimizer smoke tests, then the prospective campaign freeze."""
from __future__ import annotations
import argparse
from pathlib import Path
from unittest.mock import patch
from . import design
from .analysis import preflight, read, now
from .train import train_ppo, train_sac, predict


def run(out):
    import numpy as np
    out.mkdir(parents=True,exist_ok=True);fixtures=out/'optimizer-fixtures';fixtures.mkdir(exist_ok=True)
    original_seed=design.seed;ledger=[]
    def qa_seed(split,*key):
        # Test dependency injection maps all learner/episode identities to QA only.
        value=original_seed('qa','optimizer-smoke',split,*key);ledger.append({'requested_split':split,'actual_split':'qa','seed':value});return value
    configs=[dict(id='QA-graph-update',phase='qa',rep=0,kind='graph',mask='100000',steps=512,hp=design.PPO_COMMON.copy()),dict(id='QA-generic-update',phase='qa',rep=1,kind='ppo',mask=None,steps=512,hp=design.PPO_COMMON.copy()),dict(id='QA-sac-update',phase='qa',rep=2,kind='sac',mask=None,steps=512,hp=dict(lr=.0003,gamma=.99,train_freq=4,gradient_steps=1,learning_starts=64,batch_size=16))]
    reports=[]
    for cfg in configs:
        dst=fixtures/cfg['id'];dst.mkdir(exist_ok=True)
        with patch('b1s.v1_1.train.seed',side_effect=qa_seed):
            actor=train_sac(cfg,dst) if cfg['kind']=='sac' else train_ppo(cfg,dst)
            a=predict(actor,np.zeros(93,dtype=np.float32));assert a.shape==(12,) and np.isfinite(a).all() and (np.abs(a)<=1).all()
        resources=read(dst/'TRAINING_RESOURCES.json')
        assert resources['environment_steps']==512 and resources['initial_actor_digest']!=resources['final_actor_digest']
        reports.append(dict(id=cfg['id'],status='PASS',native_steps=512,scope='QA optimizer execution, not discovery/competence data',resources=resources))
    design.write(out/'OPTIMIZER_QA.json',dict(status='PASS',fixtures=reports,seeds=ledger,scientific_data=False))
    result=preflight(out)
    freeze=read(out/'DEVELOPMENT_FREEZE.json')
    freeze['optimizer_QA_sha256']=design.sha(out/'OPTIMIZER_QA.json');freeze['utc_before_training']=now()
    freeze['QA_training_is_not_scientific_campaign']=True
    design.write(out/'DEVELOPMENT_FREEZE.json',freeze)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
