"""CP-NB rules and isolated QA fixtures; never execute the empirical registry here.
The runtime fixtures use temporary directories and the QA namespace only.
Passing tests do not establish competence, structural stability or B1-E readiness.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import numpy as np
from . import trial as d
from . import runner as r


def normalizer():
    return dict(mean=[2.0]*93, scale=[0.5]*93)


class RuleTests(unittest.TestCase):
    def test_01_registry_and_budget(self):
        reg=d.registry();p=d.plan()
        self.assertEqual(len(reg),24)
        self.assertEqual(len({c['id'] for c in reg}),24)
        self.assertEqual(p['train_native_steps'],25165824)
        self.assertEqual(p['normalization_native_steps'],98304)
        self.assertEqual(p['evaluation_native_step_cap'],816000)
        self.assertEqual(d.BUDGETS,[524288,1048576])
        for c in reg:
            self.assertEqual(c['budgets'],d.BUDGETS)
            if c['role']!='SAC':
                self.assertTrue(all(b%c['hp']['rollout_steps']==0 for b in d.BUDGETS))
    def test_02_pairing(self):
        reg={c['id']:c for c in d.registry()}
        for role in d.ROLES:
            for rep in range(3):
                self.assertEqual(reg[f'{role}-raw-r{rep}']['initial_seed'],reg[f'{role}-standardized-r{rep}']['initial_seed'])
        self.assertEqual(reg['G0-raw-r0']['initial_seed'],reg['GD-raw-r0']['initial_seed'])
        self.assertEqual(len({reg[f'PPO-raw-r{x}']['initial_seed'] for x in range(3)}),3)
    def test_03_namespaces_separate(self):
        values=[d.seed(s,'test',i) for s in ('qa','normalization','training','evaluation') for i in range(40)]
        self.assertEqual(len(values),len(set(values)))
        self.assertEqual(d.seed('qa','stable'),d.seed('qa','stable'))
    def test_04_final_forbidden(self):
        for name in ('final','b1e','confirmation',''):
            with self.assertRaises(RuntimeError): d.seed(name,0)
        with self.assertRaises(RuntimeError): d.seed('qa',0,bits=64)
        for c in d.registry(): self.assertLess(c['initial_seed'],2**32)
    def test_05_plan_and_boundary(self):
        self.assertEqual(d.read(d.ROOT/'PLAN.json'),d.plan())
        for name in ('B1E_executed','final_seeds_generated','ready_for_b1e_confirmatory_run','ready_for_b1e_freeze','ready_for_b1e_protocol_design'):
            self.assertIs(d.plan()[name],False)
        self.assertFalse(d.plan()['structural_search'])
    def test_06_raw_identity(self):
        z=np.arange(93,dtype=np.float32)
        x=r.FixedInput(normalizer(),'raw')(z)
        np.testing.assert_array_equal(x,z)
        self.assertFalse(np.shares_memory(x,z))
    def test_07_standardization_fixed_no_clip(self):
        t=r.FixedInput(normalizer(),'standardized');z=np.full((2,93),100.,dtype=np.float32)
        before=(t.mean.tobytes(),t.scale.tobytes())
        np.testing.assert_array_equal(t(z),np.full_like(z,196.))
        self.assertEqual(before,(t.mean.tobytes(),t.scale.tobytes()))
        self.assertFalse(t.mean.flags.writeable);self.assertFalse(t.scale.flags.writeable)
    def test_08_invalid_normalizer(self):
        for data in (dict(mean=[0.]*92,scale=[1.]*93),dict(mean=[math.nan]*93,scale=[1.]*93),dict(mean=[0.]*93,scale=[0.]*93)):
            with self.assertRaises(RuntimeError): r.FixedInput(data,'standardized')
        with self.assertRaises(RuntimeError): r.FixedInput(normalizer(),'unknown')
    def test_09_invalid_observation(self):
        t=r.FixedInput(normalizer(),'raw')
        for z in (np.zeros(92),np.full(93,np.inf),np.array(2.)):
            with self.assertRaises(RuntimeError): t(z)
    def test_10_gae_streamwise(self):
        rewards=np.array([[1.,2.],[3.,4.],[5.,6.]])
        values=np.array([[.1,.2],[.3,.4],[.5,.6]])
        dones=np.array([[False,False],[True,False],[False,True]])
        last=np.array([.7,.8]);actual,targets=r.gae_returns(rewards,values,dones,last,.99,.95)
        expected=np.zeros_like(rewards)
        for j in range(2):
            a=0.
            for t in (2,1,0):
                nt=1-float(dones[t,j]);v=last[j] if t==2 else values[t+1,j]
                a=rewards[t,j]+.99*v*nt-values[t,j]+.99*.95*nt*a;expected[t,j]=a
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(targets,actual+values)
    def test_11_gae_terminal(self):
        a,t=r.gae_returns(np.array([[3.]]),np.array([[1.]]),np.array([[True]]),np.array([999.]),.99,.95)
        self.assertEqual(float(a[0,0]),2.);self.assertEqual(float(t[0,0]),3.)
    def test_12_competence_boundaries(self):
        w=dict(loss=0.,displacement=2.)
        self.assertFalse(d.competence(dict(loss=-d.DELTA,displacement=3.),w)['descriptive_competence_pass'])
        self.assertTrue(d.competence(dict(loss=-d.DELTA-1e-8,displacement=3.),w)['descriptive_competence_pass'])
        self.assertFalse(d.competence(dict(loss=-100.,displacement=3.-1e-8),w)['descriptive_competence_pass'])
    def test_13_summary_rejects_absent_nonfinite(self):
        with self.assertRaises(RuntimeError): d.summarize([])
        with self.assertRaises(RuntimeError): d.summarize([dict(loss=math.nan,displacement=0.,cycles=1)])
        row=dict(loss=2.,displacement=3.,cycles=4,fallen_count=0,package_dropped=False)
        self.assertEqual(d.summarize([row,row])['episodes'],2)
    def test_14_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json';d.write_new(p,{'value':1});before=p.read_bytes()
            d.write_new(p,{'value':1})
            with self.assertRaises(RuntimeError): d.write_new(p,{'value':2})
            self.assertEqual(p.read_bytes(),before)
    def test_15_factorial_arithmetic(self):
        cells={}
        for role in d.ROLES:
            for mode in d.MODES:
                for b in d.BUDGETS:
                    n=int(mode=='standardized');h=int(b==d.BUDGETS[1])
                    cells[f'{role}-{mode}-{b}']=dict(by_rep=[dict(rep=i,summary={k:10+i+2*n+3*h+4*n*h for k in ('loss','displacement','fall_fraction','cycles')}) for i in range(3)])
        out=d.factorial_contrasts(cells)
        for role in d.ROLES:
            self.assertEqual(out[role]['loss']['interaction']['by_rep'],[4,4,4])
            self.assertEqual(out[role]['loss']['normalization_at_low']['mean'],2)
    def guard_context(self,changes=None,attempt='1',repo=None,plan_hash='plan'):
        request=dict(namespace=d.NAMESPACE,authorize_competence_pilot=True,B1E_executed=False,final_seeds_generated=False,code_commit='code',plan_sha256='plan',protocol_sha256='protocol')
        env=dict(GITHUB_REPOSITORY=repo or d.REPO,GITHUB_REF='refs/heads/'+d.BRANCH,GITHUB_RUN_ATTEMPT=attempt,GITHUB_SHA='head')
        def fake_git(*a):
            return 'head' if a[0]=='rev-parse' else ('experiments/competence_pilot/START_REQUEST.json' if changes is None else changes)
        def fake_sha(p): return plan_hash if Path(p).name=='PLAN.json' else 'protocol'
        return env,request,fake_git,fake_sha
    def check_guard(self,**kwargs):
        env,request,git,sha=self.guard_context(**kwargs)
        with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(d,'read',return_value=request),mock.patch.object(d,'git',side_effect=git),mock.patch.object(d,'sha',side_effect=sha):
            return d.execution_guard()
    def test_16_exact_guard(self):
        self.assertTrue(self.check_guard()['authorize_competence_pilot'])
    def test_17_reruns_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'retries'): self.check_guard(attempt='2')
    def test_18_unapproved_changes_rejected(self):
        for args in (dict(changes='experiments/competence_pilot/runner.py'),dict(repo='another/repository'),dict(plan_hash='changed')):
            with self.subTest(args=args),self.assertRaises(RuntimeError): self.check_guard(**args)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        sd=d.seed('qa','runtime-fixture',self._testMethodName,bits=32)
        random.seed(sd);np.random.seed(sd);r.torch.manual_seed(sd)
    def test_01_rng_restored(self):
        before=(random.getstate(),np.random.get_state(),r.torch.get_rng_state().clone())
        with self.assertRaises(ValueError):
            with r.isolated_evaluation_rng():
                random.random();np.random.rand(5);r.torch.rand(5);raise ValueError('fixture')
        self.assertEqual(random.getstate(),before[0]);after=np.random.get_state()
        self.assertEqual(after[0],before[1][0]);np.testing.assert_array_equal(after[1],before[1][1])
        self.assertEqual(after[2:],before[1][2:]);self.assertTrue(r.torch.equal(r.torch.get_rng_state(),before[2]))
    def test_02_native_fixed_input_parity(self):
        envs=[r.NativeEnv(),r.NativeEnv()];trans=r.FixedInput(normalizer(),'standardized')
        try:
            sd=d.seed('qa','native-parity');z1,_=envs[0].reset(seed=sd);z2,_=envs[1].reset(seed=sd)
            np.testing.assert_array_equal(z1,z2)
            for _ in range(8):
                np.testing.assert_array_equal(trans(z1),(z1.astype(np.float64)-2.).astype(np.float32)*2.)
                a=np.full(12,.125,dtype=np.float32)
                x=envs[0].step(a);y=envs[1].step(a)
                np.testing.assert_array_equal(x[0],y[0]);self.assertEqual(x[1:],y[1:])
                z1,z2=x[0],y[0]
                if x[2] or x[3]: break
        finally:
            for e in envs:e.close()
    def test_03_actor_message_contract(self):
        z=r.torch.zeros(2,93)
        for mask in ('000000','111111'):
            actor=r.RoutingActor(mask);before=r.model_digest(actor)
            with r.torch.no_grad():
                mu,_,snap=actor(z,trace=True);ref=actor.reference_forward(z)
                _,_,lesion=actor(z,(0,),True)
            self.assertEqual(tuple(mu.shape),(2,12));self.assertEqual(tuple(snap['m'].shape),(2,6,8))
            r.torch.testing.assert_close(mu,ref,rtol=1e-5,atol=1e-7)
            self.assertTrue(r.torch.equal(snap['h'],lesion['h']));self.assertTrue(r.torch.equal(snap['m'],lesion['m']))
            self.assertEqual(before,r.model_digest(actor))
    def synthetic_env(self):
        import gymnasium as gym
        class Fixture(gym.Env):
            metadata={}
            def __init__(self):
                self.observation_space=gym.spaces.Box(-np.inf,np.inf,(93,),dtype=np.float32)
                self.action_space=gym.spaces.Box(-1.,1.,(12,),dtype=np.float32)
                self.t=0;self.episode_seed=0;self.state=np.zeros(93,dtype=np.float32)
            def reset(self,*,seed=None,options=None):
                super().reset(seed=seed);self.episode_seed=int(seed);self.t=0
                self.state=np.cos(np.arange(93,dtype=np.float32)/10)*.1
                return self.state.copy(),{'episode_seed':self.episode_seed}
            def step(self,action):
                a=np.asarray(action,dtype=np.float32);self.t+=1
                self.state=(.9*self.state+.05*np.resize(a,93)).astype(np.float32)
                reward=-float(np.square(a-.2).mean())
                info=dict(episode_seed=self.episode_seed,package_displacement=float(self.state[0]),fallen_count=0,package_dropped=False)
                return self.state.copy(),reward,False,self.t>=16,info
            def close(self): pass
        return Fixture
    def test_04_ppo_synthetic_post_update_checkpoints(self):
        for role,mask in (('G0','000000'),('GD','111111'),('PPO',None)):
            with tempfile.TemporaryDirectory() as tmp,mock.patch.object(r,'NativeEnv',self.synthetic_env()),mock.patch.object(r,'BUDGETS',[2048,4096]):
                out=Path(tmp);observed=[]
                cfg=dict(id='QA-'+role,role=role,mask=mask,rep=0,environment_split='qa',hp=(d.GENERIC_HP if role=='PPO' else d.GRAPH_HP).copy())
                def checkpoint(actor,steps,updates):
                    saved=r.torch.load(out/f'model-{steps}.pt',map_location='cpu',weights_only=False)
                    for name,tensor in actor.state_dict().items():self.assertTrue(r.torch.equal(tensor,saved['actor_state'][name]))
                    expected=steps//cfg['hp']['rollout_steps']*cfg['hp']['epochs']*(cfg['hp']['rollout_steps']//cfg['hp']['minibatch'])
                    self.assertEqual(updates,expected);observed.append(steps)
                resources,ledger,_=r.ppo_fit(cfg,r.FixedInput(normalizer(),'standardized'),out,checkpoint)
                self.assertEqual(observed,[2048,4096]);self.assertEqual(resources['environment_steps'],4096)
                self.assertNotEqual(resources['initial_actor_digest'],resources['final_actor_digest']);self.assertTrue(ledger)
    def test_05_sac_synthetic_post_update_checkpoints(self):
        from stable_baselines3 import SAC
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(r,'NativeEnv',self.synthetic_env()),mock.patch.object(r,'BUDGETS',[128,256]):
            out=Path(tmp);observed=[];hp=dict(d.SAC_HP,buffer_size=512,learning_starts=64,batch_size=64)
            cfg=dict(id='QA-SAC',role='SAC',mask=None,rep=0,environment_split='qa',hp=hp,initial_seed=d.seed('qa','SAC-fixture',bits=32))
            def checkpoint(model,steps,updates):
                self.assertEqual(updates,steps-hp['learning_starts'])
                with r.isolated_evaluation_rng():
                    saved=SAC.load(out/f'model-{steps}.zip',device='cpu')
                    self.assertEqual(r.model_digest(saved.actor),r.model_digest(model.actor))
                    self.assertEqual(saved._n_updates,updates)
                observed.append(steps)
            resources,ledger,_=r.sac_fit(cfg,r.FixedInput(normalizer(),'standardized'),out,checkpoint)
            self.assertEqual(observed,[128,256]);self.assertEqual(resources['environment_steps'],256)
            self.assertNotEqual(resources['initial_actor_digest'],resources['final_actor_digest']);self.assertTrue(ledger)


def execute_suite(case):
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),log=stream.getvalue(),**d.BOUNDARY)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    d.require(result.wasSuccessful(),case.__name__+' failed: '+stream.getvalue())
    return report


def unit_tests(): return execute_suite(RuleTests)


def runtime_tests():
    r.runtime()
    with r.isolated_evaluation_rng(): return execute_suite(RuntimeTests)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',action='store_true');a=p.parse_args()
    unit_tests()
    if a.runtime: runtime_tests()
