"""Unmodified native Multiwalker, centralized raw information and causal QA."""
from __future__ import annotations
import gzip
import hashlib
import importlib.metadata as im
import json
import platform
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from pettingzoo.sisl import multiwalker_v9
from .core import ROOT, RoutingActor, seed_for, model_digest, write_json, core_tests

KWARGS=dict(n_walkers=3,position_noise=.001,angle_noise=.001,forward_reward=1.0,
            terminate_reward=-100.0,fall_reward=-10.0,shared_reward=True,
            terminate_on_fall=True,remove_on_fall=True,terrain_length=200,max_cycles=500,render_mode=None)
NODES=('walker_0','walker_1','walker_2')


def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class NativeEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self):
        super().__init__()
        self.native=multiwalker_v9.parallel_env(**KWARGS)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(93,),dtype=np.float32)
        self.action_space=gym.spaces.Box(-1.0,1.0,shape=(12,),dtype=np.float32)
        self.t=0;self.episode_seed=None;self.initial_x=0.;self.finished=True

    @property
    def physics(self):return self.native.aec_env.unwrapped.env

    @staticmethod
    def flatten(obs):
        if set(obs)!=set(NODES):raise RuntimeError('Native observation keys differ from locked three-node contract')
        z=np.concatenate([obs[n] for n in NODES]).astype(np.float32)
        if z.shape!=(93,) or not np.isfinite(z).all():raise RuntimeError('Nonfinite or misshaped native observation')
        return z

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed=int(self.np_random.integers(0,2**32,dtype=np.uint32))
        self.episode_seed=int(seed)
        obs,info=self.native.reset(seed=self.episode_seed)
        self.t=0;self.finished=False;self.initial_x=float(self.physics.package.position.x)
        return self.flatten(obs),{'episode_seed':self.episode_seed}

    def step(self, action):
        if self.finished:raise RuntimeError('step after terminal; reset required')
        a=np.asarray(action,dtype=np.float32)
        if a.shape!=(12,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):
            raise ValueError('Invalid requested action: retain incident, never replace a seed')
        requests={n:a[4*i:4*i+4].copy() for i,n in enumerate(NODES)}
        obs,r,term,trunc,info=self.native.step(requests)
        self.t+=1
        if set(r)!=set(NODES):raise RuntimeError('Native reward keys differ')
        rewards=np.array([r[n] for n in NODES],dtype=np.float64)
        if not np.isfinite(rewards).all() or not np.all(rewards==rewards[0]):
            raise RuntimeError('Native common shared reward failed')
        terminated=bool(all(term.values()));truncated=bool(all(trunc.values()))
        self.finished=terminated or truncated
        z=self.flatten(obs)
        audit={'episode_seed':self.episode_seed,'native_cycle':self.t,'native_terminations':term,
               'native_truncations':trunc,'horizon_reached':self.t>=500,
               'package_displacement':float(self.physics.package.position.x)-self.initial_x,
               'fallen_count':int(np.count_nonzero(self.physics.fallen_walkers)),
               'package_dropped':bool(self.physics.game_over),
               'requested':a.tolist(),'admitted':a.tolist(),'motor_speed_setpoint':(np.tile([4.,6.,4.,6.],3)*np.sign(a)).tolist(),
               'motor_torque_limit_setpoint':(80*np.abs(a)).tolist(),
               'realized_energy_measured':False,'reward_counted_once':float(rewards.mean())}
        return z,float(rewards.mean()),terminated,truncated,audit

    def close(self):self.native.close()


def runtime_lock():
    import pettingzoo
    parent=Path(pettingzoo.__file__).parent
    expected={'sisl/multiwalker_v9.py':'555eee6a7f37d5bb908152f6680a6d7ffe882f91',
              'sisl/multiwalker/multiwalker.py':'8edf250d1dec03d4ae2ee56c9e2dff5fe50e875c',
              'sisl/multiwalker/multiwalker_base.py':'476b3950cc1146e53338ec4bf48b9291b6143a84'}
    observed={}
    for rel,sha in expected.items():
        raw=(parent/rel).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if sha!=actual:raise RuntimeError(f'Upstream source mismatch: {rel}: {actual}')
        observed[rel]={'blob':actual,'sha256':hashlib.sha256(raw).hexdigest()}
    return {'upstream_commit':'e62d6bcdc6e0c28fa19c08545a63dc8370393495','sources':observed,
            'versions':{x:im.version(x) for x in ['torch','numpy','gymnasium','pettingzoo','box2d-py','pygame','stable-baselines3']},
            'python':platform.python_version(),'platform':platform.platform(),'torch_threads':torch.get_num_threads()}


def qa(out: Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed_for('qa','instrument'))
    report={'core':core_tests(),'runtime':runtime_lock(),'scope':'QA only; no training or final evidence',
            'four_regimes':[],'replay':[],'failures':[]}
    for k,(aa,bb) in enumerate(((0.,0.),(.5,0.),(0.,.5),(.5,.5))):
        env=NativeEnv();z,_=env.reset(seed=seed_for('qa','regime',k))
        a=np.zeros(12,dtype=np.float32);a[:4]=aa;a[4:8]=bb
        nxt,r,t,tr,info=env.step(a)
        assert len(nxt)==93 and info['requested']==info['admitted']
        for i,w in enumerate(env.physics.walkers):
            if w.hull is not None:
                actual=[float(j.motorSpeed) for j in w.joints]
                assert np.array_equal(actual,info['motor_speed_setpoint'][4*i:4*i+4])
        report['four_regimes'].append({'regime':[aa,bb],'seed':env.episode_seed,'reward':r,'terminal':t,'observation_sha':array_hash(nxt)})
        env.close()
    actor=RoutingActor('101010');before=model_digest(actor)
    e1=NativeEnv();e2=NativeEnv()
    z1,_=e1.reset(seed=seed_for('qa','replay'));z2,_=e2.reset(seed=seed_for('qa','replay'))
    rng=np.random.default_rng(seed_for('qa','actions'))
    for t in range(24):
        assert np.array_equal(z1,z2)
        tensor=torch.from_numpy(z1[None])
        with torch.no_grad():
            mu,_,intact=actor(tensor,trace=True)
            smu,_,sham=actor(tensor,trace=True)
            lmu,_,lesion=actor(tensor,(0,),True)
            off,_,_=actor(tensor,(1,),True)
            assert torch.equal(mu,smu) and torch.equal(mu,off)
            assert torch.equal(intact['m'],lesion['m'])
            assert torch.equal(intact['h'],lesion['h'])
        report['replay'].append({'t':t,'state_sha':array_hash(z1),'action_change':float((mu.tanh()-lmu.tanh()).abs().max())})
        a=rng.uniform(-.2,.2,12).astype(np.float32)
        z1,r1,d1,tr1,_=e1.step(a);z2,r2,d2,tr2,_=e2.step(a)
        assert r1==r2 and d1==d2 and tr1==tr2 and np.array_equal(z1,z2)
        if d1 or tr1:break
    assert any(x['action_change']>1e-9 for x in report['replay'])
    assert model_digest(actor)==before
    e1.close();e2.close()
    env=NativeEnv();env.reset(seed=seed_for('qa','bad_input'))
    for invalid in (np.zeros(11),np.full(12,np.nan),np.full(12,1.01)):
        try:env.step(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid API input was silently admitted')
    env.close()
    env=NativeEnv();z,_=env.reset(seed=seed_for('qa','timing'));start=time.perf_counter()
    for k in range(256):
        with torch.no_grad():a=actor(torch.from_numpy(z[None]))[0].tanh()[0].numpy()
        z,r,t,tr,_=env.step(a)
        if t or tr:z,_=env.reset(seed=seed_for('qa','timing_reset',k))
    report['seconds_per_actor_plus_native_step']=(time.perf_counter()-start)/256
    env.close()
    report.update(status='PASS',instrument_implemented=True,training_run=False,B1E_executed=False,
                  final_seeds_generated=False,reachable_snapshot_sensitivity='PASS',prefix_replay='EXACT on retained QA prefixes',
                  C6='PASS algebraic routing; native 16-channel catalogue mapping NOT_APPLICABLE')
    write_json(out/'B1S_QA_REPORT.json',report)
    print(json.dumps(report,indent=2))
    return report
