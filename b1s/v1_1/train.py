"""New development runs; frozen v1 actors and native dynamics are imported unchanged."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import random
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from b1s.execution.core import RoutingActor, GenericActor, Value, log_squashed_gaussian, model_digest
from b1s.execution.instrument import NativeEnv, runtime_lock
from .design import ROOT, plan, seed, sha, write, commit, screening_registry, competition_registry


def initialize(cfg):
    s=seed(cfg['phase']+'-train','weights',cfg['rep'],cfg['kind'] if cfg['kind']=='sac' else 'ppo')
    random.seed(s);np.random.seed(s%(2**32));torch.manual_seed(s)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return s


class SequenceEnv(NativeEnv):
    """Episode seeds are explicit even when an RL library calls reset(seed=None)."""
    def __init__(self,phase,rep,slot=0):
        super().__init__();self.phase=phase;self.rep=rep;self.slot=slot;self.counter=0
    def reset(self,*,seed=None,options=None):
        # The RL library's seed initializes its learner, not this declared episode ledger.
        actual=globals()['seed'](self.phase+'-train','environment',self.rep,self.slot,self.counter)
        self.counter+=1
        obs,info=super().reset(seed=actual,options=options)
        info['library_reset_seed']=seed;return obs,info


@torch.no_grad()
def predict(actor,z,lesion=()):
    if isinstance(actor,(RoutingActor,GenericActor)):
        return actor(torch.from_numpy(np.asarray(z,dtype=np.float32)[None]),lesion)[0].tanh()[0].numpy()
    if lesion:raise ValueError('A non-graph model has no route intervention')
    return np.asarray(actor.predict(z,deterministic=True)[0],dtype=np.float32)


def evaluate(actor,rep,split,count,trace=None,case=''):
    env=NativeEnv();rows=[];handle=gzip.open(trace,'wt',encoding='utf-8') if trace else None
    try:
        for ep in range(count):
            sd=seed(split,'environment',rep,ep);z,_=env.reset(seed=sd);ret=0.;request_cost=0.;h=hashlib.sha256()
            while True:
                a=np.zeros(12,dtype=np.float32) if actor is None else predict(actor,z)
                nxt,r,term,trunc,info=env.step(a)
                event={'case':case,'replicate':rep,'episode':ep,'split':split,'seed':sd,'cycle':env.t,'z':z.tolist(),'action':a.tolist(),'reward':r,'next_z':nxt.tolist(),'audit':info}
                line=json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n';h.update(line.encode())
                if handle:handle.write(line)
                ret+=r;request_cost+=float(np.abs(a).sum());z=nxt
                if term or trunc:break
            rows.append(dict(rep=rep,episode=ep,seed=sd,split=split,loss=-ret,return_value=ret,cycles=env.t,displacement=info['package_displacement'],fallen_count=info['fallen_count'],package_dropped=info['package_dropped'],motor_request_abs_sum=request_cost,trace_sha256=h.hexdigest(),development_only=True))
    finally:
        env.close()
        if handle:handle.close()
    return rows


def save_actor(actor,critic,opt,cfg,steps,path):
    torch.save({'configuration':cfg,'actor_state':actor.state_dict(),'critic_state':critic.state_dict(),'optimizer':opt.state_dict(),'steps':steps},path)


def train_ppo(cfg,out):
    initialize(cfg);hp=cfg['hp'];steps=cfg['steps'];n_env=8
    batch=hp['rollout_steps'];horizon=batch//n_env
    assert batch%n_env==0 and steps%batch==0
    actor=RoutingActor(cfg['mask']) if cfg['kind']=='graph' else GenericActor();critic=Value()
    params=list(actor.parameters())+list(critic.parameters());opt=torch.optim.Adam(params,lr=hp['lr'],eps=hp['adam_eps'])
    envs=[SequenceEnv(cfg['phase'],cfg['rep'],i) for i in range(n_env)]
    z=np.stack([e.reset()[0] for e in envs]);returns=np.zeros(n_env);lengths=np.zeros(n_env,dtype=int)
    episodes=[];curves=[];checks=[];updates=0;initial=model_digest(actor);t0=time.perf_counter()
    milestones=plan()['calibration_checkpoints'] if cfg['phase']=='calibration' else []
    try:
        for offset in range(0,steps,batch):
            obs=[];pre=[];logps=[];values=[];rewards=[];dones=[]
            for t in range(horizon):
                zt=torch.from_numpy(z)
                with torch.no_grad():
                    mu,ls=actor(zt);p=mu+ls.exp()*torch.randn_like(mu);a=p.tanh();lp=log_squashed_gaussian(p,mu,ls);v=critic(zt)
                obs.append(z.copy());pre.append(p.numpy());logps.append(lp.numpy());values.append(v.numpy())
                rs=[];ds=[];nz=[]
                for i,e in enumerate(envs):
                    nxt,r,term,trunc,info=e.step(a[i].numpy());done=term or trunc
                    returns[i]+=r;lengths[i]+=1;rs.append(r);ds.append(done)
                    if done:
                        episodes.append(dict(seed=e.episode_seed,slot=i,episode=e.counter-1,return_value=float(returns[i]),cycles=int(lengths[i]),native_steps=offset+t*n_env+i+1,displacement=info['package_displacement'],fallen_count=info['fallen_count']))
                        nxt=e.reset()[0];returns[i]=0.;lengths[i]=0
                    nz.append(nxt)
                rewards.append(rs);dones.append(ds);z=np.stack(nz)
            with torch.no_grad():last=critic(torch.from_numpy(z)).numpy()
            vv=np.asarray(values);rr=np.asarray(rewards);dd=np.asarray(dones)
            advantage=np.zeros((horizon,n_env),dtype=np.float32);gae=np.zeros(n_env)
            for t in range(horizon-1,-1,-1):
                nt=1-dd[t].astype(float);nextv=last if t==horizon-1 else vv[t+1]
                gae=rr[t]+hp['gamma']*nextv*nt-vv[t]+hp['gamma']*hp['gae_lambda']*nt*gae;advantage[t]=gae
            targets=(advantage+vv).reshape(-1);advantage=advantage.reshape(-1);advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
            ot=torch.from_numpy(np.asarray(obs).reshape(batch,93));pt=torch.from_numpy(np.asarray(pre).reshape(batch,12));old=torch.from_numpy(np.asarray(logps).reshape(-1))
            at=torch.from_numpy(advantage);rt=torch.from_numpy(targets);al=[];vl=[];kl=[]
            for epoch in range(hp['epochs']):
                order=np.random.permutation(batch)
                for start in range(0,batch,hp['minibatch']):
                    ix=order[start:start+hp['minibatch']];mu,ls=actor(ot[ix]);lp=log_squashed_gaussian(pt[ix],mu,ls)
                    logr=lp-old[ix];ratio=logr.exp();a_loss=-torch.min(ratio*at[ix],ratio.clamp(1-hp['clip'],1+hp['clip'])*at[ix]).mean()
                    v_loss=(critic(ot[ix])-rt[ix]).square().mean();loss=a_loss+hp['value_coef']*v_loss
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO objective')
                    opt.zero_grad();loss.backward();g=nn.utils.clip_grad_norm_(params,hp['max_grad_norm'])
                    if not torch.isfinite(g):raise RuntimeError('Nonfinite gradient')
                    opt.step();updates+=1;al.append(a_loss.item());vl.append(v_loss.item());kl.append(float(((ratio-1)-logr).mean().detach()))
            done_steps=offset+batch
            if done_steps%32768==0 or done_steps==steps:
                entry=dict(native_steps=done_steps,actor_loss=float(np.mean(al)),value_loss=float(np.mean(vl)),approx_kl=float(np.mean(kl)),completed_episodes=len(episodes),elapsed_seconds=time.perf_counter()-t0,optimizer_updates=updates)
                curves.append(entry);write(out/'LEARNING_CURVE.json',curves);write(out/'TRAINING_EPISODES.json',episodes);print(cfg['id'],entry,flush=True)
            if done_steps in milestones:
                save_actor(actor,critic,opt,cfg,done_steps,out/f'checkpoint-{done_steps}.pt')
                ev=evaluate(actor,cfg['rep'],'calibration-select',plan()['calibration_selection_episodes'],case=cfg['id'])
                checks.append(dict(steps=done_steps,episodes=ev,mean_loss=float(np.mean([x['loss'] for x in ev]))));write(out/'CHECKPOINT_EVALUATIONS.json',checks)
        save_actor(actor,critic,opt,cfg,steps,out/'model.pt')
        write(out/'TRAINING_RESOURCES.json',dict(environment_steps=steps,training_seconds=time.perf_counter()-t0,optimizer_updates=updates,parameters_actor=sum(x.numel() for x in actor.parameters()),parameters_critic=sum(x.numel() for x in critic.parameters()),sampling_parallelism=n_env,rollout_is_total_transitions=batch,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,initial_actor_digest=initial,final_actor_digest=model_digest(actor),raw_observation_normalization=False,trainable_temporal_actor_memory=False))
        if initial==model_digest(actor):raise RuntimeError('No actor update')
        return actor
    finally:
        for env in envs:env.close()


def train_sac(cfg,out):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    sd=initialize(cfg);hp=cfg['hp'];steps=cfg['steps'];env=SequenceEnv(cfg['phase'],cfg['rep']);episodes=[];curves=[];checks=[]
    milestones=plan()['calibration_checkpoints'] if cfg['phase']=='calibration' else []
    class Logger(BaseCallback):
        def __init__(self):super().__init__();self.ret=0.;self.length=0
        def _on_step(self):
            self.ret+=float(self.locals['rewards'][0]);self.length+=1
            if self.locals['dones'][0]:
                inf=self.locals['infos'][0];episodes.append(dict(seed=inf['episode_seed'],slot=0,episode=len(episodes),return_value=self.ret,cycles=self.length,native_steps=self.num_timesteps,displacement=inf['package_displacement'],fallen_count=inf['fallen_count']));self.ret=0.;self.length=0
            if self.num_timesteps%32768==0:
                c=dict(native_steps=self.num_timesteps,completed_episodes=len(episodes),elapsed_seconds=time.perf_counter()-t0,optimizer_updates=self.model._n_updates)
                curves.append(c);write(out/'LEARNING_CURVE.json',curves);write(out/'TRAINING_EPISODES.json',episodes);print(cfg['id'],c,flush=True)
            if self.num_timesteps in milestones:
                self.model.save(out/f'checkpoint-{self.num_timesteps}')
                ev=evaluate(self.model,cfg['rep'],'calibration-select',plan()['calibration_selection_episodes'],case=cfg['id'])
                checks.append(dict(steps=self.num_timesteps,episodes=ev,mean_loss=float(np.mean([x['loss'] for x in ev])),callback_timing='after environment transition, before this collection block gradient updates'))
                write(out/'CHECKPOINT_EVALUATIONS.json',checks)
            return True
    model=SAC('MlpPolicy',env,learning_rate=hp['lr'],gamma=hp['gamma'],buffer_size=steps,learning_starts=hp['learning_starts'],batch_size=hp['batch_size'],tau=.005,train_freq=hp['train_freq'],gradient_steps=hp['gradient_steps'],ent_coef='auto',policy_kwargs=dict(net_arch=[64,64],activation_fn=nn.ReLU),seed=sd%(2**32),device='cpu',verbose=0)
    initial=model_digest(model.actor);t0=time.perf_counter()
    try:
        model.learn(total_timesteps=steps,callback=Logger(),progress_bar=False);model.save(out/'model')
        final=model_digest(model.actor)
        write(out/'TRAINING_RESOURCES.json',dict(environment_steps=model.num_timesteps,training_seconds=time.perf_counter()-t0,optimizer_updates=model._n_updates,parameters_actor=sum(x.numel() for x in model.actor.parameters()),parameters_critic=sum(x.numel() for x in model.critic.parameters()),target_critic_parameters=sum(x.numel() for x in model.critic_target.parameters()),sampling_parallelism=1,replay_capacity=steps,replay_entries=model.replay_buffer.size(),replay_archived=False,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,initial_actor_digest=initial,final_actor_digest=final,raw_observation_normalization=False))
        if initial==final:raise RuntimeError('No SAC actor update')
        return model
    finally:env.close()


def load(folder):
    cfg=json.loads((folder/'STARTED.json').read_text())['configuration']
    if cfg['kind']=='sac':
        from stable_baselines3 import SAC
        return SAC.load(folder/'model.zip',device='cpu')
    actor=RoutingActor(cfg['mask']) if cfg['kind']=='graph' else GenericActor();state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);actor.load_state_dict(state['actor_state']);actor.eval();return actor


def run_fit(cfg,out):
    out.mkdir(parents=True,exist_ok=False)
    write(out/'STARTED.json',dict(configuration=cfg,source_commit=commit(),plan_sha256=sha(ROOT/'PLAN.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),runtime=runtime_lock(),development_only=True,confirmatory=False))
    try:
        actor=train_sac(cfg,out) if cfg['kind']=='sac' else train_ppo(cfg,out)
        if cfg['phase']!='competitive':
            split=cfg['phase']+'-select';n=plan()['discovery_selection_episodes'] if cfg['phase']=='discovery' else plan()['calibration_selection_episodes']
            ev=evaluate(actor,cfg['rep'],split,n,trace=out/'SELECTION_TRACE.jsonl.gz',case=cfg['id']);write(out/'SELECTION.json',dict(rows=ev,mean_loss=float(np.mean([r['loss'] for r in ev])),steps=cfg['steps'],selection_uses_terminal_checkpoint=True))
        write(out/'COMPLETE.json',dict(status='COMPLETE',source_commit=commit(),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},development_only=True,confirmatory=False))
        return True
    except Exception:
        write(out/'FAILED.json',dict(status='FAILED_RETAINED',error=traceback.format_exc(),configuration=cfg,replacement_seed=False));print(traceback.format_exc(),flush=True);return False


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['screen','competitive']);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path);a=p.parse_args()
    if not 0<=a.shard<a.shards:raise ValueError('Invalid shard')
    reg=screening_registry() if a.phase=='screen' else competition_registry(json.loads(a.selection.read_text()))
    failures=[]
    for idx,cfg in enumerate(reg):
        if idx%a.shards==a.shard and not run_fit(cfg,a.out/cfg['id']):failures.append(cfg['id'])
    write(a.out/f'SHARD_{a.phase}_{a.shard}.json',dict(status='PASS' if not failures else 'FAILED',failed_fit_ids=failures,source_commit=commit()))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
