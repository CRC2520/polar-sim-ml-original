"""Development-only PPO/SAC fits, common native data and sealed search registry."""
from __future__ import annotations
import gc
import gzip
import hashlib
import json
import random
import resource
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import ROOT, RoutingActor, GenericActor, Value, log_squashed_gaussian, masks, seed_for, model_digest, write_json
from .instrument import NativeEnv, runtime_lock


def plan():return json.loads((ROOT/'PLAN.json').read_text())


def registry():
    p=plan()
    entries=[{'id':'graph-'+m,'kind':'graph','mask':m,'lr':p['ppo']['lr'],'gamma':p['ppo']['gamma']} for m in masks()+['111111']]
    for kind in ('ppo','sac'):
        k=0
        for lr in p['generic_search']['learning_rates']:
            for gamma in p['generic_search']['discount_factors']:
                entries.append({'id':f'{kind}-{k:02d}','kind':kind,'mask':None,'lr':lr,'gamma':gamma})
                k+=1
    assert len(entries)==127 and len({x['id'] for x in entries})==127
    return entries


def setup_seed(rep,kind):
    seed=seed_for('train','weights',rep,kind if kind=='sac' else 'ppo')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return seed


@torch.no_grad()
def predict(actor,z,lesion=()):
    if isinstance(actor,(RoutingActor,GenericActor)):
        return actor(torch.from_numpy(np.asarray(z,dtype=np.float32)[None]),lesion)[0].tanh()[0].numpy()
    return np.asarray(actor.predict(z,deterministic=True)[0],dtype=np.float32)


def evaluate(actor,rep,split,n,trace_path=None,case_id='',lesion=()):
    if split not in ('selection','heldout','causal','qa'):raise ValueError('Forbidden evaluation split')
    env=NativeEnv();rows=[]
    trace=gzip.open(trace_path,'wt',encoding='utf-8') if trace_path else None
    try:
        for ep in range(n):
            seed=seed_for(split,'environment',rep,ep)
            z,_=env.reset(seed=seed);ret=0.;events=hashlib.sha256();reqcost=0.
            while True:
                a=np.zeros(12,dtype=np.float32) if actor is None else predict(actor,z,lesion)
                nxt,r,t,tr,info=env.step(a)
                row={'case':case_id,'replicate':rep,'split':split,'episode':ep,'seed':seed,
                     'cycle':env.t,'z':z.tolist(),'request':a.tolist(),'reward':r,
                     'next_z':nxt.tolist(),'terminal':t,'truncated':tr,'audit':info}
                line=json.dumps(row,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n'
                events.update(line.encode())
                if trace:trace.write(line)
                ret+=r;reqcost+=float(np.abs(a).sum());z=nxt
                if t or tr:break
                if env.t>=500:raise RuntimeError('Native horizon was not enforced')
            rows.append({'replicate':rep,'split':split,'episode':ep,'seed':seed,'return':ret,'loss':-ret,
                         'cycles':env.t,'package_displacement':info['package_displacement'],
                         'fallen_count':info['fallen_count'],'package_dropped':info['package_dropped'],
                         'motor_request_abs_sum':reqcost,'trace_sha256':events.hexdigest(),
                         'development_only':True,'confirmatory':False})
    finally:
        env.close()
        if trace:trace.close()
    return rows


def train_ppo(cfg,rep,out):
    p=plan();hp=p['ppo'];steps=p['training_steps_per_fit'];rollout=hp['rollout_steps']
    assert steps%rollout==0
    setup_seed(rep,cfg['kind'])
    actor=RoutingActor(cfg['mask']) if cfg['kind']=='graph' else GenericActor()
    critic=Value();params=list(actor.parameters())+list(critic.parameters())
    opt=torch.optim.Adam(params,lr=cfg['lr'],eps=1e-5)
    env=NativeEnv();ep=0;z,_=env.reset(seed=seed_for('train','environment',rep,ep))
    initial_digest=model_digest(actor);eps=[];curves=[];epret=0.;eplen=0;updates=0
    t0=time.perf_counter()
    try:
        for offset in range(0,steps,rollout):
            obs=[];pre=[];oldlp=[];values=[];rewards=[];dones=[]
            for k in range(rollout):
                tensor=torch.from_numpy(z[None])
                with torch.no_grad():
                    mu,ls=actor(tensor);u0=mu+ls.exp()*torch.randn_like(mu)
                    action=u0.tanh();lp=log_squashed_gaussian(u0,mu,ls);v=critic(tensor)
                obs.append(z.copy());pre.append(u0[0].numpy());oldlp.append(lp.item());values.append(v.item())
                z,r,t,tr,info=env.step(action[0].numpy())
                done=t or tr;rewards.append(r);dones.append(done);epret+=r;eplen+=1
                if done:
                    eps.append({'episode':ep,'seed':env.episode_seed,'return':epret,'cycles':eplen,
                                'step':offset+k+1,'package_displacement':info['package_displacement'],'fall_count':info['fallen_count']})
                    ep+=1;epret=0.;eplen=0;z,_=env.reset(seed=seed_for('train','environment',rep,ep))
            with torch.no_grad():last_value=float(critic(torch.from_numpy(z[None]))[0])
            adv=np.zeros(rollout,dtype=np.float32);last_gae=0.
            for k in reversed(range(rollout)):
                nonterminal=1.0-float(dones[k]);vnext=last_value if k==rollout-1 else values[k+1]
                delta=rewards[k]+cfg['gamma']*vnext*nonterminal-values[k]
                last_gae=delta+cfg['gamma']*hp['gae_lambda']*nonterminal*last_gae;adv[k]=last_gae
            returns=adv+np.asarray(values,dtype=np.float32)
            adv=(adv-adv.mean())/(adv.std()+1e-8)
            ot=torch.from_numpy(np.stack(obs));pt=torch.from_numpy(np.stack(pre))
            old=torch.tensor(oldlp);at=torch.from_numpy(adv);rt=torch.from_numpy(returns)
            actor_losses=[];value_losses=[];kls=[];clipfractions=[]
            for epoch in range(hp['epochs']):
                order=np.random.permutation(rollout)
                for lo in range(0,rollout,hp['minibatch']):
                    idx=order[lo:lo+hp['minibatch']]
                    mu,ls=actor(ot[idx]);new=log_squashed_gaussian(pt[idx],mu,ls)
                    logratio=new-old[idx];ratio=logratio.exp()
                    aloss=-torch.min(ratio*at[idx],ratio.clamp(1-hp['clip'],1+hp['clip'])*at[idx]).mean()
                    vloss=(critic(ot[idx])-rt[idx]).square().mean()
                    loss=aloss+hp['value_coef']*vloss
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO objective')
                    opt.zero_grad();loss.backward()
                    grad=nn.utils.clip_grad_norm_(params,hp['max_grad_norm'])
                    if not torch.isfinite(grad):raise RuntimeError('Nonfinite PPO gradient')
                    opt.step();updates+=1
                    actor_losses.append(aloss.item());value_losses.append(vloss.item())
                    kls.append(float(((ratio-1)-logratio).mean().detach()));clipfractions.append(float(((ratio-1).abs()>hp['clip']).float().mean()))
            curve={'steps':offset+rollout,'actor_loss':float(np.mean(actor_losses)),
                   'value_loss':float(np.mean(value_losses)),'approx_kl':float(np.mean(kls)),
                   'clip_fraction':float(np.mean(clipfractions)),'episodes_completed':len(eps),
                   'last_20_episode_return':float(np.mean([x['return'] for x in eps[-20:]])) if eps else None,
                   'elapsed_seconds':time.perf_counter()-t0,'optimizer_updates':updates}
            curves.append(curve)
            if (offset+rollout)%(rollout*16)==0:print(cfg['id'],rep,curve,flush=True)
        duration=time.perf_counter()-t0
        torch.save({'actor_state':actor.state_dict(),'critic_state':critic.state_dict(),'optimizer':opt.state_dict(),
                    'configuration':cfg,'replicate':rep,'steps':steps},out/'model.pt')
        write_json(out/'LEARNING_CURVE.json',curves);write_json(out/'TRAINING_EPISODES.json',eps)
        write_json(out/'TRAINING_RESOURCES.json',{'environment_steps':steps,'optimizer_updates':updates,
                   'training_seconds':duration,'initial_actor_sha256':initial_digest,'final_actor_sha256':model_digest(actor),
                   'parameters_actor':sum(x.numel() for x in actor.parameters()),'parameters_critic':sum(x.numel() for x in critic.parameters()),
                   'trainable_log_std':actor.log_std.detach().tolist(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'train_data_retention':'episode identities, returns, curves, checkpoint and source/config; not every training observation retained',
                   'initialization_and_final_parameters_differ':initial_digest!=model_digest(actor)})
        if initial_digest==model_digest(actor):raise RuntimeError('Training did not change any actor parameter')
        return actor
    finally:env.close()


def train_sac(cfg,rep,out):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    p=plan();hp=p['sac'];steps=p['training_steps_per_fit'];seed=setup_seed(rep,'sac')
    env=NativeEnv();episodes=[];curves=[]
    class Logger(BaseCallback):
        def __init__(self):super().__init__();self.ret=0.;self.length=0
        def _on_step(self):
            self.ret+=float(self.locals['rewards'][0]);self.length+=1
            if self.locals['dones'][0]:
                info=self.locals['infos'][0]
                episodes.append({'seed':info['episode_seed'],'return':self.ret,'cycles':self.length,'step':self.num_timesteps,
                                 'package_displacement':info['package_displacement'],'fall_count':info['fallen_count']})
                self.ret=0.;self.length=0
            if self.num_timesteps%8192==0:
                c={'steps':self.num_timesteps,'episodes_completed':len(episodes),'last_20_episode_return':float(np.mean([x['return'] for x in episodes[-20:]])) if episodes else None,'elapsed_seconds':time.perf_counter()-t0}
                curves.append(c);print(cfg['id'],rep,c,flush=True)
            return True
    model=SAC('MlpPolicy',env,learning_rate=cfg['lr'],gamma=cfg['gamma'],buffer_size=steps,
              learning_starts=hp['learning_starts'],batch_size=hp['batch_size'],tau=hp['tau'],
              train_freq=hp['train_freq'],gradient_steps=hp['gradient_steps'],ent_coef='auto',
              policy_kwargs={'net_arch':[64,64],'activation_fn':nn.ReLU},seed=seed,device='cpu',verbose=0)
    initial=model_digest(model.actor);t0=time.perf_counter()
    try:
        model.learn(total_timesteps=steps,callback=Logger(),progress_bar=False)
        duration=time.perf_counter()-t0;model.save(out/'model')
        write_json(out/'LEARNING_CURVE.json',curves);write_json(out/'TRAINING_EPISODES.json',episodes)
        final=model_digest(model.actor)
        write_json(out/'TRAINING_RESOURCES.json',{'environment_steps':steps,'optimizer_updates':model._n_updates,
                   'training_seconds':duration,'initial_actor_sha256':initial,'final_actor_sha256':final,
                   'parameters_actor':sum(x.numel() for x in model.actor.parameters()),
                   'parameters_critic':sum(x.numel() for x in model.critic.parameters()),
                   'parameters_target_critic':sum(x.numel() for x in model.critic_target.parameters()),
                   'replay_capacity':steps,'actual_replay_entries':model.replay_buffer.size(),
                   'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'initialization_and_final_parameters_differ':initial!=final})
        if initial==final:raise RuntimeError('SAC actor did not update')
        return model
    finally:env.close()


def load_actor(folder: Path):
    cfg=json.loads((folder/'STARTED.json').read_text())['configuration']
    if cfg['kind']=='sac':
        from stable_baselines3 import SAC
        return SAC.load(folder/'model.zip',device='cpu')
    a=RoutingActor(cfg['mask']) if cfg['kind']=='graph' else GenericActor()
    data=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    a.load_state_dict(data['actor_state']);a.eval();return a


def run_shard(shard: int,total: int,out: Path):
    p=plan();entries=registry();failures=[]
    torch.set_num_threads(1);runtime=runtime_lock()
    for index,cfg in enumerate(entries):
        if index%total!=shard:continue
        for rep in range(p['independent_training_replicates']):
            folder=out/cfg['id']/f'rep-{rep}'
            if folder.exists():raise RuntimeError('Refuse to overwrite an existing fit')
            folder.mkdir(parents=True)
            start={'configuration':cfg,'replicate':rep,'development_only':True,'confirmatory':False,
                   'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   'plan_sha256':hashlib.sha256((ROOT/'PLAN.json').read_bytes()).hexdigest(),'runtime':runtime,
                   'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            write_json(folder/'STARTED.json',start)
            try:
                actor=train_sac(cfg,rep,folder) if cfg['kind']=='sac' else train_ppo(cfg,rep,folder)
                rows=evaluate(actor,rep,'selection',p['selection_episodes_per_replicate'],folder/'selection_traces.jsonl.gz',cfg['id'])
                write_json(folder/'SELECTION.json',{'rows':rows,'mean_loss':float(np.mean([r['loss'] for r in rows])),'status':'COMPLETE'})
                write_json(folder/'COMPLETE.json',{'status':'COMPLETE','environment_training_steps':p['training_steps_per_fit'],
                           'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.iterdir()) if f.is_file()},
                           'development_only':True,'confirmatory':False,'B1E_executed':False})
                del actor;gc.collect()
            except Exception:
                issue={'status':'TECHNICAL_FAILURE','traceback':traceback.format_exc(),'configuration':cfg,'replicate':rep,
                       'no_seed_replacement':True,'scientific_aggregate_blocked':True}
                write_json(folder/'FAILURE.json',issue);failures.append(issue)
    write_json(out/f'SHARD_{shard:02d}.json',{'shard':shard,'shards':total,'failures':failures,'status':'PASS' if not failures else 'FAIL'})
    if failures:raise RuntimeError(f'{len(failures)} fits failed; evidence retained')
