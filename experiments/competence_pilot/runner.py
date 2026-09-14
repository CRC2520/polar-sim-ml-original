"""CP-NB bounded pilot. New code; frozen scientific modules are never edited."""
from __future__ import annotations
import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import time
import traceback
import numpy as np
from .trial import *


def runtime():
    global torch, nn, RoutingActor, GenericActor, Value, model_digest, log_squashed_gaussian, NativeEnv, runtime_lock
    import torch
    from torch import nn
    from b1s.execution.core import RoutingActor, GenericActor, Value, model_digest, log_squashed_gaussian
    from b1s.execution.instrument import NativeEnv, runtime_lock
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return runtime_lock()


class FixedInput:
    def __init__(self, data, mode):
        require(mode in MODES, 'Invalid input mode')
        self.mode = mode
        self.mean = np.array(data['mean'], dtype=np.float64)
        self.scale = np.array(data['scale'], dtype=np.float64)
        require(self.mean.shape == self.scale.shape == (93,), 'Normalizer shape')
        require(np.isfinite(self.mean).all() and np.isfinite(self.scale).all() and (self.scale >= .01).all(), 'Invalid normalizer')
        self.mean.setflags(write=False); self.scale.setflags(write=False)

    def __call__(self, z):
        z = np.asarray(z, dtype=np.float32)
        require(z.ndim > 0 and z.shape[-1] == 93 and np.isfinite(z).all(), 'Invalid raw observation')
        x = z.copy() if self.mode == 'raw' else ((z.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        require(np.isfinite(x).all(), 'Nonfinite normalized observation; never replace')
        return x


def gae_returns(rewards, values, dones, last, gamma, lam):
    # Preserve the historical PPO timeout treatment (done=terminated OR truncated).
    adv = np.zeros_like(rewards, dtype=np.float32); g = np.zeros(rewards.shape[1])
    for t in range(len(rewards)-1, -1, -1):
        nt = 1 - dones[t].astype(float)
        nv = last if t == len(rewards)-1 else values[t+1]
        g = rewards[t] + gamma * nv * nt - values[t] + gamma * lam * nt * g
        adv[t] = g
    return adv, adv + values


@contextlib.contextmanager
def isolated_evaluation_rng():
    p, n, t = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(p); np.random.set_state(n); torch.set_rng_state(t)


def predict(actor, z):
    if actor is None:
        return np.zeros(12, dtype=np.float32)
    with torch.no_grad():
        if isinstance(actor, (RoutingActor, GenericActor)):
            return actor(torch.from_numpy(z[None]))[0].tanh()[0].numpy()
        return np.asarray(actor.predict(z, deterministic=True)[0], dtype=np.float32)


def evaluate(actor, transform, rep, path, budgets_label):
    rows=[]; env=NativeEnv()
    with isolated_evaluation_rng(), gzip.open(path, 'wt', encoding='utf-8', newline='\n') as f:
        try:
            for ep in range(EVAL_EPISODES):
                sd=seed('evaluation','environment',rep,ep);z,_=env.reset(seed=sd)
                ret=0.; effort=0.; h=hashlib.sha256(); large=0; coords=0; maxabs=0.; saturated=0; hidden=0
                while True:
                    x=transform(z);a=predict(actor,x)
                    maxabs=max(maxabs,float(np.abs(x).max()));large+=int((np.abs(x)>10).sum());coords+=93
                    if actor is not None and env.t % 32 == 0:
                        with torch.no_grad():
                            tx=torch.from_numpy(x[None])
                            if isinstance(actor,RoutingActor):
                                hval=actor(tx,trace=True)[2]['h']
                            elif isinstance(actor,GenericActor):
                                hval=actor.net[:2](tx)
                            else:
                                hval=None
                            if hval is not None:
                                saturated+=int((hval.abs()>.99).sum());hidden+=hval.numel()
                    nz,r,term,trunc,info=env.step(a);ret+=r;effort+=float(np.abs(a).sum())
                    event=dict(rep=rep,episode=ep,seed=sd,cycle=env.t,checkpoint=budgets_label,
                               z=z.tolist(),action=a.tolist(),reward=r,next_z=nz.tolist(),audit=info)
                    line=json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
                    h.update(line.encode());f.write(line);z=nz
                    if term or trunc:
                        break
                rows.append(dict(rep=rep,episode=ep,seed=sd,loss=-ret,return_value=ret,
                                 displacement=info['package_displacement'],cycles=env.t,
                                 fallen_count=info['fallen_count'],package_dropped=info['package_dropped'],
                                 terminated=term,truncated=trunc,motor_request_abs_sum=effort,
                                 trace_sha256=h.hexdigest(),input_max_abs=maxabs,
                                 input_coordinate_fraction_abs_gt_10=large/coords,
                                 encoder_tanh_fraction_abs_gt_099=(saturated/hidden if hidden else None)))
        finally:
            env.close()
    return rows


def normalization(rep, out):
    n=0;mean=np.zeros(93,dtype=np.float64);m2=np.zeros(93,dtype=np.float64)
    path=out/f'normalization-r{rep}.jsonl.gz';env=NativeEnv()
    try:
        with gzip.open(path,'wt',encoding='utf-8',newline='\n') as f:
            for policy in ('zero','uniform'):
                rng=np.random.default_rng(seed('normalization','actions',rep,policy))
                episode=0;sd=seed('normalization','environment',rep,policy,episode);z,_=env.reset(seed=sd)
                for k in range(CALIBRATION_STEPS//2):
                    n+=1;delta=z.astype(np.float64)-mean;mean+=delta/n;m2+=delta*(z-mean)
                    a=np.zeros(12,dtype=np.float32) if policy=='zero' else rng.uniform(-1,1,12).astype(np.float32)
                    f.write(json.dumps(dict(rep=rep,policy=policy,step=k,episode=episode,seed=sd,
                                           cycle=env.t,z=z.tolist(),action=a.tolist()),sort_keys=True,separators=(',',':'))+'\n')
                    z,r,t,tr,info=env.step(a)
                    if t or tr:
                        episode+=1;sd=seed('normalization','environment',rep,policy,episode);z,_=env.reset(seed=sd)
    finally:
        env.close()
    require(n==CALIBRATION_STEPS,'Calibration count mismatch')
    scale=np.maximum(np.sqrt(np.maximum(m2/n,0)),.01)
    data=dict(rep=rep,count=n,mean=mean.tolist(),scale=scale.tolist(),variance_population=(m2/n).tolist(),
              trace_sha256=sha(path),trace_path=path.name,uses_rewards=False,uses_evaluation_data=False,
              fixed_before_training=True,online_updates=False,clip=False)
    write_new(out/f'NORMALIZATION_{rep}.json',data)
    return data


def preflight(out):
    request=execution_guard();preservation=preserve();lock=runtime()
    require(not (ROOT/'run/DEVELOPMENT_FREEZE.json').exists(),'A previous pilot freeze already exists; no replay')
    out.mkdir(parents=True,exist_ok=False)
    from .tests import runtime_tests, unit_tests
    unit_tests();qa=runtime_tests()
    write_new(out/'QA.json',qa)
    for rep in range(REPLICATES):
        normalization(rep,out)
    paths=['experiments/competence_pilot/'+n for n in ['PROTOCOL_ES.md','PLAN.json','trial.py','runner.py','tests.py','START_REQUEST.json']]
    paths+=['.github/workflows/competence-pilot.yml','b1s/execution/core.py','b1s/execution/instrument.py','b1s/execution/requirements.txt']
    freeze=dict(status='DEVELOPMENT_FROZEN_FOR_AUTHORIZED_PILOT',namespace=NAMESPACE,
                source_commit=os.environ['GITHUB_SHA'],code_commit=request['code_commit'],
                original_scientific_source=HISTORICAL,original_scientific_base=BASE,
                workflow_run_id=os.environ['GITHUB_RUN_ID'],run_attempt=1,registry=registry(),
                plan=plan(),source_hashes={p:sha(p) for p in paths},runtime=lock,
                historical_preservation=preservation,utc_before_training=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                files={p.name:sha(p) for p in out.iterdir() if p.is_file()},**BOUNDARY)
    write_new(out/'DEVELOPMENT_FREEZE.json',freeze)
    print('PRETRAINING_FREEZE_READY',sha(out/'DEVELOPMENT_FREEZE.json'),flush=True)


def verify_frozen(frozen):
    execution_guard();preserve()
    f=read(frozen/'DEVELOPMENT_FREEZE.json')
    require(f['source_commit']==os.environ['GITHUB_SHA'] and str(f['workflow_run_id'])==os.environ['GITHUB_RUN_ID'],'Freeze identity mismatch')
    require(f['registry']==registry() and f['plan']==plan(),'Registry changed')
    for p,h in f['source_hashes'].items():
        require(sha(p)==h,'Source bytes changed: '+p)
    for p,h in f['files'].items():
        require(sha(frozen/p)==h,'Frozen artifact bytes changed: '+p)
    require(f['B1E_executed'] is False and f['final_seeds_generated'] is False,'Invalid boundary')
    return f


def training_env(rep, slot, transform, ledger, split="training"):
    class TrainEnv(NativeEnv):
        def __init__(self):
            super().__init__();self.counter=0;self.total_steps=0;self.total_return=0.;self.length=0
        def reset(self,*,seed=None,options=None):
            sd=globals()['seed'](split,'environment',rep,slot,self.counter);self.counter+=1
            raw,info=super().reset(seed=sd,options=options);self.total_return=0.;self.length=0
            return transform(raw),info
        def step(self,a):
            raw,r,t,tr,info=super().step(a);self.total_steps+=1;self.total_return+=r;self.length+=1
            if t or tr:
                ledger.append(dict(seed=self.episode_seed,slot=slot,episode=self.counter-1,
                                   slot_steps=self.total_steps,return_value=self.total_return,cycles=self.length,
                                   displacement=info['package_displacement'],fallen_count=info['fallen_count'],
                                   package_dropped=info['package_dropped'],terminated=t,truncated=tr))
            return transform(raw),r,t,tr,info
    return TrainEnv()


def ppo_fit(cfg,transform,out,checkpoint):
    hp=cfg['hp'];batch=hp['rollout_steps'];n_env=8;horizon=batch//n_env
    actor=RoutingActor(cfg['mask']) if cfg['role'] in ('G0','GD') else GenericActor();critic=Value()
    initial=model_digest(actor);params=list(actor.parameters())+list(critic.parameters())
    opt=torch.optim.Adam(params,lr=hp['lr'],eps=hp['adam_eps']);ledger=[];curves=[];updates=0;t0=time.monotonic()
    envs=[training_env(cfg['rep'],i,transform,ledger,cfg.get('environment_split','training')) for i in range(n_env)]
    z=np.stack([e.reset()[0] for e in envs])
    try:
        for offset in range(0,BUDGETS[-1],batch):
            obs=[];pre=[];logps=[];values=[];rewards=[];dones=[]
            for t in range(horizon):
                with torch.no_grad():
                    mu,ls=actor(torch.from_numpy(z));p=mu+ls.exp()*torch.randn_like(mu)
                    a=p.tanh();lp=log_squashed_gaussian(p,mu,ls);v=critic(torch.from_numpy(z))
                obs.append(z.copy());pre.append(p.numpy());logps.append(lp.numpy());values.append(v.numpy())
                rs=[];ds=[];nz=[]
                for i,e in enumerate(envs):
                    nxt,r,term,trunc,info=e.step(a[i].numpy());done=term or trunc
                    rs.append(r);ds.append(done);nz.append(e.reset()[0] if done else nxt)
                rewards.append(rs);dones.append(ds);z=np.stack(nz)
            with torch.no_grad():
                last=critic(torch.from_numpy(z)).numpy()
            adv,target=gae_returns(np.asarray(rewards),np.asarray(values),np.asarray(dones),last,hp['gamma'],hp['gae_lambda'])
            adv=adv.reshape(-1);adv=(adv-adv.mean())/(adv.std()+1e-8)
            ot=torch.from_numpy(np.asarray(obs).reshape(batch,93));pt=torch.from_numpy(np.asarray(pre).reshape(batch,12))
            old=torch.from_numpy(np.asarray(logps).reshape(-1));at=torch.from_numpy(adv);rt=torch.from_numpy(target.reshape(-1))
            al=[];vl=[];kl=[]
            for epoch in range(hp['epochs']):
                order=np.random.permutation(batch)
                for j in range(0,batch,hp['minibatch']):
                    ix=order[j:j+hp['minibatch']];mu,ls=actor(ot[ix]);lp=log_squashed_gaussian(pt[ix],mu,ls)
                    logr=lp-old[ix];ratio=logr.exp()
                    a_loss=-torch.min(ratio*at[ix],ratio.clamp(1-hp['clip'],1+hp['clip'])*at[ix]).mean()
                    v_loss=(critic(ot[ix])-rt[ix]).square().mean();loss=a_loss+hp['value_coef']*v_loss
                    require(bool(torch.isfinite(loss)),'Nonfinite PPO objective')
                    opt.zero_grad();loss.backward();g=nn.utils.clip_grad_norm_(params,hp['max_grad_norm'])
                    require(bool(torch.isfinite(g)),'Nonfinite PPO gradient');opt.step();updates+=1
                    al.append(a_loss.item());vl.append(v_loss.item());kl.append(float(((ratio-1)-logr).mean().detach()))
            steps=offset+batch
            if steps%32768==0:
                row=dict(native_steps=steps,optimizer_updates=updates,actor_loss=statistics.mean(al),
                         value_loss=statistics.mean(vl),approx_kl=statistics.mean(kl),completed_episodes=len(ledger))
                curves.append(row);print(cfg['id'],row,flush=True)
                write_new(out/f'CURVE_{steps}.json',row)
            if steps in BUDGETS:
                torch.save(dict(actor_state=actor.state_dict(),critic_state=critic.state_dict(),optimizer=opt.state_dict(),
                                configuration=cfg,steps=steps,torch_rng_state=torch.get_rng_state()),out/f'model-{steps}.pt')
                checkpoint(actor,steps,updates)
        require(initial!=model_digest(actor),'No graph/PPO actor update')
        return dict(algorithm='PPO',environment_steps=BUDGETS[-1],optimizer_updates=updates,sampling_parallelism=8,
                    actor_parameters=sum(p.numel() for p in actor.parameters()),critic_parameters=sum(p.numel() for p in critic.parameters()),
                    initial_actor_digest=initial,final_actor_digest=model_digest(actor),training_and_evaluation_seconds=time.monotonic()-t0),ledger,curves
    finally:
        for e in envs:e.close()


def sac_fit(cfg,transform,out,checkpoint):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    hp=cfg['hp'];ledger=[];curves=[];env=training_env(cfg['rep'],0,transform,ledger,cfg.get('environment_split','training'))
    class PostUpdateSAC(SAC):
        def train(self,gradient_steps,batch_size=64):
            super().train(gradient_steps,batch_size)
            if self.num_timesteps in BUDGETS:
                self.save(out/f'model-{self.num_timesteps}')
                checkpoint(self,self.num_timesteps,self._n_updates)
    class Logger(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%32768==0:
                row=dict(native_steps=self.num_timesteps,optimizer_updates_before_current_collection_update=self.model._n_updates,
                         completed_episodes=len(ledger))
                curves.append(row);write_new(out/f'CURVE_{self.num_timesteps}.json',row);print(cfg['id'],row,flush=True)
            return True
    model=PostUpdateSAC('MlpPolicy',env,learning_rate=hp['lr'],gamma=hp['gamma'],buffer_size=hp['buffer_size'],
                        learning_starts=hp['learning_starts'],batch_size=hp['batch_size'],tau=hp['tau'],
                        train_freq=hp['train_freq'],gradient_steps=hp['gradient_steps'],ent_coef='auto',
                        policy_kwargs=dict(net_arch=[64,64],activation_fn=nn.ReLU),seed=cfg['initial_seed'],device='cpu',verbose=0)
    initial=model_digest(model.actor);t0=time.monotonic()
    try:
        model.learn(total_timesteps=BUDGETS[-1],callback=Logger(),progress_bar=False)
        require(model.num_timesteps==BUDGETS[-1],'Unexpected SAC transition count')
        require(model._n_updates==BUDGETS[-1]-hp['learning_starts'],'Unexpected SAC update count')
        require(initial!=model_digest(model.actor),'No SAC actor update')
        return dict(algorithm='SAC',environment_steps=model.num_timesteps,optimizer_updates=model._n_updates,
                    sampling_parallelism=1,actor_parameters=sum(p.numel() for p in model.actor.parameters()),
                    critic_parameters=sum(p.numel() for p in model.critic.parameters()),
                    target_critic_parameters=sum(p.numel() for p in model.critic_target.parameters()),
                    replay_capacity=hp['buffer_size'],replay_entries=model.replay_buffer.size(),replay_archived=False,
                    initial_actor_digest=initial,final_actor_digest=model_digest(model.actor),
                    training_and_evaluation_seconds=time.monotonic()-t0),ledger,curves
    finally:
        env.close()


def fit(index,frozen,out):
    require(0<=index<len(registry()),'Invalid fit index');cfg=registry()[index]
    f=verify_frozen(frozen);lock=runtime();require(lock['versions']==f['runtime']['versions'] and lock['sources']==f['runtime']['sources'],'Scientific runtime mismatch');norm=read(frozen/f'NORMALIZATION_{cfg["rep"]}.json');transform=FixedInput(norm,cfg['mode'])
    out=out/cfg['id'];out.mkdir(parents=True,exist_ok=False)
    random.seed(cfg['initial_seed']);np.random.seed(cfg['initial_seed']);torch.manual_seed(cfg['initial_seed'])
    write_new(out/'STARTED.json',dict(configuration=cfg,source_commit=os.environ['GITHUB_SHA'],
                                     freeze_sha256=sha(frozen/'DEVELOPMENT_FREEZE.json'),runtime=lock,**BOUNDARY))
    write_new(out/'NORMALIZATION.json',norm);done=[]
    def checkpoint(actor,steps,updates):
        require(steps in BUDGETS and steps not in done,'Duplicate or unplanned checkpoint')
        before=model_digest(actor.actor if cfg['role']=='SAC' else actor)
        rows=evaluate(actor,transform,cfg['rep'],out/f'evaluation-{steps}.jsonl.gz',steps)
        require(before==model_digest(actor.actor if cfg['role']=='SAC' else actor),'Evaluation altered weights')
        write_new(out/f'EVALUATION_{steps}.json',dict(steps=steps,optimizer_updates=updates,
                  measured_after_budget_updates=True,episodes=rows,summary=summarize(rows)))
        done.append(steps)
    try:
        resources,ledger,curves=(sac_fit if cfg['role']=='SAC' else ppo_fit)(cfg,transform,out,checkpoint)
        require(done==BUDGETS,'Missing checkpoint evaluation')
        resources.update(input_mode=cfg['mode'],normalizer_sha256=sha(out/'NORMALIZATION.json'),
                         reward_normalization=False,normalizer_updated_during_training=False,
                         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_new(out/'TRAINING_RESOURCES.json',resources);write_new(out/'TRAINING_EPISODES.json',ledger)
        write_new(out/'LEARNING_CURVE.json',curves)
        write_new(out/'COMPLETE.json',dict(status='COMPLETE',configuration=cfg,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},**BOUNDARY))
    except Exception:
        write_new(out/'FAILED.json',dict(status='FAILED_RETAINED',configuration=cfg,error=traceback.format_exc(),replacement_seed=False,**BOUNDARY))
        raise


def aggregate(fits,frozen,out,artifact_inventory):
    verify_frozen(frozen);runtime();out.mkdir(parents=True,exist_ok=False)
    require(read(ROOT/'PLAN.json')==plan(),'Plan contract mismatch')
    cells={};all_rows={};resources={};fit_manifests={};witness=[]
    for cfg in registry():
        folder=fits/cfg['id'];c=read(folder/'COMPLETE.json');require(c['status']=='COMPLETE' and c['configuration']==cfg,'Fit absent/failed/configuration mismatch')
        for p,h in c['files'].items():
            require(sha(folder/p)==h,'Fit file hash mismatch: '+cfg['id']+'/'+p)
        started=read(folder/'STARTED.json');require(started['source_commit']==os.environ['GITHUB_SHA'] and started['freeze_sha256']==sha(frozen/'DEVELOPMENT_FREEZE.json'),'Fit provenance mismatch')
        require(sha(folder/'NORMALIZATION.json')==sha(frozen/f'NORMALIZATION_{cfg["rep"]}.json'),'Wrong fixed normalizer')
        r=read(folder/'TRAINING_RESOURCES.json');require(r['environment_steps']==BUDGETS[-1],'Incomplete budget')
        resources[cfg['id']]=r;fit_manifests[cfg['id']]=dict(complete_sha256=sha(folder/'COMPLETE.json'),files=c['files'])
        for b in BUDGETS:
            ev=read(folder/f'EVALUATION_{b}.json');rows=ev['episodes']
            require(len(rows)==EVAL_EPISODES and ev['measured_after_budget_updates'] is True,'Invalid checkpoint')
            require([r['seed'] for r in rows]==[seed('evaluation','environment',cfg['rep'],ep) for ep in range(EVAL_EPISODES)],'Evaluation panel mismatch')
            require(summarize(rows)==ev['summary'],'Episode summary mismatch')
            all_rows[f'{cfg["id"]}-{b}']=rows
    for rep in range(REPLICATES):
        norm=read(frozen/f'NORMALIZATION_{rep}.json')
        witness+=evaluate(None,FixedInput(norm,'raw'),rep,out/f'no-action-r{rep}.jsonl.gz','no-action')
    zero_by_rep={rep:summarize([x for x in witness if x['rep']==rep]) for rep in range(REPLICATES)}
    for role in ROLES:
        for mode in MODES:
            for b in BUDGETS:
                reps=[];rows=[]
                for rep in range(REPLICATES):
                    ep=all_rows[f'{role}-{mode}-r{rep}-{b}'];s=summarize(ep);rows+=ep
                    reps.append(dict(rep=rep,summary=s,competence=competence(s,zero_by_rep[rep])))
                s=summarize(rows)
                cells[f'{role}-{mode}-{b}']=dict(summary=s,competence=competence(s,summarize(witness)),by_rep=reps,
                     blocks_passing=sum(x['competence']['descriptive_competence_pass'] for x in reps),independent_training_blocks=3)
    pages=read(artifact_inventory);artifacts=[a for page in pages for a in page['artifacts']]
    expected={'cpnb-freeze'}|{'cpnb-fit-'+c['id'] for c in registry()}
    inventory=[a for a in artifacts if a['name'] in expected]
    require({a['name'] for a in inventory}==expected and len(inventory)==len(expected),'Incomplete artifact inventory')
    summary=dict(status='PILOT_COMPLETE_FOR_REVIEW',source_commit=os.environ['GITHUB_SHA'],
                 freeze_sha256=sha(frozen/'DEVELOPMENT_FREEZE.json'),workflow_run_id=os.environ['GITHUB_RUN_ID'],
                 cells=cells,factorial_contrasts=factorial_contrasts(cells),resources=resources,witness=summarize(witness),
                 witness_by_rep=zero_by_rep,fit_manifests=fit_manifests,artifacts=inventory,plan=plan(),
                 historical_preservation=preserve(),scientific_v1_1_results_replaced=False,
                 note='Pilot diagnostics only; all budgets and blocks reported. No new structure selected, no automatic continuation.',**BOUNDARY)
    write_new(out/'SUMMARY.json',summary);write_new(out/'EPISODES.json',dict(models=all_rows,no_action=witness))
    text=['# Piloto de competencia CP-NB-1.0.0','', '**PILOT_COMPLETE_FOR_REVIEW**','',
          'Desarrollo nuevo: no confirma estructura, no modifica B1-S v1.1 y no habilita B1-E.','',
          '| Rol | Entrada | Pasos | Pérdida | Desplazamiento adicional | Competencia agregada | Bloques /3 |',
          '|---|---|---:|---:|---:|---|---:|']
    for name,c in cells.items():
        role,mode,b=name.split('-');text.append(f'| {role} | {mode} | {b} | {c["summary"]["loss"]:.8g} | {c["competence"]["additional_displacement"]:.8g} | {c["competence"]["descriptive_competence_pass"]} | {c["blocks_passing"]} |')
    text+=['','Tres bloques: precisión limitada. Los presupuestos son checkpoints anidados, no réplicas adicionales.',
           'El normalizador se fijó con datos de calibración de entradas, no con evaluación; no se normalizaron recompensas.',
           'Los contrastes y las medias por bloque están en SUMMARY.json. Se conservan resultados adversos e inconclusos.',
           'B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION; ready_for_b1e_confirmatory_run=false.',
           'Los binarios tienen retención temporal en Actions (90 días); ver inventario y hashes. No se promete permanencia en Git.','']
    (out/'COMPLETION_REPORT_ES.md').write_text('\n'.join(text),encoding='utf-8')
    write_new(out/'RESULTS_MANIFEST.json',dict(status='TEXT_AND_FIT_HASHES_VERIFIED',
              source_commit=os.environ['GITHUB_SHA'],freeze_sha256=summary['freeze_sha256'],
              files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.md')},
              auxiliary_artifact_files={p.name:sha(p) for p in out.iterdir() if p.suffix=='.gz'},**BOUNDARY))
    print(json.dumps(dict(status=summary['status'],cells=len(cells),fits=24,**BOUNDARY)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','fit','aggregate']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frozen',type=Path);p.add_argument('--fits',type=Path);p.add_argument('--index',type=int);p.add_argument('--artifacts',type=Path);a=p.parse_args()
    if a.stage=='preflight':preflight(a.out)
    elif a.stage=='fit':fit(a.index,a.frozen,a.out)
    else:
        try:
            aggregate(a.fits,a.frozen,a.out,a.artifacts)
        except Exception:
            a.out.mkdir(parents=True,exist_ok=True)
            write_new(a.out/'BLOCKED.json',dict(status='PILOT_BLOCKED_WITH_RETAINED_EVIDENCE',
                source_commit=os.environ.get('GITHUB_SHA'),workflow_run_id=os.environ.get('GITHUB_RUN_ID'),
                error=traceback.format_exc(),missing_results_not_filled=True,**BOUNDARY))
            raise


if __name__=='__main__':
    main()
