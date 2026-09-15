"""Guarded B1-SC-D2 v1.0 scientific execution engine.
Scientific entry points require an immutable START_REQUEST.json. QA uses only the
``qa`` namespace and is never scientific evidence.
"""
from __future__ import annotations

import contextlib, hashlib, json, math, os, random, statistics, subprocess, time, traceback
from pathlib import Path
from typing import Sequence
import numpy as np
from experiments.b1sc_d2_v1_0 import implementation as d2

ROOT=Path(__file__).resolve().parent
REPO_ROOT=d2.REPO_ROOT
EXEC_BRANCH="research/b1sc-d2-v1.0-execution-readiness-20260915"
QUALIFIED_IMPLEMENTATION_COMMIT="8f6c46e37a6b4cf18b39f21d21b232b2339c7fe1"
START_REQUEST=ROOT/"START_REQUEST.json"
IMPLEMENTATION_FREEZE=ROOT/"run"/"IMPLEMENTATION_FREEZE.json"
EXECUTION_FREEZE=ROOT/"run"/"EXECUTION_IMPLEMENTATION_FREEZE.json"
EXECUTION_WORKFLOW=".github/workflows/b1sc-d2-v1-0-execution.yml"
HP=dict(lr=.0003,gamma=.99,rollout_steps=512,minibatch=128,epochs=4,gae_lambda=.95,clip=.2,value_coef=.5,max_grad_norm=.5,adam_eps=1e-5)
N_ENVS=8

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def write_json(path,obj,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,ensure_ascii=False)+"\n"
    if exclusive:
        with path.open("x",encoding="utf-8",newline="\n") as f:f.write(data)
    else:path.write_text(data,encoding="utf-8",newline="\n")

def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def git(*args): return subprocess.check_output(["git",*args],text=True).strip()

def runtime():
    global torch,nn,RoutingActor,Value,log_squashed_gaussian,model_digest,NativeEnv,runtime_lock
    import torch
    from torch import nn
    from b1s.execution.core import RoutingActor,Value,log_squashed_gaussian,model_digest
    from b1s.execution.instrument import NativeEnv,runtime_lock
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    return runtime_lock()

def execution_guard():
    require(os.environ.get("GITHUB_REPOSITORY")==d2.REPOSITORY,"Wrong repository")
    require(os.environ.get("GITHUB_REF")=="refs/heads/"+EXEC_BRANCH,"Wrong D2 execution branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT")=="1","Scientific reruns are not authorized")
    require(START_REQUEST.exists(),"START_REQUEST.json is absent")
    require(EXECUTION_FREEZE.exists(),"D2 execution freeze is absent")
    req,freeze=read_json(START_REQUEST),read_json(EXECUTION_FREEZE)
    require(req.get("schema")==d2.SCHEMA,"Wrong request schema")
    require(req.get("authorize_b1sc_d2_v1_0") is True,"D2 execution not explicitly authorized")
    require(req.get("authorized_run_attempt")==1,"Only run attempt 1 can be authorized")
    require(req.get("B1E_executed") is False and req.get("final_seeds_generated") is False,"B1-E/final-seed boundary violated")
    require(req.get("qualified_source_commit")==freeze["qualified_source_commit"],"Qualified source mismatch")
    require(req.get("execution_freeze_sha256")==sha256_file(EXECUTION_FREEZE),"Execution freeze mismatch")
    require(req.get("registry_sha256")==freeze["registry_sha256"],"Registry mismatch")
    require(req.get("design_freeze_sha256")==freeze["design_freeze_sha256"],"Design freeze mismatch")
    require(req.get("implementation_freeze_sha256")==freeze["implementation_freeze_sha256"],"Implementation freeze mismatch")
    require(req.get("workflow_sha256")==freeze["source_hashes"][EXECUTION_WORKFLOW],"Workflow mismatch")
    current=git("rev-parse","HEAD"); require(current==os.environ.get("GITHUB_SHA"),"Unexpected checkout")
    parent=git("rev-parse","HEAD^"); require(req.get("authorization_base_commit")==parent,"Authorization base mismatch")
    require(git("diff","--name-only",parent,current).splitlines()==["experiments/b1sc_d2_v1_0/START_REQUEST.json"],"Authorization commit contains other changes")
    for rel,expected in freeze["source_hashes"].items():
        p=REPO_ROOT/rel; require(p.is_file() and sha256_file(p)==expected,"Frozen source changed: "+rel)
    require(freeze["status"]=="D2_EXECUTION_PIPELINE_QUALIFIED_NOT_AUTHORIZED","Wrong execution freeze status")
    require(freeze["scientific_training_performed"] is False and freeze["scientific_evaluation_performed"] is False,"Qualification contains science")
    return req

def config_for(condition,block):
    require(condition in d2.CONDITIONS and 0<=int(block)<d2.BLOCKS,"Invalid D2 condition/block")
    row=next(x for x in d2.registry() if x["condition"]==condition and x["block"]==int(block))
    return dict(row,hp=HP.copy(),algorithm="PPO")
def state_digest(m): return model_digest(m)
def module_digest(m):
    h=hashlib.sha256()
    for k,t in sorted(m.state_dict().items()): h.update(k.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def paired_seed(split,block,purpose,**extra): return d2.seed(split,d2.paired_seed_identity(int(block),str(purpose),**extra))

def gae_returns(rewards,values,dones,last,gamma,lam):
    adv=np.zeros_like(rewards,dtype=np.float32); g=np.zeros(rewards.shape[1],dtype=np.float64)
    for t in range(len(rewards)-1,-1,-1):
        nt=1-dones[t].astype(float); nv=last if t==len(rewards)-1 else values[t+1]
        g=rewards[t]+gamma*nv*nt-values[t]+gamma*lam*nt*g; adv[t]=g
    return adv,adv+values

class TrainingEnv:
    def __init__(self,block,slot,ledger,split):
        require(split in ("training","qa"),"Invalid training split")
        self.env=NativeEnv(); self.block=int(block); self.slot=int(slot); self.ledger=ledger; self.split=split; self.counter=0; self.total_steps=0; self.ret=0.; self.length=0
    def reset(self):
        sd=paired_seed(self.split,self.block,"environment",slot=self.slot,episode=self.counter); self.counter+=1
        z,info=self.env.reset(seed=sd); self.ret=0.; self.length=0; return z,info
    def step(self,a):
        z,r,t,tr,info=self.env.step(a); self.total_steps+=1; self.ret+=r; self.length+=1
        if t or tr:self.ledger.append(dict(block=self.block,slot=self.slot,episode=self.counter-1,seed=self.env.episode_seed,slot_steps=self.total_steps,return_value=self.ret,cycles=self.length,displacement=float(info["package_displacement"]),fallen_count=int(info["fallen_count"]),package_dropped=bool(info["package_dropped"]),terminated=bool(t),truncated=bool(tr)))
        return z,r,t,tr,info
    def close(self): self.env.close()

@contextlib.contextmanager
def isolated_rng():
    p,n,t=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    try: yield
    finally: random.setstate(p); np.random.set_state(n); torch.set_rng_state(t)

def summarize(rows:Sequence[dict]):
    require(bool(rows),"Cannot summarize absent episodes")
    return dict(loss=statistics.mean(float(x["loss"]) for x in rows),displacement=statistics.mean(float(x["displacement"]) for x in rows),fall_fraction=statistics.mean(int(x["fallen_count"]>0) for x in rows),drop_fraction=statistics.mean(int(bool(x["package_dropped"])) for x in rows),cycles=statistics.mean(int(x["cycles"]) for x in rows),episodes=len(rows))
def deterministic_action(actor,z,lesion):
    with torch.no_grad():
        mu,_=actor(torch.from_numpy(np.asarray(z,dtype=np.float32)[None]),lesion); return mu.tanh()[0].cpu().numpy().astype(np.float32)
def eval_seed(split,block,episode):
    require(split in ("diagnostic","endpoint","causal","qa"),"Invalid evaluation split")
    return paired_seed(split,block,"panel",episode=int(episode))
def evaluation_lesion(condition,mode,decision):
    if mode=="condition_endpoint": return d2.endpoint_lesion(condition)
    if mode in ("intact","sham"): return ()
    if mode=="permanent_all_active_lesion": return d2.permanent_lesion()
    if mode=="inactive_edge_control": return d2.inactive_control_lesion()
    if mode=="window_9_40_all_active_lesion": return d2.window_9_40_lesion(decision)
    raise RuntimeError("Unknown evaluation mode")

def evaluate_actor(actor,condition,block,split,episodes,mode):
    cap=d2.DIAGNOSTIC_EPISODES if split=="diagnostic" else d2.ENDPOINT_EPISODES if split=="endpoint" else d2.CAUSAL_EPISODES if split=="causal" else 8
    require(1<=int(episodes)<=cap,"Evaluation count outside cap")
    rows=[]; env=NativeEnv(); before=state_digest(actor)
    with isolated_rng():
        try:
            for ep in range(int(episodes)):
                sd=eval_seed(split,block,ep); z,_=env.reset(seed=sd); ret=0.; h=hashlib.sha256()
                while True:
                    lesion=evaluation_lesion(condition,mode,env.t); a=deterministic_action(actor,z,lesion)
                    nz,r,t,tr,audit=env.step(a); h.update(np.asarray(z,np.float32).tobytes()+np.asarray(a,np.float32).tobytes()+np.float64(r).tobytes()+np.asarray(nz,np.float32).tobytes()); ret+=r; z=nz
                    if t or tr: break
                rows.append(dict(condition=condition,mode=mode,block=int(block),episode=ep,seed=sd,loss=-ret,return_value=ret,displacement=float(audit["package_displacement"]),cycles=int(env.t),fallen_count=int(audit["fallen_count"]),package_dropped=bool(audit["package_dropped"]),terminated=bool(t),truncated=bool(tr),trace_sha256=h.hexdigest()))
        finally: env.close()
    require(state_digest(actor)==before,"Evaluation altered actor weights"); return rows

def evaluate_no_action(block,episodes=d2.ENDPOINT_EPISODES,split="endpoint"):
    require(split in ("endpoint","qa"),"Invalid witness split"); cap=d2.ENDPOINT_EPISODES if split=="endpoint" else 8; require(1<=episodes<=cap,"Witness count outside cap")
    rows=[]; env=NativeEnv()
    try:
        for ep in range(episodes):
            sd=eval_seed(split,block,ep); z,_=env.reset(seed=sd); ret=0.; h=hashlib.sha256()
            while True:
                a=np.zeros(12,dtype=np.float32); nz,r,t,tr,audit=env.step(a); h.update(np.asarray(z,np.float32).tobytes()+a.tobytes()+np.float64(r).tobytes()+np.asarray(nz,np.float32).tobytes()); ret+=r; z=nz
                if t or tr: break
            rows.append(dict(condition="no_action",mode="no_action",block=block,episode=ep,seed=sd,loss=-ret,return_value=ret,displacement=float(audit["package_displacement"]),cycles=int(env.t),fallen_count=int(audit["fallen_count"]),package_dropped=bool(audit["package_dropped"]),terminated=bool(t),truncated=bool(tr),trace_sha256=h.hexdigest()))
    finally: env.close()
    return rows

def _train_ppo(cfg,out,budget,split,checkpoint_evaluator=None):
    hp=cfg["hp"]; batch=hp["rollout_steps"]; require(budget>0 and budget%batch==0,"Invalid budget")
    if split=="training": require(budget==d2.NATIVE_STEPS,"Scientific budget differs from freeze")
    else: require(split=="qa" and budget<=2048,"QA microfit budget exceeded")
    horizon=batch//N_ENVS; torch.manual_seed(int(cfg["initial_seed"])); actor=RoutingActor(d2.S6_MASK); critic=Value(); initial_actor=state_digest(actor); initial_A=module_digest(actor.A); initial_critic=module_digest(critic)
    params=list(actor.parameters())+list(critic.parameters()); opt=torch.optim.Adam(params,lr=hp["lr"],eps=hp["adam_eps"]); ledger=[]; curves=[]; envs=[TrainingEnv(cfg["block"],i,ledger,split) for i in range(N_ENVS)]; z=np.stack([e.reset()[0] for e in envs])
    torch.manual_seed(int(cfg["policy_sampling_seed"]) if split=="training" else paired_seed("qa",cfg["block"],"policy-sampling")); rng=np.random.default_rng(paired_seed(split,cfg["block"],"minibatches")); updates=0; t0=time.monotonic(); lesion=d2.training_lesion(cfg["condition"])
    try:
        for offset in range(0,budget,batch):
            obs=[]; pre=[]; logps=[]; values=[]; rewards=[]; dones=[]
            for _ in range(horizon):
                with torch.no_grad():
                    mu,ls=actor(torch.from_numpy(z),lesion); p=mu+ls.exp()*torch.randn_like(mu); a=p.tanh(); lp=log_squashed_gaussian(p,mu,ls); v=critic(torch.from_numpy(z))
                obs.append(z.copy()); pre.append(p.numpy()); logps.append(lp.numpy()); values.append(v.numpy()); rs=[]; ds=[]; nzs=[]
                for i,e in enumerate(envs):
                    nz,r,t,tr,_=e.step(a[i].numpy()); done=t or tr; rs.append(r); ds.append(done); nzs.append(e.reset()[0] if done else nz)
                rewards.append(rs); dones.append(ds); z=np.stack(nzs)
            with torch.no_grad(): last=critic(torch.from_numpy(z)).numpy()
            adv,target=gae_returns(np.asarray(rewards),np.asarray(values),np.asarray(dones),last,hp["gamma"],hp["gae_lambda"]); adv=adv.reshape(-1); adv=(adv-adv.mean())/(adv.std()+1e-8)
            ot=torch.from_numpy(np.asarray(obs).reshape(batch,93)); pt=torch.from_numpy(np.asarray(pre).reshape(batch,12)); old=torch.from_numpy(np.asarray(logps).reshape(-1)); at=torch.from_numpy(adv); rt=torch.from_numpy(target.reshape(-1)); als=[]; vls=[]
            for _ in range(hp["epochs"]):
                order=rng.permutation(batch)
                for j in range(0,batch,hp["minibatch"]):
                    ix=order[j:j+hp["minibatch"]]; mu,ls=actor(ot[ix],lesion); lp=log_squashed_gaussian(pt[ix],mu,ls); logr=lp-old[ix]; ratio=logr.exp(); al=-torch.min(ratio*at[ix],ratio.clamp(1-hp["clip"],1+hp["clip"])*at[ix]).mean(); vl=(critic(ot[ix])-rt[ix]).square().mean(); loss=al+hp["value_coef"]*vl; require(bool(torch.isfinite(loss)),"Nonfinite PPO objective"); opt.zero_grad(); loss.backward(); g=nn.utils.clip_grad_norm_(params,hp["max_grad_norm"]); require(bool(torch.isfinite(g)),"Nonfinite gradient"); opt.step(); updates+=1; als.append(float(al.detach())); vls.append(float(vl.detach()))
            steps=offset+batch
            if steps%32768==0 or steps==budget: curves.append(dict(native_steps=steps,optimizer_updates=updates,actor_loss=statistics.mean(als),value_loss=statistics.mean(vls),completed_episodes=len(ledger)))
            if split=="training" and steps in d2.CHECKPOINTS:
                torch.save(dict(actor_state=actor.state_dict(),critic_state=critic.state_dict(),optimizer_state=opt.state_dict(),configuration=cfg,native_steps=steps,condition=cfg["condition"]),out/f"model-{steps}.pt"); require(checkpoint_evaluator is not None,"Missing checkpoint evaluator"); checkpoint_evaluator(actor,steps)
        require(initial_actor!=state_digest(actor),"Actor did not update")
        res=dict(condition=cfg["condition"],block=cfg["block"],algorithm="PPO",environment_steps=budget,optimizer_updates=updates,sampling_parallelism=N_ENVS,split=split,initial_actor_digest=initial_actor,final_actor_digest=state_digest(actor),initial_message_A_digest=initial_A,final_message_A_digest=module_digest(actor.A),initial_critic_digest=initial_critic,final_critic_digest=module_digest(critic),seconds=time.monotonic()-t0,training_lesion=list(lesion))
        if cfg["condition"]=="S6-OFF-TRAIN": require(res["initial_message_A_digest"]==res["final_message_A_digest"],"OFF message A changed")
        return actor,critic,res,ledger,curves
    finally:
        for e in envs:e.close()

def fit(condition,block,out_root):
    execution_guard(); runtime(); cfg=config_for(condition,block); out=Path(out_root)/cfg["id"]; out.mkdir(parents=True,exist_ok=False); write_json(out/"STARTED.json",dict(status="STARTED",configuration=cfg,source_commit=os.environ["GITHUB_SHA"],execution_freeze_sha256=sha256_file(EXECUTION_FREEZE),**d2.BOUNDARY)); diagnostics={}
    def checkpoint(actor,steps):
        rows=evaluate_actor(actor,condition,block,"diagnostic",d2.DIAGNOSTIC_EPISODES,"condition_endpoint"); payload=dict(status="DIAGNOSTIC_CHECKPOINT_COMPLETE",condition=condition,block=block,native_steps=steps,summary=summarize(rows),episodes=rows,endpoint_lesion=list(d2.endpoint_lesion(condition))); write_json(out/f"DIAGNOSTIC_{steps}.json",payload); diagnostics[str(steps)]=payload["summary"]
    try:
        actor,critic,res,ledger,curves=_train_ppo(cfg,out,d2.NATIVE_STEPS,"training",checkpoint); ep=evaluate_actor(actor,condition,block,"endpoint",d2.ENDPOINT_EPISODES,"condition_endpoint"); torch.save(dict(actor_state=actor.state_dict(),critic_state=critic.state_dict(),configuration=cfg,native_steps=d2.NATIVE_STEPS),out/"model-final.pt"); write_json(out/"ENDPOINT.json",dict(status="ENDPOINT_COMPLETE",condition=condition,block=block,summary=summarize(ep),episodes=ep,endpoint_lesion=list(d2.endpoint_lesion(condition)))); write_json(out/"TRAINING_RESOURCES.json",res); write_json(out/"TRAINING_EPISODES.json",ledger); write_json(out/"LEARNING_CURVE.json",curves); require(set(map(int,diagnostics))==set(d2.CHECKPOINTS),"Missing diagnostic checkpoint"); files={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}; result=dict(status="FIT_COMPLETE",configuration=cfg,diagnostics=diagnostics,files=files,**d2.BOUNDARY); write_json(out/"COMPLETE.json",result); return result
    except Exception:
        write_json(out/"FAILED.json",dict(status="FAILED_RETAINED",configuration=cfg,error=traceback.format_exc(),replacement_seed=False,**d2.BOUNDARY)); raise

def find_fit_dir(root,condition,block):
    target=f"{condition}-b{int(block)}"; found=[]
    for marker in Path(root).rglob("COMPLETE.json"):
        try:p=read_json(marker)
        except Exception:continue
        if p.get("configuration",{}).get("id")==target and marker.parent not in found:found.append(marker.parent)
    require(len(found)==1,"Cannot resolve fit "+target); return found[0]
def verify_fit_dir(folder,cfg):
    c=read_json(Path(folder)/"COMPLETE.json"); require(c["status"]=="FIT_COMPLETE" and c["configuration"]==cfg,"Fit mismatch")
    for n,h in c["files"].items(): require(sha256_file(Path(folder)/n)==h,"Fit hash mismatch: "+n)
    s=read_json(Path(folder)/"STARTED.json"); require(s["configuration"]==cfg and s["source_commit"]==os.environ["GITHUB_SHA"],"Fit provenance mismatch")
    r=read_json(Path(folder)/"TRAINING_RESOURCES.json"); require(r["environment_steps"]==d2.NATIVE_STEPS and r["split"]=="training","Incomplete budget"); require(r["training_lesion"]==list(d2.training_lesion(cfg["condition"])),"Training lesion mismatch")
    e=read_json(Path(folder)/"ENDPOINT.json"); require(e["status"]=="ENDPOINT_COMPLETE" and len(e["episodes"])==d2.ENDPOINT_EPISODES,"Endpoint mismatch")
    for step in d2.CHECKPOINTS: require(len(read_json(Path(folder)/f"DIAGNOSTIC_{step}.json")["episodes"])==d2.DIAGNOSTIC_EPISODES,"Diagnostic mismatch")
    return c

def _bootstrap(values,label):
    vals=np.asarray(list(values),dtype=np.float64); require(vals.shape==(d2.BLOCKS,),"Bootstrap requires 8 blocks"); rng=np.random.default_rng(d2.seed("bootstrap",{"purpose":label})); reps=np.empty(d2.BOOTSTRAP_RESAMPLES)
    for i in range(d2.BOOTSTRAP_RESAMPLES): reps[i]=vals[rng.integers(0,len(vals),len(vals))].mean()
    return dict(mean=float(vals.mean()),median=float(np.median(vals)),ci95=[float(np.quantile(reps,.025)),float(np.quantile(reps,.975))],by_block=vals.tolist(),resamples=d2.BOOTSTRAP_RESAMPLES)
def _competence(model,witness):
    li=witness["loss"]-model["loss"]; dd=model["displacement"]-witness["displacement"]; return {"loss_improvement":li,"additional_displacement":dd,"pass":bool(li>d2.LOSS_MARGIN and dd>=d2.DISPLACEMENT_MARGIN)}

def aggregate_training(fits_root,out):
    execution_guard(); runtime(); out=Path(out); out.mkdir(parents=True,exist_ok=False); fits={}
    for c in d2.CONDITIONS:
        for b in range(d2.BLOCKS):
            cfg=config_for(c,b); f=find_fit_dir(fits_root,c,b); verify_fit_dir(f,cfg); fits[(c,b)]=f
    by=[]; las=[]; das=[]; pooled_on=[]; pooled_off=[]; diagnostics={str(s):[] for s in d2.CHECKPOINTS}; witnesses={}; competence={c:[] for c in d2.CONDITIONS}
    for b in range(d2.BLOCKS):
        onr=read_json(fits[("S6-ON",b)]/"TRAINING_RESOURCES.json"); offr=read_json(fits[("S6-OFF-TRAIN",b)]/"TRAINING_RESOURCES.json"); require(onr["initial_actor_digest"]==offr["initial_actor_digest"] and onr["initial_critic_digest"]==offr["initial_critic_digest"],"Paired initialization mismatch")
        one=read_json(fits[("S6-ON",b)]/"ENDPOINT.json"); offe=read_json(fits[("S6-OFF-TRAIN",b)]/"ENDPOINT.json"); ons,offs=one["summary"],offe["summary"]; pooled_on+=one["episodes"]; pooled_off+=offe["episodes"]; la=offs["loss"]-ons["loss"]; da=ons["displacement"]-offs["displacement"]; pr=d2.practical_advantage(la,da); las.append(la); das.append(da); by.append(dict(block=b,on=ons,off=offs,loss_advantage=la,displacement_advantage=da,practical_advantage=pr)); wr=evaluate_no_action(b); ws=summarize(wr); witnesses[str(b)]=ws; competence["S6-ON"].append(dict(block=b,**_competence(ons,ws))); competence["S6-OFF-TRAIN"].append(dict(block=b,**_competence(offs,ws)))
        for s in d2.CHECKPOINTS:
            od=read_json(fits[("S6-ON",b)]/f"DIAGNOSTIC_{s}.json")["summary"]; fd=read_json(fits[("S6-OFF-TRAIN",b)]/f"DIAGNOSTIC_{s}.json")["summary"]; diagnostics[str(s)].append(dict(block=b,on=od,off=fd,loss_advantage=fd["loss"]-od["loss"],displacement_advantage=od["displacement"]-fd["displacement"]))
    po,pf=summarize(pooled_on),summarize(pooled_off); ala=pf["loss"]-po["loss"]; ada=po["displacement"]-pf["displacement"]; ap=d2.practical_advantage(ala,ada); gate=d2.training_gate([x["practical_advantage"] for x in by],ap); result=dict(status="D2_TRAINING_EFFECT_COMPLETE",H_TRAIN_D2=gate["H_TRAIN_D2"],aggregate=dict(on=po,off=pf,loss_advantage=ala,displacement_advantage=ada,practical_advantage=ap),gate=gate,by_block=by,loss_advantage_bootstrap=_bootstrap(las,"d2-training-loss"),displacement_advantage_bootstrap=_bootstrap(das,"d2-training-displacement"),checkpoint_diagnostics=diagnostics,competence_vs_no_action=competence,witness_by_block=witnesses,fits_verified=d2.FITS,**d2.BOUNDARY); write_json(out/"TRAINING_EFFECT.json",result); return result

def load_actor(fit_dir,condition):
    c=read_json(Path(fit_dir)/"COMPLETE.json")["configuration"]; cfg=config_for(condition,c["block"]); p=torch.load(Path(fit_dir)/"model-final.pt",map_location="cpu"); require(p["configuration"]==cfg,"Model config mismatch"); a=RoutingActor(d2.S6_MASK); a.load_state_dict(p["actor_state"]); a.eval(); return a

def online_block(block,fits_root,out):
    execution_guard(); runtime(); cfg=config_for("S6-ON",block); f=find_fit_dir(fits_root,"S6-ON",block); verify_fit_dir(f,cfg); actor=load_actor(f,"S6-ON"); out=Path(out); out.mkdir(parents=True,exist_ok=False); modes=("intact","permanent_all_active_lesion","sham","inactive_edge_control","window_9_40_all_active_lesion"); rows={m:evaluate_actor(actor,"S6-ON",block,"causal",d2.CAUSAL_EPISODES,m) for m in modes}; ih=[x["trace_sha256"] for x in rows["intact"]]; sham=ih==[x["trace_sha256"] for x in rows["sham"]]; inactive=ih==[x["trace_sha256"] for x in rows["inactive_edge_control"]]; require(sham and inactive,"Causal control diverged"); s={k:summarize(v) for k,v in rows.items()}; lh=s["permanent_all_active_lesion"]["loss"]-s["intact"]["loss"]; dh=s["intact"]["displacement"]-s["permanent_all_active_lesion"]["displacement"]; wlh=s["window_9_40_all_active_lesion"]["loss"]-s["intact"]["loss"]; wdh=s["intact"]["displacement"]-s["window_9_40_all_active_lesion"]["displacement"]; result=dict(status="D2_ONLINE_BLOCK_COMPLETE",block=int(block),summaries=s,permanent=dict(loss_harm=lh,displacement_harm=dh,practical_harm=d2.practical_harm(lh,dh)),window_9_40=dict(loss_harm=wlh,displacement_harm=wdh,practical_harm=d2.practical_harm(wlh,wdh),primary_gate=False),controls=dict(sham_exact=sham,inactive_edge_exact=inactive),episodes=rows,**d2.BOUNDARY); write_json(out/"ONLINE_BLOCK.json",result); return result

def find_online_dir(root,block):
    found=[]
    for m in Path(root).rglob("ONLINE_BLOCK.json"):
        try:p=read_json(m)
        except Exception:continue
        if p.get("block")==int(block) and m.parent not in found:found.append(m.parent)
    require(len(found)==1,"Cannot resolve online block"); return found[0]
def aggregate_online(online_root,out):
    execution_guard(); out=Path(out); out.mkdir(parents=True,exist_ok=False); blocks=[]
    for b in range(d2.BLOCKS):
        r=read_json(find_online_dir(online_root,b)/"ONLINE_BLOCK.json"); require(r["status"]=="D2_ONLINE_BLOCK_COMPLETE","Incomplete online block"); blocks.append(r)
    controls=all(x["controls"]["sham_exact"] and x["controls"]["inactive_edge_exact"] for x in blocks); intact=[e for b in blocks for e in b["episodes"]["intact"]]; perm=[e for b in blocks for e in b["episodes"]["permanent_all_active_lesion"]]; win=[e for b in blocks for e in b["episodes"]["window_9_40_all_active_lesion"]]; si,sp,sw=summarize(intact),summarize(perm),summarize(win); lh=sp["loss"]-si["loss"]; dh=si["displacement"]-sp["displacement"]; pr=d2.practical_harm(lh,dh); gate=d2.online_gate([b["permanent"]["practical_harm"] for b in blocks],pr,controls); wlh=sw["loss"]-si["loss"]; wdh=si["displacement"]-sw["displacement"]; result=dict(status="D2_ONLINE_EFFECT_COMPLETE",H_ONLINE_D2=gate["H_ONLINE_D2"],aggregate=dict(intact=si,permanent_lesion=sp,loss_harm=lh,displacement_harm=dh,practical_harm=pr),gate=gate,by_block=[dict(block=b["block"],permanent=b["permanent"],controls=b["controls"]) for b in blocks],loss_harm_bootstrap=_bootstrap([b["permanent"]["loss_harm"] for b in blocks],"d2-online-loss"),displacement_harm_bootstrap=_bootstrap([b["permanent"]["displacement_harm"] for b in blocks],"d2-online-disp"),window_9_40_replication=dict(intact=si,lesion=sw,loss_harm=wlh,displacement_harm=wdh,practical_harm=d2.practical_harm(wlh,wdh),blocks_with_practical_harm=sum(bool(b["window_9_40"]["practical_harm"]) for b in blocks),primary_gate=False),controls_pass=controls,**d2.BOUNDARY); write_json(out/"ONLINE_EFFECT.json",result); return result

def final_aggregate(training_file,online_file,out):
    execution_guard(); out=Path(out); out.mkdir(parents=True,exist_ok=False); t=read_json(training_file); o=read_json(online_file); integrity=t.get("status")=="D2_TRAINING_EFFECT_COMPLETE" and t.get("fits_verified")==d2.FITS and o.get("status")=="D2_ONLINE_EFFECT_COMPLETE" and o.get("controls_pass") is True; adjud=d2.adjudicate(bool(t.get("H_TRAIN_D2")),bool(o.get("H_ONLINE_D2")),bool(integrity)); result=dict(status="B1SC_D2_V1_0_DEVELOPMENT_COMPLETE_FOR_REVIEW",mechanistic_adjudication=adjud,H_TRAIN_D2=bool(t.get("H_TRAIN_D2")),H_ONLINE_D2=bool(o.get("H_ONLINE_D2")),integrity_pass=bool(integrity),training_effect=t,online_effect=o,automatic_b1e_progression=False,no_polar_specificity_claim=True,**d2.BOUNDARY); write_json(out/"SUMMARY.json",result); (out/"COMPLETION_REPORT_ES.md").write_text(f"# B1-SC-D2 v1.0 — cierre mecanístico\n\n**B1SC_D2_V1_0_DEVELOPMENT_COMPLETE_FOR_REVIEW**\n\n- H_TRAIN-D2: `{result['H_TRAIN_D2']}`\n- H_ONLINE-D2: `{result['H_ONLINE_D2']}`\n- Adjudicación: `{adjud}`\n\nNo habilita B1-E automáticamente ni demuestra especificidad polar o consciencia.\n",encoding="utf-8"); write_json(out/"RESULTS_MANIFEST.json",dict(status="D2_RESULTS_PREPARED_FOR_REVIEW",files={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()},automatic_publication=False,**d2.BOUNDARY)); return result

def qa_microfit(out):
    runtime(); out=Path(out); out.mkdir(parents=True,exist_ok=False); resources={}; endpoints={}
    for c in d2.CONDITIONS:
        cfg=config_for(c,0); cfg=dict(cfg); cfg["initial_seed"]=paired_seed("qa",0,"weights"); cfg["policy_sampling_seed"]=paired_seed("qa",0,"policy-sampling"); folder=out/c; folder.mkdir(); actor,critic,r,ledger,curves=_train_ppo(cfg,folder,512,"qa"); rows=evaluate_actor(actor,c,0,"qa",2,"condition_endpoint"); resources[c]=r; endpoints[c]=summarize(rows)
    require(resources["S6-ON"]["initial_actor_digest"]==resources["S6-OFF-TRAIN"]["initial_actor_digest"],"QA initial actor mismatch"); require(resources["S6-OFF-TRAIN"]["initial_message_A_digest"]==resources["S6-OFF-TRAIN"]["final_message_A_digest"],"QA OFF A changed"); result=dict(status="D2_EXECUTION_QA_MICROFIT_PASS_NOT_SCIENTIFIC",native_steps_per_condition=512,conditions=list(d2.CONDITIONS),paired_initial_actor=True,off_message_A_unchanged=True,endpoint_smoke=endpoints,scientific_registry_consumed=False,scientific_training=False,scientific_evaluation=False,**d2.BOUNDARY); write_json(out/"QA_MICROFIT.json",result); return result
