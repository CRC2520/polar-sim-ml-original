"""R11b feasibility-audited pilot and confirmatory experiment functions."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import numpy as np
from r10_discriminating.experiments import _train_agent
from .config import E2,E3,E4,E5,GLOBAL_REQUIRED
from .core import R11Agent
from .environments import (RelationalEnv, OscillatoryReservoirV2,
                           MaintenanceQueueV2, AttributionTransferEnvV2)
from .experiments import PILOT_CONFIGS, make_r11, stable_seed, experiment2, experiment3

PILOT_B_SEEDS=tuple(range(972001,972005))
FINAL_B_SEEDS=tuple(range(973001,973013))
TRANSFER_ENVS=(("oscillatory_reservoir_v2",OscillatoryReservoirV2),
               ("maintenance_queue_v2",MaintenanceQueueV2))


def _mean(values):
    return float(np.mean(np.asarray(values,float)))


def constant_policy_audit(env_cls,seed,action,steps=320):
    env=env_cls(); obs=env.reset(seed); rewards=[]; alive=[]
    for _ in range(steps):
        obs,r,done,info=env.step(action)
        rewards.append(float(r)); alive.append(float(info["alive"]))
    return _mean(rewards),_mean(alive)


def feasibility_audit():
    rows={}
    passed=True
    for name,cls in TRANSFER_ENVS:
        per_action={}
        for action in range(4):
            vals=[constant_policy_audit(cls,stable_seed(970000+i,name),action)
                  for i in range(100)]
            per_action[str(action)]=dict(mean_reward=_mean([x[0] for x in vals]),
                                         mean_alive=_mean([x[1] for x in vals]))
        feasible=[v for v in per_action.values()
                  if v["mean_alive"]>=.95 and v["mean_reward"]>=.50]
        ok=bool(feasible) and per_action["0"]["mean_reward"]<=.10
        passed &= ok
        rows[name]=dict(pass_family=ok,actions=per_action)
    return dict(pass_audit=bool(passed),families=rows,
                scope="Evaluator-only development tapes; never supplied to an agent.")


def run_r11(wrapper,env_cls,seed,steps=320,variant="full",adapt_window=64,
            metric_start=64,update_adapter=True):
    env=env_cls(); obs=env.reset(seed); wrapper.reset_episode(obs,variant)
    rewards=[]; alive=[]
    for t in range(steps):
        p=wrapper.prepare(obs,variant,adaptation_probe=t<adapt_window)
        a=int(p["action"])
        nxt,r,done,info=env.step(a)
        wrapper.complete(obs,a,nxt,r,p,learn_base=False,terminal=done,
                         update_adapter=update_adapter)
        if t>=metric_start:
            rewards.append(float(r)); alive.append(float(info["alive"]))
        obs=nxt
    return dict(return_mean=_mean(rewards),alive_fraction=_mean(alive))


def run_r9(base,env_cls,seed,steps=320,metric_start=64):
    env=env_cls(); obs=env.reset(seed); a=copy.deepcopy(base); a.reset_episode(obs)
    rewards=[]; alive=[]
    for t in range(steps):
        p=a.prepare(obs); action=int(p["action"])
        nxt,r,done,info=env.step(action)
        a.complete_transition(obs,action,nxt,r,p,learn=False,terminal=done)
        if t>=metric_start:
            rewards.append(float(r)); alive.append(float(info["alive"]))
        obs=nxt
    return dict(return_mean=_mean(rewards),alive_fraction=_mean(alive))


def experiment4b(base,seed,cfg):
    out={}; ok=True
    for name,cls in TRANSFER_ENVS:
        s=stable_seed(seed,"R11B-E4-"+name)
        baseline=run_r9(base,cls,s)
        r11=run_r11(make_r11(base,cfg,False),cls,s)
        dR=r11["return_mean"]-baseline["return_mean"]
        dA=r11["alive_fraction"]-baseline["alive_fraction"]
        domain_ok=(r11["alive_fraction"]>=E4["alive"] and
                   r11["return_mean"]>=E4["reward"] and
                   dR>=E4["reward_improvement"] and
                   dA>=E4["alive_improvement"])
        ok &= domain_ok
        out[name]=dict(pass_domain=bool(domain_ok),r11=r11,baseline=baseline,
                       reward_improvement=dR,alive_improvement=dA)
    return dict(pass_seed=bool(ok),domains=out)


def _agebin(age):
    return 0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3


def _balanced(y,p,classes):
    y=np.asarray(y); p=np.asarray(p); vals=[]
    for c in classes:
        m=y==c
        if np.any(m): vals.append(float(np.mean(p[m]==c)))
    return float(np.mean(vals)) if vals else 0.


def _auc(score,label):
    s=np.asarray(score,float); y=np.asarray(label,int)
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return .5
    return float(np.mean(pos[:,None]>neg[None,:])+
                 .5*np.mean(pos[:,None]==neg[None,:]))


def _first_half(base,seed,cfg):
    env=AttributionTransferEnvV2(); obs=env.reset(seed)
    w=make_r11(base,cfg,False); w.reset_episode(obs,"full")
    errors=[]
    for t in range(320):
        p=w.prepare(obs,"full",adaptation_probe=t<96)
        a=int(p["action"])
        nxt,r,done,info=env.step(a)
        err=float(np.mean(np.abs(np.r_[nxt,r]-p["predictions"][a])))
        errors.append(err)
        w.complete(obs,a,nxt,r,p,learn_base=False,terminal=done,update_adapter=True)
        obs=nxt
    return w,env,obs,np.asarray(errors,float)


def _second_half(wrapper,env,obs,variant,error_threshold,collect=False):
    w=copy.deepcopy(wrapper); e=copy.deepcopy(env); current=np.asarray(obs,float).copy()
    if variant=="noMemory":
        w.base.workspace.set_modes(no_memory=True)
    rewards=[]; alive=[]; rows=[]
    for t in range(320,640):
        p=w.prepare(current,variant,adaptation_probe=False)
        cf=[]
        if collect:
            for candidate in range(4):
                ee=copy.deepcopy(e); cf.append(float(ee.step(candidate)[1]))
        a=int(p["action"])
        nxt,r,done,info=e.step(a)
        current_error=float(np.mean(np.abs(np.r_[nxt,r]-p["predictions"][a])))
        prospective_uncertainty=1.-float(p["confidence"])
        w.complete(current,a,nxt,r,p,learn_base=False,terminal=done,update_adapter=False)
        rewards.append(float(r)); alive.append(float(info["alive"]))
        if collect:
            rows.append(dict(source_true=int(info["source"]),source_pred=int(w.source_class),
                age_true=int(info["agebin"]),age_pred=int(_agebin(w.world_age)),
                uncertain=float(prospective_uncertainty),
                high_error=int(current_error>error_threshold),
                prediction_error=current_error,
                cf_correct=int(np.argmax(p["counterfactual_reward"])==np.argmax(cf))))
        current=nxt
    return dict(return_mean=_mean(rewards),alive_fraction=_mean(alive),rows=rows)


def experiment5b(base,seed,cfg):
    s=stable_seed(seed,"R11B-E5")
    w,env,obs,first_errors=_first_half(base,s,cfg)
    threshold=float(np.median(first_errors))
    full=_second_half(w,env,obs,"full",threshold,collect=True)
    nm=_second_half(w,env,obs,"noMemory",threshold)
    pc=_second_half(w,env,obs,"permuted_content",threshold)
    nc=_second_half(w,env,obs,"noCross",threshold)
    rows=full["rows"]
    ts=np.array([x["source_true"] for x in rows]); ps=np.array([x["source_pred"] for x in rows])
    ta=np.array([x["age_true"] for x in rows]); pa=np.array([x["age_pred"] for x in rows])
    uncertain=np.array([x["uncertain"] for x in rows]); high=np.array([x["high_error"] for x in rows])
    selfworld_mask=ts!=2
    selfworld=_balanced((ts[selfworld_mask]==0).astype(int),
                        (ps[selfworld_mask]==0).astype(int),(0,1)) if np.any(selfworld_mask) else 0.
    source=_balanced(ts,ps,(0,1,2)); time=_balanced(ta,pa,(0,1,2,3))
    meta=_auc(uncertain,high); cf=_mean([x["cf_correct"] for x in rows])
    valid=(full["alive_fraction"]>=E5["alive"] and all(np.any(ts==c) for c in (0,1,2)))
    base_ok=(valid and selfworld>=E5["self_world"] and source>=E5["source"] and
             time>=E5["time"] and meta>=E5["auc"] and cf>=E5["counterfactual"])
    memory_drop=full["return_mean"]-nm["return_mean"]
    content_drop=full["return_mean"]-pc["return_mean"]
    cross_drop=full["return_mean"]-nc["return_mean"]
    damages=[full["alive_fraction"]-x["alive_fraction"] for x in (nm,pc,nc)]
    lesion_ok=(memory_drop>=E5["memory_drop"] and content_drop>=E5["content_drop"] and
               cross_drop>=E5["cross_drop"] and
               max(abs(x) for x in damages)<=E5["max_alive_damage"])
    return dict(pass_seed=bool(base_ok and lesion_ok),valid=bool(valid),
                base_conjunction=bool(base_ok),lesion_conjunction=bool(lesion_ok),
                self_world=selfworld,source=source,time=time,auc=meta,counterfactual=cf,
                source_counts={str(c):int(np.sum(ts==c)) for c in (0,1,2)},
                full={k:v for k,v in full.items() if k!="rows"},
                noMemory={k:v for k,v in nm.items() if k!="rows"},
                permuted_content={k:v for k,v in pc.items() if k!="rows"},
                noCross={k:v for k,v in nc.items() if k!="rows"},
                memory_drop=memory_drop,content_drop=content_drop,cross_drop=cross_drop,
                alive_damages=damages)


def pilot_score_b(base,seed,cfg):
    e4=experiment4b(base,seed,cfg); e5=experiment5b(base,seed,cfg)
    r4=_mean([x["r11"]["return_mean"] for x in e4["domains"].values()])
    alive_min=min([x["r11"]["alive_fraction"] for x in e4["domains"].values()]+
                  [e5["full"]["alive_fraction"]])
    valid=alive_min>=.60
    score=(r4/.35+e5["counterfactual"]/.45) if valid else -1e9
    return dict(valid=bool(valid),score=float(score),alive_min=float(alive_min),E4=e4,E5=e5)


def run_pilot_b(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    audit=feasibility_audit()
    (output/"FEASIBILITY_AUDIT.json").write_text(json.dumps(audit,indent=2))
    if not audit["pass_audit"]:
        raise RuntimeError("R11b transfer feasibility prerequisite failed")
    bases={s:_train_agent(s) for s in PILOT_B_SEEDS}
    ranking=[]
    for cfg in PILOT_CONFIGS:
        rows=[pilot_score_b(bases[s],s,cfg) for s in PILOT_B_SEEDS]
        valid=all(x["valid"] for x in rows)
        score=float(np.median([x["score"] for x in rows])) if valid else -1e9
        ranking.append(dict(config=cfg,valid=bool(valid),median_score=score,rows=rows))
    ranking.sort(key=lambda x:(x["median_score"],x["config"]["name"]),reverse=True)
    if not ranking[0]["valid"]:
        selected=None
    else:
        selected=ranking[0]["config"]
    result=dict(protocol="PREREG_R11B.md",pilot_only=True,seeds=list(PILOT_B_SEEDS),
                feasibility=audit,selected=selected,ranking=ranking)
    (output/"PILOT_R11B.json").write_text(json.dumps(result,indent=2))
    lines=["# R11b pilot","","Feasibility audit: **%s**"%("PASS" if audit["pass_audit"] else "FAIL"),"",
           "Selected: **%s**"%(selected["name"] if selected else "NONE"),""]
    lines += [f"- {x['config']['name']}: valid={x['valid']}, score={x['median_score']:.6f}" for x in ranking]
    (output/"PILOT_REPORT.md").write_text("\n".join(lines)+"\n")
    return result
