"""R11 pilot/final experiment logic. Final execution requires a frozen chosen config."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import numpy as np
from r9_completion.agent import NativeAgent
from r10_discriminating.experiments import _train_agent, CyclicBufferEnv, RepairQueueEnv
from .core import R11Agent
from .environments import RelationalEnv, AttributionTransferEnv
from .config import *

PILOT_CONFIGS=(
 dict(name='A',forgetting=.985,l1=.0001,min_samples=5,reliability_margin=.002,adapter_weight=.60,risk_floor=.25),
 dict(name='B',forgetting=.995,l1=.0001,min_samples=5,reliability_margin=0.,adapter_weight=.75,risk_floor=.25),
 dict(name='C',forgetting=.97,l1=.0005,min_samples=3,reliability_margin=.002,adapter_weight=.75,risk_floor=.20),
 dict(name='D',forgetting=.985,l1=0.,min_samples=3,reliability_margin=0.,adapter_weight=.60,risk_floor=.20),
 dict(name='E',forgetting=.995,l1=.0005,min_samples=8,reliability_margin=.005,adapter_weight=.60,risk_floor=.30),
 dict(name='F',forgetting=.97,l1=.0001,min_samples=5,reliability_margin=.005,adapter_weight=.40,risk_floor=.25),
 dict(name='G',forgetting=.985,l1=.0005,min_samples=3,reliability_margin=0.,adapter_weight=.75,risk_floor=.30),
 dict(name='H',forgetting=.995,l1=0.,min_samples=8,reliability_margin=.002,adapter_weight=.40,risk_floor=.20),
)

def make_r11(base,cfg,generic=False):
    kw={k:v for k,v in cfg.items() if k!='name'}
    return R11Agent(base,generic=generic,**kw)

def stable_seed(seed,label):
    return int.from_bytes(hashlib.sha256(f'R11|{seed}|{label}'.encode()).digest()[:8],'little')&0xffffffff

def run_wrapper(wrapper,env_cls,seed,steps=320,variant='full',adapt_window=0,metric_start=0,update_adapter=True,
                collect_native=False):
    env=env_cls() if callable(env_cls) else env_cls
    obs=env.reset(seed); wrapper.reset_episode(obs,variant)
    rows=[]; rewards=[]; alive=[]
    for t in range(steps):
        pol=wrapper.prepare(obs,variant,adaptation_probe=(t<adapt_window))
        action=int(pol['action'])
        # evaluator counterfactuals never enter learning/policy
        cf=None
        if collect_native:
            cf=[]
            for a in range(4):
                ee=copy.deepcopy(env)
                cf.append(float(ee.step(a)[1]))
        nxt,r,done,info=env.step(action)
        wrapper.complete(obs,action,nxt,r,pol,learn_base=False,terminal=done,update_adapter=update_adapter)
        if t>=metric_start:
            rewards.append(float(r)); alive.append(float(info['alive']))
        if collect_native:
            rows.append(dict(t=t,source_true=int(info.get('source',2)),age_true=int(info.get('agebin',3)),
                source_pred=int(wrapper.source_class),age_pred=int(wrapper.world_age),
                confidence=float(wrapper.confidence),
                prediction_error=float(np.mean(np.abs(np.r_[nxt,r]-pol['predictions'][action]))),
                cf_correct=int(np.argmax(pol['counterfactual_reward'])==np.argmax(cf)),
                reward=float(r),alive=float(info['alive'])))
        obs=nxt
    return dict(return_mean=float(np.mean(rewards)),alive_fraction=float(np.mean(alive)),rows=rows,
                adapter_state=wrapper.adapter.state())

def run_base(base,env_cls,seed,steps=320,variant='full',metric_start=0):
    env=env_cls() if callable(env_cls) else env_cls
    a=copy.deepcopy(base); obs=env.reset(seed); a.reset_episode(obs,variant)
    rewards=[]; alive=[]
    for t in range(steps):
        p=a.prepare(obs,variant); action=int(p['action'])
        nxt,r,done,info=env.step(action)
        a.complete_transition(obs,action,nxt,r,p,learn=False,terminal=done)
        if t>=metric_start:
            rewards.append(float(r)); alive.append(float(info['alive']))
        obs=nxt
    return dict(return_mean=float(np.mean(rewards)),alive_fraction=float(np.mean(alive)))

def train_relational_adapter(wrapper,seed):
    # own-policy acquisition; identical episode/tape budget for relational and isomorphic agents.
    for ep in range(4):
        env_cls=(lambda:RelationalEnv(delay=bool(ep%2)))
        run_wrapper(wrapper,env_cls,stable_seed(seed,f'rel-train-{ep}'),steps=320,
                    adapt_window=64,metric_start=0,update_adapter=True)
    return wrapper

def experiment2(base,seed,cfg):
    rel=train_relational_adapter(make_r11(base,cfg,False),seed)
    gen=train_relational_adapter(make_r11(base,cfg,True),seed)
    out={}; ok=True
    for name,delay in [('relational',False),('relational_delay',True)]:
        ss=stable_seed(seed,'E2-'+name)
        rr=run_wrapper(copy.deepcopy(rel),lambda:RelationalEnv(delay=delay),ss,update_adapter=False)
        gg=run_wrapper(copy.deepcopy(gen),lambda:RelationalEnv(delay=delay),ss,update_adapter=False)
        dR=rr['return_mean']-gg['return_mean']; dA=rr['alive_fraction']-gg['alive_fraction']
        passed=dR>=E2['reward_margin'] and dA>=E2['alive_margin']
        ok &= passed
        out[name]=dict(relational=rr,generic=gg,reward_diff=dR,alive_diff=dA,pass_domain=bool(passed))
    # anti-overfit guard: online adaptation is permitted but no target-family pretraining.
    from r9_completion.environments import ScalarEnvironment
    ss=stable_seed(seed,'E2-ecology-guard')
    rr=run_wrapper(make_r11(base,cfg,False),lambda:ScalarEnvironment('ecology_delay9'),ss,
                   adapt_window=64,metric_start=64,update_adapter=True)
    gg=run_wrapper(make_r11(base,cfg,True),lambda:ScalarEnvironment('ecology_delay9'),ss,
                   adapt_window=64,metric_start=64,update_adapter=True)
    guard=rr['return_mean']-gg['return_mean']
    ok &= guard>=E2['ecology_guard']
    return dict(pass_seed=bool(ok),domains=out,ecology_guard_diff=guard,
                parameter_count_rel=rel.parameter_count(),parameter_count_gen=gen.parameter_count())

def experiment3(seed):
    rng=np.random.default_rng(stable_seed(seed,'E3'))
    ntr,nte=6000,3000
    X=rng.normal(size=(ntr+nte,6))
    true_gate=(X[:,0]-.8*X[:,2]+.55*X[:,4]-.35*X[:,5]>0).astype(int)
    local=.35*X[:,1]-.22*X[:,3]
    cross=.72*np.tanh(.8*X[:,0]-X[:,2]+.4*X[:,4])
    noise=rng.normal(0,.07,len(X))
    y=local+true_gate*cross+noise
    off=local; on=local+cross
    utility=((y-on)**2<(y-off)**2).astype(int)
    A=np.c_[np.ones(ntr),X[:ntr]]
    Y=utility[:ntr].astype(float)
    reg=.2*np.eye(7); reg[0,0]=0
    w=np.linalg.solve(A.T@A+reg,A.T@Y)
    Xt=X[ntr:]; yt=y[ntr:]
    score=np.c_[np.ones(nte),Xt]@w
    gate=(score>=.5)
    pred=np.where(gate,on[ntr:],off[ntr:])
    perm=Xt.copy(); perm[:,[0,2,4,5]]=perm[:,[2,4,5,0]]
    pg=(np.c_[np.ones(nte),perm]@w>=.5)
    pp=np.where(pg,on[ntr:],off[ntr:])
    label=utility[ntr:]
    def ba(y,p):
        vals=[np.mean(p[y==c]==c) for c in (0,1) if np.any(y==c)]
        return float(np.mean(vals))
    mse=lambda z:float(np.mean((yt-z)**2))
    acc=ba(label,gate.astype(int)); best=min(mse(off[ntr:]),mse(on[ntr:]))
    imp=best-mse(pred); pd=mse(pp)-mse(pred)
    passed=acc>=E3['balanced_accuracy'] and imp>=E3['mse_gain'] and pd>=E3['perm_damage']
    return dict(pass_seed=bool(passed),balanced_accuracy=acc,adaptive_mse=mse(pred),
                off_mse=mse(off[ntr:]),on_mse=mse(on[ntr:]),improvement=imp,
                permuted_mse=mse(pp),permutation_damage=pd)

def experiment4(base,seed,cfg):
    out={}; ok=True
    for name,envcls in [('cyclic_buffer',CyclicBufferEnv),('repair_queue',RepairQueueEnv)]:
        ss=stable_seed(seed,'E4-'+name)
        baseline=run_base(base,envcls,ss,metric_start=64)
        r11=run_wrapper(make_r11(base,cfg,False),envcls,ss,adapt_window=64,metric_start=64,update_adapter=True)
        dR=r11['return_mean']-baseline['return_mean']; dA=r11['alive_fraction']-baseline['alive_fraction']
        passed=(r11['alive_fraction']>=E4['alive'] and r11['return_mean']>=E4['reward'] and
                dR>=E4['reward_improvement'] and dA>=E4['alive_improvement'])
        ok &= passed
        out[name]=dict(r11=r11,baseline=baseline,reward_improvement=dR,alive_improvement=dA,
                       pass_domain=bool(passed))
    return dict(pass_seed=bool(ok),domains=out)

def _agebin(age):
    return 0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3

def _balanced(y,p,classes):
    y=np.asarray(y); p=np.asarray(p); vals=[]
    for c in classes:
        m=y==c
        if m.any(): vals.append(np.mean(p[m]==c))
    return float(np.mean(vals)) if vals else 0.

def _auc(score,label):
    s=np.asarray(score,float); y=np.asarray(label,int); pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0:return .5
    return float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))

def e5_rollout(base,seed,cfg,variant='full'):
    w=make_r11(base,cfg,False)
    res=run_wrapper(w,AttributionTransferEnv,seed,steps=640,variant=variant,
                    adapt_window=96,metric_start=320,update_adapter=True,collect_native=True)
    rows=res['rows']; half=rows[320:]
    true_src=np.array([x['source_true'] for x in half]); pred_src=np.array([x['source_pred'] for x in half])
    true_age=np.array([x['age_true'] for x in half]); pred_age=np.array([_agebin(x['age_pred']) for x in half])
    err=np.array([x['prediction_error'] for x in rows]); conf=np.array([x['confidence'] for x in rows])
    threshold=float(np.median(err[:320]))
    meta=_auc(1-conf[320:],(err[320:]>threshold).astype(int))
    cf=float(np.mean([x['cf_correct'] for x in half]))
    valid=res['alive_fraction']>=E5['alive'] and all(np.any(true_src==c) for c in (0,1,2))
    sw=true_src!=2
    selfworld=_balanced((true_src[sw]==0).astype(int),(pred_src[sw]==0).astype(int),(0,1)) if sw.any() else 0.
    source=_balanced(true_src,pred_src,(0,1,2)); time=_balanced(true_age,pred_age,(0,1,2,3))
    return dict(return_mean=res['return_mean'],alive_fraction=res['alive_fraction'],valid=bool(valid),
                self_world=selfworld,source=source,time=time,auc=meta,counterfactual=cf,
                source_counts={str(c):int(np.sum(true_src==c)) for c in (0,1,2)})

def experiment5(base,seed,cfg):
    ss=stable_seed(seed,'E5')
    full=e5_rollout(base,ss,cfg,'full')
    nm=e5_rollout(base,ss,cfg,'noMemory')
    pc=e5_rollout(base,ss,cfg,'permuted_content')
    nc=e5_rollout(base,ss,cfg,'noCross')
    md=full['return_mean']-nm['return_mean']; cd=full['return_mean']-pc['return_mean']; xd=full['return_mean']-nc['return_mean']
    damages=[full['alive_fraction']-x['alive_fraction'] for x in (nm,pc,nc)]
    base_ok=(full['valid'] and full['self_world']>=E5['self_world'] and full['source']>=E5['source'] and
             full['time']>=E5['time'] and full['auc']>=E5['auc'] and full['counterfactual']>=E5['counterfactual'])
    lesion=(md>=E5['memory_drop'] and cd>=E5['content_drop'] and xd>=E5['cross_drop'] and
            max(abs(x) for x in damages)<=E5['max_alive_damage'])
    return dict(pass_seed=bool(base_ok and lesion),base_conjunction=bool(base_ok),lesion_conjunction=bool(lesion),
                full=full,noMemory=nm,permuted_content=pc,noCross=nc,
                memory_drop=md,content_drop=cd,cross_drop=xd,alive_damages=damages)

def pilot_score(base,seed,cfg):
    e4=experiment4(base,seed,cfg); e5=experiment5(base,seed,cfg)
    r4=np.mean([v['r11']['return_mean'] for v in e4['domains'].values()])
    alive_min=min([v['r11']['alive_fraction'] for v in e4['domains'].values()]+[e5['full']['alive_fraction']])
    score=r4/.35+e5['full']['counterfactual']/.45
    return dict(score=float(score),alive_min=float(alive_min),E4=e4,E5=e5)

def run_pilot(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    bases={s:_train_agent(s) for s in PILOT_SEEDS}
    configs=[]
    for cfg in PILOT_CONFIGS:
        rows=[pilot_score(bases[s],s,cfg) for s in PILOT_SEEDS]
        valid=all(r['alive_min']>=.60 for r in rows)
        score=float(np.median([r['score'] for r in rows])) if valid else -1e9
        configs.append(dict(config=cfg,valid=valid,median_score=score,rows=rows))
    configs.sort(key=lambda x:(x['median_score'],x['config']['name']),reverse=True)
    result=dict(pilot_only=True,seeds=list(PILOT_SEEDS),ranking=configs,selected=configs[0]['config'])
    (output/'PILOT_R11.json').write_text(json.dumps(result,indent=2))
    (output/'PILOT_REPORT.md').write_text('# R11 pilot\n\nSelected: **%s**\n\n%s\n'%
        (result['selected']['name'],'\n'.join(f"- {x['config']['name']}: valid={x['valid']}, score={x['median_score']:.6f}" for x in configs)))
    return result
