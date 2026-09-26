from __future__ import annotations
import copy, hashlib, json, math
from pathlib import Path
import numpy as np
from r10_discriminating.experiments import (_train_agent,_ridge_classifier,_predict_classifier,
    _balanced_accuracy,_auc,ACTIONS)
SEEDS=tuple(range(964001,964013))
STEPS=640
REQ=9

class AttributionEnvV2:
    """Viability-preserving periodic source-attribution challenge."""
    def reset(self,seed):
        self.rng=np.random.default_rng(seed); self.t=0; self.alive=True
        self.reserve=.82; self.health=.86; self.demand=.50; self.last_world=-99
        phase=int(self.rng.integers(0,7))
        idx=np.arange(STEPS)
        self.shock_flag=((idx+phase)%7==0)|((idx+phase)%19==0)
        signs=np.where(((idx+phase)//7)%2==0,1.,-1.)
        self.shock_mag=.035*signs*self.shock_flag
        self.demand_tape=np.clip(.50+.25*np.sin(2*np.pi*idx/48.+phase*.1)+
                                 .06*np.sin(2*np.pi*idx/13.),.12,.88)
        return self.obs()
    def obs(self): return np.clip([self.reserve,self.health,self.demand],0,1)
    def step(self,action):
        flow=ACTIONS[int(action)]
        shock=float(self.shock_mag[self.t])
        old_d=self.demand; self.demand=float(self.demand_tape[self.t])
        self_effect=np.array([-.22*flow, .055*(flow/.12)-.018*(flow/.12)**2, 0.])
        world_effect=np.array([.45*shock, .75*shock, self.demand-old_d])
        self.reserve=np.clip(self.reserve+.018+self_effect[0]+world_effect[0],0,1)
        self.health=np.clip(self.health+.004+self_effect[1]+world_effect[1]-.006*self.demand,0,1)
        self.alive=bool(self.alive and self.health>.15 and self.reserve>.08)
        target=.02+.09*self.demand
        service=min(1.,flow/max(target,1e-6))
        reward=float(service*(.65+.35*self.health)*(1-.10*flow/.12) if self.alive else 0.)
        sm=float(np.linalg.norm(self_effect[:2])); wm=float(np.linalg.norm(world_effect[:2]))
        if wm>1.5*max(sm,1e-9): src=1
        elif sm>1.5*max(wm,1e-9): src=0
        else: src=2
        if self.shock_flag[self.t]: self.last_world=self.t
        age=self.t-self.last_world
        agebin=0 if age==0 else 1 if age<=2 else 2 if age<=7 else 3
        self.t+=1
        return self.obs(),reward,self.t==STEPS,dict(alive=self.alive,source=src,agebin=agebin)

def run(agent,seed,variant='full'):
    env=AttributionEnvV2(); obs=env.reset(seed)
    a=copy.deepcopy(agent); a.reset_episode(obs,variant)
    X=[]; src=[]; age=[]; unc=[]; err=[]; cf=[]; mem=[]; obs_after=[]; rewards=[]; alive=[]
    for t in range(STEPS):
        p=a.prepare(obs,variant)
        cf_rewards=[]
        for act in range(4):
            ee=copy.deepcopy(env); cf_rewards.append(ee.step(act)[1])
        action=int(p['action']); nxt,r,_,info=env.step(action)
        a.complete_transition(obs,action,nxt,r,p,learn=False)
        X.append(np.r_[p['features'],p['gate_context'],p['goals'],p['scores'],
                       p['memory_energy'],p['memory_resource']])
        src.append(info['source']); age.append(info['agebin']); unc.append(float(p['gate_context'][-1]))
        err.append(float(np.mean(np.abs(nxt-p['predictions'][action,:3]))))
        cf.append(int(np.argmax(p['predictions'][:,3])==np.argmax(cf_rewards)))
        mem.append(float(p['memory_resource'][action])); obs_after.append(nxt.copy())
        rewards.append(r); alive.append(float(info['alive'])); obs=nxt
    targets=[]; preds=[]
    for t in range(STEPS-12):
        targets.append(.5*(obs_after[t+3][0]+obs_after[t+11][0])); preds.append(mem[t])
    return dict(X=np.asarray(X),source=np.asarray(src),age=np.asarray(age),unc=np.asarray(unc),
        errors=np.asarray(err),cf=np.asarray(cf),mem_mae=float(np.mean(np.abs(np.asarray(preds)-np.asarray(targets)))),
        return_mean=float(np.mean(rewards)),alive_fraction=float(np.mean(alive)))

def evaluate(seed):
    agent=_train_agent(seed)
    b=run(agent,seed^0xE5B,'full'); split=STEPS//2
    Xtr,Xte=b['X'][:split],b['X'][split:]; st,se=b['source'][:split],b['source'][split:]
    at,ae=b['age'][:split],b['age'][split:]
    # Validity checks are evaluated before interpreting probe scores.
    mtr=st!=2; mte=se!=2
    counts={str(i):int(np.sum(se==i)) for i in range(3)}
    binary_counts={'self':int(np.sum(se==0)),'world':int(np.sum(se==1))}
    valid=(b['alive_fraction']>=.80 and min(binary_counts.values())>=40 and all(counts[str(i)]>0 for i in range(3)))
    src_model=_ridge_classifier(Xtr,st,3); src_pred=_predict_classifier(src_model,Xte)
    src_ba=_balanced_accuracy(se,src_pred)
    if mtr.sum() and mte.sum():
        sw_model=_ridge_classifier(Xtr[mtr],(st[mtr]==0).astype(int),2)
        sw_pred=_predict_classifier(sw_model,Xte[mte])
        sw_ba=_balanced_accuracy((se[mte]==0).astype(int),sw_pred)
    else: sw_ba=0.
    tm=_ridge_classifier(Xtr,at,4); tp=_predict_classifier(tm,Xte); time_ba=_balanced_accuracy(ae,tp)
    thr=float(np.median(b['errors'][:split])); auc=_auc(b['unc'][split:],(b['errors'][split:]>thr).astype(int))
    cf=float(np.mean(b['cf'][split:]))
    nm=run(agent,seed^0xE5B,'noMemory'); pc=run(agent,seed^0xE5B,'permuted_content'); nc=run(agent,seed^0xE5B,'noCross')
    md=nm['mem_mae']-b['mem_mae']; cd=b['return_mean']-pc['return_mean']; xd=b['return_mean']-nc['return_mean']
    ad=[b['alive_fraction']-x['alive_fraction'] for x in (nm,pc,nc)]
    base=sw_ba>=.70 and src_ba>=.55 and time_ba>=.45 and auc>=.65 and cf>=.45
    lesion=md>=.02 and cd>=.01 and xd>=.01 and max(abs(x) for x in ad)<=.10
    return dict(seed=seed,valid=bool(valid),pass_seed=bool(valid and base and lesion),base=bool(base),lesion=bool(lesion),
        alive=b['alive_fraction'],return_mean=b['return_mean'],source_counts=counts,binary_counts=binary_counts,
        self_world=sw_ba,source=src_ba,time=time_ba,auc=auc,counterfactual=cf,
        memory_mae=b['mem_mae'],noMemory_mae=nm['mem_mae'],memory_damage=md,
        content_drop=cd,cross_drop=xd,alive_damages=ad)

def main(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[evaluate(s) for s in SEEDS]
    counts=dict(valid=sum(r['valid'] for r in rows),base=sum(r['base'] for r in rows),
                lesion=sum(r['lesion'] for r in rows),passed=sum(r['pass_seed'] for r in rows))
    verdict='PASS' if counts['passed']>=REQ else 'FAIL'
    result=dict(status='post-confirmatory corrective; original E5 unchanged',seeds=list(SEEDS),
                required=REQ,counts=counts,verdict=verdict,records=rows)
    (out/'RESULTS_E5B.json').write_text(json.dumps(result,indent=2))
    lines=['# R10 E5b — validity-controlled corrective replication','',
      'Original E5 remains FAIL 0/12. E5b is a new post-confirmatory experiment.','',
      f"Valid seeds: {counts['valid']}/12; base precursor conjunction: {counts['base']}/12; lesion conjunction: {counts['lesion']}/12; complete: {counts['passed']}/12.",
      f"Formal E5b verdict: **{verdict}**.",'',
      'Median metrics:']
    for k in ('alive','return_mean','self_world','source','time','auc','counterfactual','memory_damage','content_drop','cross_drop'):
        lines.append(f"- {k}: **{float(np.median([r[k] for r in rows])):.6f}**")
    (out/'REPORT_E5B.md').write_text('\n'.join(lines)+'\n')
    h=hashlib.sha256()
    for p in sorted(out.iterdir()): h.update(p.name.encode()); h.update(p.read_bytes())
    (out/'SHA256.txt').write_text(h.hexdigest()+'  E5B_OUTPUT_SET\n')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True)
    main(ap.parse_args().output)
