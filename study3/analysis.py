"""Prespecified seed-level analysis and exact decision logic; no post-run tuning."""
import csv
import json
from pathlib import Path
import numpy as np
from .core import FINAL_SEEDS, CELLS, MODES

CONTRASTS=[('tension','no_K',.001),('network','retuned_zero',.001),
           ('topology_rewired','rewired',.0005),('topology_reverse','reverse',.0005),
           ('feature','generic_nonlinear',.0005)]


def interval(x,indices):
    x=np.asarray(x,float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)!=indices.shape[1]:raise ValueError('invalid paired data')
    boot=x[indices].mean(axis=1)
    return dict(mean=float(x.mean()),low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975)),
                lower99_one_sided=float(np.quantile(boot,.01)),upper99_one_sided=float(np.quantile(boot,.99)))


def classify(valid,claims,gates):
    required={name for name,_,_ in CONTRASTS}
    if set(claims)!=required or set(gates)!={'cost','post_switch','constraints'}:
        raise ValueError('missing or extra decision field')
    if not valid:return 'invalidate_study'
    if all(claims.values()) and all(gates.values()):return 'support_configured_network_on_tested_tasks'
    if claims['tension'] and claims['network'] and all(gates.values()):return 'support_network_not_exclusive_polarity'
    if claims['network'] and all(gates.values()):return 'network_benefit_without_tension_specific_evidence'
    return 'no_confirmed_network_advantage'


def analyze(rows,equivalence,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    expected={(s,tuple(c),m) for s in FINAL_SEEDS for c in CELLS for m in MODES}
    keys=[(r['seed'],tuple(r['cell']),r['mode']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('unexpected trial set')
    metrics=['loss','cost','post_switch_loss','return_loss','saturation','violations','runtime_seconds']
    index={k:r for k,r in zip(keys,rows)}
    values={m:{v:np.array([np.mean([index[(s,c,m)]['metrics'][v] for c in CELLS]) for s in FINAL_SEEDS]) for v in metrics} for m in MODES}
    rng=np.random.default_rng(917370);idx=rng.integers(0,len(FINAL_SEEDS),(20000,len(FINAL_SEEDS)))
    summaries={m:{v:float(values[m][v].mean()) for v in metrics} for m in MODES}
    contrasts={};claims={}
    for name,other,threshold in CONTRASTS:
        c=interval(values['full']['loss']-values[other]['loss'],idx)
        c.update(comparator=other,required_upper_below=-threshold)
        c['passes']=c['upper99_one_sided'] < -threshold
        contrasts[name]=c;claims[name]=c['passes']
    cost=interval(values['full']['cost']-values['retuned_zero']['cost'],idx)
    switch=interval(values['full']['post_switch_loss']-values['retuned_zero']['post_switch_loss'],idx)
    hard=sum(r['metrics']['violations'] for r in rows if r['mode']=='full')
    gates=dict(cost=cost['upper99_one_sided']<=.05,post_switch=switch['upper99_one_sided']<=.005,constraints=hard==0)
    negative=interval(values['zero_negative']['loss']-values['retuned_zero']['loss'],idx)
    valid=all(x<=1e-10 for x in equivalence.values()) and negative['lower99_one_sided']>.01
    decision=classify(valid,claims,gates)
    harm=contrasts['network']['lower99_one_sided']>.001
    result=dict(status='prospective_final_evaluation',trials=len(rows),seeds=len(FINAL_SEEDS),
                steps_per_trial=64,bootstrap_resamples=20000,summary=summaries,contrasts=contrasts,
                claims=claims,gates=gates,cost_guardrail=cost,post_switch_guardrail=switch,
                hard_violations_full=hard,equivalence_max_action_difference=equivalence,
                negative_control=negative,valid=valid,decision=decision,harm_flag=harm,
                limitations=['Conditional on the single development-selected configuration set.',
                             'Synthetic task generators designed inside the project.',
                             'Shared coupling gains tune configured templates, not arbitrary learned topology.',
                             'Approximate percentile intervals; no consciousness inference.'])
    subgroup=[]
    for cell in CELLS:
        for m in MODES:
            r=[index[(s,cell,m)]['metrics']['loss'] for s in FINAL_SEEDS]
            diff=np.asarray([index[(s,cell,'full')]['metrics']['loss']-index[(s,cell,m)]['metrics']['loss'] for s in FINAL_SEEDS])
            subgroup.append(dict(cell=list(cell),mode=m,mean_loss=float(np.mean(r)),full_minus_mode=interval(diff,idx)))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'descriptive_subgroups.json').write_text(json.dumps(subgroup,indent=2,allow_nan=False)+'\n')
    with (out/'trial_summaries.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','family','topology','resource','mode']+metrics)
        for r in rows:writer.writerow([r['seed'],*r['cell'],r['mode']]+[r['metrics'][v] for v in metrics])
    text=['# Study 3: external network evaluation','',f'Decision: **{decision}**',f'Validity: {valid}; separate harm flag: {harm}.',
          f'{len(rows)} complete runs; {len(FINAL_SEEDS)} paired seeds; 12 factorial cells; 64 transitions per run.','',
          '|Controller|Mean external MSE|Mean cost|Post-switch MSE|','|---|---:|---:|---:|']
    for m,v in summaries.items():text.append(f"|{m}|{v['loss']:.9f}|{v['cost']:.6f}|{v['post_switch_loss']:.9f}|")
    text+=['','## Five prespecified contrasts','Full minus comparator; negative favors full. One-sided 99% bounds implement alpha .05/5.','',
           '|Claim|Mean difference|95% interval|99% upper|Required upper below|Pass|','|---|---:|---|---:|---:|---|']
    for k,v in contrasts.items():text.append(f"|{k}|{v['mean']:+.9f}|[{v['low95']:+.9f}, {v['high95']:+.9f}]|{v['upper99_one_sided']:+.9f}|{v['required_upper_below']:+.6f}|{v['passes']}|")
    text+=['','## Gates and controls',json.dumps(dict(gates=gates,equivalence=equivalence,negative=negative),indent=2),'',
           'All subgroup contrasts are descriptive. No result changes the frozen Study 2 conclusion.',
           'The tested configured templates are limited implementations, not the full layered architecture or evidence of consciousness.']
    (out/'results.md').write_text('\n'.join(text)+'\n')
    tex=['\\begin{table}[H]\\centering\\small',
         '\\caption{Study 3: external tracking on 40 paired seeds, 12 factorial cells and 64 transitions per run. Lower MSE is better.}',
         '\\label{tab:study3-results}','\\begin{tabular}{lrrr}\\toprule','Controller & Tracking MSE & Cost & Post-switch MSE \\\\ \\midrule']
    for m,v in summaries.items():tex.append(m.replace('_',r'\_')+f" & {v['loss']:.6f} & {v['cost']:.4f} & {v['post_switch_loss']:.6f}"+r' \\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}']
    (out/'results_table.tex').write_text('\n'.join(tex)+'\n')
    tex=['\\begin{table}[H]\\centering\\small',
         '\\caption{Prespecified contrasts: full minus comparator. Five one-sided claims use 99\\% bootstrap upper bounds.}',
         '\\label{tab:study3-contrasts}','\\begin{tabular}{lrrrl}\\toprule','Contrast & Difference & 99\\% upper & Threshold & Pass \\\\ \\midrule']
    for k,v in contrasts.items():tex.append(k.replace('_',r'\_')+f" & {v['mean']:+.6f} & {v['upper99_one_sided']:+.6f} & {v['required_upper_below']:+.4f} & {v['passes']}"+r' \\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}']
    (out/'contrasts_table.tex').write_text('\n'.join(tex)+'\n')
    return result
