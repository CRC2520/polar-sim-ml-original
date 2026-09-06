"""Separate prospective active-K panel motivated by development, not final outcomes."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime,timezone
from urllib.request import urlopen
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ACTIVE=ROOT/'docs/STUDY3_ACTIVE_FREEZE.json'
PRIMARY='e44f783e096aafa9822ce517994780d5a89c3ae0'
SEEDS=list(range(902001,902041))
MODES=['full','no_K','retuned_zero','rewired','reverse','generic_nonlinear']


def sha(b):return hashlib.sha256(b).hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def now():return datetime.now(timezone.utc).isoformat()

def own_hashes():
    return {p:sha((ROOT/p).read_bytes()) for p in ['scripts/run_study3_active.py','docs/STUDY3_ACTIVE_PANEL_PROTOCOL.md']}


def prepare():
    if ACTIVE.exists():raise RuntimeError('active freeze already exists')
    primary=json.loads((ROOT/'docs/STUDY3_FREEZE.json').read_text())
    selection=json.loads((ROOT/'results_study3/development/selection.json').read_text())
    scores_file=ROOT/'results_study3/development/scores.json.gz'
    if sha(scores_file.read_bytes())!=selection['scores_sha256']:raise ValueError('development score checksum')
    rows=json.loads(gzip.decompress(scores_file.read_bytes()))
    selected=dict(primary['selected']);details={}
    for mode in ['full','rewired','reverse','generic_nonlinear']:
        groups=defaultdict(list)
        for r in rows:
            p=r['parameters']
            if r['mode']==mode and p['b']!=0:groups[(p['eta'],p['a'],p['b'])].append(r['metrics']['loss'])
        if len(groups)!=18 or any(len(v)!=72 for v in groups.values()):raise ValueError('incomplete active candidate set')
        ranked=sorted((float(np.mean(v)),abs(p[1])+abs(p[2]),*p) for p,v in groups.items())
        best=ranked[0];selected[mode]=dict(eta=best[2],a=best[3],b=best[4])
        details[mode]={'selected':selected[mode],'development_mean_loss':best[0],
                       'candidates':[dict(mean_loss=x[0],eta=x[2],a=x[3],b=x[4]) for x in ranked]}
    value=dict(status='ACTIVE_PANEL_FROZEN_BEFORE_ANY_FINAL_OUTCOMES',created_utc=now(),
               motivation='Primary development chose a=b=0; active causal contrast otherwise absent.',
               primary_registration=PRIMARY,primary_freeze_sha256=sha((ROOT/'docs/STUDY3_FREEZE.json').read_bytes()),
               core_hashes=primary['scientific_source_sha256'],panel_hashes=own_hashes(),selected=selected,
               selection_details=details,final_seeds=SEEDS,modes=MODES,expected_trials=2880)
    dump(ACTIVE,value);print(json.dumps({m:details[m]['selected'] for m in details},indent=2),flush=True)


def verify(commit):
    from study3.run import verify_registration
    verify_registration(PRIMARY)
    if not commit or len(commit)!=40:raise ValueError('active public registration required')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],check=True)
    pub=json.loads(urlopen(f'https://api.github.com/repos/CRC2520/polar-sim-ml-original/commits/{commit}',timeout=30).read())
    if pub['sha']!=commit:raise ValueError('public registration mismatch')
    registered=subprocess.check_output(['git','show',f'{commit}:docs/STUDY3_ACTIVE_FREEZE.json'])
    if registered!=ACTIVE.read_bytes():raise ValueError('active freeze changed')
    frozen=json.loads(registered)
    if frozen['panel_hashes']!=own_hashes():raise ValueError('active source/protocol changed')
    if frozen['primary_freeze_sha256']!=sha((ROOT/'docs/STUDY3_FREEZE.json').read_bytes()):raise ValueError('primary freeze changed')
    if frozen['final_seeds']!=SEEDS:raise ValueError('seed set changed')
    return frozen,dict(commit=commit,published_commit_utc=pub['commit']['committer']['date'],verified_utc=now())


def report(rows,difference,out):
    from study3.core import CELLS
    from study3.analysis import interval,CONTRASTS,classify
    keys=[(r['seed'],tuple(r['cell']),r['mode']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!={(s,c,m) for s in SEEDS for c in CELLS for m in MODES}:raise ValueError('incomplete active panel')
    indexed=dict(zip(keys,rows));metrics=['loss','cost','post_switch_loss','return_loss','saturation','violations','runtime_seconds']
    values={m:{v:np.array([np.mean([indexed[(s,c,m)]['metrics'][v] for c in CELLS]) for s in SEEDS]) for v in metrics} for m in MODES}
    idx=np.random.default_rng(917371).integers(0,40,(20000,40))
    summaries={m:{v:float(values[m][v].mean()) for v in metrics} for m in MODES}
    cs={};claims={}
    for name,other,threshold in CONTRASTS:
        x=interval(values['full']['loss']-values[other]['loss'],idx)
        x.update(comparator=other,required_upper_below=-threshold,passes=x['upper99_one_sided'] < -threshold)
        cs[name]=x;claims[name]=x['passes']
    cost=interval(values['full']['cost']-values['retuned_zero']['cost'],idx)
    switch=interval(values['full']['post_switch_loss']-values['retuned_zero']['post_switch_loss'],idx)
    hard=sum(r['metrics']['violations'] for r in rows if r['mode']=='full')
    gates=dict(cost=cost['upper99_one_sided']<=.05,post_switch=switch['upper99_one_sided']<=.005,constraints=hard==0)
    result=dict(status='separately_registered_active_panel_not_replacement_for_primary',trials=len(rows),seeds=40,
                summary=summaries,contrasts=cs,claims=claims,gates=gates,cost_guardrail=cost,
                post_switch_guardrail=switch,hard_violations_full=hard,
                maximum_full_vs_noK_action_difference=difference,
                decision=classify(True,claims,gates),harm_flag=cs['network']['lower99_one_sided']>.001,
                scope='Best active K configuration within the original finite grid; same developmental dataset; new final seeds.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);dump(out/'results.json',result)
    sub=[]
    for c in CELLS:
        for m in MODES:
            arr=np.array([indexed[(s,c,'full')]['metrics']['loss']-indexed[(s,c,m)]['metrics']['loss'] for s in SEEDS])
            sub.append(dict(cell=list(c),mode=m,mean_loss=float(np.mean([indexed[(s,c,m)]['metrics']['loss'] for s in SEEDS])),full_minus_mode=interval(arr,idx)))
    dump(out/'descriptive_subgroups.json',sub)
    text=['# Separately registered active-K panel','',f"Decision: **{result['decision']}**; harm flag: **{result['harm_flag']}**.",
          '2880 runs; 40 NEW final seeds; 12 cells; 6 conditions. Does not replace primary Study 3.','',
          '|Controller|MSE|Cost|Post-switch MSE|','|---|---:|---:|---:|']
    for m,v in summaries.items():text.append(f"|{m}|{v['loss']:.9f}|{v['cost']:.6f}|{v['post_switch_loss']:.9f}|")
    text+=['','|Contrast|Difference|95% interval|99% upper|Threshold|Pass|','|---|---:|---|---:|---:|---|']
    for k,v in cs.items():text.append(f"|{k}|{v['mean']:+.9f}|[{v['low95']:+.9f}, {v['high95']:+.9f}]|{v['upper99_one_sided']:+.9f}|{v['required_upper_below']:+.6f}|{v['passes']}|")
    text+=['',json.dumps(dict(gates=gates,cost=cost,post_switch=switch,max_action_difference=difference),indent=2)]
    (out/'results.md').write_text('\n'.join(text)+'\n')
    tex=['\\begin{table}[H]\\centering\\small','\\caption{Separately registered active-$K$ panel: 40 new seeds and 2,880 runs. Lower external MSE is better.}',
         '\\label{tab:study3-active}','\\begin{tabular}{lrrr}\\toprule','Controller & Tracking MSE & Cost & Post-switch MSE \\\\ \\midrule']
    for m,v in summaries.items():tex.append(m.replace('_',r'\_')+f" & {v['loss']:.6f} & {v['cost']:.4f} & {v['post_switch_loss']:.6f}"+r' \\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}'];(out/'results_table.tex').write_text('\n'.join(tex)+'\n')
    tex=['\\begin{table}[H]\\centering\\small','\\caption{Active-panel contrasts: full minus comparator, with one-sided 99\\% upper bounds within this panel.}',
         '\\label{tab:study3-active-contrasts}','\\begin{tabular}{lrrrl}\\toprule','Contrast & Difference & 99\\% upper & Threshold & Pass \\\\ \\midrule']
    for k,v in cs.items():tex.append(k.replace('_',r'\_')+f" & {v['mean']:+.6f} & {v['upper99_one_sided']:+.6f} & {v['required_upper_below']:+.4f} & {v['passes']}"+r' \\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}'];(out/'contrasts_table.tex').write_text('\n'.join(tex)+'\n')
    return result


def run_final(commit):
    from study3.core import CELLS,environment,trial
    from study3.run import ArchiveWriter,metadata
    frozen,reg=verify(commit);out=ROOT/'results_study3/active/final'
    if out.exists():raise RuntimeError('refuse to overwrite active final')
    out.mkdir(parents=True);dump(out/'STARTED.json',dict(started_utc=now(),registration=reg,environment=metadata()))
    writer=ArchiveWriter(out);rows=[];difference=0.
    try:
        for s in SEEDS:
            for c in CELLS:
                env=environment(s,c);reference=None
                for m in MODES:
                    r=trial(m,frozen['selected'],env,retain=True)
                    if m in ('full','rewired','reverse','generic_nonlinear') and r['parameters']['b']==0:raise ValueError('inactive intact model')
                    actions=np.array([f['action'] for f in r['frames']])
                    if m=='full':reference=actions
                    if m=='no_K':difference=max(difference,float(np.max(np.abs(actions-reference))))
                    writer.add(f'{s}-'+ '-'.join(c)+f'-{m}.json.gz',r)
                    rows.append({k:r[k] for k in ('seed','cell','mode','metrics')})
            print(f'ACTIVE completed seed {s}: {len(rows)} runs',flush=True)
    finally:writer.close()
    manifest=dict(status='completed_active_panel',completed_utc=now(),registration=reg,environment=metadata(),
                  frozen_sha256=sha(ACTIVE.read_bytes()),selected=frozen['selected'],core_hashes=frozen['core_hashes'],
                  panel_hashes=own_hashes(),archives=writer.archives,records=writer.records,
                  max_full_vs_noK_action_difference=difference,trials=len(rows))
    dump(out/'manifest.json',manifest);report(rows,difference,out/'generated')
    print((out/'generated/results.md').read_text(),flush=True)


def regenerate():
    from study3.core import CELLS,environment,transition,metrics,build,trial
    from study3.run import source_hashes
    out=ROOT/'results_study3/active/final';man=json.loads((out/'manifest.json').read_text())
    if man['core_hashes']!=source_hashes() or man['panel_hashes']!=own_hashes():raise ValueError('source changed')
    grouped=defaultdict(list)
    for x in man['records']:grouped[x['archive']].append(x)
    rows=[];reference={};difference=0.;replays=0;replay_max=0.
    for arc in man['archives']:
        path=out/arc['path']
        if sha(path.read_bytes())!=arc['sha256']:raise ValueError('archive checksum')
        with tarfile.open(path,'r') as tf:
            names=tf.getnames();wanted=[x['name'] for x in grouped[arc['path']]]
            if len(set(names))!=len(names) or set(names)!=set(wanted):raise ValueError('archive membership')
            for item in grouped[arc['path']]:
                b=tf.extractfile(item['name']).read();raw=gzip.decompress(b)
                if sha(b)!=item['sha256'] or sha(raw)!=item['raw_sha256']:raise ValueError('record checksum')
                r=json.loads(raw);env=environment(r['seed'],r['cell']);m=r['mode']
                if r['parameters']!=build(m,man['selected'],r['seed']).study_parameters:raise ValueError('configuration mismatch')
                for f in r['frames']:
                    t=f['t'];o=f['observation']
                    for key in ('target','costs','weights','allowed','chi'):
                        if not np.array_equal(o[key],env[key][t]):raise ValueError('observation mismatch')
                    if o['budget']!=float(env['budget'][t]):raise ValueError('budget mismatch')
                    if not np.allclose(f['effect'],transition(env,t,np.asarray(f['action'])),atol=1e-13,rtol=0):raise ValueError('effect mismatch')
                scored=metrics(r['frames'])
                if any(abs(scored[k]-r['metrics'][k])>1e-13 for k in scored):raise ValueError('outcome mismatch')
                actions=np.array([f['action'] for f in r['frames']]);key=(r['seed'],tuple(r['cell']))
                if m=='full':reference[key]=actions
                if m=='no_K':difference=max(difference,float(np.max(np.abs(actions-reference[key]))))
                if r['seed'] in (SEEDS[0],SEEDS[-1]) and tuple(r['cell']) in (CELLS[0],CELLS[-1]):
                    replay=trial(m,man['selected'],env);a=np.array([f['action'] for f in replay['frames']])
                    delta=float(np.max(np.abs(actions-a)));replay_max=max(replay_max,delta);replays+=1
                    if delta>1e-12 or abs(replay['metrics']['loss']-r['metrics']['loss'])>1e-13:raise ValueError('controller replay mismatch')
                rows.append({k:r[k] for k in ('seed','cell','mode','metrics')})
    if difference!=man['max_full_vs_noK_action_difference'] or replays!=24:raise ValueError('audit count mismatch')
    result=report(rows,difference,out/'regenerated')
    if result!=json.loads((out/'generated/results.json').read_text()):raise ValueError('analysis mismatch')
    dump(out/'REGENERATION.json',dict(status='all_2880_records_verified',checked_utc=now(),
                 controller_replays=replays,max_replay_action_difference=replay_max,
                 manifest_sha256=sha((out/'manifest.json').read_bytes())))
    print('All active-panel records verified; 24 controller replays verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','final','regenerate']);p.add_argument('--registration-sha');args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='final':run_final(args.registration_sha)
    else:regenerate()
