"""P0 diagnosis, development calibration, public freeze, final evaluation and replay."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import gzip,hashlib,json,os,platform,subprocess,tarfile
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from study3.run import ArchiveWriter
from integrated_polar.layers import CalibratedMonitor
from .p0 import run_diagnosis
from .tasks import DEV_SEEDS,FINAL_SEEDS,CELLS,MODES,run_trial,jsonable
from .evaluate import analyze,verify_record,CAPABILITY_SEEDS,capability_probe,coordination_probe,calibration_metrics,interval
ROOT=Path(__file__).resolve().parents[1]
BASE='cea382a7fd1486bc8403ee071c47432405ae57c2'
FREEZE=ROOT/'docs/P0P2_FREEZE.json';OUT=ROOT/'results_p0p2'


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')
def source_hashes():
    paths=[]
    for folder in ('integrated_polar','gap_resolution'):paths.extend((ROOT/folder).glob('*.py'))
    paths.extend([ROOT/'docs/P0P2_PROTOCOL.md',ROOT/'tests/test_integrated_polar.py',ROOT/'tests/test_gap_evaluation.py'])
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(paths)}
def environment_info():return {'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
                              'packages':subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines()}


def develop():
    if FREEZE.exists():raise RuntimeError('already frozen; preserve prior registration')
    root=OUT/'development';root.mkdir(parents=True,exist_ok=False)
    dump(root/'STARTED.json',{'started':stamp(),'environment':environment_info()})
    # P0 precedes P1 outcome inspection; never changes historical Study 3 outputs.
    records,p0=run_diagnosis();dump(root/'P0_DIAGNOSIS.json',p0)
    writer=ArchiveWriter(root)
    for r in records:writer.add('p0-'+str(r['seed'])+'-'+'-'.join(map(str,r['cell']))+'-'+r['condition']+'.json.gz',r)
    writer.close();dump(root/'P0_MANIFEST.json',{'archives':writer.archives,'records':writer.records})
    del records
    rows=[];scores=[];labels=[];score_records=[];equivalence=0.
    for seed in DEV_SEEDS:
        for cell in CELLS:
            reference=None
            for mode in MODES:
                rec=run_trial(seed,cell,mode,retain=False)
                rows.append({k:rec[k] for k in ('seed','cell','mode','metrics')})
                actions=np.array([f['action'] for f in rec['frames']])
                if mode=='full':
                    reference=actions
                    for f in rec['frames']:
                        scores.append(f['pre_feedback_score']);labels.append(f['success'])
                        score_records.append({'seed':seed,'cell':list(cell),'t':f['t'],'score':f['pre_feedback_score'],'success':f['success']})
                if mode=='generic_equivalent':equivalence=max(equivalence,float(np.max(abs(reference-actions))))
        print(f'DEVELOPMENT integrated seed {seed}: {len(rows)} runs',flush=True)
    monitor=CalibratedMonitor().fit(scores,labels);dump(root/'MONITOR.json',monitor.snapshot())
    dump(root/'P1_DEVELOPMENT.json',analyze(rows,equivalence,DEV_SEEDS))
    dump(root/'DEVELOPMENT_ROWS.json',rows);dump(root/'CALIBRATION_ROWS.json',score_records)
    freeze={'status':'FROZEN_BEFORE_P1_P2_FINAL','created_utc':stamp(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'scientific_source_sha256':source_hashes(),'monitor':monitor.snapshot(),'final_seeds':FINAL_SEEDS,
            'capability_seeds':CAPABILITY_SEEDS,'expected_p1_trials':1056,'configuration':'AgentConfig defaults; no post-development parameter tuning',
            'development_files':{p.name:sha(p.read_bytes()) for p in root.iterdir() if p.suffix=='.json'},
            'environment':environment_info(),'old_studies_base':BASE}
    dump(FREEZE,freeze)
    print(json.dumps({'P0':p0,'P1_development':json.loads((root/'P1_DEVELOPMENT.json').read_text()),'monitor':monitor.snapshot()},indent=2),flush=True)


def check_registration(commit):
    if not isinstance(commit,str) or len(commit)!=40:raise ValueError('public registration SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],check=True)
    b=subprocess.check_output(['git','show',f'{commit}:docs/P0P2_FREEZE.json'])
    if b!=FREEZE.read_bytes():raise RuntimeError('freeze differs from public registration')
    freeze=json.loads(b)
    if freeze['scientific_source_sha256']!=source_hashes():raise RuntimeError('scientific sources changed after freeze')
    public=json.loads(urlopen(f'https://api.github.com/repos/CRC2520/polar-sim-ml-original/commits/{commit}',timeout=30).read())
    for name,h in freeze['development_files'].items():
        if sha((OUT/'development'/name).read_bytes())!=h:raise RuntimeError('development evidence altered')
    when=public['commit']['committer']['date']
    if datetime.fromisoformat(when.replace('Z','+00:00'))>datetime.now(timezone.utc):raise RuntimeError('future registration')
    return freeze,{'commit':commit,'public_commit_utc':when,'verified_utc':stamp(),'freeze_sha256':sha(b)}


def final(commit):
    freeze,registration=check_registration(commit);root=OUT/'final';root.mkdir(parents=True,exist_ok=False)
    dump(root/'STARTED.json',{'started_utc':stamp(),'registration':registration,'environment':environment_info()})
    monitor=CalibratedMonitor.restore(freeze['monitor']);writer=ArchiveWriter(root);rows=[];eq=0.
    for seed in FINAL_SEEDS:
        for cell in CELLS:
            reference=None
            for mode in MODES:
                rec=run_trial(seed,cell,mode,monitor,retain=True)
                actions=np.array([f['action'] for f in rec['frames']])
                if mode=='full':reference=actions
                if mode=='generic_equivalent':eq=max(eq,float(np.max(abs(actions-reference))))
                verify_record(rec)
                writer.add(f'p1-{seed}-{cell[0]}-{cell[1]}-{mode}.json.gz',rec)
                rows.append({k:rec[k] for k in ('seed','cell','mode','metrics')})
        print(f'FINAL integrated seed {seed}: {len(rows)} runs',flush=True)
    writer.close();summary=analyze(rows,eq);dump(root/'P1_RESULTS.json',summary);dump(root/'P1_ROWS.json',rows)
    # P2 is separately measured, not inferred from P1 task performance.
    capability=[];coordination=[];monitor_frames=[];review_rows=[];monitor_writer=ArchiveWriter(root/'p2')
    (root/'p2').mkdir(exist_ok=True)
    for seed in CAPABILITY_SEEDS:
        capability.append(capability_probe(seed))
        for comm in (True,False):coordination.append(coordination_probe(seed,comm))
        for cell in CELLS:
            base=run_trial(seed,cell,'full',monitor,retain=True)
            review=run_trial(seed,cell,'full',monitor,retain=True,review=True)
            monitor_frames.extend([{k:f[k] for k in ('probability','success','pre_feedback_score')} for f in base['frames']])
            review_rows.append({'seed':seed,'cell':list(cell),'base':base['metrics'],'review':review['metrics'],
                                'review_fraction':float(np.mean([f['requested_review'] for f in review['frames']]))})
            monitor_writer.add(f'p2-{seed}-{cell[0]}-{cell[1]}-base.json.gz',base)
            monitor_writer.add(f'p2-{seed}-{cell[0]}-{cell[1]}-review.json.gz',review)
        print(f'P2 completed seed {seed}',flush=True)
    monitor_writer.close()
    idx=np.random.default_rng(951777).integers(0,len(CAPABILITY_SEEDS),(10000,len(CAPABILITY_SEEDS)))
    delta=np.array([next(x['loss'] for x in coordination if x['seed']==s and x['communication'])-next(x['loss'] for x in coordination if x['seed']==s and not x['communication']) for s in CAPABILITY_SEEDS])
    monitor_result=calibration_metrics(monitor_frames,monitor)
    p2={'status':'bounded_specific_capabilities','seeds':len(CAPABILITY_SEEDS),
        'source_attribution_accuracy':sum(r['source_correct'] for r in capability)/sum(r['source_trials'] for r in capability),
        'ambiguous_source_abstention_rate':sum(r['ambiguous_abstentions'] for r in capability)/sum(r['source_trials'] for r in capability),
        'normative_pass_count':sum(r['normative_pass'] for r in capability),'semantic_max_error':max(r['semantic_max_error'] for r in capability),
        'learned_physical_signs_all_correct':all(r['learned_fill_drain'][0][0]>0 and r['learned_fill_drain'][0][1]<0 for r in capability),
        'coordination_loss_difference':interval(delta,idx,upper=.975),'coordination_violations':sum(r['violations'] for r in coordination),
        'monitor':monitor_result,'review_fraction':float(np.mean([r['review_fraction'] for r in review_rows])),
        'review_minus_base_loss':float(np.mean([r['review']['loss']-r['base']['loss'] for r in review_rows])),
        'scope':'Institutional rules, own/other candidate effects, typed tank vocabulary, limited autonomous resource coordination; not general ethics, human semantic grounding or consciousness.'}
    dump(root/'P2_RESULTS.json',p2);dump(root/'P2_CAPABILITY_RECORDS.json',capability);dump(root/'P2_COORDINATION_RECORDS.json',coordination)
    dump(root/'P2_MONITOR_FRAMES.json',monitor_frames);dump(root/'P2_REVIEW_ROWS.json',review_rows)
    manifest={'status':'completed','completed_utc':stamp(),'registration':registration,'source_sha256':source_hashes(),'p1_trials':len(rows),
              'p1_archives':writer.archives,'p1_records':writer.records,'p2_archives':monitor_writer.archives,'p2_records':monitor_writer.records,
              'generic_equivalence':eq,'result_sha256':{p.name:sha(p.read_bytes()) for p in root.iterdir() if p.suffix=='.json'},'monitor':freeze['monitor']}
    dump(root/'MANIFEST.json',manifest);print(json.dumps({'P1':summary,'P2':p2},indent=2),flush=True)


def iter_records(root,archives,records):
    by_archive={}
    for rec in records:by_archive.setdefault(rec['archive'],[]).append(rec)
    for arc in archives:
        path=root/arc['path']
        if sha(path.read_bytes())!=arc['sha256']:raise ValueError('archive hash mismatch')
        with tarfile.open(path,'r') as tf:
            members=tf.getnames();expected=by_archive[arc['path']]
            if len(members)!=len(set(members)) or set(members)!={r['name'] for r in expected}:raise ValueError('unexpected archive content')
            for r in expected:
                data=tf.extractfile(r['name']).read()
                if sha(data)!=r['sha256']:raise ValueError('record hash mismatch')
                raw=gzip.decompress(data)
                if sha(raw)!=r['raw_sha256']:raise ValueError('raw hash mismatch')
                yield json.loads(raw)


def regenerate():
    root=OUT/'final';man=json.loads((root/'MANIFEST.json').read_text())
    if man['source_sha256']!=source_hashes():raise RuntimeError('different scientific sources')
    for name,h in man['result_sha256'].items():
        if sha((root/name).read_bytes())!=h:raise ValueError('result file changed')
    rows=[];eq=0.;reference={};replaymax=0.;replays=0;monitor=CalibratedMonitor.restore(man['monitor'])
    for r in iter_records(root,man['p1_archives'],man['p1_records']):
        verify_record(r);rows.append({k:r[k] for k in ('seed','cell','mode','metrics')})
        case=(r['seed'],tuple(r['cell']));actions=np.array([f['action'] for f in r['frames']])
        if r['mode']=='full':reference[case]=actions
        if r['mode']=='generic_equivalent':eq=max(eq,float(np.max(abs(actions-reference[case]))))
        if r['seed']==FINAL_SEEDS[0] and r['mode'] in ('full','diagonal_plan','no_priorities','no_memory','myopic'):
            replay=run_trial(r['seed'],tuple(r['cell']),r['mode'],monitor,retain=False)
            replaymax=max(replaymax,float(np.max(abs(actions-np.array([f['action'] for f in replay['frames']])))));replays+=1
    res=analyze(rows,eq)
    if res!=json.loads((root/'P1_RESULTS.json').read_text()):raise ValueError('regenerated P1 outcomes differ')
    p2_count=0
    for r in iter_records(root/'p2',man['p2_archives'],man['p2_records']):verify_record(r);p2_count+=1
    if p2_count!=192 or replaymax>1e-8:raise ValueError('incomplete P2 or failed policy replay')
    p2_records=json.loads((root/'P2_CAPABILITY_RECORDS.json').read_text());coord=json.loads((root/'P2_COORDINATION_RECORDS.json').read_text())
    # Replay both local-policy coordination arms on the first fixed capability seed.
    for old in [r for r in coord if r['seed']==CAPABILITY_SEEDS[0]]:
        new=coordination_probe(old['seed'],old['communication'])
        if abs(new['loss']-old['loss'])>1e-10:raise ValueError('coordination replay differs')
    monitor_frames=json.loads((root/'P2_MONITOR_FRAMES.json').read_text())
    if calibration_metrics(monitor_frames,monitor)!=json.loads((root/'P2_RESULTS.json').read_text())['monitor']:raise ValueError('calibration reanalysis differs')
    dump(root/'REGENERATION.json',{'status':'verified','p1_records':len(rows),'p2_trajectory_records':p2_count,
                                  'controller_replays':replays,'maximum_replay_action_difference':replaymax,
                                  'coordination_replays':2,'checked_utc':stamp(),'manifest_sha256':sha((root/'MANIFEST.json').read_bytes()),
                                  'checks':['archive/record hashes','frozen source','expected P1 design','all physical transitions','outcome recomputation','generic equivalence','P1 decision','monitor scores','policy replays']})
    print('All final trajectory records and outcome regeneration verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['develop','final','regenerate']);p.add_argument('--registration-sha');a=p.parse_args()
    if a.stage=='develop':develop()
    elif a.stage=='final':final(a.registration_sha)
    else:regenerate()
