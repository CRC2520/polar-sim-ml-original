"""Develop, register, execute once, and independently regenerate Study 3."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
from datetime import datetime,timezone
from urllib.request import urlopen
import numpy as np
from .core import DEV_SEEDS,FINAL_SEEDS,CELLS,MODES,TUNED,candidates,environment,transition,trial,metrics,build
from .analysis import analyze

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/'docs/STUDY3_FREEZE.json'


def sha(data):return hashlib.sha256(data).hexdigest()

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def now():return datetime.now(timezone.utc).isoformat()

def source_hashes():
    paths=[ROOT/'network_tension.py',ROOT/'docs/STUDY3_PROTOCOL.md',ROOT/'tests/test_study3.py']
    paths+=list((ROOT/'polar').glob('*.py'))+list((ROOT/'study3').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(paths)}

def metadata():
    return dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                packages=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines())


def develop():
    if FREEZE.exists():raise RuntimeError('freeze already exists; do not overwrite development')
    out=ROOT/'results_study3/development';out.mkdir(parents=True,exist_ok=False)
    started=now();selected={};all_scores=[];total=0
    envs=[environment(s,c) for s in DEV_SEEDS for c in CELLS]
    for mode in TUNED:
        scored=[]
        for pars in candidates(mode):
            losses=[]
            for env in envs:
                record=trial(mode,{mode:pars},env,retain=False)
                losses.append(record['metrics']['loss']);total+=1
                all_scores.append(dict(mode=mode,parameters=pars,seed=env['seed'],cell=env['cell'],metrics=record['metrics']))
            mean=float(np.mean(losses));scored.append((mean,abs(pars['a'])+abs(pars['b']),pars['eta'],pars['a'],pars['b']))
            print(f'DEV {mode} {pars} MSE={mean:.9f}',flush=True)
        winner=min(scored);selected[mode]=dict(eta=winner[2],a=winner[3],b=winner[4])
    assert total==8208
    payload=json.dumps(all_scores,allow_nan=False,separators=(',',':')).encode()
    (out/'scores.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    summary=dict(status='development_only',started_utc=started,finished_utc=now(),runs=total,selected=selected,
                 candidates_per_mode={m:len(candidates(m)) for m in TUNED},
                 scores_sha256=sha((out/'scores.json.gz').read_bytes()),environment=metadata())
    dump(out/'selection.json',summary)
    freeze=dict(status='FROZEN_BEFORE_FINAL_REQUIRES_PUBLIC_REGISTRATION',created_utc=now(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scientific_source_sha256=source_hashes(),selected=selected,development_seeds=DEV_SEEDS,
                final_seeds=FINAL_SEEDS,expected_final_trials=5760,
                development_summary_sha256=sha((out/'selection.json').read_bytes()),
                protocol='docs/STUDY3_PROTOCOL.md',environment=metadata())
    dump(FREEZE,freeze);print(json.dumps(summary,indent=2),flush=True)


def verify_registration(commit):
    if not commit or len(commit)!=40:raise ValueError('a public registration commit is required')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],check=True)
    data=json.loads(urlopen(f'https://api.github.com/repos/CRC2520/polar-sim-ml-original/commits/{commit}',timeout=30).read())
    if data['sha']!=commit:raise RuntimeError('registration SHA mismatch')
    published=datetime.fromisoformat(data['commit']['committer']['date'].replace('Z','+00:00'))
    if datetime.now(timezone.utc)<published:raise RuntimeError('registration is in the future')
    registered=subprocess.check_output(['git','show',f'{commit}:docs/STUDY3_FREEZE.json'])
    if registered!=FREEZE.read_bytes():raise RuntimeError('freeze differs from registered bytes')
    freeze=json.loads(registered)
    if freeze['scientific_source_sha256']!=source_hashes():raise RuntimeError('scientific source changed after freeze')
    if freeze['final_seeds']!=FINAL_SEEDS:raise RuntimeError('seed set changed')
    selection=ROOT/'results_study3/development/selection.json'
    if sha(selection.read_bytes())!=freeze['development_summary_sha256']:raise RuntimeError('development evidence altered')
    return freeze,dict(commit=commit,published_commit_utc=published.isoformat(),verified_utc=now(),freeze_sha256=sha(registered))


class ArchiveWriter:
    def __init__(self,out):
        self.out=Path(out);self.part=0;self.size=0;self.records=[];self.archives=[];self.tar=None
    def close_part(self):
        if self.tar is not None:
            self.tar.close();self.archives.append(dict(path=self.name,sha256=sha((self.out/self.name).read_bytes())))
            self.tar=None
    def add(self,name,payload):
        raw=json.dumps(payload,allow_nan=False,separators=(',',':')).encode();data=gzip.compress(raw,compresslevel=6,mtime=0)
        if self.tar is None or self.size+len(data)>6_000_000:
            self.close_part();self.part+=1;self.name=f'traces-{self.part:03d}.tar';self.tar=tarfile.open(self.out/self.name,'w');self.size=0
        info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0
        self.tar.addfile(info,io.BytesIO(data));self.size+=len(data)+1024
        self.records.append(dict(archive=self.name,name=name,sha256=sha(data),raw_sha256=sha(raw)))
    def close(self):self.close_part()


def execute_final(registration):
    freeze,reg=verify_registration(registration)
    out=ROOT/'results_study3/final'
    if out.exists():raise RuntimeError('final output exists; never overwrite final evidence')
    out.mkdir(parents=True);dump(out/'STARTED.json',dict(started_utc=now(),registration=reg,environment=metadata()))
    writer=ArchiveWriter(out);rows=[];equiv={'generic_equivalent':0.,'signed_intensity':0.}
    try:
        for seed in FINAL_SEEDS:
            for cell in CELLS:
                env=environment(seed,cell);reference=None
                for mode in MODES:
                    rec=trial(mode,freeze['selected'],env,retain=True)
                    actions=np.asarray([f['action'] for f in rec['frames']])
                    if mode=='full':reference=actions
                    if mode in equiv:equiv[mode]=max(equiv[mode],float(np.max(np.abs(actions-reference))))
                    name=f'{seed}-'+'-'.join(cell)+f'-{mode}.json.gz'
                    writer.add(name,rec)
                    rows.append({k:rec[k] for k in ('seed','cell','mode','metrics')})
            print(f'FINAL completed seed {seed}: {len(rows)} runs',flush=True)
    finally:writer.close()
    manifest=dict(status='completed_final',completed_utc=now(),registration=reg,environment=metadata(),
                  scientific_source_sha256=source_hashes(),selected=freeze['selected'],expected_trials=5760,
                  trials=len(rows),archives=writer.archives,records=writer.records,equivalence=equiv)
    dump(out/'manifest.json',manifest)
    analyze(rows,equiv,out/'generated')
    print((out/'generated/results.md').read_text(),flush=True)


def regenerate(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['scientific_source_sha256']!=source_hashes():raise RuntimeError('reporter/scientific source differs from run')
    grouped={}
    for rec in manifest['records']:grouped.setdefault(rec['archive'],[]).append(rec)
    rows=[];seen=set();actions={};eq={'generic_equivalent':0.,'signed_intensity':0.}
    for arc in manifest['archives']:
        path=directory/arc['path']
        if sha(path.read_bytes())!=arc['sha256']:raise ValueError('archive checksum failed')
        with tarfile.open(path,'r') as tf:
            members=tf.getnames()
            wanted=[x['name'] for x in grouped[arc['path']]]
            if len(members)!=len(set(members)) or set(members)!=set(wanted):raise ValueError('unexpected archive members')
            for item in grouped[arc['path']]:
                data=tf.extractfile(item['name']).read()
                if sha(data)!=item['sha256']:raise ValueError('record checksum failed')
                raw=gzip.decompress(data)
                if sha(raw)!=item['raw_sha256']:raise ValueError('raw record checksum failed')
                r=json.loads(raw);key=(r['seed'],tuple(r['cell']),r['mode'])
                if key in seen:raise ValueError('duplicate trial')
                seen.add(key)
                expected=build(r['mode'],manifest['selected'],r['seed']).study_parameters
                if r['parameters']!=expected:raise ValueError('parameter mismatch')
                env=environment(r['seed'],r['cell'])
                for f in r['frames']:
                    t=f['t'];o=f['observation']
                    for k in ('target','costs','weights','allowed','chi'):
                        if not np.array_equal(np.asarray(o[k]),env[k][t]):raise ValueError('observation replay mismatch')
                    if o['budget']!=float(env['budget'][t]):raise ValueError('budget mismatch')
                    if not np.allclose(f['effect'],transition(env,t,np.asarray(f['action'])),rtol=0,atol=1e-13):raise ValueError('effect replay mismatch')
                recomputed=metrics(r['frames'])
                if any(abs(recomputed[k]-r['metrics'][k])>1e-13 for k in recomputed):raise ValueError('outcome mismatch')
                rows.append({k:r[k] for k in ('seed','cell','mode','metrics')})
                case=key[:2]
                if r['mode']=='full':actions[case]=np.asarray([f['action'] for f in r['frames']])
                if r['mode'] in eq:
                    eq[r['mode']]=max(eq[r['mode']],float(np.max(np.abs(np.asarray([f['action'] for f in r['frames']])-actions[case]))))
    if len(rows)!=5760 or eq!=manifest['equivalence']:raise ValueError('incomplete final/equivalence mismatch')
    result=analyze(rows,eq,directory/'regenerated')
    if json.loads((directory/'generated/results.json').read_text())!=result:raise ValueError('regenerated conclusions differ')
    dump(directory/'REGENERATION.json',dict(status='all_5760_records_verified',checked_utc=now(),
                checks=['archive/record hashes','complete unique trial set','64 transitions','frozen configuration',
                        'exogenous observation replay','effect replay','outcome recalculation','equivalence','identical decision'],
                manifest_sha256=sha((directory/'manifest.json').read_bytes())))
    print('Verified all records and identical results',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['develop','final','regenerate']);p.add_argument('--registration-sha');p.add_argument('--directory',default='results_study3/final');args=p.parse_args()
    if args.stage=='develop':develop()
    elif args.stage=='final':execute_final(args.registration_sha)
    else:regenerate(ROOT/args.directory)
