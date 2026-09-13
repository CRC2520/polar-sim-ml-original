"""B1-S static publication; does not import or execute scientific project code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b861e732e5f5a667d50cf96023d1b4ebcbfc7125'
WORKFLOW = '.github/workflows/b1s-documentary-pivot.yml'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(items):
    result = {}
    for k, v in items:
        require(k not in result, f'Duplicate JSON key: {k}')
        result[k] = v
    return result


def load(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(text, edits):
    for e in edits:
        require(e['old'] and text.count(e['old']) == 1, f'Invalid exact anchor: {e["id"]}')
        text = text.replace(e['old'], e['new'], 1)
    return text


def tree(ref):
    result = {}
    for line in git('ls-tree', '-r', '-z', ref).split(b'\0'):
        if line:
            value, path = line.split(b'\t', 1)
            result[path.decode()] = value.decode()
    return result


def validate(prepare=False):
    state = load(ROOT / 'b1s/TRANSITION.json')
    require(state['status'] == 'B1S_DESIGN', 'Design status required')
    require(state['bases']['code']['commit'] == BASE, 'Base changed')
    require(state['effective_program_decision']['B1E_v2'] == 'ON_HOLD_PENDING_STRUCTURAL_DISCOVERY', 'Hold missing')
    for k in ('new_experiments_run','training_run','new_seeds_generated','final_seeds_generated','B1E_executed','scientific_code_modified'):
        require(state[k] is False, f'Unauthorized activity: {k}')
    for k in ('ready_for_b1s_implementation','ready_for_b1s_experiments','ready_for_b1s_confirmation','domain_selected','catalogue_map_complete'):
        require(state['readiness'][k] is False, f'Premature readiness: {k}')
    for k in ('ready_for_b1e_freeze_now','ready_for_b1e_confirmatory_run_now'):
        require(state['effective_program_decision'][k] is False, 'Hold overridden')
    patch = load(ROOT / 'b1s/EDITORIAL_PATCH.json')
    require(patch['kind'] == 'code' and patch['base_commit'] == BASE, 'Patch base mismatch')
    require(len(patch['files']) == 1 and patch['files'][0]['path'] == 'README.md', 'Scope mismatch')
    item = patch['files'][0]
    before = git('show', f'{BASE}:README.md')
    require(hashlib.sha1(b'blob '+str(len(before)).encode()+b'\0'+before).hexdigest() == item['base_blob_sha'], 'README base blob mismatch')
    after = transform(before.decode('utf-8'), item['edits']).encode('utf-8')
    current = (ROOT / 'README.md').read_bytes()
    require(current in (before, after), 'Concurrent README change')
    if prepare:
        (ROOT / 'README.md').write_bytes(after)
    else:
        require(current == after, 'README not updated')
    allowed = lambda p: p == 'README.md' or p == WORKFLOW or p.startswith('b1s/')
    changed = git('diff','--name-only',BASE,'--').decode().splitlines()
    untracked = git('ls-files','--others','--exclude-standard').decode().splitlines()
    require(all(allowed(p) for p in changed + untracked), 'Change outside documentary allowlist')
    original, now = tree(BASE), tree('HEAD')
    protected = 0
    for path, entry in original.items():
        if path != 'README.md':
            require(now.get(path) == entry, f'Historical Git object or mode changed: {path}')
            protected += 1
    pairs = load(ROOT / 'b1s/TRANSLATION_PAIRS.json')
    require(pairs['pairs'] == [], 'Unexpected local translation claim')
    translation = {'version':1,'pairs':[], 'scope':pairs['scope'], 'spanish_research_plan':pairs['spanish_research_plan']}
    index = ROOT / 'b1s/TRANSLATION_INDEX.json'
    if prepare:
        index.write_text(json.dumps(translation,indent=2)+'\n',encoding='utf-8')
    else:
        require(load(index) == translation, 'Translation reference changed')
    files = sorted(p for p in (ROOT/'b1s').rglob('*') if p.is_file() and p.name != 'PUBLICATION.json' and '__pycache__' not in p.parts)
    files += [ROOT/'README.md', ROOT/WORKFLOW]
    payload = {
        'version':'1.0.0','status':'B1S_DESIGN_PUBLISHED','kind':'code',
        'source_base_commit':BASE,
        'content_commit_before_generated_editorial_commit':git('rev-parse','HEAD').decode().strip(),
        'commit_semantics':'The enclosing publication commit is identified by Git, not self-hashed here.',
        'historical_git_objects_and_modes_preserved':protected,
        'editorial_changes':[{'path':'README.md','before_sha256':sha(before),'after_sha256':sha(after),'edit_ids':[e['id'] for e in item['edits']]}],
        'artifact_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},
        'new_experiments_run':False,'final_seeds_generated':False,'B1E_executed':False,
        'scientific_code_modified':False,'historical_results_changed':False,
        'B1E_v2':'ON_HOLD_PENDING_STRUCTURAL_DISCOVERY',
        'validation_scope':'Static JSON, exact README edits and historical Git objects/modes; no scientific experiment replay.'}
    out = ROOT/'b1s/PUBLICATION.json'
    if prepare:
        out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    else:
        saved=load(out)
        payload['content_commit_before_generated_editorial_commit']=saved['content_commit_before_generated_editorial_commit']
        require(saved == payload,'Publication manifest stale')
    print(json.dumps({'status':'PASS','protected_git_objects':protected,'hashed_artifacts':len(payload['artifact_sha256']),'experiments_run':False},indent=2))


def self_test():
    require(transform('abc',[{'id':'x','old':'b','new':'d'}]) == 'adc','Transformation failed')
    for action in (lambda:transform('aaa',[{'id':'x','old':'a','new':'z'}]),lambda:json.loads('{"x":1,"x":2}',object_pairs_hook=unique)):
        try:
            action()
        except ValueError:
            continue
        raise ValueError('Negative test accepted invalid input')
    print('{"self_tests":"PASS","positive_cases":1,"negative_cases":2}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['code'],default='code')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        validate(args.prepare)
