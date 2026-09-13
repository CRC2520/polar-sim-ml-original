"""Read-only v1.1 validation with immutable v1 pins and additive path allowlist."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PINS = {
    'code': ('c27235deb1688c105bfe96597dfd233108307895', 'b1/B1D_MANIFEST.json', 'e5de2f406cf34ea5ce122443890ac9344e4df68206edd267dd1ff7eaadb7dc2b'),
    'document': ('d8926e834256d559bb1855fce429cf4753ce5c9e', 'b1/B1D_FREEZE.json', 'e5494377608d5fae6133244091fe87bf6a8205dee2159f94d3950a1e723f8691'),
}
WORKFLOW = '.github/workflows/b1d-v1-1-validation.yml'
B0_SHA = '7d9abe376401ca53c8e8f9f519b75c6bb227dfc49749059bb72eb24050978cc1'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('duplicate JSON key: ' + k)
        out[k] = v
    return out

def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def safe(root, path):
    target = (root / path).resolve()
    if Path(path).is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError('unsafe artifact path: ' + path)
    return target

def hashes(root, mapping):
    for path, expected in mapping.items():
        if not re.fullmatch('[a-f0-9]{64}', expected) or digest(safe(root, path)) != expected:
            raise ValueError('hash mismatch: ' + path)

def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])

def allowed(path):
    return path.startswith('b1/v1_1/') or path == WORKFLOW

def tree(root, ref):
    records = {}
    for row in git(root, 'ls-tree', '-rz', ref).split(b'\0'):
        if row:
            meta, path = row.split(b'\t', 1)
            records[path.decode()] = meta.decode()
    return records

def closure_evidence(root, kind, sparse=False):
    prefix = root / 'b1/v1_1'
    evidence = prefix if kind == 'code' else prefix / 'code_exports'
    result = read(evidence / 'B1D_V1_1_RESULTS.json')
    p5 = read(evidence / 'p5/results/RESULTS.json')
    resource = read(evidence / 'RESOURCE_MATCHING_CERTIFICATE.json')
    export = read(evidence / 'EXPORTER_V1_1_AUDIT.json')
    stats = read(evidence / 'B1E_STATISTICAL_REDESIGN.json')
    sim = read(evidence / 'statistics/SIMULATION_RESULTS.json')
    capacity = read(evidence / 'B1E_RESOURCE_CERTIFICATE.json')
    for obj in (result, p5, resource, export, stats, sim, capacity):
        if obj.get('development_only') is not True or obj.get('confirmatory') is not False or obj.get('reusable_as_final') is not False:
            raise ValueError('development-only evidence flags missing')
        if obj.get('final_seeds_generated', False) or obj.get('B1E_executed', False):
            raise ValueError('prohibited confirmatory activity')
    if p5['status'] != 'PASS' or p5['C6_status'] != 'PASS' or p5['episode_count'] != 288 or p5['event_count'] != 27648:
        raise ValueError('P5/C6 completeness gate')
    if not p5['physical_bound_pass'] or not p5['full_refill_witness_separate'] or p5['positive_utility_eligible'] or p5['positive_pairing_specificity_eligible']:
        raise ValueError('P5 scientific disposition gate')
    if any(p5[k] for k in ('controller_failures','guardrail_violations','resource_cap_binding')):
        raise ValueError('P5 execution gate')
    if resource['status'] != 'CERTIFIED_WITH_EXPLICIT_SCOPE' or resource['blocker_status'] != 'RESOLVED' or resource['resource_cap_binding']:
        raise ValueError('resource certification gate')
    if len(resource['required_axes']) != 16 or resource['axis_rows'] != 288:
        raise ValueError('incomplete resource axes')
    for pilot in ('P5','P6','P7'):
        for comp in ('C0','C1','C2','C3','C4'):
            if resource['invariants_pass'][pilot][comp] is not True:
                raise ValueError('decisive resource invariant failure')
    if not resource['old_vs_new_action_and_memory_equal'] or result['P7']['scientific_rerun'] or result['P7']['retuning']:
        raise ValueError('P7 preservation gate')
    ar = export['archive_replay']
    if export['status'] != 'PASS' or not ar['coverage_pass'] or ar['diagnostic_rows_exported'] != 96 or ar['validated_records'] != 82176:
        raise ValueError('export completeness gate')
    if ar['original_rows_byte_identical'] != 42 or ar['previously_derived_rows_byte_identical'] != 54 or export['tests']['status'] != 'PASS':
        raise ValueError('export provenance/corruption gate')
    if export['QA']['trace_records'] != 1536 or export['QA']['controller_failures'] or export['QA']['guardrail_violations']:
        raise ValueError('export end-to-end QA gate')
    if not export['historical_preservation']['all_unchanged'] or export['original_export_status'] != 'FAILED_RETAINED_UNCHANGED':
        raise ValueError('historical incident preservation gate')
    if stats['status'] != 'PROSPECTIVE_METHOD_VALIDATED' or sim['status'] != 'PASS' or stats['N_v2'] != capacity['N'] or stats['N_v2'] != sim['N_v2']:
        raise ValueError('prospective fixed-N method gate')
    if not capacity['feasible'] or capacity['disk']['all_failures_total_bytes'] >= capacity['disk']['available_free_bytes']:
        raise ValueError('resource feasibility gate')
    if kind == 'code':
        fpath = 'b1/v1_1/p5/P5_SOURCE_FREEZE.json'
        freeze = read(root / fpath)
        if digest(root / fpath) != p5['freeze_sha256']:
            raise ValueError('P5 source freeze changed')
        hashes(root, freeze['artifact_sha256'])
        sys.path.insert(0, str(root))
        from b1.v1_1.p5.verify_results import verify
        verified = verify(evidence / 'p5/results')
        if verified['status'] != 'PASS':
            raise ValueError('P5 archive verification failed')
        if not sparse:
            source = p5['source_commit']
            git(root, 'merge-base', '--is-ancestor', source, 'HEAD')
            if git(root, 'show', source + ':' + fpath) != (root / fpath).read_bytes():
                raise ValueError('published P5 freeze identity mismatch')
            for path, expected in freeze['artifact_sha256'].items():
                if hashlib.sha256(git(root, 'show', source + ':' + path)).hexdigest() != expected:
                    raise ValueError('published P5 source mismatch: ' + path)
            if 'b1/v1_1/p5/results/RESULTS.json' in tree(root, source):
                raise ValueError('P5 result already existed at prospective source freeze')
    return True

def validate(root, kind, sparse=False, pre_freeze=False):
    root = Path(root).resolve()
    base, oldpath, oldhash = PINS[kind]
    if digest(root / oldpath) != oldhash:
        raise ValueError('immutable v1 manifest changed')
    old = read(root / oldpath)
    hashes(root, old['artifact_hashes'])
    b0path = root / ('b0/B0_FREEZE.json' if kind == 'document' else 'b1/contracts/frozen_b0/B0_FREEZE.json')
    if digest(b0path) != B0_SHA:
        raise ValueError('B0 freeze changed')
    if kind == 'document':
        hashes(root, read(b0path)['artifact_sha256'])
    git_count = None
    if not sparse:
        git(root, 'merge-base', '--is-ancestor', base, 'HEAD')
        before, after = tree(root, base), tree(root, 'HEAD')
        for path, meta in before.items():
            if after.get(path) != meta:
                raise ValueError('historical tree changed: ' + path)
            mode, typ, sha = meta.split()
            p = root / path
            data = os.readlink(p).encode() if mode == '120000' else p.read_bytes()
            if typ != 'blob' or hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != sha:
                raise ValueError('historical working bytes changed: ' + path)
            if mode in {'100644', '100755'} and bool(p.stat().st_mode & 0o111) != (mode == '100755'):
                raise ValueError('historical executable mode changed: ' + path)
        for path in after.keys() - before.keys():
            if not allowed(path):
                raise ValueError('addition outside allowlist: ' + path)
        for name in git(root, 'diff', '--name-only', '-z', 'HEAD').split(b'\0'):
            if name and not allowed(name.decode()):
                raise ValueError('working change outside allowlist: ' + name.decode())
        git_count = len(before)
    prefix = root / 'b1/v1_1'
    code_translations = prefix / 'translations/es/TRANSLATION_INDEX.json'
    if kind == 'code' and code_translations.exists():
        ti = read(code_translations)
        if ti['translation_count'] != len(ti['entries']):
            raise ValueError('translation count mismatch')
        for row in ti['entries']:
            if digest(safe(root, row['source_path'])) != row['source_sha256'] or digest(safe(root, row['translation_path'])) != row['translation_sha256']:
                raise ValueError('code translation provenance mismatch')
    structured = 0
    for p in prefix.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if re.search(r'(^|/)(final_seeds|confirmatory_results|b1e_results)([./_]|$)', str(p.relative_to(prefix)), re.I):
            raise ValueError('prohibited final execution artifact: ' + str(p))
        if p.suffix == '.json':
            read(p)
            structured += 1
        if p.suffix == '.csv':
            rows = list(csv.reader(p.open(newline='')))
            if not rows or len(set(rows[0])) != len(rows[0]) or any(len(r) != len(rows[0]) for r in rows):
                raise ValueError('invalid CSV: ' + str(p))
            structured += 1
    name = 'B1D_V1_1_MANIFEST.json' if kind == 'code' else 'B1D_V1_1_FREEZE.json'
    own = 'b1/v1_1/' + name
    if not pre_freeze:
        m = read(prefix / name)
        hashes(root, m['artifact_hashes'])
        expected = {str(p.relative_to(root)) for p in prefix.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'} - {own}
        expected.add(WORKFLOW)
        if set(m['artifact_hashes']) != expected:
            raise ValueError('v1.1 manifest inventory mismatch')
        if m.get('final_seeds_generated') is not False or m.get('B1E_executed') is not False or m.get('ready_for_b1e_confirmatory_run') is not False:
            raise ValueError('confirmatory execution prohibition missing')
        matrix = list(csv.DictReader((prefix / 'BLOCKER_CLOSURE_MATRIX.csv').open()))
        ids = {'B1-CONFLICT-P5-C4-REFILL-001','B1-RESOURCE-CERTIFICATE-PENDING','B1D-ORIGINAL-EXPORT-PROVENANCE-INCIDENT','B1E-RESOURCE-INFEASIBLE-CURRENT-DISK'}
        if len(matrix) != 4 or {r['blocker_id'] for r in matrix} != ids:
            raise ValueError('blocker matrix cardinality')
        blocked = [r['blocker_id'] for r in matrix if r['v1_1_status'] == 'STILL_BLOCKED']
        if any(r['v1_1_status'] not in {'RESOLVED','MITIGATED','STILL_BLOCKED','NOT_APPLICABLE'} for r in matrix):
            raise ValueError('invalid blocker status')
        if m['status'] not in {'B1D_BLOCKED', 'B1D_COMPLETE'} or (blocked and m['status'] != 'B1D_BLOCKED'):
            raise ValueError('unsupported completion claim')
        if sorted(m['unresolved_blockers']) != sorted(blocked):
            raise ValueError('manifest/matrix blocker mismatch')
        if m['status'] == 'B1D_COMPLETE' or m.get('ready_for_b1e_freeze'):
            if any(r['v1_1_status'] != 'RESOLVED' for r in matrix):
                raise ValueError('completion requires resolved essential blockers')
            closure_evidence(root, kind, sparse)
            if not m.get('CI_runs') or any(r.get('conclusion') != 'success' or not re.fullmatch('[a-f0-9]{40}', r.get('commit','')) for r in m['CI_runs']):
                raise ValueError('actual successful content CI evidence required')
    elif (prefix / ('B1D_V1_1_RESULTS.json' if kind == 'code' else 'code_exports/B1D_V1_1_RESULTS.json')).exists():
        closure_evidence(root, kind, sparse)
    if kind == 'document' and (prefix / 'CODE_EXPORT_PROVENANCE.json').exists():
        provenance = read(prefix / 'CODE_EXPORT_PROVENANCE.json')
        if provenance['repository'] != 'CRC2520/polar-sim-ml-original' or not re.fullmatch('[a-f0-9]{40}', provenance['commit']):
            raise ValueError('invalid pinned code repository/commit')
        for row in provenance['exports']:
            p = safe(root, row['local_path'])
            if digest(p) != row['sha256']:
                raise ValueError('import mismatch: ' + str(p))
            data = p.read_bytes()
            if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != row['git_blob_sha']:
                raise ValueError('import Git blob mismatch')
        ti = prefix / 'TRANSLATION_INDEX.json'
        if ti.exists():
            for row in read(ti)['translations']:
                if digest(root / row['source_path']) != row['source_sha256'] or digest(root / row['spanish_path']) != row['spanish_sha256']:
                    raise ValueError('translation provenance mismatch')
    return {'status': 'PASS', 'scope': 'sparse_local_hash_validation' if sparse else 'full_git_ancestry_trees_working_bytes', 'historical_git_blobs_verified': git_count, 'v1_artifact_hashes_verified': len(old['artifact_hashes']), 'v1_1_structured_files': structured, 'final_seeds_generated': False, 'B1E_executed': False}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='.')
    p.add_argument('--kind', choices=PINS, required=True)
    p.add_argument('--sparse', action='store_true')
    p.add_argument('--pre-freeze', action='store_true')
    a = p.parse_args()
    print(json.dumps(validate(a.repo_root, a.kind, a.sparse, a.pre_freeze), indent=2))
