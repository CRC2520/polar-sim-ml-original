"""Read-only reconstruction and checkpoint replay audit for integrated R8.

Lives outside r8_completion, the frozen scientific source tree. Accept a campaign
directory containing seed_* directly or shard*/seed_*. No training or final
campaign is started. Evidence files are never replaced or edited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    if os.environ.get(variable) != '1':
        raise RuntimeError(f'Run with {variable}=1 to match the experiment environment')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r8_completion.integrated import load_agent, rollout
from r8_completion.io import atomic_json

DOMAINS = ('id', 'ood_delay2', 'ood_delay9')
VARIANTS = ('full', 'noEcology', 'noMemory', 'noNetwork', 'miscredit',
            'coordinate_equivalent', 'generic_joint', 'generic_flat', 'network_fixed', 'network_random')
HORIZONS = (1, 4, 12)
RETURN_HORIZON = 40
CLAIMS = ('E_credit', 'E_utility', 'T_utility', 'I_generic')
MARGINS = {
    'E_credit': {'relative_mse_reduction': .10, 'absolute_mse_reduction': 1e-4},
    'E_utility': {'resource_gain': .02, 'alive_noninferiority': -.01},
    'T_utility': {'resource_gain': .01, 'alive_noninferiority': -.005},
    'I_generic': {'alive_gain': .01, 'resource_noninferiority': -.02},
}
TOLERANCE = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def load_trace(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def assert_close(actual, expected, label, tolerance=TOLERANCE):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(label+': invalid shape or nonfinite data')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    if error > tolerance:
        raise AssertionError(f'{label}: absolute reconstruction error {error}')
    return error


def verify_complete(directory):
    manifest_path = directory/'COMPLETE.json'
    manifest = json_read(manifest_path)
    if manifest.get('complete') is not True:
        raise AssertionError('Manifest not marked complete: '+str(manifest_path))
    if manifest.get('metadata', {}).get('sources_unchanged') is not True:
        raise AssertionError('Manifest reports sources changed during execution')
    seen, zip_count, array_count = set(), 0, 0
    for entry in manifest['files']:
        path = (directory/entry['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or path in seen:
            raise AssertionError('Duplicate or escaping artifact path')
        seen.add(path)
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise AssertionError('Artifact size/SHA256 mismatch: '+str(path))
        if path.suffix == '.npz':
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise AssertionError('Invalid NPZ CRC: '+str(path))
                names = archive.namelist()
                if len(set(names)) != len(names):
                    raise AssertionError('Duplicate ZIP members: '+str(path))
            with np.load(path, allow_pickle=False) as z:
                if set(z.files) != set(entry.get('arrays', {})):
                    raise AssertionError('Manifest does not cover every NPZ array')
                for name in z.files:
                    value = z[name]
                    if value.dtype.hasobject:
                        raise AssertionError('Object arrays are forbidden')
                    expected = entry.get('arrays', {}).get(name)
                    if expected is not None and (list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']):
                        raise AssertionError('Manifest array schema mismatch')
                array_count += len(z.files)
            zip_count += 1
    expected_files = {path.resolve() for path in directory.iterdir()
                      if path.is_file() and path.name != 'COMPLETE.json'}
    if expected_files != seen:
        raise AssertionError('COMPLETE does not cover every evidence file')
    source = json_read(directory/'SOURCE.json')
    source_drift = {}
    for name, digest in source['source_sha256'].items():
        current = sha256(ROOT/name) if (ROOT/name).is_file() else None
        if current != digest:
            source_drift[name] = {'recorded': digest, 'current': current}
    return {'manifest_sha256': sha256(manifest_path), 'files_checked': len(seen),
            'npz_crc_checked': zip_count, 'arrays_schema_checked': array_count,
            'source_drift': source_drift, 'complete_metadata': manifest.get('metadata')}


def native_metrics(trace):
    """Independent metric reconstruction from physical posttransition arrays."""
    meta = json.loads(str(trace['metadata']))
    c = meta['config']
    capacity = c['agents']*c['capacity_per_agent']
    alive_before = trace['alive_before']
    alive_denominator = max(1, int(np.count_nonzero(alive_before)))
    action = trace['action']
    chosen_feasible = np.take_along_axis(trace['feasible'], action[..., None], axis=-1)[..., 0]
    metrics = {
        'alive_fraction': float(np.mean(trace['alive_after'])),
        'resource_fraction': float(np.mean(trace['resource_after_DIAGNOSTIC'])/capacity),
        'public_resource_fraction': float(np.mean(np.clip(trace['public_sequence'][1:, ..., 0], 0, 1))),
        'action_mean': float(np.sum(action)/alive_denominator),
        'guard_restriction_fraction': float(np.mean(~trace['feasible'])),
        'guard_violation_fraction': float(np.count_nonzero((~chosen_feasible) & alive_before)/alive_denominator),
        'uniform_probe_fraction': float(np.count_nonzero(trace['uniform_probe'])/alive_denominator),
    }
    tension = np.mean(np.abs(trace['previous_prediction']-trace['poles']), axis=-1)
    tension += .2*trace['poles'][..., 0]*trace['poles'][..., 1]
    timing = {
        'tension_formula_error': assert_close(tension, trace['tension'], 'operational tension'),
        'prediction_timing_error': assert_close(trace['previous_prediction'][1:], trace['issued_prediction'][:-1], 'previous forecast timing'),
        'next_poles_timing_error': assert_close(trace['next_poles'][:-1], trace['poles'][1:], 'next factual poles'),
        'initial_forecast_equals_current': bool(np.array_equal(trace['previous_prediction'][0], trace['poles'][0])),
    }
    return metrics, timing, meta


def prediction_metrics(checkpoint, trace):
    """No calls to trajectory_samples or predictive_metrics in scientific code."""
    count = len(trace['action'])-RETURN_HORIZON
    valid = trace['uniform_probe'][:count] & trace['alive_before'][:count]
    time, group, individual = np.nonzero(valid)
    if len(time) == 0:
        raise AssertionError('Prediction diagnosis has no randomized own-action probes')
    x = trace['features'][time, group, individual]
    a = trace['action'][time, group, individual]
    future = np.stack([trace['public_sequence'][time+h, group, individual] for h in HORIZONS], axis=1)
    result = {}
    for name, key in (('aligned', 'head_coef'), ('miscredit', 'shifted_head_coef')):
        block_coef = checkpoint[key].reshape(4, 16, 6)[a]
        pred = np.einsum('ni,nik->nk', x, block_coef).reshape(len(x), 3, 2)
        squared = (pred-future)**2
        result[name] = {'mse': float(np.mean(squared)),
                        'mse_by_horizon_variable': np.mean(squared, axis=0).tolist(),
                        'n_randomized_probes': len(time)}
    return result


def reconstruct_claims(prediction, native):
    by_domain = {claim: {} for claim in CLAIMS}
    values = {}
    for domain in DOMAINS:
        b = prediction[domain]['miscredit']['mse']
        c = prediction[domain]['aligned']['mse']
        credit_delta = b-c
        required = max(.10*b, 1e-4)
        full = native[domain]['full']
        no_eco = native[domain]['noEcology']
        no_net = native[domain]['noNetwork']
        joint = native[domain]['generic_joint']
        dr_eco = full['resource_fraction']-no_eco['resource_fraction']
        da_eco = full['alive_fraction']-no_eco['alive_fraction']
        dr_net = full['resource_fraction']-no_net['resource_fraction']
        da_net = full['alive_fraction']-no_net['alive_fraction']
        da_joint = full['alive_fraction']-joint['alive_fraction']
        dr_joint = full['resource_fraction']-joint['resource_fraction']
        by_domain['E_credit'][domain] = bool(credit_delta >= required)
        by_domain['E_utility'][domain] = bool(dr_eco >= .02 and da_eco >= -.01)
        by_domain['T_utility'][domain] = bool(dr_net >= .01 and da_net >= -.005)
        by_domain['I_generic'][domain] = bool(da_joint >= .01 and dr_joint >= -.02)
        values[domain] = {'credit_delta_mse': credit_delta, 'credit_required_delta': required,
                          'eco_delta_resource': dr_eco, 'eco_delta_alive': da_eco,
                          'network_delta_resource': dr_net, 'network_delta_alive': da_net,
                          'generic_delta_alive': da_joint, 'generic_delta_resource': dr_joint}
    return {'success': {claim: all(outcomes.values()) for claim, outcomes in by_domain.items()},
            'domain_success': by_domain, 'effects': values}


def audit_seed(directory, allow_source_drift=False):
    integrity = verify_complete(directory)
    if integrity['source_drift'] and not allow_source_drift:
        raise AssertionError('Scientific source differs from recorded SOURCE.json: '+str(directory))
    saved = json_read(directory/'result.json')
    seed = saved['seed']
    if set(saved['native']) != set(DOMAINS) or set(saved['prediction']) != set(DOMAINS):
        raise AssertionError('Result lacks the complete three-domain protocol')
    if any(set(saved['native'][domain]) != set(VARIANTS) for domain in DOMAINS):
        raise AssertionError('Result lacks the complete ten-variant protocol')
    checkpoint = load_trace(directory/'frozen_agent.npz')
    checkpoint_meta = json.loads(str(checkpoint['metadata']))
    if checkpoint_meta['seed'] != seed:
        raise AssertionError('Checkpoint seed mismatch')
    networks = {name: json.loads(str(checkpoint[field])) for name, field in
                (('learned', 'network_state_json'), ('joint', 'generic_network_state_json'))}
    learned_diag, joint_diag = (networks[name]['fit_diagnostics'] for name in ('learned', 'joint'))
    if learned_diag['training_arrays_sha256'] != joint_diag['training_arrays_sha256']:
        raise AssertionError('Learned/Joint acquisition data mismatch')
    if (learned_diag['local_parameters']+learned_diag['cross_parameters'] != 1536
            or joint_diag['local_parameters']+joint_diag['cross_parameters'] != 1536):
        raise AssertionError('Capacity mismatch')
    native, prediction, timing, equivalence, zero_gate = {}, {}, {}, {}, {}
    representation = {}
    maximum_metric_error, maximum_prediction_error = 0., 0.
    for domain in DOMAINS:
        native[domain], timing[domain] = {}, {}
        physical = {}
        for variant, expected in saved['native'][domain].items():
            trace = load_trace(directory/f'test_{domain}_{variant}.npz')
            metrics, causal_timing, meta = native_metrics(trace)
            if meta['seed'] != seed or meta['episode'] != 700 or meta['domain'] != domain or meta['variant'] != variant or meta['learn']:
                raise AssertionError('Test trace identity mismatch')
            for key, value in metrics.items():
                maximum_metric_error = max(maximum_metric_error, assert_close(value, expected[key], f'{seed}/{domain}/{variant}/{key}'))
            if str(trace['transition_sha256']) != expected['transition_sha256']:
                raise AssertionError('Stored transition identifier mismatch')
            if metrics['guard_violation_fraction'] != 0 or metrics['uniform_probe_fraction'] != 0:
                raise AssertionError('Native evaluation violates declared guard/probe contract')
            native[domain][variant], timing[domain][variant] = metrics, causal_timing
            if variant in ('full', 'noNetwork', 'coordinate_equivalent', 'generic_flat'):
                physical[variant] = {key: value.copy() for key, value in trace.items()
                                     if key != 'metadata'}
            if variant == 'full':
                live_p = trace['poles'][trace['alive_before']]
                flat = live_p.reshape(-1, 16)
                joined = np.column_stack((flat, trace['tension'][trace['alive_before']]))
                representation[domain] = {
                    'scope': 'full native test, observations with alive_before=true',
                    'observations': len(flat),
                    'centered_poles_16_rank': int(np.linalg.matrix_rank(flat-flat.mean(0))),
                    'centered_poles_and_tension_24_rank': int(np.linalg.matrix_rank(joined-joined.mean(0))),
                    'desire_authenticity_complement_error': float(np.max(np.abs(live_p[:,4,0]+live_p[:,7,1]-1.))),
                    'limit_recognition_complement_error': float(np.max(np.abs(live_p[:,4,1]+live_p[:,7,0]-1.))),
                    'both_channels_above_0_05_fraction_by_pair': np.mean(np.all(live_p > .05, axis=-1), axis=0).tolist(),
                }
        equivalence[domain] = {}
        for variant in ('coordinate_equivalent', 'generic_flat'):
            ref, other = physical['full'], physical[variant]
            equivalence[domain][variant] = {
                'actions_exact': bool(np.array_equal(ref['action'], other['action'])),
                'alive_exact': bool(np.array_equal(ref['alive_after'], other['alive_after'])),
                'resource_exact': bool(np.array_equal(ref['resource_after_DIAGNOSTIC'], other['resource_after_DIAGNOSTIC'])),
                'max_scores_error': float(np.max(np.abs(ref['scores']-other['scores']))),
            }
        applicable = checkpoint_meta['network_gate'] == 0.
        exact_keys = {key: bool(np.array_equal(value, physical['noNetwork'][key])) for key, value in physical['full'].items()}
        zero_gate[domain] = {'applicable': applicable, 'all_fields_exact': all(exact_keys.values()), 'field_checks': exact_keys}
        if applicable and not all(exact_keys.values()):
            raise AssertionError('Gate0 full/noNetwork discrepancy')
        trace = load_trace(directory/f'prediction_{domain}.npz')
        meta = json.loads(str(trace['metadata']))
        if meta['episode'] != 650 or meta['learn'] or meta['seed'] != seed or meta['domain'] != domain:
            raise AssertionError('Prediction episode identity mismatch')
        prediction[domain] = prediction_metrics(checkpoint, trace)
        for name in ('aligned', 'miscredit'):
            for key in ('mse', 'mse_by_horizon_variable', 'n_randomized_probes'):
                maximum_prediction_error = max(maximum_prediction_error, assert_close(
                    prediction[domain][name][key], saved['prediction'][domain][name][key],
                    f'{seed}/{domain}/{name}/{key}'))
    for variant, selection in saved['selection'].items():
        if selection['selection_uses_test'] or selection['episode'] != 500:
            raise AssertionError('Gate selection uses wrong split')
        if len(selection['candidates']) != 12:
            raise AssertionError('Unequal gate selection budget')
        candidates = {}
        for name, value in selection['candidates'].items():
            match = re.fullmatch(r'ecology=([\d.]+),network=([\d.]+)', name)
            if match is None:
                raise AssertionError('Unparseable development candidate')
            candidates[tuple(map(float, match.groups()))] = value
        if set(candidates) != {(w, g) for w in (0., 2., 8.) for g in (0., .25, .5, 1.)}:
            raise AssertionError('Development candidate budget mismatch')
        reference = candidates[(0., 0.)]['alive_fraction']
        feasible = [(key, value) for key, value in candidates.items()
                    if value['alive_fraction'] >= reference-.005]
        chosen = max(feasible, key=lambda item: (item[1]['public_resource_fraction'], -item[0][0], -item[0][1]))[0]
        if list(chosen) != selection['selected']:
            raise AssertionError('Gate selection violates viability or selection objective')
        keys = ('ecology_weight', 'network_gate') if variant == 'full' else ('generic_ecology_weight', 'generic_gate')
        if chosen != tuple(checkpoint_meta[key] for key in keys):
            raise AssertionError('Selected gates differ from frozen checkpoint')
    coordinate_matches = all(equivalence[d]['coordinate_equivalent']['actions_exact'] for d in DOMAINS)
    if coordinate_matches != saved['coordinate_action_equal']:
        raise AssertionError('Coordinate outcome flag mismatch')
    return {'seed': seed, 'directory': str(directory), 'integrity': integrity,
            'native': native, 'prediction': prediction, 'causal_timing': timing,
            'equivalence': equivalence, 'gate_zero_checks': zero_gate,
            'checkpoint_gates': {key: checkpoint_meta[key] for key in
                                 ('ecology_weight', 'network_gate', 'generic_ecology_weight', 'generic_gate')},
            'network_identification': {'learned': learned_diag, 'joint': joint_diag},
            'representation_diagnostics_secondary': representation,
            'maximum_native_metric_error': maximum_metric_error,
            'maximum_prediction_metric_error': maximum_prediction_error,
            'criteria': reconstruct_claims(prediction, native)}


def checkpoint_replay(directory):
    records = []
    agent = load_agent(directory/'frozen_agent.npz')
    for domain in DOMAINS:
        original = load_trace(directory/f'test_{domain}_full.npz')
        meta = json.loads(str(original['metadata']))
        replay, outcome = rollout(agent, meta['episode'], domain, 'full', learn=False,
                                  ecology_weight=meta['ecology_weight'], gate=meta['gate'],
                                  steps=meta['config']['steps'])
        if set(replay) != set(original):
            raise AssertionError('Replay trace key mismatch')
        differences = [key for key in original if not np.array_equal(original[key], replay[key])]
        if differences:
            raise AssertionError(f'Replay mismatch {domain}: {differences}')
        records.append({'seed': agent.seed, 'domain': domain, 'episode': meta['episode'],
                        'arrays_compared': len(original), 'all_arrays_exact': True,
                        'transition_sha256': outcome['transition_sha256']})
    return records


def audit(input_dir, expected_seeds=None, allow_source_drift=False):
    input_dir = Path(input_dir).resolve()
    directories = [path for path in input_dir.rglob('seed_*') if path.is_dir()
                   and re.fullmatch(r'seed_\d+', path.name)]
    if re.fullmatch(r'seed_\d+', input_dir.name):
        directories.append(input_dir)
    directories = sorted(set(directories), key=lambda path: (int(path.name.split('_')[1]), str(path)))
    seeds = [int(path.name.split('_')[1]) for path in directories]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need unique, nonempty seed directories; select one campaign root')
    if expected_seeds is not None and seeds != sorted(expected_seeds):
        raise ValueError(f'Expected seeds {sorted(expected_seeds)}, observed {seeds}')
    if allow_source_drift and any(not 940101 <= seed <= 940199 for seed in seeds):
        raise ValueError('Source drift exemption is restricted to observed development seeds')
    if any(not (path/'COMPLETE.json').is_file() for path in directories):
        raise ValueError('Campaign includes incomplete seed directories')
    before = {str(path/'COMPLETE.json'): sha256(path/'COMPLETE.json') for path in directories}
    rows = []
    for directory in directories:
        rows.append(audit_seed(directory, allow_source_drift))
        print(f'audited integrated seed {rows[-1]["seed"]}', flush=True)
    replay = checkpoint_replay(directories[0])
    if any(sha256(path) != digest for path, digest in before.items()):
        raise AssertionError('Evidence changed during audit')
    successes = {claim: [row['criteria']['success'][claim] for row in rows] for claim in CLAIMS}
    return {'audit_schema': 'r8-integrated-audit-v1', 'input': str(input_dir), 'seeds': seeds,
            'audit_script_sha256': sha256(__file__), 'audit_threads': {'OPENBLAS_NUM_THREADS': 1, 'OMP_NUM_THREADS': 1},
            'pilot_thread_limitation': ('Pilot V2 explicitly set OPENBLAS_NUM_THREADS=1; OMP_NUM_THREADS was not explicitly set by its runner.' if allow_source_drift else None),
            'expected_domains': DOMAINS, 'primary_margins': MARGINS,
            'all_artifact_hashes_and_crcs_verified': True,
            'all_native_metrics_reconstructed': True, 'all_prediction_metrics_reconstructed': True,
            'source_drift_permitted_for_pilot': allow_source_drift,
            'evidence_unchanged': True, 'no_training_or_new_campaign_executed': True,
            'first_seed_three_domain_checkpoint_replay': replay,
            'success_vectors': successes, 'wins': {claim: sum(v) for claim, v in successes.items()},
            'coordinate_domain_action_matches': sum(row['equivalence'][d]['coordinate_equivalent']['actions_exact'] for row in rows for d in DOMAINS),
            'generic_flat_domain_action_matches': sum(row['equivalence'][d]['generic_flat']['actions_exact'] for row in rows for d in DOMAINS),
            'total_seed_domains': len(rows)*3,
            'gate_zero_seeds': sum(row['checkpoint_gates']['network_gate'] == 0 for row in rows),
            'maximum_native_metric_error': max(row['maximum_native_metric_error'] for row in rows),
            'maximum_prediction_metric_error': max(row['maximum_prediction_metric_error'] for row in rows),
            'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--expected-seeds', nargs='+', type=int)
    p.add_argument('--allow-pilot-source-drift', action='store_true')
    a = p.parse_args()
    if a.report.resolve().is_relative_to(a.input.resolve()):
        p.error('Audit report must be outside the evidence directory')
    if a.allow_pilot_source_drift and a.expected_seeds and any(not 940101 <= s <= 940199 for s in a.expected_seeds):
        p.error('Source drift exemption is restricted to development seeds')
    result = audit(a.input, a.expected_seeds, a.allow_pilot_source_drift)
    atomic_json(a.report, result)
    print(json.dumps({key: result[key] for key in ('seeds', 'wins', 'gate_zero_seeds',
          'coordinate_domain_action_matches', 'maximum_native_metric_error', 'maximum_prediction_metric_error')}, indent=2))


if __name__ == '__main__':
    main()
