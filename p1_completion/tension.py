"""P1 intervention and external-task audit of the unchanged W/K extension.

The network is configured, never learned here. Its mechanism and usefulness are
different endpoints. See TENSION_DESIGN.md for the pre-final specification.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_network_v01'
# Load the original package in a dedicated namespace: avoid collision with the
# other P1 branches that import the (identical) contextual parent as `polar`.
spec = importlib.util.spec_from_file_location('_p1_frozen_polar', SOURCE/'polar/__init__.py',
                                             submodule_search_locations=[str(SOURCE/'polar')])
frozen = importlib.util.module_from_spec(spec)
sys.modules.setdefault('_p1_frozen_polar', frozen)
spec.loader.exec_module(frozen)
saved_polar = {k: v for k, v in sys.modules.items() if k == 'polar' or k.startswith('polar.')}
for k in list(saved_polar):
    del sys.modules[k]
sys.modules['polar'] = frozen
sys.modules['polar.model'] = sys.modules['_p1_frozen_polar.model']
nspec = importlib.util.spec_from_file_location('_p1_frozen_network', SOURCE/'network_tension.py')
network = importlib.util.module_from_spec(nspec)
nspec.loader.exec_module(network)
for k in [k for k in sys.modules if k == 'polar' or k.startswith('polar.')]:
    del sys.modules[k]
sys.modules.update(saved_polar)
from _p1_frozen_polar.model import ModelConfig, ContextualPolarModel
from _p1_frozen_polar.tasks import make_trial, observation, jsonable
NetworkTensionModel = network.NetworkTensionModel

VARIANTS = ('full', 'no_network', 'w_only', 'k_only', 'edge_lesion',
            'reversed', 'permuted', 'coordinate', 'generic_flat', 'recurrent')
TASK_VARIANTS = VARIANTS + ('recurrent_off',)
PILOT_SEEDS = tuple(range(930201, 930204))
FINAL_SEEDS = tuple(range(934001, 934031))
TRACE_KEYS = ('q_before', 'action', 'proposal', 'effective_target', 'planning_gain')
NETWORK_KEYS = ('tension', 'mismatch', 'conflict', 'base_proposal', 'state_term',
                'tension_term', 'eta', 'desired_effort')


def canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = [ROOT/'tension.py', ROOT/'tests_tension.py', ROOT/'TENSION_DESIGN.md']
    files += sorted(p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def check_frozen_blobs():
    manifest = json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
    for path, entry in manifest['files'].items():
        content = (SOURCE/path).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob != entry['git_blob']:
            raise ValueError('Original-source blob mismatch: '+path)


def couplings(agents=1, variant='full', seed=0):
    """Seven-type directed cycle, eighth isolated; configured test hypothesis."""
    pairs = agents*8
    w, k = np.zeros((pairs*2, pairs*2)), np.zeros((pairs*2, pairs))
    permutation = np.random.default_rng(np.random.SeedSequence([seed, 827])).permutation(8)
    for unit in range(agents):
        for s in range(7):
            d = (s+1) % 7
            if variant == 'edge_lesion' and (s, d) == (0, 1):
                continue
            if variant == 'reversed':
                s, d = d, s
            if variant == 'permuted':
                s, d = int(permutation[s]), int(permutation[d])
            s, d = unit*8+s, unit*8+d
            w[2*d, 2*s], w[2*d+1, 2*s+1] = .12, .08
            k[2*d, s], k[2*d+1, s] = .18, -.14
    if variant in ('no_network', 'recurrent_off', 'k_only'):
        w.fill(0)
    if variant in ('no_network', 'recurrent_off', 'w_only'):
        k.fill(0)
    return w, k


class FlatGeneric(NetworkTensionModel):
    """Same function in flat non-semantic coordinates: consistency control."""
    def propose_action(self, effective_target, weights, horizon):
        super().propose_action(effective_target, weights, horizon)
        n = self._network_trace
        u = np.asarray(n['base_proposal']).ravel()
        eta = np.asarray(n['eta']).ravel()
        generic_features = np.asarray(n['tension']).ravel()
        out = u + eta*(self.W@self.q.ravel()+self.K@generic_features)
        self._network_trace['network_proposal'] = out.reshape(self.shape).tolist()
        return out.reshape(self.shape)


class RecurrentNetwork(NetworkTensionModel):
    """Original generic gradient recurrence plus unchanged W/K mechanisms."""
    def propose_action(self, effective_target, weights, horizon):
        super().propose_action(effective_target, weights, horizon)
        n = self._network_trace
        eta = np.asarray(n['eta'])
        weight = np.asarray(weights)/max(float(np.max(weights)), 1e-12)
        gain = self._planning_gain
        base = np.clip(self.q+eta*weight*gain*(effective_target-gain*self.q), 0., 1.)
        result = base+eta*(np.asarray(n['state_term'])+np.asarray(n['tension_term']))
        n['base_proposal'], n['network_proposal'] = base.tolist(), result.tolist()
        n['base_update_rule'] = 'capacity_matched_projected_gradient'
        return result


class TensionClamp(NetworkTensionModel):
    """Controlled mediator intervention, not a change to the original source."""
    forced_tension = None

    def propose_action(self, effective_target, weights, horizon):
        super().propose_action(effective_target, weights, horizon)
        if self.forced_tension is None:
            raise ValueError('Mediator clamp requires same-time sham tension')
        n = self._network_trace
        forced = np.asarray(self.forced_tension, float)
        if forced.shape != self.shape[:-1] or not np.isfinite(forced).all():
            raise ValueError('Invalid sham tension')
        n['unclamped_tension'] = n['tension']
        n['tension'] = forced.tolist()
        edges = self.K*forced.reshape(1, -1)
        term = edges.sum(axis=1).reshape(self.shape)
        result = np.asarray(n['base_proposal'])+np.asarray(n['eta'])*(np.asarray(n['state_term'])+term)
        n['tension_term'], n['tension_edges'] = term.tolist(), edges.tolist()
        n['network_proposal'] = result.tolist()
        return result


def model(variant, seed, agents=1, mechanism=True):
    cfg = ModelConfig(agents=agents, types=8, seed=seed,
                      use_memory=not mechanism, use_self_model=not mechanism,
                      representation='signed_intensity' if variant == 'coordinate' else 'dual_pole')
    w, k = couplings(agents, variant, seed)
    cls = (FlatGeneric if variant == 'generic_flat' else
           RecurrentNetwork if variant in ('recurrent', 'recurrent_off') else
           TensionClamp if variant == 'tau_clamp' else NetworkTensionModel)
    return cls(cfg, state_coupling=w, tension_coupling=k,
               incompatibility=np.full((agents, 8), .5 if mechanism else 0.))


def compact_trace(m):
    t = m.last_trace
    r = {key: np.asarray(t[key]) for key in TRACE_KEYS}
    r.update({key: np.asarray(t['network'][key]) for key in NETWORK_KEYS})
    r['resource_scale'] = np.asarray(t['constraints']['resource_scale'])
    r['cost'] = np.asarray(t['constraints']['resource_cost'])
    return r


def stack_records(records):
    return {key: np.stack([r[key] for r in records]) for key in records[0]}


def pulse_pair(seed, source, pole, sigma, variant, steps=12):
    rng = np.random.default_rng(np.random.SeedSequence([seed, source, pole, 179]))
    initial = rng.uniform(.15, .45, (1, 8, 2))
    targets = np.clip(initial[None]+rng.normal(0, sigma, (steps, 1, 8, 2)), 0, 1)
    # Clamp's sham follows the unmodified full controller; its τ is a prescribed
    # mediator input for the perturbed branch at the corresponding same time.
    sham = model('full' if variant == 'tau_clamp' else variant, seed)
    pulse = model(variant, seed)
    sham.q, pulse.q = initial, initial
    changed = initial.copy()
    changed[0, source, pole] += .2
    pulse.q = changed
    a, b = [], []
    for target in targets:
        obs = {'target': target, 'budget': 16.}
        u0 = sham.act(obs)
        if variant == 'tau_clamp':
            pulse.forced_tension = np.asarray(sham.last_trace['network']['tension'])
        u1 = pulse.act(obs)
        sham.learn({'effect': u0})
        pulse.learn({'effect': u1})
        a.append(compact_trace(sham)); b.append(compact_trace(pulse))
    arrays = {'initial': initial, 'intervened_initial': changed, 'targets': targets,
              'W': pulse.W, 'K': pulse.K, 'chi': pulse.chi}
    arrays.update({'sham_'+k: v for k, v in stack_records(a).items()})
    arrays.update({'pulse_'+k: v for k, v in stack_records(b).items()})
    return arrays


def graph_distances(w, k, source):
    adjacency = np.abs(w).reshape(8, 2, 8, 2).sum(axis=(1, 3)) + np.abs(k).reshape(8, 2, 8).sum(axis=1)
    dist = np.full(8, -1, dtype=int); dist[source] = 0
    queue = [source]
    for s in queue:
        for dest in np.flatnonzero(adjacency[:, s]):
            if dist[dest] < 0:
                dist[dest] = dist[s]+1; queue.append(int(dest))
    return dist


def pulse_metrics(a, source):
    delta = a['pulse_action']-a['sham_action']
    amplitude = np.max(np.abs(delta), axis=-1)[:, 0, :]
    onset = [int(np.flatnonzero(amplitude[:, i] > 1e-10)[0]+1)
             if np.any(amplitude[:, i] > 1e-10) else None for i in range(8)]
    distances = graph_distances(a['W'], a['K'], source)
    nonlocal_mask = np.arange(8) != source
    off_path = distances < 0
    violations = ((a['pulse_action'] < -1e-12).any() or (a['pulse_action'] > 1+1e-12).any()
                  or (a['pulse_cost'] > 16+1e-12).any())
    return {'max_cross_polarity_effect': float(np.max(amplitude[:, nonlocal_mask])),
            'cross_effect_l1': float(np.sum(np.abs(delta[:, :, nonlocal_mask, :]))),
            'off_path_max': float(np.max(amplitude[:, off_path])) if off_path.any() else 0.,
            'onset_steps': onset, 'graph_distances': distances.tolist(),
            'early_arrivals': sum(onset[i] is not None and onset[i] < distances[i]
                                  for i in range(8) if distances[i] > 0),
            'reached_expected_nodes': sum(onset[i] is not None for i in range(8) if distances[i] > 0),
            'expected_reachable_nodes': int(np.count_nonzero(distances > 0)),
            'resource_scale_min': float(min(a['pulse_resource_scale'].min(), a['sham_resource_scale'].min())),
            'violations': int(violations)}


def task_trial(seed, task, variant):
    trial = make_trial(task, seed, 'heldout', agents=3, types=8)
    m = model(variant, seed, agents=3, mechanism=False)
    records = []
    for f in trial.steps:
        action = m.act(observation(f))
        effect = np.clip(action*f['gains']+f['noise'], 0., 1.)
        m.learn({'effect': effect})
        rec = compact_trace(m)
        rec.update(effect=effect, target=f['target'], observed=f['observed'], gains=f['gains'],
                   weights=f['weights'], costs=f['costs'], allowed=f['allowed'], noise=f['noise'],
                   oracle_action=f['oracle_action'], budget=np.asarray(f['budget']))
        records.append(rec)
    arrays = stack_records(records)
    arrays.update(W=m.W, K=m.K, chi=m.chi)
    return arrays


def task_metrics(a):
    axes = (1, 2, 3)
    mse = np.sum(a['weights']*(a['effect']-a['target'])**2, axis=axes)/np.sum(a['weights'], axis=axes)
    oracle_effect = np.clip(a['oracle_action']*a['gains']+a['noise'], 0., 1.)
    oracle_mse = np.sum(a['weights']*(oracle_effect-a['target'])**2, axis=axes)/np.sum(a['weights'], axis=axes)
    regret = mse-oracle_mse
    violations = ((a['action'] < -1e-10).any(axis=axes) | (a['action'] > 1+1e-10).any(axis=axes)
                  | (a['cost'] > a['budget']+1e-8)
                  | np.any((np.abs(a['action']) > 1e-10) & ~a['allowed'], axis=axes))
    recovery = []
    for start in (16, 32, 48, 64, 80):
        candidates = [t-start for t in range(start, start+14) if (regret[t:t+3] <= .015).all()]
        recovery.append(candidates[0] if candidates else None)
    observed = [x for x in recovery if x is not None]
    return {'mean_mse': float(mse.mean()), 'mean_regret': float(regret.mean()),
            'mean_cost': float(a['cost'].mean()), 'hard_violations': int(violations.sum()),
            'recovery_steps': recovery, 'recovery_censored': 5-len(observed),
            'mean_observed_recovery': float(np.mean(observed)) if observed else None}


def paired_summary(differences, confidence=.95, seed=82937):
    d = np.asarray(differences)
    rng = np.random.default_rng(seed)
    estimates = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    alpha = 1-confidence
    return {'n_seeds': len(d), 'mean_difference': float(d.mean()), 'confidence': confidence,
            'ci_low': float(np.quantile(estimates, alpha/2)),
            'ci_high': float(np.quantile(estimates, 1-alpha/2))}


def run(phase, out, registration=None):
    if phase not in ('pilot', 'final'):
        raise ValueError('Phase must be pilot or final')
    check_frozen_blobs()
    out = Path(out)
    hashes = source_hashes()
    if phase == 'final':
        if registration is None:
            raise ValueError('Final requires --registration with tension_source_sha256 before running')
        reg = json.loads(Path(registration).read_text())
        if reg.get('tension_source_sha256') != hashes:
            raise ValueError('Final source hashes differ from prior registration')
    if out.exists():
        raise FileExistsError('Output cannot be overwritten: '+str(out))
    out.mkdir(parents=True)
    (out/'raw').mkdir()
    seeds = PILOT_SEEDS if phase == 'pilot' else FINAL_SEEDS
    started = {'phase': phase, 'seeds': seeds, 'source_sha256': hashes,
               'registration_sha256': sha(registration) if registration else None,
               'source_commit': '2d2d51120409ad39559e1311897bd6b99bbf748e',
               'status': 'configured_network_not_learned_topology'}
    (out/'RUN_STARTED.json').write_text(json.dumps(started, indent=2)+'\n')
    pulse_rows, task_rows, files = [], [], {}
    runtime = {v: 0. for v in TASK_VARIANTS}
    equivalence = {'coordinate_pulse_max': 0., 'generic_flat_pulse_max': 0.,
                   'coordinate_task_max': 0., 'generic_flat_task_max': 0.}
    for seed in seeds:
        raw = {}
        for source in range(8):
            for pole in range(2):
                for sigma in (0., .02):
                    ref = None
                    for v in VARIANTS+('tau_clamp',):
                        data = pulse_pair(seed, source, pole, sigma, v)
                        if v == 'full':
                            ref = data['pulse_action'].copy()
                        if v in ('coordinate', 'generic_flat'):
                            key = v+'_pulse_max'
                            equivalence[key] = max(equivalence[key], float(np.max(np.abs(data['pulse_action']-ref))))
                        ident = f'p{source}_{pole}_n{sigma}_{v}'
                        raw.update({ident+'__'+k: value for k, value in data.items()})
                        pulse_rows.append(dict(seed=seed, source=source, pole=pole, sigma=sigma, variant=v,
                                               **pulse_metrics(data, source)))
        for task in ('switching_memory', 'gain_resource_shift'):
            ref = None
            for v in TASK_VARIANTS:
                t0 = time.perf_counter()
                data = task_trial(seed, task, v)
                runtime[v] += time.perf_counter()-t0
                if v == 'full':
                    ref = data['action'].copy()
                if v in ('coordinate', 'generic_flat'):
                    key = v+'_task_max'
                    equivalence[key] = max(equivalence[key], float(np.max(np.abs(data['action']-ref))))
                raw.update({f'task_{task}_{v}__'+k: value for k, value in data.items()})
                task_rows.append(dict(seed=seed, task=task, variant=v, **task_metrics(data)))
        path = out/'raw'/f'seed_{seed}.npz'
        np.savez_compressed(path, **raw)
        files[str(path.relative_to(out))] = sha(path)
        print(f'tension {phase}: seed {seed} complete', flush=True)
    summary = summarize(pulse_rows, task_rows, equivalence, len(seeds))
    summary['runtime_task_seconds'] = runtime
    summary['counts'] = {'seeds': len(seeds), 'pulse_pairs': len(pulse_rows),
                         'pulse_controller_steps': len(pulse_rows)*12*2,
                         'task_episodes': len(task_rows), 'task_controller_steps': len(task_rows)*96}
    for name, value in [('pulse_rows.json', pulse_rows), ('task_rows.json', task_rows), ('summary.json', summary)]:
        (out/name).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')
        files[name] = sha(out/name)
    (out/'MANIFEST.json').write_text(json.dumps({'files': files, **started}, indent=2)+'\n')
    return summary


def summarize(pulse_rows, task_rows, equivalence, n):
    pulses = {}
    for v in VARIANTS+('tau_clamp',):
        r = [x for x in pulse_rows if x['variant'] == v]
        pulses[v] = {'max_off_path_effect': max(x['off_path_max'] for x in r),
                     'mean_cross_effect_l1': float(np.mean([x['cross_effect_l1'] for x in r])),
                     'early_arrivals': sum(x['early_arrivals'] for x in r),
                     'reached_nodes': sum(x['reached_expected_nodes'] for x in r),
                     'expected_nodes': sum(x['expected_reachable_nodes'] for x in r),
                     'resource_scale_min': min(x['resource_scale_min'] for x in r),
                     'violations': sum(x['violations'] for x in r)}
    tasks, comparisons = {}, {}
    for task in ('switching_memory', 'gain_resource_shift'):
        selected = [x for x in task_rows if x['task'] == task]
        by_variant = {v: sorted([x for x in selected if x['variant'] == v], key=lambda x: x['seed']) for v in TASK_VARIANTS}
        tasks[task] = {v: {k: float(np.mean([x[k] for x in rows])) for k in
                                ('mean_mse', 'mean_regret', 'mean_cost', 'hard_violations', 'recovery_censored')}
                       for v, rows in by_variant.items()}
        comparisons[task] = {}
        for v in TASK_VARIANTS:
            if v == 'full':
                continue
            d = [a['mean_regret']-b['mean_regret'] for a, b in zip(by_variant['full'], by_variant[v])]
            comparison = paired_summary(d, .975 if v == 'no_network' else .95)
            comparison['sign'] = 'negative favors full; positive favors comparator'
            comparison['status'] = 'primary_family_two_tasks' if v == 'no_network' else 'secondary_descriptive'
            comparisons[task][v] = comparison
    primary = [comparisons[t]['no_network'] for t in comparisons]
    no_violations = all(x['hard_violations'] == 0 for x in task_rows)
    return {'evidence_scope': 'fixed_WK_engineering_mechanism_and_external_task_test_not_consciousness',
            'n_seeds': n, 'mechanism': pulses, 'equivalence_max_abs': equivalence,
            'task_means': tasks, 'paired_regret_full_minus_comparator': comparisons,
            'practical_benefit_both_tasks': all(x['ci_high'] < -.005 for x in primary) and no_violations,
            'practical_harm_by_task': {t: c['no_network']['ci_low'] > .005 for t, c in comparisons.items()},
            'all_constraints_satisfied': no_violations,
            'equivalence_pass': max(equivalence.values()) <= 1e-12,
            'no_network_cross_effect_zero': pulses['no_network']['mean_cross_effect_l1'] == 0,
            'learned_topology': False}


def verify(out):
    """Rebuild every endpoint from raw arrays; no controllers are rerun."""
    out = Path(out)
    manifest = json.loads((out/'MANIFEST.json').read_text())
    for path, digest in manifest['files'].items():
        if sha(out/path) != digest:
            raise ValueError('Artifact checksum mismatch: '+path)
    pulse_rows, task_rows = [], []
    equivalence = {'coordinate_pulse_max': 0., 'generic_flat_pulse_max': 0.,
                   'coordinate_task_max': 0., 'generic_flat_task_max': 0.}
    trace_errors = []

    def read_prefix(archive, prefix):
        return {k[len(prefix):]: archive[k] for k in archive.files if k.startswith(prefix)}

    def check_trace(data, branch='', clamp=False):
        get = lambda k: data[branch+k]
        before, action = get('q_before'), get('action')
        # Reconstruct the actual matrix action independently of stored terms.
        state = (before.reshape(len(before), -1) @ data['W'].T).reshape(before.shape)
        tau = get('tension').reshape(len(before), -1)
        tension = (tau @ data['K'].T).reshape(before.shape)
        reconstructed = get('base_proposal')+get('eta')*(state+tension)
        allowed = data.get('allowed', np.ones(action.shape, bool))
        projected = np.clip(reconstructed, 0, 1)*allowed*get('resource_scale')[:, :, None, None]
        errors = [np.max(np.abs(reconstructed-get('proposal'))), np.max(np.abs(projected-action)),
                  np.max(np.abs(before[1:]-action[:-1]))]
        mismatch = np.mean(np.abs(get('desired_effort')-before), axis=-1)
        conflict = data['chi'][None]*before[..., 0]*before[..., 1]
        errors.extend([np.max(np.abs(mismatch-get('mismatch'))), np.max(np.abs(conflict-get('conflict')))])
        if not clamp:
            errors.append(np.max(np.abs(mismatch+conflict-get('tension'))))
        trace_errors.append(float(max(errors)))

    for seed in manifest['seeds']:
        with np.load(out/'raw'/f'seed_{seed}.npz', allow_pickle=False) as archive:
            for source in range(8):
                for pole in range(2):
                    for sigma in (0., .02):
                        ref = None
                        for v in VARIANTS+('tau_clamp',):
                            data = read_prefix(archive, f'p{source}_{pole}_n{sigma}_{v}__')
                            check_trace(data, 'sham_')
                            check_trace(data, 'pulse_', clamp=v == 'tau_clamp')
                            if v == 'tau_clamp' and not np.array_equal(data['pulse_tension'], data['sham_tension']):
                                raise ValueError('Mediator clamp differs from sham')
                            if v == 'full':
                                ref = data['pulse_action'].copy()
                            if v in ('coordinate', 'generic_flat'):
                                key = v+'_pulse_max'
                                equivalence[key] = max(equivalence[key], float(np.max(np.abs(data['pulse_action']-ref))))
                            pulse_rows.append(dict(seed=seed, source=source, pole=pole, sigma=sigma, variant=v,
                                                   **pulse_metrics(data, source)))
            for task in ('switching_memory', 'gain_resource_shift'):
                ref = None
                for v in TASK_VARIANTS:
                    data = read_prefix(archive, f'task_{task}_{v}__')
                    check_trace(data)
                    if v == 'full':
                        ref = data['action'].copy()
                    if v in ('coordinate', 'generic_flat'):
                        key = v+'_task_max'
                        equivalence[key] = max(equivalence[key], float(np.max(np.abs(data['action']-ref))))
                    task_rows.append(dict(seed=seed, task=task, variant=v, **task_metrics(data)))
    summary = summarize(pulse_rows, task_rows, equivalence, len(manifest['seeds']))
    saved = json.loads((out/'summary.json').read_text())
    for key, value in summary.items():
        if canonical(value) != canonical(saved[key]):
            raise ValueError('Raw summary reconstruction mismatch: '+key)
    for path, rows in [('pulse_rows.json', pulse_rows), ('task_rows.json', task_rows)]:
        if canonical(rows) != canonical(json.loads((out/path).read_text())):
            raise ValueError('Raw row reconstruction mismatch: '+path)
    result = {'verified_raw_files': len(manifest['seeds']), 'pulse_pairs': len(pulse_rows),
              'task_episodes': len(task_rows), 'max_equation_reconstruction_error': max(trace_errors),
              'all_rows_and_summary_exact': True, 'controllers_rerun': False,
              'source_unchanged': source_hashes() == manifest['source_sha256']}
    if max(trace_errors) > 1e-12:
        raise ValueError('Equation reconstruction error exceeds 1e-12')
    (out/'VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('pilot', 'final'))
    p.add_argument('--out', required=True)
    p.add_argument('--registration')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.out), indent=2))
        return
    if a.phase is None:
        p.error('--phase is required unless --verify')
    s = run(a.phase, a.out, a.registration)
    print(json.dumps({k: s[k] for k in ('counts', 'equivalence_pass', 'practical_benefit_both_tasks')}, indent=2))


if __name__ == '__main__':
    main()
