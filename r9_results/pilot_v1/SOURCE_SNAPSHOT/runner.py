"""Prospective R9 native acquisition, selection and untouched evaluation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from r8_completion.io import atomic_json, atomic_npz, atomic_text, sha256, write_manifest
from r9_completion.agent import NativeAgent
from r9_completion.config import (PROTOCOL, VERSION, PILOT_SEEDS, FINAL_SEEDS,
                                  KINDS, VARIANTS, DOMAINS)
from r9_completion.environments import ScalarEnvironment

ROOT = Path(__file__).resolve().parents[1]


def tape_seed(seed, episode, domain):
    value = f'R9|{seed}|{episode}|{domain}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little')


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def rollout(agent, domain, episode, variant='full', *, learn=False, epsilon=0.,
            gate_override=None, collect_events=True):
    environment = ScalarEnvironment(domain)
    tape_domain = 'ecology_train' if domain == 'ecology_delay9' else domain
    env_seed = tape_seed(agent.seed, episode, tape_domain)
    obs = environment.reset(env_seed)
    agent.reset_episode(obs, variant)
    random = np.random.default_rng(tape_seed(agent.seed, episode, 'policy'))
    probe_uniform = random.random(PROTOCOL['steps'])
    probe_actions = random.integers(0, 4, size=PROTOCOL['steps'])
    rows = {key: [] for key in ('observation', 'policy_observation', 'next_observation',
            'features', 'target', 'action', 'reward', 'alive', 'reserve', 'service',
            'constraint', 'scores', 'q_scores', 'predictions', 'raw_predictions',
            'local_predictions', 'cross_predictions', 'gate', 'gate_context',
            'gate_context_used', 'lower_energy', 'feasible', 'fallback', 'goals',
            'memory_energy', 'memory_resource', 'poles', 'tension', 'previous_prediction',
            'issued_pole_predictions', 'lag_predictions', 'probe', 'mask_violation',
            'workspace_update_count', 'goal_event_count')}
    events = []
    initial_parameters = agent.parameter_digest()
    for step in range(PROTOCOL['steps']):
        policy_obs = obs.copy()
        if variant in ('observation_shock', 'dense_shock') and PROTOCOL['shock_steps'][0] <= step < PROTOCOL['shock_steps'][1]:
            policy_obs = np.clip(policy_obs+PROTOCOL['shock_magnitude']*np.array([1., -1., 1.]), 0., 1.)
        policy = agent.prepare(policy_obs, variant, gate_override=gate_override)
        probe = bool(probe_uniform[step] < epsilon)
        action = int(probe_actions[step] if probe else policy['action'])
        nxt, reward, terminated, info = environment.step(action)
        # Neither privileged info nor the unperturbed observation enters policy
        # for a shocked step. Only observed transition is used by online Q.
        agent.complete_transition(policy_obs, action, nxt, reward, policy, learn=learn, terminal=terminated)
        updates = copy.deepcopy(getattr(agent.workspace, 'memory_updates', []))
        goal_events = copy.deepcopy(getattr(agent.workspace, 'last_events', []))
        if collect_events and (updates or goal_events):
            events.append(_jsonable(dict(step=step, memory=updates, goals=goal_events)))
        values = dict(observation=obs, policy_observation=policy_obs, next_observation=nxt,
                      features=policy['features'], target=np.r_[nxt, reward], action=action,
                      reward=reward, alive=info['alive'], reserve=info['reserve'],
                      service=info['service'], constraint=info['constraint_violation'],
                      probe=probe, mask_violation=not bool(policy['feasible'][action]),
                      workspace_update_count=len(updates), goal_event_count=len(goal_events))
        for key in ('scores', 'q_scores', 'predictions', 'raw_predictions',
                    'local_predictions', 'cross_predictions', 'gate', 'gate_context',
                    'gate_context_used', 'lower_energy', 'feasible', 'fallback', 'goals',
                    'memory_energy', 'memory_resource', 'poles', 'tension', 'previous_prediction',
                    'issued_pole_predictions', 'lag_predictions'):
            values[key] = policy[key]
        for key, value in values.items():
            rows[key].append(np.asarray(value).copy())
        obs = nxt
        # An absorbing death is still followed to the predeclared horizon.
    trace = {key: np.asarray(value) for key, value in rows.items()}
    if not learn and agent.parameter_digest() != initial_parameters:
        raise ValueError('Learned parameters changed during frozen-policy rollout')
    metrics = dict(reward_mean=float(trace['reward'].mean()),
                   alive_fraction=float(trace['alive'].mean()),
                   reserve_mean=float(trace['reserve'].mean()),
                   service_mean=float(trace['service'].mean()),
                   constraint_fraction=float(trace['constraint'].mean()),
                   gate_active_fraction=float(np.mean(trace['gate'] > 0)),
                   fallback_fraction=float(trace['fallback'].mean()),
                   mask_violations=int(trace['mask_violation'].sum()),
                   probes=int(trace['probe'].sum()),
                   observed_updates=int(trace['workspace_update_count'].sum()),
                   goal_events=int(trace['goal_event_count'].sum()),
                   all_finite=bool(all(np.isfinite(x).all() for x in trace.values())))
    if not metrics['all_finite'] or (epsilon == 0 and metrics['mask_violations']):
        raise ValueError('Invalid native trace')
    metadata = dict(seed=agent.seed, kind=agent.kind, domain=domain, episode=episode,
                    variant=variant, environment_seed=env_seed,
                    environment_tape_sha256=environment.tape_digest(),
                    policy_parameter_sha256_before=initial_parameters,
                    policy_parameter_sha256_after=agent.parameter_digest(),
                    epsilon=epsilon, learn=learn, gate_override=gate_override,
                    measured_steps=len(trace['reward']))
    return trace, metrics, metadata, events


def save_rollout(directory, result, files):
    trace, metrics, metadata, events = result
    directory = Path(directory)
    files.append(atomic_npz(directory/'trace.npz', trace))
    files.append(atomic_json(directory/'metrics.json', dict(metrics=metrics, metadata=metadata)))
    if events:
        content = ''.join(json.dumps(x, separators=(',', ':'), allow_nan=False)+'\n' for x in events)
        files.append(atomic_text(directory/'workspace_events.jsonl', content))


def concatenate(traces):
    return {key: np.concatenate([t[key] for t in traces], axis=0)
            for key in ('features', 'action', 'target', 'gate_context')}


def select_modes(records):
    def select(names):
        viable = [n for n in names if records[n]['alive_fraction'] >= PROTOCOL['absolute_alive_floor']]
        candidates = viable or list(names)
        if viable:
            return max(candidates, key=lambda n: (records[n]['reward_mean'], -names.index(n)))
        return max(candidates, key=lambda n: (records[n]['alive_fraction'], records[n]['reward_mean'], -names.index(n)))
    full_names = ('off', 'context_zero', 'context_small', 'context_large')
    constant_names = ('off', 'constant_quarter', 'constant_half', 'constant_one')
    return select(full_names), select(constant_names)


def train_kind(seed, kind, output, files):
    agent = NativeAgent(seed, kind)
    trained = []
    for episode in range(PROTOCOL['training_episodes']):
        result = rollout(agent, 'ecology_train', 100+episode, learn=True,
                         epsilon=PROTOCOL['epsilon'], gate_override='constant_half')
        save_rollout(output/'training'/kind/f'episode_{episode:02d}', result, files)
        trained.append(result[0])
        if (episode+1) % PROTOCOL['fit_every_episodes'] == 0:
            agent.fit_model(concatenate(trained), episode+1)
            files.append(atomic_json(output/'training'/kind/f'fit_{episode+1:02d}.json',
                                     dict(model=agent.model.state_dict(), fit=agent.fit_log[-1])))
    for episode in range(PROTOCOL['calibration_episodes']):
        result = rollout(agent, 'ecology_train', 200+episode, epsilon=PROTOCOL['epsilon'],
                         gate_override='constant_half')
        save_rollout(output/'calibration'/kind/f'episode_{episode:02d}', result, files)
        if episode == 0:
            agent.calibrate_uncertainty(result[0])
        else:
            agent.fit_gate(result[0])
    base = agent.state_dict()
    records = {}
    for mode in PROTOCOL['selection_candidates']:
        candidate = NativeAgent.from_state_dict(base)
        result = rollout(candidate, 'ecology_train', 300, gate_override=mode)
        save_rollout(output/'selection'/kind/mode, result, files)
        records[mode] = result[1]
    agent.gate_mode, agent.constant_mode = select_modes(records)
    agent.selection_record = dict(candidates=records, selected=agent.gate_mode,
                                  constant_selected=agent.constant_mode,
                                  candidate_workspace_updates_not_retained=True,
                                  total_training_transitions=PROTOCOL['training_episodes']*PROTOCOL['steps'],
                                  total_calibration_transitions=PROTOCOL['calibration_episodes']*PROTOCOL['steps'],
                                  total_selection_transitions=len(records)*PROTOCOL['steps'])
    files.append(atomic_json(output/'checkpoints'/f'{kind}.json', agent.state_dict()))
    return agent


def run_seed(seed, phase, output):
    start = time.perf_counter()
    if phase == 'final':
        if seed not in FINAL_SEEDS:
            raise ValueError('Unregistered final seed')
        from r9_completion.provenance import verify
        frozen = verify()
        authorization = ROOT/'r9_results/FINAL_EXECUTION_AUTHORIZATION.json'
        auth = json.loads(authorization.read_text())
        if not auth.get('authorized') or auth['freeze_sha256'] != frozen['freeze_sha256']:
            raise ValueError('No matching pre-execution authorization')
    else:
        if seed not in PILOT_SEEDS:
            raise ValueError('Unregistered development seed')
        frozen = {'status': 'development, not confirmatory'}
    output = Path(output)/f'seed_{seed}'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    files, agents = [], {}
    for kind in KINDS:
        agents[kind] = train_kind(seed, kind, output, files)
    metrics = {}
    evaluated_domains = DOMAINS if phase == 'final' else DOMAINS[:2]
    for domain in evaluated_domains:
        metrics[domain] = {}
        for variant in VARIANTS:
            kind = variant if variant in ('dense', 'scalar', 'fixed') else 'dense' if variant == 'dense_shock' else 'full'
            agent = NativeAgent.from_state_dict(agents[kind].state_dict())
            result = rollout(agent, domain, 700, variant)
            save_rollout(output/'evaluation'/domain/variant, result, files)
            metrics[domain][variant] = result[1]
    if phase == 'final':
        after = verify()
        if after != frozen:
            raise ValueError('Final source changed')
    summary = dict(version=VERSION, phase=phase, seed=seed, metrics=metrics,
                   source=frozen, elapsed_seconds=time.perf_counter()-start,
                   trained_kinds=list(KINDS), evaluated_variants=list(VARIANTS),
                   evaluated_domains=list(evaluated_domains),
                   own_acquisition_digests={k: a.model.fit_diagnostics['training_arrays_sha256']
                                            for k, a in agents.items()})
    files.append(atomic_json(output/'summary.json', summary))
    write_manifest(output/'COMPLETE.json', files, metadata=dict(seed=seed, phase=phase, source=frozen))
    return dict(seed=seed, phase=phase, elapsed_seconds=summary['elapsed_seconds'], files=len(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'final'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for seed in args.seeds:
        print(json.dumps(run_seed(seed, args.phase, args.output)), flush=True)


if __name__ == '__main__':
    main()
