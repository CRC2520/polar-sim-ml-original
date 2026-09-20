"""Audit mutation tests on temporary synthetic data; no environment is run."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from r9_completion.audit import (AuditError, audit_events, audit_manifest,
    audit_model, audit_rollout, independent_endpoints, independent_inference,
    read_npz, sha256)
from r9_completion.network import SparseActionModel


def synthetic_metrics(n=4):
    domains = ('ecology_train', 'ecology_delay9', 'inventory_transfer', 'thermal_transfer')
    variants = ('full', 'noCross', 'constant_gate', 'permuted_gate', 'scalar', 'permuted_content', 'dense')
    return {seed: {domain: {variant: dict(reward_mean=.75 if variant == 'full' else .5,
                                        alive_fraction=1.) for variant in variants}
                   for domain in domains} for seed in range(n)}


def dry_actor_fixture(directory):
    """Exercise an actor against constant public values, never physical dynamics."""
    from r9_completion.agent import NativeAgent
    agent = NativeAgent(950101)
    observation = np.array([.7, .6, .2])
    agent.reset_episode(observation)
    checkpoint = agent.state_dict()
    before = agent.parameter_digest()
    rows, events = {}, []
    for t in range(320):
        policy = agent.prepare(observation)
        action = policy['action']
        agent.complete_transition(observation, action, observation, .5, policy, learn=False)
        memory, goals = copy.deepcopy(agent.workspace.memory_updates), copy.deepcopy(agent.workspace.last_events)
        events.append(dict(step=t, memory=memory, goals=goals))
        values = dict(observation=observation, policy_observation=observation, next_observation=observation,
                      target=np.r_[observation, .5], action=action, reward=.5, alive=True,
                      reserve=.7, service=.5, constraint=False, probe=False, mask_violation=False,
                      workspace_update_count=len(memory), goal_event_count=len(goals))
        for key in ('features', 'predictions', 'raw_predictions', 'local_predictions', 'cross_predictions',
                    'scores', 'q_scores', 'gate', 'gate_context', 'gate_context_used', 'goals',
                    'feasible', 'lower_energy', 'memory_energy', 'memory_resource', 'poles',
                    'tension', 'previous_prediction', 'issued_pole_predictions', 'lag_predictions', 'fallback'):
            values[key] = policy[key]
        for key, value in values.items():
            rows.setdefault(key, []).append(np.asarray(value).copy())
    trace = {key: np.asarray(values) for key, values in rows.items()}
    np.savez_compressed(directory / 'trace.npz', **trace)
    meta = dict(seed=950101, kind='full', domain='ecology_train', episode=700, variant='full',
                measured_steps=320, epsilon=0., learn=False, gate_override=None,
                policy_parameter_sha256_before=before, policy_parameter_sha256_after=agent.parameter_digest())
    metrics = dict(reward_mean=.5, alive_fraction=1., reserve_mean=float(trace['reserve'].mean()),
        service_mean=.5, constraint_fraction=0., gate_active_fraction=float((trace['gate'] > 0).mean()),
        fallback_fraction=float(trace['fallback'].mean()), mask_violations=0, probes=0,
        observed_updates=int(trace['workspace_update_count'].sum()), goal_events=int(trace['goal_event_count'].sum()), all_finite=True)
    (directory / 'metrics.json').write_text(json.dumps(dict(metrics=metrics, metadata=meta)))
    (directory / 'workspace_events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    return checkpoint, trace


class AuditContracts(unittest.TestCase):
    def test_independent_holm_and_conjunctive_reconstruction(self):
        data = synthetic_metrics()
        data[0]['ecology_delay9']['permuted_gate']['reward_mean'] = .75
        data[1]['thermal_transfer']['dense']['reward_mean'] = .75
        result = independent_inference(data, range(4))
        by_name = {row['claim']: row for row in result['rows']}
        self.assertEqual(by_name['GATE']['seed_success'], [False, True, True, True])
        self.assertEqual(by_name['TRANSFER']['seed_success'], [True, False, True, True])
        self.assertEqual(by_name['REL']['p_one_sided_exact'], .0625)
        self.assertTrue(all(not row['supported'] for row in result['rows']))
        with self.assertRaises(AuditError):
            independent_inference(data, range(5))

    def test_death_guards_and_fixed_denominator(self):
        result = independent_endpoints([1., 1., 0., 0.], [1, 1, 0, 0], horizon=4)
        self.assertEqual(result['reward_mean'], .5)
        with self.assertRaises(AuditError):
            independent_endpoints([1., 1., .1, 0.], [1, 1, 0, 0], horizon=4)
        with self.assertRaises(AuditError):
            independent_endpoints([1., 0., 0., 0.], [1, 0, 1, 0], horizon=4)
        with self.assertRaises(AuditError):
            independent_endpoints([1., 1.], [1, 1], horizon=4)

    def test_saved_model_direct_audit_detects_mutated_normalization_and_capacity(self):
        rng = np.random.default_rng(949111)
        x, a = rng.normal(size=(24, 6)), np.tile(np.arange(4), 6)
        y = .1 * x[:, :2] + rng.normal(0, .01, (24, 2))
        for mode in ('sparse', 'dense', 'fixed'):
            model = SparseActionModel(6, 2, mode=mode).fit(x, a, y)
            state = model.state_dict()
            audit_model(state, x, a, y)
            altered = copy.deepcopy(state)
            altered['feature_mean_'][0] += .01
            with self.assertRaises(AuditError):
                audit_model(altered, x, a, y)
            altered = copy.deepcopy(state)
            altered['fit_diagnostics']['active_slope_coefficients'] += 1
            with self.assertRaises(AuditError):
                audit_model(altered, x, a, y)
            with self.assertRaises(AuditError):
                audit_model(state, x, a, y + .01)

    def test_npz_rejects_object_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'object.npz'
            np.savez(path, invalid=np.asarray([{}], dtype=object))
            with self.assertRaises(ValueError):
                read_npz(path)

    def test_manifest_detects_modified_unlisted_and_escaping_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'a.json'
            path.write_text('{}')
            manifest = dict(complete=True, files=[dict(path='a.json', bytes=path.stat().st_size, sha256=sha256(path))])
            complete = root / 'COMPLETE.json'
            complete.write_text(json.dumps(manifest))
            audit_manifest(root)
            (root / 'extra.json').write_text('{}')
            with self.assertRaises(AuditError):
                audit_manifest(root)
            (root / 'extra.json').unlink()
            path.write_text('[]')
            with self.assertRaises(AuditError):
                audit_manifest(root)
            manifest['files'][0]['path'] = '../a.json'
            complete.write_text(json.dumps(manifest))
            with self.assertRaises(AuditError):
                audit_manifest(root)

    def test_dry_actor_raw_ledger_and_delayed_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, trace = dry_actor_fixture(root)
            _, _, _, events = audit_rollout(root, checkpoint=checkpoint)
            self.assertGreater(events['memory_events'], 900)
            plain = root / 'workspace_events.jsonl'
            original_events = plain.read_bytes()
            compressed = root / 'workspace_events.jsonl.gz'
            compressed.write_bytes(gzip.compress(original_events, mtime=0))
            with self.assertRaisesRegex(AuditError, 'Duplicate compressed'):
                audit_events(plain, trace, 'full')
            plain.unlink()
            self.assertEqual(audit_events(plain, trace, 'full'), events)
            compressed.unlink()
            plain.write_bytes(original_events)
            require_original = trace['target'].copy()
            trace['target'][17, 1] += .1
            np.savez_compressed(root / 'trace.npz', **trace)
            with self.assertRaisesRegex(AuditError, 'Factual target'):
                audit_rollout(root, checkpoint=checkpoint)
            trace['target'] = require_original
            np.savez_compressed(root / 'trace.npz', **trace)
            path = root / 'workspace_events.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            first = next(row for row in rows if row['memory'])
            original_key = first['memory'][0]['key']
            first['memory'][0]['key'] += ':invented'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(AuditError, 'Memory key'):
                audit_events(path, trace, 'full')
            first['memory'][0]['key'] = original_key
            removed = first['memory'].pop(0)
            trace['workspace_update_count'][first['step']] -= 1
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(AuditError, 'Incomplete factual delayed'):
                audit_events(path, trace, 'full')
            first['memory'].insert(0, removed)
            trace['workspace_update_count'][first['step']] += 1
            first['memory'][0]['action'] = (first['memory'][0]['action'] + 1) % 4
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(AuditError, 'unexecuted action'):
                audit_events(path, trace, 'full')


if __name__ == '__main__':
    unittest.main()
