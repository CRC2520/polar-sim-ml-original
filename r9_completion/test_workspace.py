"""Mechanistic contracts; these fixtures are not R9 efficacy experiments."""
import copy
import json
import unittest
from dataclasses import replace

import numpy as np

from r9_completion.workspace import (CognitiveWorkspace, HORIZONS, route_message,
                                     viability_consumer, resource_consumer)


class WorkspaceTests(unittest.TestCase):
    def test_unseen_predictions_are_persistence_and_own_history_only(self):
        w = CognitiveWorkspace()
        first = w.observe([.7, .6, .3])
        np.testing.assert_array_equal(first.consequences.delayed_estimates,
                                     np.tile([.7, .6, .3], (4, 3, 1)))
        self.assertEqual(len(w.memory_updates), 0)
        second = w.observe([.6, .65, .3], prev_action=2, prev_prediction=[.5, .5, .2])
        np.testing.assert_allclose(second.state.slope, [-.1, .05, 0.])
        np.testing.assert_allclose(second.state.prediction_error, [.1, .15, .1])
        np.testing.assert_array_equal(w.history_array()[:, 3], [-1, 2])
        with self.assertRaises(TypeError):
            w.observe([.5, .5, .5], future_state=[.4, .4, .4])

    def test_delayed_targets_arrive_once_and_reconstruct_from_factual_sequence(self):
        w = CognitiveWorkspace()
        observations = np.array([[.7-.01*t, .4+.005*t, .3] for t in range(15)])
        all_updates = []
        for t, obs in enumerate(observations):
            w.observe(obs, prev_action=None if t == 0 else (t-1) % 4)
            all_updates.extend(copy.deepcopy(w.memory_updates))
            self.assertLessEqual(len(w.pending), 11)
        self.assertEqual(len(all_updates), sum(max(0, len(observations)-h) for h in HORIZONS))
        unique = set()
        for u in all_updates:
            self.assertEqual(u['destination_time']-u['origin_time'], u['horizon'])
            self.assertEqual(u['action'], u['origin_time'] % 4)
            np.testing.assert_array_equal(u['observed'], observations[u['destination_time']])
            np.testing.assert_allclose(u['delta'], observations[u['destination_time']]-observations[u['origin_time']])
            unique.add((u['origin_time'], u['horizon']))
        self.assertEqual(len(unique), len(all_updates))

    def test_episode_reset_keeps_cues_without_cross_episode_targets(self):
        w = CognitiveWorkspace()
        for t in range(8):
            w.observe([.5, .4, .2], prev_action=None if t == 0 else 2)
        memory, absolute = copy.deepcopy(w.memory), w.absolute_time
        self.assertTrue(w.pending)
        w.reset_episode()
        self.assertEqual(w.memory, memory)
        self.assertEqual(w.pending, [])
        self.assertEqual(w.history_array().shape, (0, 4))
        message = w.observe([.5, .4, .2])
        self.assertEqual(message.time, 0)
        self.assertEqual(message.absolute_time, absolute+1)
        self.assertEqual(w.memory_updates, [])
        self.assertGreater(np.asarray(message.consequences.counts).sum(), 0)

    def test_recall_new_evidence_retained_revision_changes_later_forecast(self):
        w = CognitiveWorkspace()
        # Independent episodes produce the same cue/action outcome, then the
        # same cue is recalled before a contradictory factual consequence.
        for _ in range(4):
            w.reset_episode()
            w.observe([.5, .4, .2])
            w.observe([.45, .4, .2], prev_action=2)
        w.reset_episode()
        before_message = w.observe([.5, .4, .2])
        key = w._key(w._cue([.5, .4, .2]), 2, 1)
        baseline = copy.deepcopy(w.memory[key])
        frozen = CognitiveWorkspace.from_state_dict(json.loads(json.dumps(w.state_dict())))
        frozen.set_modes(frozen_memory=True)
        w.observe([.75, .65, .2], prev_action=2)
        frozen.observe([.75, .65, .2], prev_action=2)
        update = next(u for u in w.memory_updates if u['key'] == key)
        self.assertTrue(update['recalled'])
        self.assertTrue(update['reconsolidated'])
        self.assertEqual(update['before'], baseline['mean_delta'])
        self.assertNotEqual(update['before'], update['after'])
        self.assertEqual(frozen.memory[key], baseline)
        self.assertFalse(frozen.memory_updates[0]['update_applied'])
        w.reset_episode()
        after_message = w.observe([.5, .4, .2])
        self.assertGreater(viability_consumer(after_message)['energy_forecast'][2],
                           viability_consumer(before_message)['energy_forecast'][2])
        np.testing.assert_array_equal(w.memory[key]['mean_delta'], update['after'])

    def test_lesions_distinguish_recall_update_and_explicit_memory(self):
        w = CognitiveWorkspace()
        for t in range(18):
            w.observe([.5-.002*t, .4+.002*t, .2], prev_action=None if t == 0 else 1)
        frozen = CognitiveWorkspace.from_state_dict(w.state_dict())
        removed = CognitiveWorkspace.from_state_dict(w.state_dict())
        frozen.set_modes(frozen_memory=True)
        removed.set_modes(no_memory=True)
        old_memory = copy.deepcopy(frozen.memory)
        frozen_message = frozen.observe([.46, .44, .2], prev_action=1)
        removed_message = removed.observe([.46, .44, .2], prev_action=1)
        self.assertEqual(frozen.memory, old_memory)
        self.assertGreater(np.asarray(frozen_message.consequences.counts).sum(), 0)
        self.assertEqual(removed.memory, {})
        self.assertEqual(removed.pending, [])
        self.assertEqual(removed_message.state.history_count, 1)
        np.testing.assert_array_equal(removed_message.state.slope, 0.)
        np.testing.assert_array_equal(removed_message.consequences.counts, 0)

    def test_same_contradiction_without_reactivation_uses_ordinary_update(self):
        # Isolate the retrieval marker, holding the learned entry, pending
        # origin and arriving factual observation identical in both copies.
        prepared = CognitiveWorkspace()
        for _ in range(4):
            prepared.reset_episode()
            prepared.observe([.5, .4, .2])
            prepared.observe([.45, .4, .2], prev_action=2)
        prepared.reset_episode()
        prepared.observe([.5, .4, .2])
        recalled = CognitiveWorkspace.from_state_dict(prepared.state_dict())
        inactive = CognitiveWorkspace.from_state_dict(prepared.state_dict())
        inactive.recalled.clear()
        self.assertEqual(recalled.memory, inactive.memory)
        key = prepared._key(prepared._cue([.5, .4, .2]), 2, 1)
        for workspace in (recalled, inactive):
            workspace.observe([.75, .65, .2], prev_action=2)
        hot = next(e for e in recalled.memory_updates if e['key'] == key)
        cold = next(e for e in inactive.memory_updates if e['key'] == key)
        for field in ('before', 'delta', 'prediction_error', 'count_before'):
            self.assertEqual(hot[field], cold[field])
        self.assertGreaterEqual(cold['prediction_error'], .08)
        self.assertTrue(hot['recalled'] and hot['reconsolidated'])
        self.assertFalse(cold['recalled'] or cold['reconsolidated'])
        self.assertTrue(cold['update_applied'])
        self.assertEqual(hot['learning_rate'], .6)
        self.assertEqual(cold['learning_rate'], 1. / (cold['count_before'] + 1))
        self.assertNotEqual(hot['after'], cold['after'])
        self.assertEqual(inactive.memory[key]['mean_delta'], cold['after'])

    def test_goals_persist_then_revise_and_abandon_under_observed_change(self):
        w = CognitiveWorkspace()
        first = w.observe([.8, .2, .1])
        self.assertEqual(first.goals.primary, 'survival')
        for _ in range(3):
            msg = w.observe([.8, .2, .1])
            self.assertEqual(msg.goals.weights, first.goals.weights)
        events = []
        for _ in range(18):
            final = w.observe([.2, .9, .1])
            events.extend(w.last_events)
        self.assertEqual(final.goals.primary, 'reserve')
        self.assertNotEqual(final.goals.weights, first.goals.weights)
        self.assertTrue(any(e['type'] == 'goal_abandoned' and e['goal'] == 'survival' for e in events))
        self.assertTrue(any(e['type'] == 'goal_revised' for e in events))
        fixed = CognitiveWorkspace(no_goal_revision=True)
        for obs in ([.8, .2, .1],)*5+([.2, .9, .1],)*8:
            msg = fixed.observe(obs)
            np.testing.assert_array_equal(msg.goals.weights, [.5, .3, .2])
        self.assertEqual(fixed.counters['goal_revisions'], 0)

    def test_typed_routes_equal_bandwidth_and_consumers_are_independent(self):
        w = CognitiveWorkspace()
        for t in range(16):
            msg = w.observe([.8-.02*t, .3+.02*t, .1], prev_action=None if t == 0 else t % 4)
        intact_vector = msg.payload_vector()
        for consumer in ('viability', 'resources'):
            for mode in ('permuted', 'content_lesion'):
                intervention = route_message(msg, mode, consumer)
                self.assertEqual(intact_vector.shape, intervention.payload_vector().shape)
                self.assertEqual(intact_vector.nbytes, intervention.payload_vector().nbytes)
                self.assertEqual(msg.time, intervention.time)
                if mode == 'permuted':
                    np.testing.assert_array_equal(np.sort(intact_vector), np.sort(intervention.payload_vector()))
        changed_goals = replace(msg, goals=replace(msg.goals, weights=(.01, .01, .98)))
        np.testing.assert_array_equal(viability_consumer(msg)['energy_forecast'],
                                     viability_consumer(changed_goals)['energy_forecast'])
        w.block_viability = True
        v, r = w.consume(msg)
        self.assertFalse(np.array_equal(v['energy_forecast'], viability_consumer(msg)['energy_forecast']))
        np.testing.assert_array_equal(r['resource_forecast'], resource_consumer(msg)['resource_forecast'])
        np.testing.assert_array_equal(msg.payload_vector(), intact_vector)

    def test_json_checkpoint_replays_pending_memory_goals_and_events_exactly(self):
        w = CognitiveWorkspace()
        for t in range(19):
            w.observe([.5-.005*t, .3+.003*t, .2], prev_action=None if t == 0 else t % 4)
        other = CognitiveWorkspace.from_state_dict(json.loads(json.dumps(w.state_dict(), allow_nan=False)))
        for t in range(16):
            obs = [.3+.006*t, .4+.004*t, .1]
            a = w.observe(obs, prev_action=t % 4, prev_prediction=[.3, .4, .1], uncertainty=.03)
            b = other.observe(obs, prev_action=t % 4, prev_prediction=[.3, .4, .1], uncertainty=.03)
            self.assertEqual(a.to_dict(), b.to_dict())
            self.assertEqual(w.memory_updates, other.memory_updates)
            self.assertEqual(w.last_events, other.last_events)
        self.assertEqual(w.state_dict(), other.state_dict())

    def test_checkpoint_storage_bound_and_history_copy(self):
        w = CognitiveWorkspace()
        for t in range(200):
            w.observe([.5, .4, .2], prev_action=None if t == 0 else 1)
        self.assertLessEqual(len(w.memory_updates), 3)
        self.assertLessEqual(len(w.last_events), 3)
        self.assertLessEqual(len(w.pending), 11)
        self.assertLessEqual(len(w.history_array()), 16)
        self.assertLess(len(json.dumps(w.state_dict())), 20000)
        history = w.history_array()
        history[:] = -99
        self.assertTrue((w.history_array()[:, :3] >= 0).all())

    def test_same_native_checkpoint_both_consumers_change_actual_action(self):
        # Construct a transparent deterministic checkpoint to expose causal
        # routes. This is a mechanism witness, not an efficacy-trained agent.
        from r9_completion.agent import NativeAgent
        agent = NativeAgent(950080)
        action = np.tile(np.arange(4), 8)
        targets = np.column_stack((np.full(32, .7), np.full(32, .23),
                                   np.full(32, .01), np.where(action == 0, 1., .5)))
        agent.model.fit(np.zeros((32, 48)), action, targets)
        obs = np.array([.7, .7, .01])
        cue = agent.workspace._cue(obs)
        for a in range(4):
            for h in HORIZONS:
                delta = [(-.6 if a == 1 else .2), (-.6 if a in (0, 3) else .15), 0.]
                agent.workspace.memory[agent.workspace._key(cue, a, h)] = dict(
                    mean_delta=delta, count=100, last_episode=0, last_time=0)
        checkpoint = json.loads(json.dumps(agent.state_dict()))
        outputs = {}
        for variant in ('full', 'block_viability', 'block_resources'):
            clone = NativeAgent.from_state_dict(checkpoint)
            clone.reset_episode(obs, variant)
            outputs[variant] = clone.prepare(obs, variant)
        self.assertEqual(outputs['full']['action'], 2)
        self.assertEqual(outputs['block_viability']['action'], 0)
        self.assertEqual(outputs['block_resources']['action'], 1)
        np.testing.assert_array_equal(outputs['full']['memory_resource'], outputs['block_viability']['memory_resource'])
        np.testing.assert_array_equal(outputs['full']['memory_energy'], outputs['block_resources']['memory_energy'])
        for variant in ('block_viability', 'block_resources'):
            np.testing.assert_array_equal(outputs['full']['predictions'], outputs[variant]['predictions'])

    def test_actual_environment_transition_uses_only_public_observations(self):
        from r9_completion.agent import NativeAgent
        from r9_completion.environments import ScalarEnvironment
        env, agent = ScalarEnvironment('ecology_train'), NativeAgent(950081)
        obs = env.reset(950081)
        agent.reset_episode(obs)
        for _ in range(15):
            policy = agent.prepare(obs)
            nxt, reward, done, info_eval = env.step(policy['action'])
            agent.complete_transition(obs, policy['action'], nxt, reward, policy, learn=False)
            obs = nxt
        self.assertGreater(agent.workspace.counters['formations'], 0)
        self.assertTrue(any(u['horizon'] == 12 for u in agent.workspace.memory_updates))
        self.assertEqual(agent.workspace.time, 14)


if __name__ == '__main__':
    unittest.main()
