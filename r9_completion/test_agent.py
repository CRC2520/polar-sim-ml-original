"""Integration contracts motivated by concrete causal/fairness risks."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from r9_completion.agent import NativeAgent, state_index
from r9_completion.runner import select_modes, rollout, save_rollout
from r9_completion.config import PROTOCOL


def fixture(kind='full'):
    obj = NativeAgent(950701, kind)
    rng = np.random.default_rng(950702)
    x = rng.uniform(0, 1, (120, 48))
    a = np.arange(120) % 4
    y = np.clip(.3+.4*x[:, [0, 4, 2, 6]]+.02*a[:, None], 0, 1)
    obj.model.fit(x, a, y)
    return obj


class NativeContracts(unittest.TestCase):
    def test_event_compression_is_lossless_and_deterministic(self):
        events = [dict(step=3, memory=[dict(delta=[.2, -.1])], goals=[])]
        expected = ''.join(json.dumps(x, separators=(',', ':'), allow_nan=False)+'\n'
                           for x in events).encode('utf-8')
        result = ({'reward': np.array([.5])}, {'reward_mean': .5}, {}, events)
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary)/'left', Path(temporary)/'right'
            save_rollout(left, result, [])
            save_rollout(right, result, [])
            raw = (left/'workspace_events.jsonl.gz').read_bytes()
            self.assertEqual(gzip.decompress(raw), expected)
            self.assertEqual(raw, (right/'workspace_events.jsonl.gz').read_bytes())

    def test_forecast_precedes_future_observation(self):
        left = fixture()
        right = NativeAgent.from_state_dict(json.loads(json.dumps(left.state_dict())))
        obs = np.array([.6, .7, .3])
        pl, pr = left.prepare(obs), right.prepare(obs)
        np.testing.assert_array_equal(pl['issued_pole_predictions'], pr['issued_pole_predictions'])
        left.complete_transition(obs, 2, np.array([.1, .2, .9]), .2, pl)
        right.complete_transition(obs, 2, np.array([.9, .8, .1]), .8, pr)
        np.testing.assert_array_equal(left.previous_pole_prediction, right.previous_pole_prediction)
        np.testing.assert_array_equal(left.previous_tension, right.previous_tension)
        self.assertEqual(left.parameter_digest(), right.parameter_digest())

    def test_off_removes_both_cross_routes(self):
        obj = fixture()
        obj.gate_mode = 'off'
        other = NativeAgent.from_state_dict(obj.state_dict())
        obs = np.array([.6, .7, .3])
        full, lesion = obj.prepare(obs), other.prepare(obs, 'noCross')
        for key in ('lag_predictions', 'features', 'tension', 'scores', 'predictions', 'issued_pole_predictions'):
            np.testing.assert_array_equal(full[key], lesion[key])
        self.assertEqual(full['gate'], 0.)

    def test_scalar_padding_not_duplicate_regularization(self):
        obj = NativeAgent(950701, 'scalar')
        poles, tension = np.ones((8, 2)), np.arange(32).reshape(8, 4)/32.
        x = obj._pack(poles, tension)
        np.testing.assert_array_equal(x[16:].reshape(8, 4)[:, 0], tension.sum(-1))
        np.testing.assert_array_equal(x[16:].reshape(8, 4)[:, 1:], 0.)
        self.assertEqual(x.shape, (48,))

    def test_terminal_td_uses_only_observed_reward(self):
        obj = NativeAgent(950701)
        obj.q[:] = 9.
        obs, nxt = np.array([.6, .7, .3]), np.array([.5, .8, .2])
        policy = obj.prepare(obs)
        obj.complete_transition(obs, 1, nxt, .4, policy, learn=True, terminal=True)
        expected = 9.+PROTOCOL['q_alpha']*(np.array([.4, .1, -.1])-9.)
        np.testing.assert_allclose(obj.q[state_index(obs), :, 1], expected)

    def test_predictive_error_proxy_varies_before_calibration(self):
        obj = NativeAgent(950701)
        obs = np.array([.6, .7, .3])
        first = obj.prepare(obs)
        obj.complete_transition(obs, 0, np.array([.4, .4, .8]), .2, first)
        second = obj.prepare(np.array([.4, .4, .8]))
        self.assertGreater(second['gate_context'][-1], first['gate_context'][-1])

    def test_equal_four_candidate_selection_and_safety(self):
        records = {name: dict(reward_mean=.5, alive_fraction=1.)
                   for name in PROTOCOL['selection_candidates']}
        records['context_zero'] = dict(reward_mean=1., alive_fraction=.79)
        records['constant_quarter']['reward_mean'] = .6
        full, constant = select_modes(records)
        self.assertEqual(full, 'off')
        self.assertEqual(constant, 'constant_quarter')
        self.assertEqual(len(records), 7)

    def test_native_checkpoint_replay_and_frozen_parameters(self):
        obj = fixture()
        checkpoint = obj.state_dict()
        trace1, metrics1, meta1, _ = rollout(obj, 'ecology_train', 950, collect_events=False)
        clone = NativeAgent.from_state_dict(json.loads(json.dumps(checkpoint)))
        trace2, metrics2, meta2, _ = rollout(clone, 'ecology_train', 950, collect_events=False)
        self.assertEqual(metrics1, metrics2)
        self.assertEqual(meta1, meta2)
        for key in trace1:
            np.testing.assert_array_equal(trace1[key], trace2[key])
        self.assertEqual(meta1['policy_parameter_sha256_before'], meta1['policy_parameter_sha256_after'])


if __name__ == '__main__':
    unittest.main()
