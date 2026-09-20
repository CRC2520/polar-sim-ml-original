import tempfile
import unittest
from pathlib import Path
import numpy as np
from collective.bridge_v2.engine import BridgeConfig, BridgeEngine


class BridgeEngineTests(unittest.TestCase):
    def config(self, **kw):
        args = dict(agents=4, groups=2, generations=2, steps=12, tail_generations=2, replace_groups=1, warmup=0)
        args.update(kw)
        return BridgeConfig(**args)

    def test_pooling_conserves_harvest_and_equalizes_actual_income(self):
        c = self.config()
        base = BridgeEngine(c, [11], .5, 'A', 'base')
        pool = BridgeEngine(c, [11], .5, 'A', 'resource_pool')
        with tempfile.TemporaryDirectory() as directory:
            a = base.episode(0, learn=False, trace_path=Path(directory)/'a.npz', trace_mode='full')
            b = pool.episode(0, learn=False, trace_path=Path(directory)/'b.npz', trace_mode='full')
            with np.load(Path(directory)/'a.npz') as ta, np.load(Path(directory)/'b.npz') as tb:
                np.testing.assert_array_equal(ta['action'], tb['action'])
                np.testing.assert_array_equal(ta['resource_after'], tb['resource_after'])
                np.testing.assert_allclose(tb['income'].sum(-1), tb['take'].sum(-1), atol=1e-14)
                np.testing.assert_allclose(np.ptp(tb['income'], axis=-1), 0, atol=1e-14)
                self.assertGreater(np.max(np.abs(tb['transfer'])), 0)
                self.assertGreater(np.max(np.abs(tb['vitality_after']-ta['vitality_after'])), 0)
            self.assertLess(b['pool_conservation_error'][0], 1e-12)
            self.assertEqual(b['pool_income_spread'][0], 0)

    def test_copy_equalization_is_only_transmission_and_conformity_negative_control(self):
        c = self.config(transmission='conformity')
        a = BridgeEngine(c, [12], .5, 'A', 'base')
        b = BridgeEngine(c, [12], .5, 'A', 'copy_score_equal')
        ha, ta = a.run(); hb, tb = b.run()
        for key in ha:
            np.testing.assert_array_equal(ha[key], hb[key])
        self.assertEqual(a.state_digest(), b.state_digest())
        fitness = np.array([[[.2, .1, .6, .9]]])
        test = BridgeEngine(self.config(), [13], .5, 'A', 'copy_score_equal')
        test.state['lineage'][:] = [0, 0, 1, 1]
        outcome = {'alive_agent_time': np.repeat(fitness, 2, axis=1), 'group_alive': np.ones((1, 2))}
        record = test.evolve(0, outcome)
        np.testing.assert_allclose(record['copy_score'][..., :2].mean(-1), record['copy_score'][..., 2:].mean(-1))

    def test_coordinate_control_preserves_scores_and_trajectories(self):
        c = self.config()
        full = BridgeEngine(c, [14,15], .5, 'C', 'full')
        equivalent = BridgeEngine(c, [14,15], .5, 'C', 'coordinate_equivalent')
        idx, _ = full.observe(np.zeros((2,2)))
        np.testing.assert_allclose(full.decision_scores(idx), equivalent.decision_scores(idx), atol=1e-14)
        a,_=full.run(); b,_=equivalent.run()
        np.testing.assert_array_equal(a['alive_fraction'], b['alive_fraction'])
        np.testing.assert_array_equal(a['action_mean'], b['action_mean'])
        for key in full.state:
            np.testing.assert_allclose(full.state[key], equivalent.state[key], atol=1e-13)

    def test_qv_policy_ablation_keeps_learning_distinct_from_learning_ablation(self):
        c = self.config()
        no_policy = BridgeEngine(c,[16],.5,'C','qv_policy_off')
        no_learning = BridgeEngine(c,[16],.5,'C','qv_learning_off')
        no_policy.episode(0); no_learning.episode(0)
        self.assertGreater(np.max(np.abs(no_policy.state['qv'])), 0)
        self.assertEqual(np.max(np.abs(no_learning.state['qv'])), 0)

    def test_endogenous_initial_composition_is_prior_not_forcing(self):
        c = self.config()
        e = BridgeEngine(c,[17],.5,'C','full')
        self.assertEqual(e.state['lineage'].mean(), .5)
        self.assertGreater(np.max(np.abs(e.state['qg'])), 0)
        control=e.clone(); control.state['lineage'] ^= 1
        e.episode(0);control.episode(0)
        for key in ('qt','qv','qg','resource','vitality','alive'):
            np.testing.assert_array_equal(e.state[key],control.state[key])

    def test_exploration_masks_match_exactly(self):
        c = self.config(epsilon=.5)
        directed=BridgeEngine(c,[18],.5,'C','full')
        blind=BridgeEngine(c,[18],.5,'C','blind_exploration')
        a,b=directed.random_tape(0),blind.random_tape(0)
        np.testing.assert_array_equal(a['explore'],b['explore'])
        self.assertTrue(a['explore'].any())
        np.testing.assert_array_equal(a['gumbel'],b['gumbel'])

    def test_snapshot_restore_and_compact_exact_replay(self):
        c=self.config()
        e=BridgeEngine(c,[19],.5,'C','full')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            e.episode(0);e.save_snapshot(path/'snapshot.npz')
            replay,_=BridgeEngine.load_snapshot(path/'snapshot.npz')
            self.assertEqual(e.state_digest(),replay.state_digest())
            a=e.episode(1,reset_physical=False,trace_path=path/'a.npz')
            b=replay.episode(1,reset_physical=False,trace_path=path/'b.npz')
            self.assertEqual(e.state_digest(),replay.state_digest())
            with np.load(path/'a.npz') as aa,np.load(path/'b.npz') as bb:
                self.assertEqual(set(aa.files),set(bb.files))
                for key in aa.files:
                    np.testing.assert_array_equal(aa[key],bb[key])
            self.assertNotIn('initial_qt',aa.files)

    def test_invalid_composition_and_variant_raise(self):
        with self.assertRaises(ValueError): BridgeEngine(self.config(),[1],.35,'C','full')
        with self.assertRaises(ValueError): BridgeEngine(self.config(),[1],.5,'C','unknown')
        with self.assertRaises(KeyError):
            e=BridgeEngine(self.config(),[1],.5,'C','full');del e.state['qv'];e.episode(0)


if __name__=='__main__':unittest.main()
