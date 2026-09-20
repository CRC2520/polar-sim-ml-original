import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from collective.bridge_v2.engine import BridgeConfig, BridgeEngine
from collective.bridge_v2.history_transfer import (canonical_hash, paired_interval, transplant,
    perform_episode, replay_case, transfer_start)

class HistoryTransferTests(unittest.TestCase):
    def engine(self):
        config = replace(BridgeConfig(), agents=4, groups=2, replace_groups=1, steps=12,
                         generations=2, tail_generations=1, regeneration_delay=3)
        return BridgeEngine(config, [917001, 917002], .5, study="C", variant="full")

    def test_transplant_rejects_unmatched_present(self):
        a, b = self.engine(), self.engine()
        b.state["resource"] += 1
        with self.assertRaisesRegex(ValueError, "Present state"):
            transplant(a, b, ["qt"])

    def test_q_intervention_preserves_other_state_and_is_independent(self):
        a, b = self.engine(), self.engine()
        b.state["qt"] += 7
        child = transplant(a, b, ["qt"])
        np.testing.assert_array_equal(child.state["qt"], b.state["qt"])
        for key in a.state:
            if key != "qt":
                np.testing.assert_array_equal(child.state[key], a.state[key])
        child.state["qt"][:] = 8
        self.assertTrue(np.all(b.state["qt"] == 7))

    def test_frozen_transfer_learning_state_is_unchanged(self):
        a = self.engine()
        a.episode(0)
        b = transfer_start(a, replace(a.config, growth_mean=.12), "trained")
        before = canonical_hash({k: b.state[k] for k in ("qt", "qv", "qg", "visits")})
        b.episode(200, reset_physical=False, learn=False)
        self.assertEqual(before, canonical_hash({k: b.state[k] for k in ("qt", "qv", "qg", "visits")}))

    def test_saved_trace_reconstructs_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = perform_episode(self.engine(), Path(tmp) / "probe", "case", generation=100,
                                     learn=True, steps=12)
            result = replay_case(tmp, record)
            self.assertTrue(result["exact_trace"] and result["exact_final_state"] and result["exact_metrics"])

    def test_paired_missing_data_raises(self):
        with self.assertRaisesRegex(ValueError, "missing/non-finite"):
            paired_interval([1, np.nan], [0, 0])
        self.assertEqual(paired_interval([1, 2, 3], [0, 1, 2])["ci"], [1., 1.])

if __name__ == "__main__":
    unittest.main()
