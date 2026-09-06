import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_functional_probes import PROTOCOL, json_value, regenerate, run_seed, summarize_seed


class FunctionalProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL.read_text())
        cls.record = run_seed(cls.protocol["seeds"][0], cls.protocol)

    def test_memory_intervention_does_not_reintroduce_hidden_target_or_action_state(self):
        conditions = self.record["memory"]["conditions"]
        for result in conditions.values():
            self.assertFalse(np.asarray(result["trace"]["observed"]).any())
            self.assertFalse(np.asarray(result["trace"]["observed_target"]).any())
            self.assertFalse(np.asarray(result["before"]["q"]).any())
        self.assertTrue(np.asarray(conditions["intact"]["trace"]["action"]).any())
        self.assertFalse(np.asarray(conditions["erase"]["trace"]["action"]).any())
        self.assertFalse(np.asarray(conditions["unknown_cue"]["trace"]["action"]).any())
        np.testing.assert_allclose(conditions["edit_opposite"]["trace"]["action"],
                                   np.asarray(conditions["intact"]["trace"]["action"])[..., ::-1])

    def test_prediction_lesions_have_matched_actions_and_source_contamination_reaches_estimate(self):
        capability = self.record["capability"]
        for traces in capability["training"].values():
            self.assertEqual(len(traces), self.protocol["capability_learning_steps"])
        for replay, correct, foreign in zip(capability["replay_actions"], capability["training"]["intact"],
                                            capability["training"]["foreign_feedback"]):
            np.testing.assert_array_equal(correct["action"], replay)
            np.testing.assert_array_equal(foreign["action"], replay)
        for result in capability["conditions"].values():
            np.testing.assert_allclose(result["prediction_probe"]["action"], .25)
        m = summarize_seed(self.record)
        self.assertGreater(m["self_reset_prediction_mse_increase"], 0)
        self.assertGreater(m["self_foreign_prediction_mse_increase"], 0)

    def test_allocator_test_changes_content_and_preserves_total_demand(self):
        original = self.record["allocator"]["original"]
        permuted = self.record["allocator"]["content_permuted"]
        self.assertFalse(np.array_equal(original["proposal"], permuted["proposal"]))
        np.testing.assert_allclose(original["trace"]["demand"], permuted["trace"]["demand"])
        np.testing.assert_allclose(original["trace"]["allocations"], permuted["trace"]["allocations"])

    def test_regeneration_from_trace_and_missing_seed_rejection(self):
        protocol = {**self.protocol, "seeds": [self.record["seed"]]}
        artifact = {"protocol": protocol, "records": [self.record]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "traces.json.gz").write_bytes(gzip.compress(json.dumps(artifact, default=json_value).encode(), mtime=0))
            first = regenerate(out)
            second = regenerate(out)
            self.assertEqual(first, second)
            protocol["seeds"].append(99999)
            (out / "traces.json.gz").write_bytes(gzip.compress(json.dumps(artifact, default=json_value).encode(), mtime=0))
            with self.assertRaises(ValueError):
                regenerate(out)


if __name__ == "__main__":
    unittest.main()
