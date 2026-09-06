import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from polar.tasks import make_trial, observation, oracle_action
from polar.baselines import make_controller
from polar.evaluation import summarize_run, sustained_recovery, paired_effect, step_metrics


class EvaluationTests(unittest.TestCase):
    def simulate(self, name, seed=101):
        trial = make_trial("switching_memory", seed, "development")
        model = make_controller(name, seed)
        records = []
        for frame in trial.steps:
            action = model.act(observation(frame))
            effect = np.clip(action*frame["gains"]+frame["noise"], 0, 1)
            model.learn({"effect": effect})
            records.append(dict(action=action, effect=effect))
        return trial, records

    def test_degenerate_controls_cannot_pass(self):
        for name in ["zero", "uniform", "disconnected"]:
            trial, records = self.simulate(name)
            summary = summarize_run(trial.steps, records)
            self.assertFalse(summary["external_pass"], name)
            self.assertEqual(summary["hard_violations"], 0)
        trial, records = self.simulate("zero")
        self.assertEqual(summarize_run(trial.steps, records)["mean_node_homogeneity"], 1.)

    def test_same_seed_and_coordinates_are_equivalent(self):
        _, a = self.simulate("dual_pole")
        _, b = self.simulate("dual_pole")
        _, c = self.simulate("signed_intensity")
        np.testing.assert_array_equal([r["action"] for r in a], [r["action"] for r in b])
        np.testing.assert_allclose([r["action"] for r in a], [r["action"] for r in c], atol=1e-12)

    def test_hidden_targets_and_gains_not_in_observation(self):
        frame = make_trial("switching_memory", 101, "development").steps[48]
        obs = observation(frame)
        self.assertFalse(obs["target"].any())
        self.assertNotIn("gains", obs)
        self.assertNotIn("oracle_action", obs)

    def test_recovery_censored_and_sustained(self):
        self.assertIsNone(sustained_recovery([1, 1, 1, 1], 0, 4))
        self.assertEqual(sustained_recovery([1, .01, .01, .01], 0, 4), 1)
        self.assertIsNone(sustained_recovery([.01, 1, .01, .01], 0, 4))

    def test_paired_units_and_undefined_zero_variance(self):
        e = paired_effect([1, 2, 3, 4], [0, 1, 2, 3])
        self.assertEqual(e["mean_difference"], 1)
        self.assertEqual(e["ci95_low"], 1)
        self.assertIsNone(e["paired_dz"])
        with self.assertRaises(ValueError):
            paired_effect([1, 2], [1])

    def test_oracle_respects_constraints_and_beats_zero(self):
        for frame in make_trial("gain_resource_shift", 1001, "heldout").steps:
            action = frame["oracle_action"]
            effect = np.clip(action*frame["gains"]+frame["noise"], 0, 1)
            m = step_metrics(frame, action, effect)
            self.assertFalse(m["hard_violation"])
            self.assertAlmostEqual(m["regret"], 0.)
            self.assertGreaterEqual(step_metrics(frame, np.zeros_like(action), np.clip(frame["noise"],0,1))["regret"], -1e-8)

    def test_incomplete_trace_is_error(self):
        trial, records = self.simulate("zero")
        with self.assertRaises(ValueError):
            summarize_run(trial.steps, records[:-1])

    def test_reports_rebuild_from_traces_and_reject_tampering(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root/"scripts"))
        from run_benchmarks import run
        from regenerate_reports import regenerate
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            protocol = json.loads((root/"docs/evaluation_protocol.json").read_text())
            protocol["controllers"] = ["dual_pole", "recurrent", "zero"]
            path = temp/"protocol.json"
            path.write_text(json.dumps(protocol))
            output = temp/"run"
            run(output, path, smoke=True)
            before = (output/"generated/per_run.csv").read_bytes()
            (output/"run_summaries.json").write_text("[]")
            regenerate(output)
            self.assertEqual(before, (output/"generated/per_run.csv").read_bytes())
            with (output/"environment.jsonl.gz").open("ab") as f:
                f.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                regenerate(output)


if __name__ == "__main__":
    unittest.main()
