"""Tests isolate the diagnostic intervention itself from the model under study."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from polar.baselines import make_controller

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diagnose_study1", ROOT/"scripts/diagnose_study1.py")
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


class Study1DiagnosisTests(unittest.TestCase):
    def test_missing_mandatory_measure_is_explicit_failure(self):
        rule = dict(mean_regret_max=.015, hard_violations_max=0,
                    effect_discrimination_min=.15, recovery_fraction_min=.6, recall_regret_max=.015)
        summary = dict(mean_regret=.002, hard_violations=0,
                       effect_discrimination=None, recovery_fraction=1., recall_regret=None)
        failures = diagnosis.failure_reasons(summary, rule)
        self.assertEqual([f["metric"] for f in failures], ["effect_discrimination"])
        self.assertIsNone(failures[0]["value"])
        self.assertIsNone(failures[0]["excess"])

    def test_gate_withholds_only_weak_excitation_update(self):
        model = make_controller("dual_pole", 1, agents=1, types=1)
        action = model.act({"target": np.array([[[.01, .7]]])})
        diagnosis.gate_capability_update(model, action, np.clip(action*2, 0, 1))
        self.assertEqual(model.self_model.gain_hat[0, 0, 0], 1.)
        self.assertEqual(model.self_model.counts[0, 0, 0], 0)
        self.assertAlmostEqual(model.self_model.gain_hat[0, 0, 1], 1.35)
        self.assertEqual(model.self_model.counts[0, 0, 1], 1)
        self.assertFalse(model._pending_feedback)

    def test_reset_is_time_local_and_preserves_memory(self):
        protocol = json.loads((ROOT/"results_corrected/contextual/protocol.json").read_text())
        _, base, _ = diagnosis.execute(1012, "dual_pole", protocol)
        _, reset, _ = diagnosis.execute(1012, "reset_capacity_recall", protocol)
        for original, changed in zip(base[:48], reset[:48]):
            for key in original:
                if key != "variant":
                    self.assertEqual(original[key], changed[key])
        self.assertEqual(reset[48]["gain_high_mean"], 1.)
        self.assertEqual(base[48]["stored_target_mse"], reset[48]["stored_target_mse"])
        self.assertEqual(base[48]["confidence_high_mean"], reset[48]["confidence_high_mean"])
        self.assertNotEqual(base[48]["action_high_mean"], reset[48]["action_high_mean"])


if __name__ == "__main__":
    unittest.main()
