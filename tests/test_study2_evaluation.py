import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

from study2.environments import EnvironmentConfig, build_environment, observation, transition_details
from study2.evaluation import (ArtifactReader, pack_artifacts, write_json, read_json, sha256,
                               source_hashes, summarize_trial, recovery, sample_size, contrast, regenerate)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study2_runner", ROOT/"scripts/run_study2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Study2EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(ROOT/"docs/study2_protocol.json")
        cls.environment = build_environment(EnvironmentConfig("cooperative_inventory", "paired", 51999))
        cls.trace = runner.run_trial(cls.environment, "paired", 51999)

    def test_inventory_has_real_stockout_nonlinearity_and_validity_mask(self):
        frame = self.environment["frames"][0]
        details = transition_details(frame, np.zeros(8), np.zeros(8))
        self.assertTrue(np.all(details["state"] >= 0))
        self.assertTrue(np.any(details["stockout"] > 0))
        self.assertTrue(np.any(~details["transition_valid"]))
        self.assertFalse(np.array_equal(details["state"], details["raw_state"]))

    def test_hidden_reference_is_feasible_fixed_point_and_never_observed(self):
        for family in self.protocol["families"]:
            for regime in self.protocol["regimes"]:
                env = build_environment(EnvironmentConfig(family, regime, 51999))
                for frame in env["frames"][::32]:
                    deterministic = copy.deepcopy(frame)
                    deterministic["noise"] = np.zeros(8).tolist()
                    expected = transition_details(deterministic, frame["target"], frame["reference_action"])["state"]
                    np.testing.assert_allclose(expected, frame["target"], rtol=0, atol=1e-14)
                    obs = observation(frame, np.asarray(env["initial_state"]))
                    self.assertNotIn("matrix", obs)
                    self.assertNotIn("reference_action", obs)
                    self.assertLessEqual(np.dot(frame["costs"], frame["reference_action"]), frame["budget"])

    def test_masked_state_is_not_leaked_and_environment_keys_do_not_enter_policy(self):
        frame = copy.deepcopy(self.environment["frames"][1])
        frame["observed"] = [True, False]*4
        a, b = np.zeros(8), np.arange(8.)
        b[::2] = 0.
        obs_a, obs_b = observation(frame, a), observation(frame, b)
        np.testing.assert_array_equal(obs_a["state"], obs_b["state"])
        frame["matrix"] = np.full((8,8), 999.).tolist()
        frame["reference_action"] = [999.]*8
        np.testing.assert_array_equal(observation(frame, a)["state"], obs_a["state"])

    def test_full_trace_missing_and_inconsistent_state_raise(self):
        result = summarize_trial(self.environment, self.trace, self.protocol)
        self.assertIsNotNone(result["mean_stockout"])
        broken = copy.deepcopy(self.trace)
        broken["records"][2].pop("state_before")
        with self.assertRaises(KeyError):
            summarize_trial(self.environment, broken, self.protocol)
        broken = copy.deepcopy(self.trace)
        broken["records"][2]["state_after"][0] += .01
        with self.assertRaisesRegex(ValueError, "transition"):
            summarize_trial(self.environment, broken, self.protocol)

    def test_recovery_censored_is_missing_not_zero(self):
        censored = recovery([1.]*10, 0, 10, .025, 3)
        immediate = recovery([0.]*10, 0, 10, .025, 3)
        self.assertIsNone(censored["steps"])
        self.assertEqual(censored["status"], "right_censored")
        self.assertEqual(immediate["steps"], 0)
        self.assertEqual(immediate["status"], "recovered")

    def test_packing_roundtrip_and_tamper_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = {"x": [.12345678912345678, None], "mask": [True, False]}
            relative = "traces/test.json.gz"
            write_json(base/relative, payload)
            digest = sha256(base/relative)
            metadata = pack_artifacts(base, [relative])
            self.assertFalse((base/relative).exists())
            reader = ArtifactReader(base, metadata)
            self.assertEqual(reader.read(relative, digest), payload)
            reader.close()
            archive = base/metadata["archives"][0]["path"]
            archive.write_bytes(archive.read_bytes()[:-1]+b"x")
            with self.assertRaisesRegex(ValueError, "integrity"):
                ArtifactReader(base, metadata)

    def test_sample_size_uses_variance_not_mean_effect(self):
        rows = []
        for i, seed in enumerate(self.protocol["pilot_seeds"]):
            for family in self.protocol["families"]:
                for regime in self.protocol["regimes"]:
                    for controller, loss in (("paired", .01+i*.001), ("paired_lesion", .03)):
                        rows.append(dict(seed=seed, family=family, regime=regime, controller=controller, mean_tracking_loss=loss))
        first = sample_size(rows, self.protocol)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["controller"] == "paired":
                row["mean_tracking_loss"] += 5.
        second = sample_size(changed, self.protocol)
        self.assertEqual(first["chosen_n"], second["chosen_n"])
        self.assertAlmostEqual(first["pilot_sd"], second["pilot_sd"])
        self.assertFalse(set(first["final_seeds"]) & set(self.protocol["pilot_seeds"]))
        c = contrast(rows, "paired", "paired_lesion", "mean_tracking_loss", self.protocol)
        self.assertEqual(c["n"], len(self.protocol["pilot_seeds"]))

    def test_missing_expected_trials_and_source_changes_block_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = dict(split="pilot", source_sha256=source_hashes(ROOT), protocol=self.protocol, seeds=self.protocol["pilot_seeds"], trials=[])
            write_json(base/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing"):
                regenerate(base/"manifest.json", base/"generated")
            manifest["source_sha256"]["study2/controllers.py"] = "0"*64
            write_json(base/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "source"):
                regenerate(base/"manifest.json", base/"generated")

    def test_final_mode_requires_registration_before_creating_data_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"final"
            with self.assertRaisesRegex(ValueError, "registration"):
                runner.run("final", out)
            self.assertFalse(out.exists())
            out.mkdir()
            (out/"sentinel").write_text("existing data")
            with self.assertRaises(FileExistsError):
                runner.run("final", out)
            self.assertEqual((out/"sentinel").read_text(), "existing data")


if __name__ == "__main__":
    unittest.main()
