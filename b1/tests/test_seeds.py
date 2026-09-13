"""Only tuning seeds are derived. No test generates a calibration/final seed."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from b1.contracts import load_contracts
from b1.seeds import DevelopmentSeeds, SeedPolicyError, verify_tuning_freeze


CONTRACTS = load_contracts()


class DevelopmentSeedTests(unittest.TestCase):
    def setUp(self):
        self.seeds = DevelopmentSeeds(contracts=CONTRACTS)

    def test_exact_frozen_hash_rule_and_replay(self):
        stream = "cell=0;episode=0;epoch=0;event=initial-mapping"
        encoded = f"PD-B1-D-v1|{CONTRACTS.source_commit}|P6|tuning|0|{stream}"
        expected = int.from_bytes(sha256(encoded.encode("utf-8")).digest()[:8], "big")
        actual = self.seeds.derive("P6", "tuning", 0, stream)
        self.assertEqual(actual, expected)
        self.assertEqual(self.seeds.derive("P6", "tuning", 0, stream), actual)
        self.assertEqual(len(self.seeds.usage_log), 2)
        self.assertTrue(self.seeds.usage_log[0]["development_only"])
        self.assertFalse(self.seeds.usage_log[0]["reusable_as_final"])

    def test_counter_streams_do_not_depend_on_call_order(self):
        a = self.seeds.derive("P7", "evaluation-development", 0, "cell=0;epoch=8;event=drift")
        self.seeds.derive("P7", "evaluation-development", 0, "cell=2;epoch=24;event=drift")
        other = DevelopmentSeeds(contracts=CONTRACTS)
        other.derive("P7", "evaluation-development", 0, "cell=2;epoch=24;event=drift")
        self.assertEqual(a, other.derive("P7", "evaluation-development", 0, "cell=0;epoch=8;event=drift"))

    def test_training_evaluation_and_pilots_disjoint_observed_test_streams(self):
        seeds = {self.seeds.derive(pilot, role, index, "cell=0;epoch=0;event=test")
                 for pilot in ("P5", "P6", "P7")
                 for role in ("training", "tuning", "evaluation-development")
                 for index in (0, 31)}
        self.assertEqual(len(seeds), 18)

    def test_calibration_rejected_before_hashing(self):
        with self.assertRaises(SeedPolicyError):
            self.seeds.derive("P7", "calibration", 32, "cell=0;event=forbidden-unopened")
        self.assertEqual(self.seeds.usage_log, [])
        self.assertFalse(self.seeds.policy()["calibration_open"])

    def test_final_roles_and_outside_indices_always_rejected(self):
        for role, index in (("training-final", 0), ("evaluation-final", 0),
                            ("tuning", 32), ("calibration", 31), ("tuning", -1),
                            ("tuning", True), ("tuning", 64)):
            with self.subTest(role=role, index=index):
                with self.assertRaises(SeedPolicyError):
                    self.seeds.derive("P5", role, index, "event=blocked")
        self.assertEqual(self.seeds.usage_log, [])

    def test_ambiguous_substream_and_source_rejected(self):
        for stream in ("", "a|b", "a\nb", None):
            with self.assertRaises(SeedPolicyError):
                self.seeds.derive("P5", "tuning", 0, stream)
        with self.assertRaises(SeedPolicyError):
            DevelopmentSeeds("0" * 40, contracts=CONTRACTS)

    def test_historical_collision_does_not_select_replacement(self):
        seed = self.seeds.derive("P6", "tuning", 0, "event=collision-test")
        guarded = DevelopmentSeeds(contracts=CONTRACTS, historical_seeds=[seed])
        with self.assertRaises(SeedPolicyError):
            guarded.derive("P6", "tuning", 0, "event=collision-test")
        self.assertEqual(guarded.usage_log, [])

    def test_token_cannot_be_replaced_with_unverified_bool_or_dict(self):
        for token in (True, {"status": "TUNING_CLOSED"}, None):
            with self.assertRaises(SeedPolicyError):
                self.seeds.open_calibration(token)

    def make_freeze_fixture(self, root):
        # Artificial closure fixture tests the file verifier; it is not a claim
        # that the research tuning block was executed or calibration was opened.
        artifacts = {"fixture.py": b"# synthetic verifier fixture\n",
                     "config.json": b'{"fixture":true}\n',
                     "tuning.json": json.dumps({"bundle_indices": list(range(32)),
                                                 "development_only": True}).encode()}
        for name, data in artifacts.items():
            (root / name).write_bytes(data)
        manifest = {"status": "TUNING_CLOSED", "source_commit": CONTRACTS.source_commit,
                    "tuning_bundle_indices": list(range(32)), "configuration_frozen": True,
                    "calibration_comparator_identities_masked": True,
                    "calibration_consumed": False, "implementation_files": ["fixture.py"],
                    "configuration_file": "config.json", "tuning_ledger_file": "tuning.json",
                    "artifact_sha256": {name: sha256(data).hexdigest() for name, data in artifacts.items()}}
        path = root / "freeze.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_freeze_can_be_verified_without_deriving_holdout_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_freeze_fixture(Path(tmp))
            token = verify_tuning_freeze(path, contracts=CONTRACTS)
            self.seeds.open_calibration(token)
            self.seeds.authorize("P7", "calibration", 32, "event=authorization-only")
            self.assertEqual(self.seeds.usage_log, [])
            self.assertFalse(self.seeds.policy()["final_seeds_generated"])

    def test_changed_frozen_config_rejects_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_freeze_fixture(root)
            (root / "config.json").write_text("{}")
            with self.assertRaises(SeedPolicyError):
                verify_tuning_freeze(path, contracts=CONTRACTS)
            self.assertEqual(self.seeds.usage_log, [])

    def test_stale_verified_token_cannot_open_changed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_freeze_fixture(root)
            token = verify_tuning_freeze(path, contracts=CONTRACTS)
            (root / "config.json").write_text("{}")
            with self.assertRaises(SeedPolicyError):
                self.seeds.open_calibration(token)
            self.assertFalse(self.seeds.policy()["calibration_open"])

    def test_consumed_calibration_not_reusable_as_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_freeze_fixture(Path(tmp))
            data = json.loads(path.read_text())
            data["calibration_consumed"] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(SeedPolicyError):
                verify_tuning_freeze(path, contracts=CONTRACTS)


if __name__ == "__main__":
    unittest.main()
