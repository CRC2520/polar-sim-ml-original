"""Adversarial preservation checks use temporary Git fixtures, never historical studies."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from b1.validate_b1 import (git_blob, read_json, safe_artifact, sha256,
                            validate_artifact_hashes, verify_preservation)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "--quiet")
        self.git("config", "user.name", "B1 preservation test")
        self.git("config", "user.email", "b1-test@example.invalid")
        (self.root / "historical.json").write_text('{"decision":"unchanged"}\n')
        self.git("add", "historical.json")
        self.git("commit", "--quiet", "-m", "test: immutable fixture")
        source = self.git("rev-parse", "HEAD").strip()
        self.baseline = {"code_source_commit": source, "historical_realization_commit": source,
                         "blob_count": 1, "blobs": [{"path": "historical.json", "mode": "100644",
                           "sha": git_blob(self.root / "historical.json")}]} 

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              text=True, check=True).stdout

    def check(self):
        return verify_preservation(self.root, self.baseline, enforce_pins=False)

    def test_valid_isolated_addition(self):
        (self.root / "b1").mkdir()
        (self.root / "b1/new.py").write_text("VALUE = 1\n")
        self.git("add", "b1")
        self.git("commit", "--quiet", "-m", "test: isolated addition")
        self.assertEqual(self.check()["errors"], [])

    def test_changed_historical_working_bytes_rejected(self):
        (self.root / "historical.json").write_text('{"decision":"changed"}\n')
        self.assertTrue(any("working" in e for e in self.check()["errors"]))

    def test_committed_historical_change_rejected(self):
        (self.root / "historical.json").write_text('{}\n')
        self.git("add", "historical.json")
        self.git("commit", "--quiet", "-m", "test: adversarial historical change")
        self.assertTrue(any("HEAD blob" in e for e in self.check()["errors"]))

    def test_unapproved_addition_rejected(self):
        (self.root / "replacement.py").write_text("pass\n")
        self.git("add", "replacement.py")
        self.git("commit", "--quiet", "-m", "test: adversarial outside addition")
        self.assertTrue(any("allowlist" in e for e in self.check()["errors"]))

    def test_omitting_source_inventory_does_not_hide_history(self):
        self.baseline["blobs"] = []
        self.baseline["blob_count"] = 0
        self.assertTrue(any("complete immutable source tree" in e for e in self.check()["errors"]))

    def test_sparse_mode_never_claims_remote_integrity(self):
        result = verify_preservation(self.root, self.baseline, sparse=True, enforce_pins=False)
        self.assertFalse(result["remote_integrity_verified"])
        self.assertEqual(result["validation_scope"], "sparse_local_only")

    def test_manifest_hash_detects_changed_export(self):
        manifest = {"artifact_hashes": {"historical.json": sha256(self.root / "historical.json")}}
        self.assertEqual(validate_artifact_hashes(self.root, manifest, "b1/manifest.json"), [])
        (self.root / "historical.json").write_text("{}")
        self.assertTrue(validate_artifact_hashes(self.root, manifest, "b1/manifest.json"))

    def test_duplicate_json_keys_rejected(self):
        (self.root / "duplicate.json").write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_json(self.root / "duplicate.json")

    def test_path_escape_rejected(self):
        for name in ("../escape", "/absolute", "b1/../../escape", "b1\\escape"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_artifact(self.root, name)


if __name__ == "__main__":
    unittest.main()
