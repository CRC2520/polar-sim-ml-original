"""Small temporary fixtures only; no campaign outputs are read or packaged."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_r8_evidence as pack


class PackageFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r8-package-fixture-")
        self.base = Path(self.tmp.name)
        self.source = self.base / "POLAR_reconciled"
        self.source.mkdir()
        self._put("r8_results/REVIEW.json", b'{"fixture": true}\n')
        self._put("r8_results/Q_final/results.json", b'{"adverse_fixture": true}\n')
        self._put("r8_results/A_final/record.bin", b"A" * 71)
        self._put("r8_results/threshold_final/record.bin", b"T" * 71)
        self._put("r8_results/audits/AUDIT.json", b'{"fixture": true}\n')
        self._put("r8_results/Q_pilot_v1/old.bin", b"old adverse data")
        self._put("r8_results/Q_pilot_v2/new.bin", b"new data")
        self._put("results/r8_integrated_pilot_v1/record.bin", b"pilot v1")
        self._put("results/r8_integrated_pilot_v2/record.bin", b"pilot v2")
        self._put("r8_results/audits/résumé.txt", b"UTF8 paths remain relative")
        self._put("r8_results/SOURCE_INTEGRATED.py", b"source snapshot excluded")
        self._put("r8_results/__pycache__/cache.pyc", b"excluded")
        self._put("r8_completion/FREEZE_R8.json", b"Git-backed protocol excluded")
        self._put("p1_results/old.json", b"old evidence excluded")
        self._put("results/p1_ecology_final/old.npz", b"invalid old NPZ must not be read")
        for seed in range(942001, 942006):
            p = self.source / f"r8_results/integrated_final/shard_01/seed_{seed}/trace.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            # A valid inner ZIP tests integrity without consuming scientific arrays.
            with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as z:
                z.writestr("fixture.bin", bytes([seed % 255]) * 4000)

    def tearDown(self):
        self.tmp.cleanup()

    def _put(self, name, content):
        p = self.source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def test_inventory_scope_versions_and_stat_only(self):
        with patch.object(Path, "open", side_effect=AssertionError("inventory opened content")):
            records = pack.inventory(self.source)
        paths = [r.path for r in records]
        self.assertEqual(len(paths), 15)
        self.assertFalse(any(p.endswith((".py", ".pyc")) for p in paths))
        self.assertFalse(any("p1_" in p or p.startswith("r8_completion") for p in paths))
        self.assertIn("r8_results/Q_pilot_v1/old.bin", paths)
        self.assertIn("r8_results/Q_pilot_v2/new.bin", paths)
        finals = [r for r in records if r.category == "population_final"]
        self.assertEqual(len(finals), 3)

    def test_two_seed_preference_and_strict_byte_cut(self):
        records = pack.inventory(self.source)
        parts = pack.partition(records, max_bytes=15_000)
        integrated = [p for p in parts if p["category"] == "integrated_final"]
        self.assertEqual(len(integrated), 3)
        self.assertEqual([len({r.seed for r in p["records"]}) for p in integrated], [2, 2, 1])
        small = pack.partition(records, max_bytes=7_000)
        self.assertEqual(sum(p["category"] == "integrated_final" for p in small), 5)
        self.assertTrue(all(p["estimated_bytes"] <= 7_000 for p in small))

    def test_deterministic_publication_complete_coverage_and_hashes(self):
        a = pack.package(self.source, self.base / "out-a", max_bytes=15_000, source_commit="a" * 40)
        b = pack.package(self.source, self.base / "out-b", max_bytes=15_000, source_commit="a" * 40)
        self.assertEqual(a, b)
        self.assertEqual(a["nested_npz_zip_crc_checked"], 5)
        covered = []
        for item in a["archives"]:
            first = self.base / "out-a" / item["filename"]
            second = self.base / "out-b" / item["filename"]
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), item["sha256"])
            self.assertEqual(first.stat().st_size, item["bytes"])
            self.assertLessEqual(item["bytes"], 15_000)
            with zipfile.ZipFile(first) as z:
                self.assertIsNone(z.testzip())
                manifest = json.loads(z.read(pack.MANIFEST_NAME))
                self.assertTrue(manifest["independently_extractable"])
                for record in manifest["members"]:
                    self.assertEqual(z.read(record["path"]), (self.source / record["path"]).read_bytes())
                    covered.append(record["path"])
        self.assertEqual(sorted(covered), [r.path for r in pack.inventory(self.source)])
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(json.loads((self.base / "out-a/R8_EVIDENCE_INDEX.json").read_text()), a)

    def test_corrupt_inner_npz_rejected_before_publication(self):
        p = next(self.source.glob("r8_results/integrated_final/*/*/trace.npz"))
        data = bytearray(p.read_bytes())
        data[100] ^= 0xFF
        p.write_bytes(data)
        with self.assertRaises(zipfile.BadZipFile):
            pack.package(self.source, self.base / "corrupt-out")
        self.assertFalse((self.base / "corrupt-out").exists())

    def test_mutated_source_during_copy_rejected(self):
        original = pack._copy_member
        once = []
        def change(z, source, record, expected):
            if not once:
                once.append(True)
                (source / record.path).write_bytes(b"changed")
            return original(z, source, record, expected)
        with patch.object(pack, "_copy_member", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                pack.package(self.source, self.base / "mutation-out")
        self.assertFalse((self.base / "mutation-out/R8_EVIDENCE_INDEX.json").exists())
        self.assertFalse(list((self.base / "mutation-out").glob(".r8-package-*")))

    def test_no_overwrite_and_nested_destination(self):
        output = self.base / "once"
        pack.package(self.source, output)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            pack.package(self.source, output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        with self.assertRaisesRegex(ValueError, "outside"):
            pack.package(self.source, self.source / "r8_results/package")
        with self.assertRaisesRegex(ValueError, "absolute"):
            pack.package(self.source, Path("relative-out"))

    def test_symlinks_and_oversized_files_rejected(self):
        (self.source / "r8_results/leak.bin").symlink_to(self.base / "outside.bin")
        with self.assertRaisesRegex(ValueError, "regular"):
            pack.inventory(self.source)
        (self.source / "r8_results/leak.bin").unlink()
        self._put("r8_results/big.bin", b"x" * 10_000)
        with self.assertRaisesRegex(ValueError, "single file"):
            pack.partition(pack.inventory(self.source), max_bytes=7_000)


if __name__ == "__main__":
    unittest.main()
