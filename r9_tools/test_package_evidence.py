"""Small temporary fixtures only; never reads or packages scientific evidence."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from r9_tools.package_evidence import (
    AUTHORIZATION, CHECKSUMS, FREEZE, INDEX, MAX_BYTES,
    digest_file, inventory, json_bytes, package_evidence, partition,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else json_bytes(data))


def fixture(root):
    write(root / FREEZE, {"pilot_seeds": [101], "final_seeds": [201, 202]})
    frozen = digest_file(root / FREEZE)["sha256"]
    write(root / AUTHORIZATION, {"authorized": True, "freeze_sha256": frozen,
                               "source_commit": "a" * 40})
    write(root / "r9_results/SOURCE_PUBLICATION.json", {"source_commit": "a" * 40})
    write(root / "r9_results/pilot_v1/SOURCE_RECORD.json", {"fixture": True})
    write(root / "r9_results/report.md", b"Fixture report only.\n")
    write(root / "r9_results/helper.py", b"excluded = True\n")
    write(root / "r9_results/SOURCE_SNAPSHOT/config.json", {"excluded": True})
    write(root / "r9_results/__pycache__/cache.bin", b"excluded")
    for phase, seed in (("pilot", 101), ("final", 201), ("final", 202)):
        directory = root / f"r9_results/{phase}/data/seed_{seed}"
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture.npy", b"CRC transport fixture, not a scientific array.\n")
        write(directory / "trace.npz", data.getvalue())
        write(directory / "events.jsonl.gz", gzip.compress(b'{"fixture":true}\n', mtime=0))
        write(directory / "summary.json", {"phase": phase, "seed": seed})
        for i in range(7):
            write(directory / f"record_{i}.bin", bytes([i + seed % 200]) * 1024)
        entries = [dict(path=p.relative_to(directory).as_posix(), **{
            k: v for k, v in digest_file(p).items() if k != "crc32"})
                   for p in sorted(directory.rglob("*")) if p.is_file()]
        write(directory / "COMPLETE.json", {
            "complete": True, "metadata": {
                "phase": phase, "seed": seed,
                "source": {"freeze_sha256": frozen} if phase == "final" else {}},
            "files": entries})
    return frozen


class EvidencePackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r9-package-fixture-")
        self.root = Path(self.tmp.name) / "source"
        self.root.mkdir()
        self.freeze = fixture(self.root)
        self.output = Path(self.tmp.name) / "package"

    def tearDown(self):
        self.tmp.cleanup()

    def test_bounded_deterministic_round_trip_and_exclusions(self):
        result = package_evidence(self.root, self.output, maximum=16384)
        self.assertGreater(len(result["archives"]), 1)
        self.assertEqual(result["closure"]["source_commit"], "a" * 40)
        self.assertFalse(result["closure"]["freeze_included_in_archives"])
        restored = Path(self.tmp.name) / "restored"
        all_paths, metadata_paths = set(), set()
        for row in result["archives"]:
            path = self.output / row["filename"]
            self.assertLessEqual(path.stat().st_size, 16384)
            self.assertEqual(digest_file(path), {k: row[k] for k in ("bytes", "sha256", "crc32")})
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                archive.extractall(restored)
            for item in row["files"]:
                self.assertNotIn(item["path"], all_paths)
                all_paths.add(item["path"])
                self.assertEqual(digest_file(restored / item["path"]),
                                 {k: item[k] for k in ("bytes", "sha256", "crc32")})
            for item in row["metadata_files"]:
                self.assertNotIn(item["path"], metadata_paths)
                metadata_paths.add(item["path"])
        self.assertEqual(all_paths, {r.path for r in inventory(self.root)})
        self.assertNotIn(FREEZE, all_paths)
        self.assertIn("r9_results/pilot_v1/SOURCE_RECORD.json", all_paths)
        self.assertFalse(any(".py" in p or "SOURCE_SNAPSHOT" in p for p in all_paths))
        self.assertEqual(result["inner_container_crc_checks"], {"npz_or_zip": 3, "gzip": 3})
        for line in (self.output / CHECKSUMS).read_text().splitlines():
            expected, filename = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), expected)
        self.assertTrue(json.loads((self.output / INDEX).read_text())["complete"])
        second = package_evidence(self.root, Path(self.tmp.name) / "second", maximum=16384)
        self.assertEqual(result, second)
        with self.assertRaises(FileExistsError):
            package_evidence(self.root, self.output, maximum=16384)

    def test_incomplete_final_rejected_before_publication(self):
        shutil.rmtree(self.root / "r9_results/final/data/seed_202")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            package_evidence(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_completion_marker_rejected(self):
        (self.root / "r9_results/final/data/seed_201/COMPLETE.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete seed"):
            package_evidence(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_recorded_hash_corruption_rejected(self):
        (self.root / "r9_results/final/data/seed_201/record_0.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            package_evidence(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_inner_container_corruption_rejected(self):
        write(self.root / "r9_results/audits/broken.jsonl.gz", b"not gzip")
        with self.assertRaises(OSError):
            package_evidence(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_oversized_single_file_and_invalid_limits_rejected(self):
        write(self.root / "r9_results/large.bin", b"x" * 18000)
        with self.assertRaisesRegex(ValueError, "whole file"):
            package_evidence(self.root, self.output, maximum=16384)
        with self.assertRaises(ValueError):
            partition(inventory(self.root), self.freeze, MAX_BYTES + 1)
        self.assertFalse(self.output.exists())

    def test_symlinks_recursive_output_and_commit_mismatch_rejected(self):
        (self.root / "r9_results/link.bin").symlink_to(self.root / FREEZE)
        with self.assertRaisesRegex(ValueError, "regular"):
            package_evidence(self.root, self.output)
        (self.root / "r9_results/link.bin").unlink()
        with self.assertRaisesRegex(ValueError, "outside"):
            package_evidence(self.root, self.root / "r9_results/package")
        with self.assertRaisesRegex(ValueError, "differs"):
            package_evidence(self.root, self.output, source_commit="b" * 40)


if __name__ == "__main__":
    unittest.main()
