"""Contracts that prevent incomplete or silently replaced scientific evidence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from r8_completion.io import atomic_json, atomic_npz, sha256, verify_npz, write_manifest


class EvidenceIOContracts(unittest.TestCase):
    def test_arrays_roundtrip_and_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.npz"
            record = atomic_npz(path, x=np.arange(12).reshape(3, 4))
            self.assertEqual(record["sha256"], sha256(path))
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive["x"], np.arange(12).reshape(3, 4))
            with self.assertRaises(FileExistsError):
                atomic_npz(path, x=np.zeros(1))
            self.assertEqual(record["sha256"], sha256(path))

    def test_corruption_prevents_manifest_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.npz"
            record = atomic_npz(path, x=np.arange(10))
            path.write_bytes(path.read_bytes()[:80])
            with self.assertRaises(Exception):
                verify_npz(path)
            manifest = Path(directory) / "complete.json"
            with self.assertRaises(Exception):
                write_manifest(manifest, [record])
            self.assertFalse(manifest.exists())

    def test_nonfinite_json_and_pickle_objects_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                atomic_json(Path(directory) / "bad.json", {"metric": float("nan")})
            with self.assertRaises(TypeError):
                atomic_npz(Path(directory) / "bad.npz", x=np.array([{}], dtype=object))


if __name__ == "__main__":
    unittest.main()
