"""Validate ZIP CRC and complete NumPy loading of all final P1.E NPZ containers."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p1_ecology_final_20260920"
manifest = json.loads((OUT / "MANIFEST.json").read_text())["files"]
npz_files = sorted(OUT.rglob("*.npz"))
failures = []
members = arrays = 0
for path in npz_files:
    relative = str(path.relative_to(OUT))
    try:
        with zipfile.ZipFile(path) as container:
            members += len(container.infolist())
            bad = container.testzip()
            if bad is not None:
                raise ValueError(f"ZIP CRC error: {bad}")
        with np.load(path, allow_pickle=False) as container:
            for key in container.files:
                np.asarray(container[key])
                arrays += 1
    except Exception as exc:
        failures.append({"file":relative,"error":repr(exc)})
hash_failures = []
for relative, expected in manifest.items():
    path = OUT / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        hash_failures.append(relative)
report = {
    "all_checks_pass": not failures and not hash_failures,
    "npz_count": len(npz_files),
    "zip_members_crc_checked": members,
    "numpy_arrays_loaded": arrays,
    "containers_with_crc_or_load_failure": failures,
    "manifest_artifacts_rechecked": len(manifest),
    "hash_failures": hash_failures,
    "source_or_original_outputs_modified": False,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / "p1_results/ECOLOGY_CONTAINER_INTEGRITY.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
if not report["all_checks_pass"]:
    raise SystemExit(1)
