"""Create once, or verify, the prospective P1 source and design record."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "p1_completion/FREEZE_P1.json"
ENGINE_HASH = "48330c1a88a2312182bb488dae0096e2ccf674c6fe2b0ca7ddf6fa9b8a29dccc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    paths = [p for p in (ROOT / "p1_completion").rglob("*")
             if p.is_file() and p.suffix in (".py", ".md", ".json")
             and "__pycache__" not in p.parts and p != FREEZE]
    paths += [ROOT / "collective/bridge_v2/engine.py"]
    return sorted(paths)


def verify(path=FREEZE):
    frozen = json.loads(Path(path).read_text())
    checks = {name: (ROOT / name).is_file() and sha(ROOT / name) == digest
              for name, digest in frozen["source_sha256"].items()}
    if not all(checks.values()):
        raise ValueError("Frozen source changed: " + str([k for k, v in checks.items() if not v]))
    return {"source_files": len(checks), "all_hashes_match": True,
            "freeze_sha256": sha(path), "final_seeds": frozen["final_seeds"]}


def create():
    from p1_completion import population, ecology, tension
    assert sha(ROOT / "collective/bridge_v2/engine.py") == ENGINE_HASH
    if FREEZE.exists():
        raise FileExistsError("The prospective record may not be overwritten")
    record = {
        "campaign": "P1 completion 2026-09-20",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prospective private versioned source freeze; descriptive campaign",
        "base_commit": "c6361ee1a5f85aa9ff1f1514f1f4d37afc8013a2",
        "tension_upstream_commit": "2d2d51120409ad39559e1311897bd6b99bbf748e",
        "scope": "P1_OVERVIEW.md and module designs; not historical confirmatory A/C",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths()},
        "population_source_sha256": population.source_hashes(),
        "tension_source_sha256": tension.source_hashes(),
        "population_config": asdict(population.config_default()),
        "population_grid": population.GRID,
        "population_variants": {"A": population.VARIANTS_A, "C": population.P1_VARIANTS_C},
        "factorial_flags": population.FACTORIAL_FLAGS,
        "ecology_protocol": ecology.PROTOCOL,
        "final_seeds": {"A": list(population.FINAL_SEEDS["A"]),
                        "C": list(population.FINAL_SEEDS["C"]),
                        "E": ecology.PROTOCOL["final_seeds"],
                        "T": list(tension.FINAL_SEEDS)},
        "excluded_exposed_seeds": list(range(931001, 931031)),
        "exposure_record": population.SEED_DEVIATION,
        "pilot_review": "Independent causal review completed; 32 new contract tests passed. "
                        "Adverse pilot outcomes retained without parameter selection.",
        "final_audit_plan": {
            "all": "verify unchanged source and every recorded artifact digest; retain all conditions",
            "A": "replay success/copy_score_equal/p50 seed index0 and success/resource_pool/p50 index29; check conformity score control and gen0 physics over grid",
            "C": "replay success full, coordinate_equivalent, anchor_off_short_vitality_blind, generic_shared_target at p50 seed index0; compare full/coordinate all16 grid-rule cells",
            "E": "record exact native replay every seed; verify file hashes and first-seed saved-policy replay; independently regenerate primary summaries",
            "T": "reconstruct metrics and equations from all raw traces; replay first final seed selected pulses and all external-task variants",
            "inference": "seed-level paired intervals; report p-star non-identification and adverse effects; no global confirmatory PASS",
        },
    }
    with FREEZE.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create() if args.create else verify(), indent=2))


if __name__ == "__main__":
    main()
