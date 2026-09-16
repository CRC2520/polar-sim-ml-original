"""Integrity-only aggregation for a completed B1-SC-D3 32-fit execution.

This validates completeness and immutable initialization lineage. It does not
adjudicate H_INFO_UTILITY or H_INFO_CAUSAL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ("GLOBAL-G0", "GLOBAL-S6", "LOCAL-G0", "LOCAL-S6")
EXPECTED_SNAPSHOTS = {
    0: "30ad184c683b471eb198aeeaeb1a24b835cefd1f36f0090d7da074cb7b7a6756",
    1: "07d69728675ec90a2aa024d85f4153939bccf5a62fc4414b8209874f03deb737",
    2: "096dc088d21180f972e48e6ed1870463244b5ee4d47392880b7f700084e85b7b",
    3: "2ae02b44def6e41e12f18e0672cd2afa2406c2a409fe3be3e33a99e8690dbf9d",
    4: "24eced5465c7ad649d5c9d47fe4407765c0e700e4a46b304e7b67b548112d849",
    5: "7c4b337ae420d6c20d7abc6ccd9ec04677cc07389c0414a12174cad9feb9a199",
    6: "7f0be21fe5f192a226a055bec8b734cef9ae8a5d59487d452ad5b9a481d01c91",
    7: "6cc4ed0b22c35988bde5a94fc5f6e571951d37a79f6ffb3fc7b8c268b40fadc1",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path) -> dict:
    results = list(root.rglob("RUN_RESULT.json"))
    require(len(results) == 32, f"Expected 32 RUN_RESULT.json files, found {len(results)}")
    seen = set()
    rows = []
    by_block = {b: [] for b in range(8)}
    for path in results:
        r = read_json(path)
        key = (r["condition"], int(r["block"]))
        require(key not in seen, f"Duplicate D3 fit: {key}")
        seen.add(key)
        require(r["status"] == "D3_SCIENTIFIC_FIT_COMPLETE", f"Incomplete D3 fit: {key}")
        require(r["split"] == "training" and r["scientific_evidence"] is True, f"Non-scientific fit artifact: {key}")
        require(r["scientific_training_performed"] is True, f"Scientific training flag missing: {key}")
        require(int(r["environment_steps"]) == 1048576, f"Wrong D3 budget: {key}")
        require(r["B1E_executed"] is False and r["final_seeds_generated"] is False, f"B1-E boundary violation: {key}")
        snap = r["initialization"]["snapshot_sha256"]
        require(snap == EXPECTED_SNAPSHOTS[key[1]], f"Frozen init mismatch: {key}")
        require(r["authorization"]["qualified_runner_commit"] == "e05223aad09b75e075d6c4fac87565d95531779c", f"Runner lineage mismatch: {key}")
        by_block[key[1]].append(snap)
        fit_dir = path.parent
        require((fit_dir / "ENDPOINT.json").is_file(), f"Missing endpoint: {key}")
        require((fit_dir / "model-final.pt").is_file(), f"Missing final model: {key}")
        diagnostics = sorted(fit_dir.glob("DIAGNOSTIC_*.json"))
        require(len(diagnostics) == 4, f"Missing D3 diagnostics: {key}")
        if key[0] == "LOCAL-S6":
            require((fit_dir / "CAUSAL.json").is_file(), f"Missing causal panel: {key}")
            require((fit_dir / "NO_ACTION_WITNESS.json").is_file(), f"Missing witness: {key}")
        else:
            require(not (fit_dir / "CAUSAL.json").exists(), f"Unexpected primary causal panel outside LOCAL-S6: {key}")
        rows.append({"condition": key[0], "block": key[1], "snapshot_sha256": snap})

    require(seen == {(c, b) for c in CONDITIONS for b in range(8)}, "D3 condition/block matrix incomplete")
    for block, snaps in by_block.items():
        require(len(snaps) == 4 and len(set(snaps)) == 1, f"Block {block} did not share identical initialization bytes")

    return {
        "schema": "B1-SC-D3-SCIENTIFIC-INTEGRITY-1.0.0-20260916",
        "status": "D3_SCIENTIFIC_EXECUTION_COMPLETE_INTEGRITY_PASS_NOT_ADJUDICATED",
        "fits_verified": 32,
        "blocks_verified": 8,
        "conditions_verified": list(CONDITIONS),
        "native_steps_per_fit": 1048576,
        "total_native_steps": 33554432,
        "local_s6_causal_panels_verified": 8,
        "initialization_pairing": "PASS",
        "qualified_runner_commit": "e05223aad09b75e075d6c4fac87565d95531779c",
        "B1E_executed": False,
        "final_seeds_generated": False,
        "adjudication_performed": False,
        "fits": sorted(rows, key=lambda x: (x["condition"], x["block"])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    payload = validate(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
