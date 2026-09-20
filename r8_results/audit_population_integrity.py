"""Read-only integrity closure for the three frozen population campaigns.

This audit lives outside the frozen scientific source and draws no random seeds.
"""
from pathlib import Path
import argparse
import json

from r8_completion.io import atomic_json, sha256, verify_npz
from r8_completion.provenance import verify


def audit(root):
    root = Path(root)
    source_before = verify()
    result = {"source_before": source_before, "campaigns": {}}
    seeds = list(range(941001, 941031))
    totals = {"cells": 0, "recorded_files": 0, "npz_containers": 0, "npz_arrays": 0, "bytes": 0}
    for name, count, summary in (("Q_final", 48, "Q_FINAL_SUMMARY.json"),
                                 ("threshold_final", 18, "THRESHOLD_FINAL_SUMMARY.json"),
                                 ("A_final", 30, "CAUSAL_A_FINAL_SUMMARY.json")):
        directory = root / name
        manifests = sorted(directory.glob("*/COMPLETE.json"))
        if len(manifests) != count:
            raise ValueError(f"{name}: expected {count} completed cells, found {len(manifests)}")
        campaign = {"cells": [], "summary_sha256": sha256(directory / summary)}
        for manifest_path in manifests:
            cell = manifest_path.parent
            manifest = json.loads(manifest_path.read_text())
            protocol = json.loads((cell / "PROTOCOL.json").read_text())
            if protocol["seeds"] != seeds:
                raise ValueError(f"Unexpected final seeds: {cell}")
            row = {"cell": cell.name, "manifest_sha256": sha256(manifest_path), "files": []}
            for record in manifest["files"]:
                path = cell / record["path"]
                if sha256(path) != record["sha256"]:
                    raise ValueError(f"SHA mismatch: {path}")
                detail = {"path": record["path"], "sha256": record["sha256"], "bytes": path.stat().st_size}
                if path.suffix == ".npz":
                    verified = verify_npz(path)
                    detail["arrays"] = verified["arrays"]
                    totals["npz_containers"] += 1
                    totals["npz_arrays"] += len(verified["arrays"])
                totals["recorded_files"] += 1
                totals["bytes"] += path.stat().st_size
                row["files"].append(detail)
            campaign["cells"].append(row)
            totals["cells"] += 1
        result["campaigns"][name] = campaign
    result["totals"] = totals
    result["source_after"] = verify()
    result["all_recorded_sha256_match"] = True
    result["all_npz_crc_and_safe_load_valid"] = True
    result["all_final_seed_lists_match"] = True
    result["scope"] = "Integrity audit only; independent trajectory replay and metric reconstruction are reported separately."
    atomic_json(root / "POPULATION_FINAL_INTEGRITY.json", result)
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    audit(parser.parse_args().root)
