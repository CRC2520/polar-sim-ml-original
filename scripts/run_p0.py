#!/usr/bin/env python3
"""Regenerate all corrected P0 experiments, preserving original result folders."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from p2_baseline import main as p2
from p6_stimulus_mapping import main as p6
from p7_learnedM import run_all as p7, parser as p7_parser
from p_mini_iacl import run_and_export as iacl
from p_mini_trauma import compare_and_export as perturbation
from report_utils import REPORT_VERSION, compress_trace

def run_suite(out, seeds, plots=True, compress=True):
    out = Path(out)
    records = []
    torch.set_num_threads(1)
    for seed in seeds:
        directory = out / f"seed_{seed}"
        p2(out_dir=str(directory / "p2"), seed=seed, plots=plots)
        p6(out_dir=str(directory / "p6"), seed=seed, plots=plots)
        args = p7_parser().parse_args(["--out", str(directory / "p7"), "--seed", str(seed),
                                      "--no-ethics", "--no-homeostasis", "--quiet"]
                                     + ([] if plots else ["--no-plots"]))
        p7(args)
        iacl(out_dir=str(directory / "iacl"), seed=seed, plots=plots)
        perturbation(out_root=str(directory / "perturbation"), seed=seed, plots=plots)
        files = [
            ("p2", directory / "p2", "p2_baseline"),
            ("p6", directory / "p6", "p6_stimulus"),
            ("p7", directory / "p7" / "baseline", "p7"),
            ("p7_no_modulation", directory / "p7" / "no_modulation", "p7"),
            ("p7_no_homeostasis", directory / "p7" / "no_homeostasis", "p7"),
            ("iacl", directory / "iacl", "mini_iacl"),
            ("perturb_reduced", directory / "perturbation" / "attenuated_input", "attenuated_input"),
            ("perturb_no_reduction", directory / "perturbation" / "unattenuated_input", "unattenuated_input"),
        ]
        suffix = ".gz" if compress else ""
        for scenario, folder, name in files:
            trace_path = folder / f"{name}_timeseries.json"
            payload = json.loads(trace_path.read_text())
            control = folder / f"{name}_control_timeseries.json"
            records.append({"scenario": scenario, "seed": seed,
                            "trace_json": str(trace_path.relative_to(out)) + suffix,
                            "trace_csv": str((folder / f"{name}_timeseries.csv").relative_to(out)) + suffix,
                            "summary_json": str((folder / f"{name}_summary.json").relative_to(out)),
                            "paired_control_json": str(control.relative_to(out)) + suffix if control.exists() else None,
                            "config": payload["metadata"].get("config"),
                            "protocol": payload["metadata"].get("protocol")})
        if compress:
            for extension in ("*_timeseries.json", "*_timeseries.csv"):
                for trace_file in sorted(directory.rglob(extension)):
                    compress_trace(trace_file)
        print(f"Completed corrected P0 experiments for seed {seed}")
    manifest = {"version": REPORT_VERSION, "seeds": list(seeds), "step_indexing": "zero_based",
                "torch_version": torch.__version__, "threads": 1, "compression": "gzip-mtime0" if compress else None, "scenarios": records,
                "recovery_rule": {"reference": "matched_unstimulated_control", "tolerance_rms": .05,
                                  "sustained_steps": 3, "search": "after last nonzero stimulus"},
                "interpretation": "Descriptive dynamics only; no task-performance or consciousness conclusions."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_corrected/p0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027, 2028, 2029])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be distinct")
    run_suite(args.out, args.seeds, plots=not args.no_plots, compress=not args.no_compress)

if __name__ == "__main__":
    main()
