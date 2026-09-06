#!/usr/bin/env python3
"""Rebuild P0 summaries/tables/plots solely from saved traces, without simulation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_utils import export_run, latex_summary_table, bar_compare, save_summary, load_json, compress_trace

LABELS = {
    "p2": "P2 corrected baseline", "p6": "P6 signed input: Deseo", "p7": "P7 Learned M",
    "p7_no_modulation": "P7 No modulation or override",
    "p7_no_homeostasis": "P7 No baseline restoring drive", "iacl": "Mini-IACL 3 agents",
    "perturb_reduced": "External input attenuated", "perturb_no_reduction": "External input unattenuated",
}

def rebuild(index_path, plots=True):
    index_path = Path(index_path)
    manifest = load_json(index_path)
    root = index_path.parent
    by_seed = {}
    for record in manifest["scenarios"]:
        trace_path = root / record["trace_json"]
        payload = load_json(trace_path)
        control = load_json(root / record["paired_control_json"])["timeseries"] if record["paired_control_json"] else None
        name = trace_path.name.removesuffix(".gz").removesuffix("_timeseries.json")
        row = export_run(str(trace_path.parent), name, payload["timeseries"], LABELS[record["scenario"]],
                         metadata=payload["metadata"], control=control, plots=plots)
        by_seed.setdefault(record["seed"], {})[record["scenario"]] = row
        if trace_path.suffix == ".gz":
            for candidate in trace_path.parent.glob("*_timeseries.*"):
                if candidate.suffix in (".json", ".csv"):
                    compress_trace(candidate)
    for seed, rows in by_seed.items():
        seed_dir = root / f"seed_{seed}"
        p7 = [rows[key] for key in ("p7", "p7_no_modulation", "p7_no_homeostasis")]
        perturbation = [rows[key] for key in ("perturb_reduced", "perturb_no_reduction")]
        latex_summary_table(str(seed_dir / "p7"), "p7_ablation", "Matched stimulus and seed in learned-M ablations.", "tab:p7_ablation", p7)
        latex_summary_table(str(seed_dir / "perturbation"), "persistent_perturbation",
                            "External perturbation amplitude manipulation; not reconsolidation.",
                            "tab:persistent_perturbation", perturbation)
        if plots:
            bar_compare(str(seed_dir / "p7"), "p7_ablation", p7, "P7 descriptive dynamics")
            bar_compare(str(seed_dir / "perturbation"), "persistent_perturbation", perturbation, "Persistent external input: descriptors")
        combined = seed_dir / "p7" / "p7_all_summary.json"
        previous = load_json(combined) if combined.exists() else {"seed": seed}
        previous["rows"] = p7
        save_summary(str(seed_dir / "p7"), "p7_all", previous)
        save_summary(str(seed_dir / "perturbation"), "persistent_perturbation_all", {"rows": perturbation, "seed": seed})
    return len(manifest["scenarios"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="results_corrected/p0/index.json")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(f"Regenerated {rebuild(args.index, not args.no_plots)} scenarios from saved traces.")
