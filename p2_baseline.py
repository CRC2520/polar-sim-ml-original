"""P2 corrected descriptive baseline. Original files remain under legacy/v2_0."""
from dataclasses import asdict
from engine_v2_corrected import TensionEngine, EngineConfig
from report_utils import export_run

def main(out_dir="results_corrected/p2_baseline", steps=20, N=120, K=10, seed=2025, plots=True):
    cfg = EngineConfig(N=N, K=K, steps=steps, seed=seed, learn_M=False)
    eng = TensionEngine(cfg)
    ts = eng.run(print_console=False)
    return export_run(out_dir, "p2_baseline", ts, "P2 corrected baseline",
                      metadata={"experiment": "P2", "config": asdict(cfg), "seed": seed,
                                "polarity_types": 8, "state_units": N, "external_stimulus": False},
                      plots=plots)

if __name__ == "__main__":
    main()
