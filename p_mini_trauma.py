"""Compatibility filename for an external persistent-perturbation experiment.

The previous 'trauma/reconsolidation' claims were unsupported: this script alters
external input amplitude. Internal memory is tested separately in the P2 model.
"""
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from engine_v2_corrected import TensionEngine, EngineConfig, DEVICE
from report_utils import export_run, latex_summary_table, bar_compare, save_summary

@dataclass
class PersistentPerturbationCfg:
    N: int = 160
    K: int = 12
    steps: int = 28
    seed: int = 2025
    learn_M: bool = False
    polarity_types: int = 8
    pulse_step: int = 10
    pulse_magnitude: float = .8
    decay_rate: float = .15
    attenuate_external_input: bool = True
    attenuation_strength: float = .5
    attenuation_delay: int = 3
    attenuation_duration: int = 6
    release_step: int | None = None
    target_polarity_indices: list = field(default_factory=lambda: [3, 4, 5])
    tag: str = "External perturbation with attenuation"

class PersistentPerturbation:
    def __init__(self, cfg):
        if not 0 <= cfg.pulse_step < cfg.steps:
            raise ValueError("pulse_step must address an executed zero-indexed step")
        if not 0 < cfg.pulse_magnitude <= 1 or cfg.decay_rate < 0:
            raise ValueError("Require 0 < pulse_magnitude <= 1 and decay_rate >= 0")
        if cfg.polarity_types < 1 or not cfg.target_polarity_indices:
            raise ValueError("Explicit polarity types and nonempty targets are required")
        if any(not 0 <= i < cfg.polarity_types for i in cfg.target_polarity_indices):
            raise ValueError("Target polarity index out of bounds")
        if not 0 <= cfg.attenuation_strength <= 1 or cfg.attenuation_delay < 0 or cfg.attenuation_duration < 1:
            raise ValueError("Invalid external attenuation schedule")
        if cfg.release_step is not None and not cfg.pulse_step < cfg.release_step <= cfg.steps:
            raise ValueError("release_step must follow the pulse and not exceed steps")
        self.cfg = cfg
        self.eng = TensionEngine(EngineConfig(N=cfg.N, K=cfg.K, steps=cfg.steps,
                                              seed=cfg.seed, learn_M=cfg.learn_M))
        self.ts = None

    def attenuation(self, elapsed):
        cfg = self.cfg
        if not cfg.attenuate_external_input or elapsed < cfg.attenuation_delay:
            return 1.
        fraction = min(1., (elapsed - cfg.attenuation_delay) / cfg.attenuation_duration)
        return 1. - cfg.attenuation_strength * fraction

    def stimulus_at(self, step):
        cfg = self.cfg
        vector = np.zeros(cfg.N)
        if step < cfg.pulse_step or (cfg.release_step is not None and step >= cfg.release_step):
            return vector, 1.
        elapsed = step - cfg.pulse_step
        factor = self.attenuation(elapsed)
        value = -cfg.pulse_magnitude * math.exp(-cfg.decay_rate * elapsed) * factor
        # Unit i is explicitly assigned type i % polarity_types; no state inference.
        for i in range(cfg.N):
            if i % cfg.polarity_types in cfg.target_polarity_indices:
                vector[i] = value
        return vector, factor

    def run(self, out_dir, print_console=True):
        if self.eng.t:
            raise RuntimeError("Experiment instance has already advanced")
        os.makedirs(out_dir, exist_ok=True)
        self.eng.ts["external_attenuation_factor"] = []
        self.eng.ts["acute_pulse"] = []
        for step in range(self.cfg.steps):
            stimulus, factor = self.stimulus_at(step)
            self.eng.step(stimulus=torch.as_tensor(stimulus, dtype=torch.float32, device=DEVICE))
            self.eng.ts["external_attenuation_factor"].append(factor)
            self.eng.ts["acute_pulse"].append(step == self.cfg.pulse_step)
            if print_console:
                print(f"[Persistent input] step={step} stimulus_rms={np.sqrt(np.mean(stimulus**2)):.5f}")
        self.ts = self.eng.ts
        return self.ts

def compare_and_export(steps=28, out_root="results_corrected/persistent_perturbation",
                       N=160, K=12, seed=2025, plots=True, release_step=None):
    rows = []
    for attenuate, folder in ((True, "attenuated_input"), (False, "unattenuated_input")):
        tag = "External input attenuated" if attenuate else "External input unattenuated"
        cfg = PersistentPerturbationCfg(steps=steps, N=N, K=K, seed=seed,
                                        attenuate_external_input=attenuate, tag=tag, release_step=release_step)
        experiment = PersistentPerturbation(cfg)
        out_dir = os.path.join(out_root, folder)
        ts = experiment.run(out_dir=out_dir, print_console=False)
        control = TensionEngine(experiment.eng.cfg)
        control.run(print_console=False)
        rows.append(export_run(out_dir, folder, ts, tag,
                               metadata={"experiment": "Persistent external perturbation", "protocol": asdict(cfg),
                                         "config": asdict(experiment.eng.cfg), "seed": seed,
                                         "interpretation": "Attenuation changes external input, not internal memory."},
                               control=control.ts, plots=plots))
    latex_summary_table(out_root, "persistent_perturbation",
                        "External perturbation amplitude manipulation; not reconsolidation.",
                        "tab:persistent_perturbation", rows)
    if plots:
        bar_compare(out_root, "persistent_perturbation", rows, "Persistent external input: descriptors")
    save_summary(out_root, "persistent_perturbation_all", {"rows": rows, "seed": seed})
    return rows

# Only the class names remain as import aliases; obsolete keyword names fail
# explicitly, preventing silent reuse of 'reconsolidation' claims.
TraumaCfg = PersistentPerturbationCfg
MiniTrauma = PersistentPerturbation

if __name__ == "__main__":
    compare_and_export()
