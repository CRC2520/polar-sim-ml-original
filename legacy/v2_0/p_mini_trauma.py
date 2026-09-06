#!/usr/bin/env python
# p_mini_trauma.py — Mini-Trauma loop: acute shock + lingering bias with/without reconsolidation.
# Produces per-run CSV/JSON/PNG and a comparison LaTeX table + bars.

import os
import sys
import json
import math
import traceback
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

# ----- Engine + reporting -----
try:
    from engine_v1_locked import TensionEngine, EngineConfig, DEVICE
except Exception as e:
    print("[ERROR] Could not import engine_v1_locked:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    from report_utils import (
        save_timeseries,
        save_summary,
        plot_hgi_inc,
        latex_summary_table,
        bar_compare,
    )
except Exception as e:
    print("[ERROR] Could not import report_utils:", e)
    traceback.print_exc()
    sys.exit(1)


# ========================= Config =========================
@dataclass
class TraumaCfg:
    # Engine
    N: int = 160
    K: int = 12
    steps: int = 28
    seed: int = 2025
    learn_M: bool = False

    # Core (small) space size used to shape trauma stimuli
    core_dim: int = 8  # <- explicit core dimension (was inferred from ts["A"])

    # Trauma protocol
    trauma_step: int = 10              # when the acute shock hits (0-indexed in loop prints as step+1)
    trauma_mag: float = 0.8            # magnitude of acute negative pulse in small/core space
    decay_lambda: float = 0.15         # lingering decay rate (per step)
    with_reconsolidation: bool = True  # enable reconsolidation
    rho: float = 0.50                  # reconsolidation factor (0=no recon effect, 1=full neutralization)
    recon_delay: int = 3               # steps after trauma to begin reconsolidation window
    recon_len: int = 6                 # length of reconsolidation window (steps)

    # Which core indices to target (length Lsmall, typically 8)
    # By default simulate impact on nodes ~ {Desire/Limit, Control/Surrender, Freedom/Order}
    target_core_idx: Optional[List[int]] = None  # e.g., [4, 3, 5]

    # Logging titles
    tag: str = "Trauma+Recon"


# ========================= Runner =========================
class MiniTrauma:
    """
    Applies:
      - An acute, negative 'shock' stimulus at trauma_step in a small/core space (Lsmall, usually 8).
      - A lingering bias after the shock that decays exponentially (lambda).
      - Optional reconsolidation that attenuates lingering bias during a window (rho).
    Mapping from small/core space -> engine N is done by tiling/repeat to match N.
    """

    def __init__(self, cfg: TraumaCfg):
        self.cfg = cfg
        self.eng = TensionEngine(
            EngineConfig(
                N=cfg.N,
                K=cfg.K,
                steps=cfg.steps,
                seed=cfg.seed,
                learn_M=cfg.learn_M,
            )
        )
        # default targets if not provided
        if cfg.target_core_idx is None:
            # safe defaults within [0 .. core_dim-1]
            cd = int(cfg.core_dim)
            # pick up to three distinct indices spread across the core space
            picks = [min(4, cd-1), min(3, cd-1), min(5, cd-1)] if cd >= 6 else list(range(cd))
            self.cfg.target_core_idx = sorted(set(i for i in picks if 0 <= i < cd))

        self.ts = None  # will hold engine timeseries after run

    # ---------- helpers ----------
    def _small_dim(self) -> int:
        """Return configured core (small) dimension; no dependency on engine logs."""
        return int(self.cfg.core_dim)

    def _expand_to_engine_dim(self, v_small: np.ndarray) -> np.ndarray:
        """Expand small vector (Lsmall) to engine N by tiling or repeat/trim."""
        N = self.cfg.N
        L = len(v_small)
        if L == N:
            return v_small.astype(float, copy=False)
        if L == 0:
            return np.zeros(N, dtype=float)
        if N % L == 0:
            reps = N // L
            return np.tile(v_small, reps).astype(float, copy=False)
        reps = int(np.ceil(N / L))
        return np.tile(v_small, reps)[:N].astype(float, copy=False)

    def _acute_shock_small(self, Lsmall: int) -> np.ndarray:
        """Build an acute negative pulse in small/core space."""
        s = np.zeros(Lsmall, dtype=float)
        for i in self.cfg.target_core_idx:
            if 0 <= i < Lsmall:
                s[i] = -abs(self.cfg.trauma_mag)
        return s

    def _lingering_small(self, acute_small: np.ndarray, t: int) -> np.ndarray:
        """Exponential lingering from the acute pulse (t steps after trauma)."""
        lam = self.cfg.decay_lambda
        return acute_small * math.exp(-lam * max(t, 0))

    def _apply_recon(self, lingering_small: np.ndarray, t_after: int) -> np.ndarray:
        """
        Apply reconsolidation window:
          - starts at trauma_step + recon_delay
          - lasts recon_len steps
          - attenuates lingering by (1 - rho) linearly across window
        """
        if not self.cfg.with_reconsolidation:
            return lingering_small
        start = self.cfg.recon_delay
        end = start + self.cfg.recon_len
        if t_after < start:
            return lingering_small
        if t_after >= end:
            # after the window, lingering remains attenuated by full rho
            return lingering_small * (1.0 - self.cfg.rho)
        # inside window: linear ramp of attenuation
        frac = (t_after - start) / max(1, (end - start))
        att = 1.0 - self.cfg.rho * frac
        return lingering_small * max(att, 0.0)

    # ---------- main ----------
    def run(self, out_dir: str, print_console: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        if print_console:
            print(f"[TRAUMA] Start: steps={self.cfg.steps}, tag={self.cfg.tag}, recon={self.cfg.with_reconsolidation}")

        # warm-up 1 step to ensure A is logged
        self.eng.step(stimulus=None)

        Lsmall = self._small_dim()
        acute_small = self._acute_shock_small(Lsmall)

        for t in range(self.cfg.steps):
            stim_np = None

            if t == self.cfg.trauma_step:
                # acute shock
                stim_np = self._expand_to_engine_dim(acute_small)
                if print_console:
                    print(f"[TRAUMA] Acute shock at step {t+1}: |stim_small|={np.linalg.norm(acute_small):.3f}")

            elif t > self.cfg.trauma_step:
                # lingering with optional reconsolidation
                t_after = t - self.cfg.trauma_step
                linger_small = self._lingering_small(acute_small, t_after)
                linger_small = self._apply_recon(linger_small, t_after)
                stim_np = self._expand_to_engine_dim(linger_small)

            # step engine
            stim_torch = None if stim_np is None else torch.tensor(stim_np, dtype=torch.float32, device=DEVICE)
            self.eng.step(stimulus=stim_torch)

            # print last metrics
            H = self.eng.ts["HGI"][-1]
            I = self.eng.ts["INC"][-1]
            S = self.eng.ts["SAT"][-1]
            a = self.eng.ts["alpha"][-1]
            o = self.eng.ts["override"][-1]
            if print_console:
                print(f"[TRAUMA] Step {t+1:02d}: HGI={H:.4f}, INC={I:.4f}, SAT={S:.2f}, α={a:.3f}, override={o}")

        self.ts = self.eng.ts
        return self.ts


# ========================= Compare & export =========================
def compare_and_export(steps: int = 28, out_root: str = "results_p_mini_trauma_compare"):
    os.makedirs(out_root, exist_ok=True)

    # With reconsolidation
    cfg1 = TraumaCfg(steps=steps, with_reconsolidation=True, tag="Trauma+Recon")
    run1 = MiniTrauma(cfg1)
    ts1 = run1.run(out_dir=os.path.join(out_root, "with_recon"), print_console=True)

    # Without reconsolidation
    cfg2 = TraumaCfg(steps=steps, with_reconsolidation=False, rho=0.0, tag="Trauma (no Recon)")
    run2 = MiniTrauma(cfg2)
    ts2 = run2.run(out_dir=os.path.join(out_root, "no_recon"), print_console=True)

    # Save artifacts for each
    def per_run_artifacts(tag_dir: str, tag_name: str, ts: dict):
        # figure
        plot_hgi_inc(tag_dir, tag_name, ts, f"Mini-Trauma: {tag_name}")
        # summary
        H = np.array(ts["HGI"]); I = np.array(ts["INC"])
        row = {
            "Config": tag_name,
            "HGI_final": float(H[-1]),
            "INC_final": float(I[-1]),
            "HGI_mean": float(H.mean()),
            "INC_mean": float(I.mean()),
            "Interventions": float(np.mean(ts.get("override", [0]))),
            "Recovery": ""  # optional: infer from INC dip/recovery if needed
        }
        save_timeseries(tag_dir, tag_name, ts)
        save_summary(tag_dir, tag_name, row)
        return row

    row1 = per_run_artifacts(os.path.join(out_root, "with_recon"), "Trauma+Recon", ts1)
    row2 = per_run_artifacts(os.path.join(out_root, "no_recon"), "Trauma (no Recon)", ts2)

    # Combined table + bars
    rows = [row1, row2]
    latex_summary_table(
        out_root,
        "mini_trauma",
        caption="Mini-trauma loop: reconsolidation vs none.",
        label="tab:mini_trauma",
        rows=rows,
    )
    bar_compare(out_root, "mini_trauma", rows, "Mini-Trauma — Final HGI/INC")

    print("[TRAUMA] Artifacts written to:", out_root)


if __name__ == "__main__":
    try:
        compare_and_export()
    except Exception as e:
        print("[TRAUMA] FAILED:", e)
        traceback.print_exc()
        sys.exit(1)
