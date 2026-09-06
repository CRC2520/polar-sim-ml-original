#!/usr/bin/env python
# p_mini_iacl.py — Mini-IACL (multi-agent consensus) with CSV/PNG/LaTeX outputs

import os
import sys
import json
import traceback
import numpy as np
import torch

# ---- engine + utils ----
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
        latex_summary_table,
        plot_hgi_inc,
    )
except Exception as e:
    print("[ERROR] Could not import report_utils:", e)
    traceback.print_exc()
    sys.exit(1)


class MiniIACL:
    """
    Minimal multi-agent wrapper:
      - Creates num_agents engines (no learned M here).
      - Each step: compute group mean state from logged A; apply a small
        consensus stimulus gamma_g * (A_mean - A_i) to each agent.
      - Logs group-averaged HGI/INC/SAT/alpha/override each step.
    """

    def __init__(
        self,
        num_agents: int = 3,
        N: int = 64,
        K: int = 8,
        steps: int = 24,
        seed: int = 2025,
        gamma_group_start: float = 0.00,
        gamma_group_end: float = 0.28,
    ):
        self.num_agents = num_agents
        self.steps = steps
        self.gamma_group_start = gamma_group_start
        self.gamma_group_end = gamma_group_end

        self.agents = []
        for i in range(num_agents):
            cfg = EngineConfig(
                N=N,
                K=K,
                steps=steps,
                seed=seed + i,
                learn_M=False,  # fixed M for this demo
            )
            self.agents.append(TensionEngine(cfg))

        # group time series (averages across agents)
        self.ts = {
            "step": [],
            "HGI": [],
            "INC": [],
            "SAT": [],
            "alpha": [],
            "override": [],
        }

    def _current_A(self, eng: TensionEngine) -> np.ndarray:
        """Get last logged state vector A from engine timeseries."""
        if "A" in eng.ts and len(eng.ts["A"]) > 0:
            return np.asarray(eng.ts["A"][-1], dtype=float)
        # fallback: try attribute if engine exposes it
        if hasattr(eng, "A"):
            return np.asarray(getattr(eng, "A"), dtype=float)
        # last resort: zero vec (should not happen after warm-up)
        return np.zeros(8, dtype=float)

    def _engine_N(self, eng) -> int:
        """Infer engine state dimension N from its M matrix."""
        if hasattr(eng, "M"):
            M = eng.M
            if hasattr(M, "shape"):
                return int(M.shape[0])
            try:
                return int(M.detach().cpu().shape[0])
            except Exception:
                pass
        # Fallback to length of last logged A if present
        a = self._current_A(eng)
        return int(len(a))

    def _expand_to_engine_dim(self, v_small: np.ndarray, N_eng: int) -> np.ndarray:
        """
        Expand a small vector (e.g., 8D core) to the engine's N (e.g., 64D).
        Strategy:
          - if lengths match: return as is
          - if divisible: tile evenly
          - else: repeat & trim
        """
        L = int(len(v_small))
        if L == N_eng:
            return v_small.astype(float, copy=False)

        if L == 0:
            return np.zeros(N_eng, dtype=float)

        if N_eng % L == 0:
            reps = N_eng // L
            return np.tile(v_small, reps).astype(float, copy=False)

        # fallback: repeat & trim
        reps = int(np.ceil(N_eng / L))
        v = np.tile(v_small, reps)[:N_eng]
        return v.astype(float, copy=False)

    def run(self, out_dir: str = "results_p_mini_iacl", print_console: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        if print_console:
            print(f"[IACL] Start: agents={self.num_agents}, steps={self.steps}")

        # Warm-up once so eng.ts["A"] exists
        for eng in self.agents:
            eng.step(stimulus=None)

        for t in range(self.steps):
            # 1) Compute group mean from current logged states (likely 8D)
            A_stack_small = np.stack([self._current_A(eng) for eng in self.agents], axis=0)  # [A, L]
            A_mean_small = A_stack_small.mean(axis=0)  # [L]
            gamma_g = float(np.interp(t, [0, self.steps - 1],
                                      [self.gamma_group_start, self.gamma_group_end]))

            # 2) Advance each agent with expanded consensus stimulus
            for i, eng in enumerate(self.agents):
                A_i_small = A_stack_small[i]                       # length L (e.g., 8)
                N_eng = self._engine_N(eng)                       # target length (e.g., 64)

                # consensus pull in small space
                stim_small = gamma_g * (A_mean_small - A_i_small)  # [L]

                # expand to engine's N
                stim_np = self._expand_to_engine_dim(stim_small, N_eng)  # [N_eng]

                # torch tensor on correct device
                stim = torch.tensor(stim_np, dtype=torch.float32, device=DEVICE)
                eng.step(stimulus=stim)

            # 3) Log group averages
            H = float(np.mean([eng.ts["HGI"][-1] for eng in self.agents]))
            I = float(np.mean([eng.ts["INC"][-1] for eng in self.agents]))
            S = float(np.mean([eng.ts["SAT"][-1] for eng in self.agents]))
            a = float(np.mean([eng.ts["alpha"][-1] for eng in self.agents]))
            ov = float(np.mean([float(eng.ts["override"][-1]) for eng in self.agents]))

            self.ts["step"].append(t)
            self.ts["HGI"].append(H)
            self.ts["INC"].append(I)
            self.ts["SAT"].append(S)
            self.ts["alpha"].append(a)
            self.ts["override"].append(ov)

            if print_console:
                print(f"[IACL] Step {t+1}: HGI={H:.4f}, INC={I:.4f}, SAT={S:.2f}, α={a:.3f}, override%={ov:.2f}")

        return self.ts


def run_and_export(out_dir: str = "results_p_mini_iacl", agents: int = 3, steps: int = 24):
    env = MiniIACL(num_agents=agents, steps=steps)
    ts = env.run(out_dir=out_dir, print_console=True)

    # Figure
    plot_hgi_inc(out_dir, "mini_iacl", ts, "Mini-IACL: Group Harmony/Coherence")

    # Summary row
    H = np.array(ts["HGI"])
    I = np.array(ts["INC"])
    row = {
        "Config": f"IACL {agents} agents",
        "HGI_final": float(H[-1]),
        "INC_final": float(I[-1]),
        "HGI_mean": float(H.mean()),
        "INC_mean": float(I.mean()),
        "Interventions": float(np.mean(ts["override"])),
        "Recovery": "",
    }

    # Artifacts
    save_timeseries(out_dir, "mini_iacl", ts)
    save_summary(out_dir, "mini_iacl", row)
    latex_summary_table(
        out_dir,
        "mini_iacl",
        caption="Mini-IACL multi-agent averages.",
        label="tab:mini_iacl",
        rows=[row],
    )
    print("[IACL] Artifacts written to:", out_dir)


if __name__ == "__main__":
    try:
        run_and_export()
    except Exception as e:
        print("[IACL] FAILED:", e)
        traceback.print_exc()
        sys.exit(1)
