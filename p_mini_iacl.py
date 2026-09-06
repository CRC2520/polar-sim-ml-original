"""Mini-IACL: corrected synchronous consensus control, not a consciousness test."""
import os
from dataclasses import asdict

import numpy as np
import torch
from engine_v2_corrected import TensionEngine, EngineConfig, DEVICE
from report_utils import export_run, save_timeseries

class MiniIACL:
    def __init__(self, num_agents=3, N=64, K=8, steps=24, seed=2025,
                 gamma_group_start=0., gamma_group_end=.28):
        if num_agents < 2 or steps < 1:
            raise ValueError("Mini-IACL requires at least two agents and one step")
        if not 0 <= gamma_group_start <= .5 or not 0 <= gamma_group_end <= .5:
            raise ValueError("Group coupling must remain in [0,.5] to bound signed stimuli")
        self.num_agents, self.steps = num_agents, steps
        self.gamma_group_start, self.gamma_group_end = gamma_group_start, gamma_group_end
        self.agents = [TensionEngine(EngineConfig(N=N, K=K, steps=steps, seed=seed + i, learn_M=False))
                       for i in range(num_agents)]
        self.config = {"num_agents": num_agents, "N": N, "K": K, "steps": steps, "seed": seed,
                       "agent_seeds": [seed + i for i in range(num_agents)],
                       "gamma_group_start": gamma_group_start, "gamma_group_end": gamma_group_end,
                       "schedule": "synchronous_pre_step_snapshot_no_warmup"}
        fields = ("step", "HGI", "INC", "SAT", "alpha", "override", "continuous_modulation",
                  "hgi_guard", "inc_guard", "numeric_clip", "override_effect", "A_before", "A", "stimulus",
                  "group_mean_before", "gamma_group", "coupling_rms")
        self.ts = {field: [] for field in fields}

    def _current_A(self, eng):
        """Fail on missing, malformed, or nonfinite states; never invent a state."""
        if not hasattr(eng, "current_activation"):
            raise ValueError("Engine has no current_activation state API")
        if not hasattr(eng, "N") or not isinstance(eng.N, int) or eng.N < 1:
            raise ValueError("Engine has no valid state dimension N")
        state = eng.current_activation
        if callable(state):
            state = state()
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=float)
        if state.shape != (eng.N,):
            raise ValueError(f"Activation shape {state.shape} differs from required ({eng.N},)")
        if not np.isfinite(state).all():
            raise ValueError("Activation state contains nonfinite values")
        return state.copy()

    def run(self, out_dir="results_corrected/mini_iacl", print_console=True):
        if self.ts["step"]:
            raise RuntimeError("A MiniIACL instance can run only once; create a new seeded instance")
        os.makedirs(out_dir, exist_ok=True)
        for t in range(self.steps):
            # Every agent reads the same simulation instant, before any agent advances.
            snapshots = np.stack([self._current_A(eng) for eng in self.agents])
            mean = snapshots.mean(axis=0)
            fraction = t / max(1, self.steps - 1)
            gamma = self.gamma_group_start + fraction * (self.gamma_group_end - self.gamma_group_start)
            stimuli = gamma * (mean - snapshots)
            for eng, stimulus in zip(self.agents, stimuli):
                eng.step(stimulus=torch.as_tensor(stimulus, dtype=torch.float32, device=DEVICE))
            self.ts["step"].append(t)
            for key in ("HGI", "INC", "SAT", "alpha", "override", "continuous_modulation",
                        "hgi_guard", "inc_guard", "numeric_clip", "override_effect"):
                self.ts[key].append(float(np.mean([eng.ts[key][-1] for eng in self.agents])))
            self.ts["A_before"].append(snapshots.tolist())
            self.ts["A"].append([self._current_A(eng).tolist() for eng in self.agents])
            self.ts["stimulus"].append(stimuli.tolist())
            self.ts["group_mean_before"].append(mean.tolist())
            self.ts["gamma_group"].append(float(gamma))
            self.ts["coupling_rms"].append(float(np.sqrt(np.mean(stimuli ** 2))))
            if print_console:
                print(f"[IACL] step={t} coupling_rms={self.ts['coupling_rms'][-1]:.6f}")
        return self.ts

def run_and_export(out_dir="results_corrected/mini_iacl", agents=3, steps=24,
                   N=64, K=8, seed=2025, plots=True):
    env = MiniIACL(num_agents=agents, steps=steps, N=N, K=K, seed=seed)
    ts = env.run(out_dir=out_dir, print_console=False)
    # Continuous coupling does not provide a post-withdrawal recovery interval.
    control = MiniIACL(num_agents=agents, steps=steps, N=N, K=K, seed=seed,
                       gamma_group_start=0., gamma_group_end=0.)
    control.run(out_dir=out_dir, print_console=False)
    metadata = {"experiment": "Mini-IACL", "config": env.config,
                "interpretation": "Engine descriptors averaged over agents; flags are per-agent event fractions."}
    row = export_run(out_dir, "mini_iacl", ts, f"Mini-IACL {agents} agents",
                     metadata=metadata, control=control.ts, plots=plots)
    for i, (engine, control_engine) in enumerate(zip(env.agents, control.agents)):
        save_timeseries(out_dir, f"mini_iacl_agent_{i}", engine.ts,
                        {"agent_index": i, "config": asdict(engine.cfg), "seed": engine.cfg.seed})
        save_timeseries(out_dir, f"mini_iacl_agent_{i}_control", control_engine.ts,
                        {"agent_index": i, "config": asdict(control_engine.cfg),
                         "seed": control_engine.cfg.seed, "role": "uncoupled_control"})
    return row

if __name__ == "__main__":
    run_and_export()
