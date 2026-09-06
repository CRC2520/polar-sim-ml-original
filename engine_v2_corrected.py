"""Corrected, auditable legacy dynamics (v2.0.1); not a consciousness measure.

Canonical mathematical specification: docs/SPEC_LEGACY_CORRECTED.md.
The original engine_v1_locked.py remains an immutable historical reference.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
import platform
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

VERSION = "2.0.1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLARITY_TYPES = (
    "Poder vs Vulnerabilidad", "Placer vs Dolor", "Integración vs Fragmentación",
    "Control vs Rendición", "Deseo vs Límite", "Libertad vs Orden",
    "Preservación vs Transformación", "Reconocimiento vs Autenticidad",
)


def seed_all(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mutual_topk_mask(W, k):
    """Legacy name: UNDIRECTED UNION of row top-k proposals, not intersection.

    Self proposals may be selected and are then removed, as in v1.0.
    Consequently K is not the final undirected degree.
    """
    indices = torch.topk(W, k=k, dim=1).indices
    mask = torch.zeros_like(W)
    mask.scatter_(1, indices, 1.0)
    mask = ((mask + mask.T) > 0).to(W.dtype)
    mask.fill_diagonal_(0.0)
    return mask


def neighbor_avg(a, mask):
    return (mask @ a) / mask.sum(dim=1).clamp_min(1.0)


def HGI_graph(a, mask):
    """Graph activation homogeneity; 1 for uniform or edgeless states."""
    differences = ((a[:, None] - a[None, :]) ** 2) * mask
    return 1.0 - (differences.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean() / 4.0


def cos_t(u, v):
    """Exact cosine with zero-vector convention 0 (INC then has neutral 0.5)."""
    denom = u.norm() * v.norm()
    if denom.detach().item() == 0.0:
        # Keep a valid zero gradient when a learned trajectory crosses zero.
        return (u.sum() + v.sum()) * 0.0
    return (torch.dot(u, v) / denom).clamp(-1.0, 1.0)


def INC_weighted(a_t, a_tm1, mu, lam_T=0.30):
    """Temporal/self-EMA alignment, not narrative, purpose, or consciousness."""
    return 0.5 * (1.0 + lam_T * cos_t(a_t, a_tm1) + (1.0 - lam_T) * cos_t(a_t, mu))


def SAT(a, thr=0.80):
    return (a.abs() > thr).to(a.dtype).mean()


def Laplacian_energy(a, mask):
    return ((a - neighbor_avg(a, mask)) ** 2).mean()


@dataclass
class EngineConfig:
    N: int = 120
    K: int = 10
    steps: int = 12
    seed: int = 2025
    lam_T: float = 0.30
    # Compatibility names only: this controller has no ethical semantics.
    ethics_kappa: float = 1.08
    ethics_tau_ovr: float = 0.68
    ethics_wH: float = 0.45
    ethics_wC: float = 0.45
    ethics_wS: float = 0.10
    ethics_alpha_min: float = 0.60
    inc_floor: float = 0.96
    inc_gain: float = 0.18
    hgi_floor: float = 0.95
    hgi_gain: float = 0.10
    beta_mu: float = 0.06
    gamma_cons_start: float = 0.18
    gamma_cons_end: float = 0.34
    stim_step: int = 6
    stim_scale: float = 0.5
    learn_M: bool = False
    lr_M: float = 0.0035
    reg_M: float = 5e-4
    no_ethics: bool = False
    no_homeostasis: bool = False
    # Unambiguous v2 controls.
    disable_modulation: bool = False
    memory_retention: float = 0.90
    dt: float = 1.0
    tension_clip: float = 1.2
    override_clip: float = 1.0
    saturation_threshold: float = 0.80
    homeostasis_gain: float = 0.05
    max_grad_norm: float = 1.0
    max_abs_M: float = 1.0

    def validate(self):
        for key in ("N", "K", "steps", "seed", "stim_step"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if self.N < 2 or not 1 <= self.K < self.N:
            raise ValueError("Require N >= 2 and 1 <= K < N")
        if self.steps < 1 or self.seed < 0 or self.seed >= 2**32 or self.stim_step < 0:
            raise ValueError("Require steps >= 1, 0 <= seed < 2**32 and stim_step >= 0")
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("lam_T", "memory_retention", "inc_floor", "hgi_floor", "ethics_alpha_min"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        for key in ("ethics_kappa", "ethics_tau_ovr", "ethics_wH", "ethics_wC", "ethics_wS",
                    "inc_gain", "hgi_gain", "beta_mu", "gamma_cons_start", "gamma_cons_end",
                    "reg_M", "homeostasis_gain"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("dt", "tension_clip", "override_clip", "lr_M", "max_grad_norm", "max_abs_M"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.override_clip > self.tension_clip:
            raise ValueError("override_clip must not exceed tension_clip")
        if not 0 < self.saturation_threshold < math.tanh(self.tension_clip):
            raise ValueError("saturation_threshold must be reachable: 0 < threshold < tanh(tension_clip)")
        if abs(self.stim_scale) > 1:
            raise ValueError("stim_scale must be in [-1, 1]")


class TensionEngine:
    def __init__(self, cfg: EngineConfig, nodes=None):
        cfg.validate()
        self.cfg = copy.deepcopy(cfg)
        seed_all(cfg.seed)
        self.N, self.t = cfg.N, 0
        if nodes is None:
            self.nodes = [POLARITY_TYPES[i % len(POLARITY_TYPES)] for i in range(self.N)]
        else:
            if len(nodes) != self.N or any(not isinstance(n, str) or not n.strip() for n in nodes):
                raise ValueError("nodes must contain exactly N nonempty unit labels; eight polarity types are not N units")
            self.nodes = list(nodes)
        self.B = (torch.randn(self.N) * 0.05).to(DEVICE)
        self.tensions = ((torch.rand(self.N) * 2 - 1).to(DEVICE) * 0.2 + self.B).clamp(-cfg.tension_clip, cfg.tension_clip)
        self.U = torch.zeros_like(self.tensions)
        W0 = torch.softmax(torch.rand(self.N, self.N).to(DEVICE), dim=1)
        self.mask = mutual_topk_mask(W0, cfg.K)
        self.M = torch.zeros(self.N, self.N, device=DEVICE)
        if cfg.learn_M:
            self.M = nn.Parameter(self.M)
            self.opt = optim.Adam([self.M], lr=cfg.lr_M)
        A0 = torch.tanh(self.tensions).detach()
        self.mu_t = A0.clone()
        self.A_prev = A0.clone()
        scalar_keys = ("step", "time", "HGI", "INC", "SAT", "alpha", "override", "override_effect",
                       "risk", "varA", "lapE", "continuous_modulation", "hgi_guard", "inc_guard",
                       "numeric_clip", "modulation_delta_norm", "learning_loss", "gradient_norm",
                       "M_clipped", "gamma_cons", "HGI_before", "INC_before", "SAT_before")
        vector_keys = ("A_before", "A", "tension_before", "tension", "mu_before", "mu_after", "stimulus",
                       "delta", "proposed_delta", "homeostasis")
        self.ts = {key: [] for key in scalar_keys + vector_keys}
        self.ts.update(seed=cfg.seed, version=VERSION, config=asdict(self.cfg))
        with open(__file__, "rb") as source:
            source_sha = hashlib.sha256(source.read()).hexdigest()
        self.metadata = {
            "engine_version": VERSION, "engine_sha256": source_sha,
            "config": asdict(self.cfg), "seed": cfg.seed,
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "device": str(DEVICE), "dtype": str(self.tensions.dtype),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(), "torch_build_config": torch.__config__.show(),
            "unit_labels": self.nodes, "polarity_types": list(POLARITY_TYPES),
            "initial_state": {"B": self._vector(self.B), "tension": self._vector(self.tensions),
                              "A": self._vector(A0), "mu": self._vector(self.mu_t),
                              "mask": self._vector(self.mask), "M": self._vector(self.M)},
            "metric_semantics": {"HGI": "graph_activation_homogeneity",
                                 "INC": "temporal_self_ema_alignment",
                                 "SAT": "fraction_above_activation_threshold"},
            "legacy_alias": "no_ethics disables dynamical modulation and risk override; no ethical evaluation is implemented",
        }
        self.ts["metadata"] = self.metadata

    @staticmethod
    def _vector(value):
        return value.detach().cpu().tolist()

    @property
    def current_activation(self):
        self._validate_vector("tensions", self.tensions)
        return torch.tanh(self.tensions).detach().clone()

    @property
    def A(self):
        return self.current_activation

    def _validate_vector(self, name, value):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (self.N,):
            raise ValueError(f"{name} must be a tensor of shape ({self.N},)")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} contains nonfinite values")

    def modulation_alpha(self, HGI, INC, SATv):
        cfg = self.cfg
        risk = cfg.ethics_wH * (1.0 - HGI) + cfg.ethics_wC * (1.0 - INC) + cfg.ethics_wS * SATv
        if cfg.disable_modulation or cfg.no_ethics:
            return torch.ones_like(risk), torch.zeros_like(risk, dtype=torch.bool), risk
        alpha = (1.0 / (1.0 + cfg.ethics_kappa * risk)).clamp(cfg.ethics_alpha_min, 1.0)
        return alpha, risk > cfg.ethics_tau_ovr, risk

    def ethics_alpha(self, HGI, INC, SATv):
        """Deprecated compatibility spelling for modulation_alpha; not ethics."""
        return self.modulation_alpha(HGI, INC, SATv)

    def homeostasis(self, A):
        return torch.zeros_like(A) if self.cfg.no_homeostasis else self.cfg.homeostasis_gain * (self.B - A)

    def step(self, stimulus=None):
        cfg = self.cfg
        for name, value in (("tensions", self.tensions), ("mu_t", self.mu_t), ("A_prev", self.A_prev), ("B", self.B)):
            self._validate_vector(name, value)
        for name, value in (("mask", self.mask), ("M", self.M)):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (self.N, self.N) or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be a finite ({self.N}, {self.N}) tensor")
        if self.tensions.abs().max().item() > cfg.tension_clip + 1e-6:
            raise ValueError("tensions lie outside configured bounds")
        stim = torch.zeros_like(self.tensions)
        if stimulus is not None:
            self._validate_vector("stimulus", stimulus)
            if stimulus.abs().max().item() > 1.0:
                raise ValueError("stimulus must be in [-1, 1]")
            stim = stimulus.detach().to(device=DEVICE, dtype=self.tensions.dtype).clone()
        A = torch.tanh(self.tensions)
        mu_before = self.mu_t.detach().clone()
        tension_before = self.tensions.detach().clone()
        H, I, Sv = HGI_graph(A, self.mask), INC_weighted(A, self.A_prev, mu_before, cfg.lam_T), SAT(A, cfg.saturation_threshold)
        alpha, override, risk = self.modulation_alpha(H, I, Sv)
        A_nb = neighbor_avg(A, self.mask)
        fraction = min(self.t / max(1, cfg.steps - 1), 1.0)
        gamma = cfg.gamma_cons_start + (cfg.gamma_cons_end - cfg.gamma_cons_start) * fraction
        homeo = self.homeostasis(A)
        base_delta = gamma * (A_nb - A) + cfg.beta_mu * (mu_before - A) + homeo
        if cfg.learn_M:
            base_delta = base_delta + self.M @ A
        proposed_delta = cfg.dt * alpha * (base_delta + stim)
        numeric_clip = False

        def clip_numeric(value):
            nonlocal numeric_clip
            if not torch.isfinite(value).all().item():
                raise ValueError("Nonfinite proposed state; integration aborted")
            numeric_clip |= bool((value.abs() > cfg.tension_clip).any().detach().item())
            return value.clamp(-cfg.tension_clip, cfg.tension_clip)

        next_tension = clip_numeric(self.tensions + proposed_delta)
        A_next = torch.tanh(next_tension)
        H_chk = HGI_graph(A_next, self.mask)
        I_chk = INC_weighted(A_next, A, mu_before, cfg.lam_T)
        miss_H = max(0.0, cfg.hgi_floor - H_chk.detach().item())
        miss_I = max(0.0, cfg.inc_floor - I_chk.detach().item())
        hgi_guard = cfg.hgi_gain > 0 and H_chk.detach().item() < cfg.hgi_floor
        if hgi_guard:
            next_tension = clip_numeric(next_tension + cfg.dt * cfg.hgi_gain * (1 + miss_H) * (A_nb - A))
            A_next = torch.tanh(next_tension)
        inc_guard = cfg.inc_gain > 0 and INC_weighted(A_next, A, mu_before, cfg.lam_T).detach().item() < cfg.inc_floor
        if inc_guard:
            next_tension = clip_numeric(next_tension + cfg.dt * cfg.inc_gain * (1 + 2 * miss_I) * (mu_before - A_next))
            A_next = torch.tanh(next_tension)
        override_flag = bool(override.detach().item())
        override_effect = override_flag and bool((next_tension.abs() > cfg.override_clip).any().detach().item())
        if override_flag:
            next_tension = next_tension.clamp(-cfg.override_clip, cfg.override_clip)
            A_next = torch.tanh(next_tension)
        H_post = HGI_graph(A_next, self.mask)
        I_post = INC_weighted(A_next, A, mu_before, cfg.lam_T)
        smooth = Laplacian_energy(A_next, self.mask)
        learning_loss, gradient_norm, M_clipped = None, None, False
        if cfg.learn_M:
            loss = 1.2 * (1.0 - H_post) + 0.6 * (1.0 - I_post) + 0.02 * smooth + cfg.reg_M * (self.M ** 2).mean()
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite learning loss; step aborted")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.M.grad is None or not torch.isfinite(self.M.grad).all().item():
                raise ValueError("Missing or nonfinite M gradient; step aborted")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_([self.M], cfg.max_grad_norm, error_if_nonfinite=True).item())
            self.opt.step()
            with torch.no_grad():
                if not torch.isfinite(self.M).all().item():
                    raise ValueError("Nonfinite learned M; step aborted")
                M_clipped = bool((self.M.abs() > cfg.max_abs_M).any().item())
                self.M.clamp_(-cfg.max_abs_M, cfg.max_abs_M)
            learning_loss = float(loss.detach().item())
        self.A_prev = A.detach().clone()
        self.mu_t = cfg.memory_retention * mu_before + (1.0 - cfg.memory_retention) * A_next.detach()
        self.tensions = next_tension.detach().clone()
        self.U = homeo.detach().clone()
        flags = dict(continuous_modulation=bool(alpha.detach().item() < 1.0), override=override_flag,
                     override_effect=override_effect, hgi_guard=bool(hgi_guard), inc_guard=bool(inc_guard),
                     numeric_clip=numeric_clip)
        values = dict(step=self.t, time=(self.t + 1) * cfg.dt,
                      HGI=float(H_post.detach().item()), INC=float(I_post.detach().item()),
                      SAT=float(SAT(A_next, cfg.saturation_threshold).detach().item()),
                      alpha=float(alpha.detach().item()), risk=float(risk.detach().item()),
                      varA=float(A_next.detach().var(unbiased=False).item()), lapE=float(smooth.detach().item()),
                      modulation_delta_norm=float((cfg.dt * (alpha - 1) * (base_delta + stim)).detach().norm().item()),
                      learning_loss=learning_loss, gradient_norm=gradient_norm, M_clipped=M_clipped,
                      gamma_cons=gamma, HGI_before=float(H.detach().item()), INC_before=float(I.detach().item()),
                      SAT_before=float(Sv.detach().item()), **flags)
        vectors = dict(A_before=A, A=A_next, tension_before=tension_before, tension=self.tensions,
                       mu_before=mu_before, mu_after=self.mu_t, stimulus=stim, delta=self.tensions - tension_before,
                       proposed_delta=proposed_delta, homeostasis=homeo)
        values.update({key: self._vector(value) for key, value in vectors.items()})
        for key, value in values.items():
            self.ts[key].append(value)
        self.t += 1
        return A_next.detach().clone(), flags

    def export_payload(self):
        return {"metadata": copy.deepcopy(self.metadata), "timeseries": copy.deepcopy(self.ts)}

    def run(self, out_dir=None, print_console=True):
        """Run cfg.steps NO-STIMULUS steps. Experiments own their explicit schedule."""
        for _ in range(self.cfg.steps):
            _, flags = self.step()
            if print_console:
                print(f"Step {self.t}: HGI={self.ts['HGI'][-1]:.4f}, INC={self.ts['INC'][-1]:.4f}, "
                      f"SAT={self.ts['SAT'][-1]:.2f}, alpha={self.ts['alpha'][-1]:.3f}, flags={flags}")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "timeseries.json"), "w", encoding="utf-8") as target:
                json.dump(self.export_payload(), target, indent=2, ensure_ascii=False, allow_nan=False)
        return self.ts
