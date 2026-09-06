
# engine_v1_locked.py — deterministic, versioned core (v1.0)
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, os, math, hashlib, random
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mutual_topk_mask(W, k):
    topk = torch.topk(W, k=k, dim=1).indices
    mask = torch.zeros_like(W)
    mask.scatter_(1, topk, 1.0)
    mask = ((mask + mask.T) > 0).float()
    mask.fill_diagonal_(0.0)
    return mask

def neighbor_avg(a, mask):
    deg = mask.sum(dim=1).clamp_min(1.0)
    return (mask @ a) / deg

def HGI_graph(a, mask):
    # 1 - normalized neighbor squared diffs
    deg = mask.sum(dim=1).clamp_min(1.0)
    diff = ((a[:, None] - a[None, :]) ** 2) * mask
    c_i = diff.sum(dim=1) / deg
    return 1.0 - c_i.mean() / 4.0

def cos_t(u, v):
    nu=u.norm()+1e-8; nv=v.norm()+1e-8
    return torch.dot(u, v)/(nu*nv)

def INC_weighted(a_t, a_tm1, mu, lam_T=0.30):
    lam_G = 1.0 - lam_T
    raw = lam_T*cos_t(a_t, a_tm1) + lam_G*cos_t(a_t, mu)
    return 0.5*(1.0 + raw)

def SAT(a, thr=0.85):
    return torch.mean((torch.abs(a)>thr).float())

def Laplacian_energy(a, mask):
    A_nb = neighbor_avg(a, mask)
    return ((a - A_nb)**2).mean()

@dataclass
class EngineConfig:
    N:int=120
    K:int=10
    steps:int=12
    seed:int=2025
    lam_T:float=0.30
    ethics_kappa:float=1.08
    ethics_tau_ovr:float=0.68
    ethics_wH:float=0.45
    ethics_wC:float=0.45
    ethics_wS:float=0.10
    ethics_alpha_min:float=0.60
    inc_floor:float=0.96
    inc_gain:float=0.18
    hgi_floor:float=0.95
    hgi_gain:float=0.10
    beta_mu:float=0.06
    gamma_cons_start:float=0.18
    gamma_cons_end:float=0.34
    stim_step:int=6
    stim_scale:float=0.5
    learn_M:bool=False
    lr_M:float=0.0035
    reg_M:float=5e-4
    no_ethics:bool=False
    no_homeostasis:bool=False

class TensionEngine:
    def __init__(self, cfg:EngineConfig, nodes=None):
        self.cfg = cfg
        seed_all(cfg.seed)
        self.N = cfg.N
        self.t = 0

        # node names (Spanish polarity set if provided)
        self.nodes = nodes or [
            'Poder vs Vulnerabilidad', 'Placer vs Dolor', 'Integración vs Fragmentación',
            'Control vs Rendición', 'Deseo vs Límite', 'Libertad vs Orden',
            'Preservación vs Transformación', 'Reconocimiento vs Autenticidad'
        ]
        # repeat labels if needed
        if len(self.nodes) < self.N:
            self.nodes = (self.nodes * (self.N // len(self.nodes) + 1))[:self.N]

        # Baselines B_i, tensions, homeostasis U_i
        self.B = (torch.randn(self.N)*0.05).to(DEVICE)
        self.tensions = (torch.rand(self.N)*2-1).to(DEVICE) * 0.2 + self.B
        self.U = torch.zeros_like(self.tensions)

        # Influence W, symmetric mask
        W0 = torch.rand(self.N, self.N).to(DEVICE)
        W0 = torch.softmax(W0, dim=1)
        self.mask = mutual_topk_mask(W0, cfg.K)

        # Modulation matrix M (learnable optional)
        self.M = torch.zeros(self.N, self.N, device=DEVICE)
        if cfg.learn_M:
            self.M = nn.Parameter(self.M)
            self.opt = optim.Adam([self.M], lr=cfg.lr_M)

        # running goal mu_t (EMA of activations)
        A0 = torch.tanh(self.tensions).detach()
        self.mu_t = A0.clone()
        self.A_prev = A0.clone()

        # logging
        self.ts = {"step":[], "HGI":[], "INC":[], "SAT":[], "alpha":[], "override":[],
                   "varA":[], "lapE":[], "seed":cfg.seed}

    # ethics alpha
    def ethics_alpha(self, HGI, INC, SATv):
        if self.cfg.no_ethics:
            return torch.tensor(1.0, device=DEVICE), torch.tensor(False, device=DEVICE), torch.tensor(0.0, device=DEVICE)
        R = self.cfg.ethics_wH*(1.0-HGI) + self.cfg.ethics_wC*(1.0-INC) + self.cfg.ethics_wS*SATv
        alpha = 1.0/(1.0 + self.cfg.ethics_kappa*R)
        alpha = torch.clamp(alpha, min=self.cfg.ethics_alpha_min, max=1.0)
        return alpha, (R>self.cfg.ethics_tau_ovr), R

    def homeostasis(self, A):
        if self.cfg.no_homeostasis:
            return torch.zeros_like(A)
        # simple homeostatic drive toward baseline B (utility gradient)
        return 0.05*(self.B - A)

    def step(self, stimulus=None):
        cfg = self.cfg
        A = torch.tanh(self.tensions)

        # metrics
        H = HGI_graph(A, self.mask)
        I = INC_weighted(A, self.A_prev, self.mu_t, cfg.lam_T)
        S_v = SAT(A, 0.85)
        alpha, ovr, _ = self.ethics_alpha(H, I, S_v)

        # neighbor influence + homeostasis + optional M
        A_nb = neighbor_avg(A, self.mask)
        gc = cfg.gamma_cons_start + (cfg.gamma_cons_end - cfg.gamma_cons_start) * (self.t / max(1, cfg.steps-1))
        base_delta = gc*(A_nb - A) + cfg.beta_mu*(self.mu_t - A) + self.homeostasis(A)

        if cfg.learn_M:
            base_delta = base_delta + (self.M @ A)

        # external stimulus (vector in [-1,1])
        stim_vec = torch.zeros_like(A)
        if stimulus is not None:
            stim_vec = stimulus.to(DEVICE)
        # total change with ethics scaling
        delta = alpha * (base_delta + stim_vec)

        tensions_next = torch.clamp(self.tensions + delta, -1.2, 1.2)
        A_next = torch.tanh(tensions_next)

        # Guardrails (error-scaled)
        H2_chk = HGI_graph(A_next, self.mask)
        I2_chk = INC_weighted(A_next, A, self.mu_t, cfg.lam_T)
        miss_H = max(0.0, cfg.hgi_floor - H2_chk.item())
        miss_I = max(0.0, cfg.inc_floor - I2_chk.item())
        hgi_gain_eff = cfg.hgi_gain * (1.0 + 1.0*miss_H)
        inc_gain_eff = cfg.inc_gain * (1.0 + 2.0*miss_I)
        hgi_flag = False; inc_flag = False
        if H2_chk.item() < cfg.hgi_floor:
            tensions_next = torch.clamp(tensions_next + hgi_gain_eff*(A_nb - A), -1.2, 1.2)
            A_next = torch.tanh(tensions_next); hgi_flag=True
        if INC_weighted(A_next, A, self.mu_t, cfg.lam_T).item() < cfg.inc_floor:
            tensions_next = torch.clamp(tensions_next + inc_gain_eff*(self.mu_t - A_next), -1.2, 1.2)
            A_next = torch.tanh(tensions_next); inc_flag=True

        if ovr:
            tensions_next = torch.clamp(tensions_next, -1.0, 1.0)
            A_next = torch.tanh(tensions_next)

        # learn M if enabled
        if cfg.learn_M:
            H2 = HGI_graph(A_next, self.mask); I2 = INC_weighted(A_next, A, self.mu_t, cfg.lam_T)
            smooth = Laplacian_energy(A_next, self.mask)
            loss = 1.2*(1.0 - H2) + 0.6*(1.0 - I2) + 0.02*smooth + self.cfg.reg_M*(self.M**2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()

        # update state + logs
        self.A_prev = A.detach().clone()
        self.mu_t = 0.90*self.mu_t + 0.10*A_next.detach()
        self.tensions = tensions_next.detach()

        H2 = HGI_graph(A_next, self.mask); I2 = INC_weighted(A_next, self.A_prev, self.mu_t, cfg.lam_T)
        self.ts["step"].append(self.t)
        self.ts["HGI"].append(float(H2.item()))
        self.ts["INC"].append(float(I2.item()))
        self.ts["SAT"].append(float(SAT(A_next).item()))
        self.ts["alpha"].append(float(alpha.item()))
        self.ts["override"].append(bool(ovr.item() if hasattr(ovr,'item') else bool(ovr)))
        self.ts["varA"].append(float(A_next.var().item()))
        self.ts["lapE"].append(float(Laplacian_energy(A_next, self.mask).item()))
        self.t += 1

        return A_next, dict(hgi_guard=hgi_flag, inc_guard=inc_flag)

    # run loop
    def run(self, out_dir=None, print_console=True):
        os.makedirs(out_dir or ".", exist_ok=True)
        for s in range(self.cfg.steps):
            stim_vec = None
            # default: no stimulus; external scripts can inject
            A_next, flags = self.step(stimulus=stim_vec)
            if print_console:
                print(f"Step {s+1}: HGI={self.ts['HGI'][-1]:.4f}, INC={self.ts['INC'][-1]:.4f}, SAT={self.ts['SAT'][-1]:.2f}, α={self.ts['alpha'][-1]:.3f}, guards(H/I)={flags['hgi_guard']}/{flags['inc_guard']}")
        if out_dir:
            with open(os.path.join(out_dir, "timeseries.json"), "w") as f:
                json.dump({"timeseries": self.ts}, f, indent=2)
        return self.ts
