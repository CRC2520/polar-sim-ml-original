"""
POLAR v2 — simulación tabular con metacontrol aprendido.
Ver HIPOTESIS_v2.md (registro previo). Etiquetas: [M] [L] [S] [I] [GT] [E].
"""
import numpy as np

N, KR, METAB, XMAX = 10, 200.0, 0.3, 10.0
ACT = np.array([0, 1, 2, 3]); NA = 4
NE, NRB = 4, 5; NS = NE * NRB
SIG = [0.02, 0.2]; OMG = [0.0, 1.0]; LAMG = [0.0, 1.0]
ARMS = [(s, o, l) for s in SIG for o in OMG for l in LAMG]  # 8 brazos
WIN = 20
CFG = dict(c_ca=3.0, delay_L=0, obs_sd=0.0, k_win=10, long_test=4500, train_steps=15000, test_lo=0.07, test_hi=0.26, test_P=600, shock=0.4, lamG_scale=1.0, alpha=0.1, ep_len=3000, rand_phase=False, death_pen=5.0, R_term=0.0, gamma_long=0.97, w_Qv=1.0, eff=1.0, diff_c=0.0)

BASE = dict(meta=False, fixed=(0.05, 0.5, 1.0), w_fixed=1.0)
CONDS = {}
for _k in range(21):
    CONDS[f"A_k{_k}"] = dict(BASE, gterm="mix", k=_k)
    CONDS[f"B_k{_k}"] = dict(meta=False, fixed=(0.05, 0.5, 0.0), taxmix=True, k=_k)


def threat_bin(x):   # [S] amenaza propia
    return np.where(x < 3, 2, np.where(x < 6, 1, 0))


def res_bin(R):      # [S] estado del recurso, sin ponderar por w
    return 2 if R < 60 else (1 if R < 120 else 0)


class Agents:
    def __init__(self, rng):
        self.rng = rng
        self.Qt = np.zeros((N, NS, NA)); self.Qv = np.zeros((N, NS, NA)); self.Qg = np.zeros((N, NS, NA))
        self.Nsa = np.zeros((N, NS, NA))
        self.M = np.zeros((N, 9, len(ARMS))); self.Mn = np.zeros((N, 9, len(ARMS)))
        self.w = np.zeros((N, N))
        self.d = np.zeros((N, N)); self.b0 = np.zeros(N)
        self.c = np.zeros(N)
        self.trait = np.zeros(N)
        self.A = np.tile(np.eye(4) * 1e-2, (N, 1, 1)); self.bv = np.zeros((N, 4)); self.beta = np.zeros(N)

    def reset_agent(self, i):
        for T in (self.Qt, self.Qv, self.Qg, self.Nsa, self.M, self.Mn):
            T[i] = 0
        self.w[i] = 0; self.w[:, i] = 0; self.d[i] = 0; self.d[:, i] = 0; self.b0[i] = 0


def env_r(t, phase, off=0):
    if phase == "train":
        return 0.165 + 0.085 * np.sin(2 * np.pi * (t + off) / 1000)
    lo, hi = CFG['test_lo'], CFG['test_hi']
    return (lo + hi) / 2 + (hi - lo) / 2 * np.sin(2 * np.pi * t / CFG['test_P'])


def episode(ag, cfg, steps, phase, rng, log, env_rng=None):
    env_rng = rng if env_rng is None else env_rng
    R = KR * 0.8; x = np.full(N, 5.0); alive = np.ones(N, bool)
    idx = np.arange(N)
    arm = np.zeros(N, int); ctx = np.zeros(N, int)
    win_dx = np.zeros(N); win_death = np.zeros(N)
    harv_win = np.zeros(N); prev_harv = None; take_win = np.zeros(N)
    thr_hist = []; alive_sum = 0.0; first_death = None; alive_agent = np.zeros(N)
    lag = rng.integers(200, 400)
    off = env_rng.integers(0, 1000) if (CFG['rand_phase'] and phase == 'train') else 0
    R_sum = 0.0; harv_cum = np.zeros(N); collapse = None
    L, KW = CFG['delay_L'], CFG['k_win']
    R_start, T_hist, r_hist, Robs_hist, take_hist = [], [], [], [], []
    death_R = []; dstar = []; beta_log = []
    R_obs = R
    if cfg.get('w_fixed') is not None:
        ag.w[:] = cfg['w_fixed']; np.fill_diagonal(ag.w, 0)
    for t in range(steps):
        r = env_r(t, phase, off)
        if phase == "test" and t == 300:
            R *= CFG['shock']
        if phase == "test" and t == 900:
            for i in range(int(round(0.3 * N)) if N > 3 else 0):          # cambio de composición
                ag.reset_agent(i); x[i] = 5.0; alive[i] = True
        tb_now = threat_bin(x); thr_hist.append(tb_now.copy())

        # --- metacontrol cada WIN pasos [L] ---
        if t % WIN == 0:
            if t > 0 and cfg["meta"]:
                rew = win_dx - 10 * win_death           # señal: solo viabilidad individual
                upd = alive | (win_death > 0)
                ag.M[idx[upd], ctx[upd], arm[upd]] += 0.1 * (rew[upd] - ag.M[idx[upd], ctx[upd], arm[upd]])
                if phase == 'train':
                    ag.Mn[idx[upd], ctx[upd], arm[upd]] += 1
            # dependencia estimada D̂ [L]: cosecha futura propia ~ cosecha de otros
            if prev_harv is not None:
                h = prev_harv / (WIN * 3)
                pred = ag.b0 + ag.d @ h
                err = take_win / WIN - pred
                ag.d += 0.05 * np.outer(err, h); ag.b0 += 0.05 * err
                np.fill_diagonal(ag.d, 0)
                ag.w = np.clip(ag.w + 0.2 * np.clip(-ag.d, 0, 1) - 0.02 * ag.w, 0, 1)
                if cfg.get('w_fixed') is not None:
                    ag.w[:] = cfg['w_fixed']
                np.fill_diagonal(ag.w, 0)
            prev_harv = harv_win.copy(); harv_win[:] = 0; take_win[:] = 0
            win_dx[:] = 0; win_death[:] = 0
            if cfg["meta"]:
                if cfg.get("ctx_threat") == "none":
                    tb = np.ones(N, int)
                elif cfg.get("ctx_threat") == "sham":
                    tb = thr_hist[-lag] if len(thr_hist) > lag else rng.integers(0, 3, N)
                else:
                    tb = tb_now
                ctx = tb * 3 + res_bin(R)
                vals = ag.M[idx, ctx].copy()
                if cfg.get("no_directed"):
                    vals[:, [k for k, a in enumerate(ARMS) if a[1] > 0]] = -1e9
                greedy = vals.argmax(1)
                explore = rng.random(N) < 0.1
                allowed = [k for k, a in enumerate(ARMS) if not (cfg.get("no_directed") and a[1] > 0)]
                arm = np.where(explore, rng.choice(allowed, N), greedy)
                if log is not None and phase == "test":
                    for i in np.where(alive)[0]:
                        log.append((res_bin(R), ARMS[arm[i]][2]))

        if cfg["meta"]:
            sig = np.array([ARMS[k][0] for k in arm]); omg = np.array([ARMS[k][1] for k in arm])
            lg = np.array([ARMS[k][2] for k in arm])
        else:
            sig, omg, lg = (np.full(N, v) for v in cfg["fixed"])

        # --- política [M]: azar ciego separado de exploración dirigida ---
        eb = np.clip((x / XMAX * NE).astype(int), 0, NE - 1)
        R_obs = R + (env_rng.normal(0, CFG['obs_sd'] * KR / 200) if CFG['obs_sd'] > 0 else 0.0)
        Robs_hist.append(R_obs)
        s = eb * NRB + min(max(int(R_obs / KR * NRB), 0), NRB - 1)
        U = 1 / np.sqrt(ag.Nsa[idx, s] + 1)
        score = ag.Qt[idx, s] + CFG['w_Qv'] * ag.Qv[idx, s] + CFG['lamG_scale'] * lg[:, None] * ag.Qg[idx, s] + omg[:, None] * U
        g = rng.gumbel(size=(N, NA))
        a = (score / 0.05 + g).argmax(1)
        blind = rng.random(N) < sig
        a = np.where(blind, rng.integers(0, NA, N), a)
        a[~alive] = 0

        take = ACT[a] * R / KR * CFG['eff']
        if take.sum() > R:
            take *= R / take.sum()
        R_prev = R
        R_start.append(R); T_hist.append(take.sum()); r_hist.append(r); take_hist.append(take.copy())
        R_after = R - take.sum()
        Rl = R_start[-1 - L] if (L > 0 and len(R_start) > L) else R_after
        R = max(R_after + r * Rl * (1 - Rl / KR), 0.0)
        dx = np.where(alive, take - METAB, 0.0)
        x = np.clip(x + dx, -1, XMAX)
        died = alive & (x <= 0); alive = alive & ~died
        death_R += [R] * int(died.sum())
        alive_sum += alive.mean(); alive_agent += alive; R_sum += R; harv_cum += take
        if collapse is None and (R < 5 * KR / 200 or not alive.any()):
            collapse = t
        if died.any() and first_death is None:
            first_death = t
        win_dx += dx; win_death += died; harv_win += ACT[a] * alive; take_win += take

        vi = dx - CFG['death_pen'] * died
        c = CFG['c_ca']
        gterm = cfg.get('gterm')
        if gterm == 'marginal':
            pen = take
        elif gterm == 'global_sum':
            pen = np.full(N, take.sum())
        elif gterm == 'global_mean':
            pen = np.full(N, take.mean())
        elif gterm == 'sham':
            pen = take[(idx + rng.integers(1, N)) % N]
        elif gterm == 'trait':
            pen = take * ag.trait
        elif gterm == 'mix':
            pen = take * (idx < cfg['k'])
        elif gterm == 'est_use':
            pen = np.maximum(0.0, -ag.beta) * take
        else:
            pen = np.zeros(N)
        gr = (ag.w * vi[None, :]).sum(1) / max(N - 1, 1) - c * pen
        vi_own = vi - c * take if cfg.get('tax') else vi
        if cfg.get('taxmix'):
            vi_own = vi - c * take * (idx < cfg['k'])
        eb2 = np.clip((x / XMAX * NE).astype(int), 0, NE - 1)
        R_obs2 = R + (env_rng.normal(0, CFG['obs_sd'] * KR / 200) if CFG['obs_sd'] > 0 else 0.0)
        s2 = eb2 * NRB + min(max(int(R_obs2 / KR * NRB), 0), NRB - 1)
        # --- estimador interno [L][TR: R_obs] ---
        n_h = len(Robs_hist)
        if cfg.get('est') and n_h > KW + 1:
            lagsh = 150 if cfg.get('est') == 'sham' else 0
            if n_h > KW + lagsh + 1:
                own = np.sum(take_hist[-KW - lagsh: len(take_hist) - lagsh], axis=0)
                r0 = Robs_hist[-1 - KW] / KR
                y = R_obs2 - Robs_hist[-1 - KW]
                f = np.stack([own, np.ones(N), np.full(N, r0), np.full(N, r0 ** 2)], 1)
                ag.A = 0.9995 * ag.A + f[:, :, None] * f[:, None, :]
                ag.bv = 0.9995 * ag.bv + f * y
                if t % 20 == 0:
                    ag.beta = np.linalg.solve(ag.A + 1e-3 * np.eye(4), ag.bv[:, :, None])[:, 0, 0]
        # --- D* contrafactual [GT] (solo experimentador) ---
        nt = len(R_start)
        if cfg.get('est') and t % 20 == 0 and nt > KW + L + 1:
            def roll(delta):
                Rs = list(R_start[:nt - KW])
                Rc = R_start[nt - KW]
                for u in range(nt - KW, nt):
                    Rs.append(Rc)
                    Ra = Rc - (T_hist[u] + delta)
                    Rlu = Rs[u - L] if L > 0 else Ra
                    Rc = max(Ra + r_hist[u] * Rlu * (1 - Rlu / KR), 0.0)
                return Rc
            dd = 0.01
            dstar.append((roll(dd) - roll(0.0)) / (KW * dd))
            beta_log.append(ag.beta.copy())
        live = alive | died; cont = ~died
        for Q, rew, gm in ((ag.Qt, take, 0.5), (ag.Qv, vi_own, CFG['gamma_long']), (ag.Qg, gr, CFG['gamma_long'])):
            tgt = rew + gm * Q[idx, s2].max(1) * cont
            Q[idx, s, a] += np.where(live, CFG['alpha'] * (tgt - Q[idx, s, a]), 0)
        ag.Nsa[idx, s, a] += live

        if not alive.any():
            return dict(surv=0.0, steps=t + 1, alive_mean=alive_sum / steps, first_death=first_death,
                        alive_agent=alive_agent / steps, R_mean=R_sum / steps, collapse=collapse,
                        harv_var=float(np.var(harv_cum)), harv_mean=float(np.mean(harv_cum)),
                        death_R=death_R, dstar=dstar, beta_log=beta_log)
    return dict(surv=alive.mean(), steps=steps, alive_mean=alive_sum / steps,
                first_death=first_death if first_death is not None else steps, alive_agent=alive_agent / steps,
                R_mean=R_sum / steps, collapse=collapse if collapse is not None else steps,
                harv_var=float(np.var(harv_cum)), harv_mean=float(np.mean(harv_cum)),
                        death_R=death_R, dstar=dstar, beta_log=beta_log)


def probe_H3(ag):
    """[I] Intervención do(amenaza propia) con estado del recurso fijo."""
    out = {}
    for lvl, name in ((0, "baja"), (2, "alta")):
        D, W = [], []
        for rb in range(3):
            c = lvl * 3 + rb
            k = ag.M[:, c].argmax(1)
            for kk in k:
                s_, o_, _ = ARMS[kk]
                D.append(o_ / (o_ + s_ / 0.2 + 0.01)); W.append(o_)
        out[name] = (np.mean(D), np.mean(W))
    return out


def value_gap(ag, rb):
    gaps = []
    for tb in range(3):
        c = tb * 3 + rb
        for i in range(N):
            m, n = ag.M[i, c], ag.Mn[i, c]
            l1 = [m[j] for j, a in enumerate(ARMS) if a[2] == 1 and n[j] > 0]
            l0 = [m[j] for j, a in enumerate(ARMS) if a[2] == 0 and n[j] > 0]
            if l1 and l0:
                gaps.append(max(l1) - max(l0))
    return float(np.mean(gaps)) if gaps else np.nan


def probe_H3v(ag):
    out = {}
    for lvl, name in ((0, "baja"), (2, "alta")):
        om, sg = [], []
        for rb in range(3):
            c = lvl * 3 + rb
            for i in range(N):
                vis = ag.Mn[i, c] > 0
                if not vis.any():
                    continue
                vals = np.where(vis, ag.M[i, c], -1e9)
                s_, o_, _ = ARMS[vals.argmax()]
                om.append(o_); sg.append(s_)
        out[name] = (float(np.mean(om)) if om else np.nan, float(np.mean(sg)) if sg else np.nan)
    return out


def run(cond, seed):
    import copy
    rng = np.random.default_rng(seed)
    cfg = CONDS[cond]; ag = Agents(rng)
    done, n_ep, tr_surv = 0, 0, []
    env_rng = np.random.default_rng([seed, 7])
    dst, bl = [], []
    while done < CFG['train_steps']:
        o = episode(ag, cfg, CFG['ep_len'], "train", rng, None, env_rng)
        done += o["steps"]; n_ep += 1; tr_surv.append(o["surv"])
        dst += o["dstar"]; bl += o["beta_log"]
    out = dict(train_surv=float(np.mean(tr_surv)), train_eps=n_ep)
    for name, steps in (("short", 1500), ("long", CFG['long_test'])):
        te = episode(copy.deepcopy(ag), cfg, steps, "test", rng, None, np.random.default_rng([seed, 8]))
        dR = np.array(te["death_R"])
        out.update({f"{name}_alive": te["alive_mean"], f"{name}_R": te["R_mean"],
                    f"{name}_first_death": te["first_death"], f"{name}_collapse": te["collapse"],
                    f"{name}_deaths": int(len(dR)), f"{name}_starv_frac": float((dR >= 0.3 * KR).mean()) if len(dR) else np.nan,
                    f"{name}_harv_sd": float(np.sqrt(te["harv_var"])),
                    f"{name}_alive_agent": np.asarray(te["alive_agent"]).tolist()})
    if cfg.get("est"):
        late_d = np.array(dst[-150:]); dbar = float(late_d.mean())
        out.update(dstar_mean=dbar, dstar_sd=float(late_d.std()), beta_final=ag.beta.tolist(),
                   rel_err=float(np.mean(np.abs(ag.beta - dbar)) / abs(dbar)),
                   beta_mean=float(ag.beta.mean()))
    return out


if __name__ == "__main__":
    import json, sys
    from multiprocessing import Pool
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    jobs = [(c, s) for c in CONDS for s in range(seeds)]
    with Pool() as p:
        outs = p.starmap(run, jobs)
    res = {}
    for (c, s), o in zip(jobs, outs):
        res.setdefault(c, []).append(o)
    json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "results_v21.json", "w"), default=lambda v: float(v) if np.isscalar(v) else str(v))
    for c, rs in res.items():
        f = lambda k: np.nanmean([r[k] for r in rs])
        print(f"{c:15s} prueba={f('test_surv'):.2f} pasos={f('test_steps'):5.0f} "
              f"entren={f('train_surv'):.2f} eps={f('train_eps'):.1f} w={f('w_mean'):.2f} "
              f"λG(Rbajo)={f('lamG_lowR'):.2f} λG(Ralto)={f('lamG_highR'):.2f}")
