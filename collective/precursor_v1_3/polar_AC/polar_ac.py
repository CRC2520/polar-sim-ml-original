"""Estudios A (igualación del beneficio privado) y C (puente entre capas). Ver HIPOTESIS_AC.md"""
import copy, sys, json
import numpy as np
import polar_ac_core as P

G, NA_, GENS, WARM, STEPS, KREP, MU = 6, 10, 25, 5, 1000, 2, 0.01
P.N = NA_; P.KR = 20.0 * NA_
BASE_CFG = dict(rand_phase=True, death_pen=50.0, gamma_long=0.99, diff_c=0.0, R_term=0.0,
                lamG_scale=1.0, alpha=0.1, c_ca=3.0, delay_L=5, obs_sd=10.0, w_Qv=1.0)
ROWS = ("Qt", "Qv", "Qg", "Nsa")

def evolve(regime, k0, seed, equalize=False, arch="BASE"):
    P.CFG.update(BASE_CFG)
    cond = dict(meta=False, fixed=(0.05, 0.5, 1.0), w_fixed=1.0, gterm="trait")
    if arch == "SIN_ANCLA":       P.CFG.update(w_Qv=0.0)
    if arch == "HORIZONTE_CORTO": P.CFG.update(gamma_long=0.5)
    if arch == "SIN_DIRIGIDA":    cond["fixed"] = (0.05, 0.0, 1.0)
    rng = np.random.default_rng(seed); env = np.random.default_rng([seed, 7])
    groups = []
    for _ in range(G):
        ag = P.Agents(rng); ag.trait = np.zeros(NA_); ag.trait[:k0] = 1
        groups.append(ag)
    prop, alive = [], []
    for gen in range(GENS):
        fg, fi = [], []
        for ag in groups:
            o = P.episode(ag, cond, STEPS, "train", rng, None, env)
            fg.append(o["alive_mean"]); fi.append(np.asarray(o["alive_agent"]))
        fg = np.array(fg)
        prop.append(float(np.mean([ag.trait.mean() for ag in groups]))); alive.append(float(fg.mean()))
        if gen >= WARM:
            for ag, f in zip(groups, fi):
                if equalize:                                   # compensa el déficit de los contenidos
                    m = ag.trait > 0.5
                    if m.any() and (~m).any():
                        f = f + max(0.0, f[~m].mean() - f[m].mean()) * ag.trait
                if regime == "CONF":
                    mm = ag.trait.mean()
                    maj = 1.0 if mm > 0.5 else (0.0 if mm < 0.5 else float(rng.integers(2)))
                    same = np.where(ag.trait == maj)[0]; i_ = int(rng.integers(NA_))
                    if len(same) and ag.trait[i_] != maj:
                        src = int(rng.choice(same))
                        for nm in ROWS: getattr(ag, nm)[i_] = getattr(ag, nm)[src]
                        ag.trait[i_] = maj
                else:
                    lo, hi = int(np.argmin(f)), int(np.argmax(f))
                    if f[hi] > f[lo]:
                        for nm in ROWS: getattr(ag, nm)[lo] = getattr(ag, nm)[hi]
                        ag.trait[lo] = ag.trait[hi]
            order = np.argsort(fg + rng.normal(0, 1e-9, G))
            for l, w in zip(order[:KREP], order[-KREP:]):
                new = copy.deepcopy(groups[w]); new.rng = rng; groups[int(l)] = new
        for ag in groups:
            ag.trait = np.where(rng.random(NA_) < MU, 1 - ag.trait, ag.trait)
    return dict(prop=prop, alive=alive)

if __name__ == "__main__":
    study, arg, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    out = []
    for s in range(a, b):
        if study == "A":
            reg, eq = arg.split(":"); out.append(evolve(reg, 6, s, equalize=(eq == "eq")))
        else:
            out.append(evolve("CONF", 5, s, arch=arg))
    json.dump(out, open(f"AC_{study}_{arg.replace(':','-')}_{a}.json", "w"))
    print(study, arg, a, "hecho", flush=True)
