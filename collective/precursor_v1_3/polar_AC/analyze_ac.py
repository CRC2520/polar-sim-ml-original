import json, glob, numpy as np
from scipy import stats
S=0.10; B=5000; rng=np.random.default_rng(3)
def load(pat):
    out=[]
    for f in sorted(glob.glob(pat)): out+=json.load(open(f))
    return np.array([np.mean(x['prop'][20:25]) for x in out]), np.array([np.mean(x['alive'][20:25]) for x in out])
print("=== ESTUDIO A: igualación del beneficio privado (inicio 0,60) ===")
print("régimen              sin igualar    con igualación   efecto   p (Mann-Whitney)")
eff={}
for reg,nm in (("AMBOS","imitación del éxito"),("CONF","conformidad")):
    b,ab = load(f"AC_A_{reg}-base_*.json"); e,ae = load(f"AC_A_{reg}-eq_*.json")
    p = stats.mannwhitneyu(e,b,alternative="greater").pvalue
    eff[reg]=(b,e)
    print(f"{nm:20s} {b.mean():.3f} (n={len(b)})   {e.mean():.3f} (n={len(e)})    {e.mean()-b.mean():+.3f}   p={p:.4f}")
    print(f"                     vivos {ab.mean():.2f}          vivos {ae.mean():.2f}")
    if reg=="AMBOS":
        ok = (e.mean()-b.mean())>=S and p<0.05
        print(f"  H-A1: {'SE CUMPLE' if ok else 'NO SE CUMPLE'}")
dA=eff['AMBOS'][1].mean()-eff['AMBOS'][0].mean(); dC=eff['CONF'][1].mean()-eff['CONF'][0].mean()
bs=[]
for _ in range(B):
    r=lambda v: rng.choice(v,len(v)).mean()
    bs.append((r(eff['AMBOS'][1])-r(eff['AMBOS'][0])) - (r(eff['CONF'][1])-r(eff['CONF'][0])))
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"  H-A2 (interacción): efecto {dA:+.3f} (éxito) vs {dC:+.3f} (conformidad); diferencia {dA-dC:+.3f} IC95 [{lo:+.3f}, {hi:+.3f}]")
print(f"        → {'SE CUMPLE' if lo>0 else 'NO SE CUMPLE'}")
print("\n=== ESTUDIO C: puente entre capas (conformidad, inicio 0,50) ===")
base,abase = load("AC_C_BASE_*.json")
print(f"BASE                 contención {base.mean():.3f} (n={len(base)}), vivos {abase.mean():.2f}")
ps=[]; names=[]
for v,nm in (("SIN_ANCLA","sin crítico de viabilidad"),("HORIZONTE_CORTO","horizonte corto"),("SIN_DIRIGIDA","sin exploración dirigida")):
    x,ax = load(f"AC_C_{v}_*.json")
    p = stats.mannwhitneyu(x,base).pvalue
    ps.append(p); names.append((nm,x,ax,p))
order=sorted(range(3), key=lambda i: ps[i]); adj={}; prev=0
for rank,i in enumerate(order): prev=adj[i]=min(1.0,max(prev,(3-rank)*ps[i]))
for i,(nm,x,ax,p) in enumerate(names):
    d=x.mean()-base.mean()
    print(f"{nm:28s} {x.mean():.3f}  efecto {d:+.3f}  vivos {ax.mean():.2f}  p={p:.4f}  p(Holm)={adj[i]:.4f}  {'difiere' if (abs(d)>=S and adj[i]<0.05) else 'no difiere'}")
sig=[i for i in range(3) if abs(names[i][1].mean()-base.mean())>=S and adj[i]<0.05]
print(f"\nH-C1: {'SE CUMPLE' if sig else 'NO SE CUMPLE'} ({len(sig)} de 3 variantes difieren de BASE)")
d0=names[0][1].mean()-base.mean()
print(f"H-C2 (sin ancla reduce la contención): efecto {d0:+.3f} → {'SE CUMPLE' if (d0<=-S and adj[0]<0.05) else 'NO SE CUMPLE'}")
