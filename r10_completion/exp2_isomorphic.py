"""R10 E2 — exact isomorphism and matched-support POLAR-vs-GENERIC control."""
from __future__ import annotations
import numpy as np
from .core import sigmoid, ridge_fit, ridge_predict, orthogonal

POLAR_SUPPORTS=(
    (0,1,2,3,8,9,16,17),
    (4,5,6,7,10,11,18,19),
    (8,9,10,11,12,13,20,21),
    (0,1,12,13,14,15,22,23),
)

def _features(rng,n):
    raw=sigmoid(rng.normal(size=(n,16)))
    interactions=np.column_stack([raw[:,2*i]*raw[:,2*i+1] for i in range(8)])
    return np.c_[raw,interactions]

def _truth(seed):
    rng=np.random.default_rng(seed)
    w=np.zeros((24,4))
    for a,support in enumerate(POLAR_SUPPORTS):
        signs=rng.choice([-1.,1.],len(support))
        w[list(support),a]=signs*rng.uniform(.65,1.05,len(support))
        outside=[i for i in range(24) if i not in support]
        extra=rng.choice(outside,3,replace=False)
        w[extra,a]=rng.normal(0,.12,3)
    bias=rng.normal(0,.15,4)
    return w,bias

def _fit_supported(x,y,supports,lam=.02):
    models=[]
    for a,s in enumerate(supports):
        models.append((tuple(s),ridge_fit(x[:,s],y[:,a],lam)))
    return models

def _predict_supported(x,models):
    return np.column_stack([ridge_predict(x[:,s],w) for s,w in models])

def _metrics(pred,true):
    action=np.argmax(pred,axis=1)
    chosen=true[np.arange(len(true)),action]
    return dict(return_mean=float(np.mean(chosen)),life_fraction=float(np.mean(chosen>=.40)),
                action=action,chosen=chosen)

def run(seed):
    rng=np.random.default_rng(seed)
    xtr=_features(rng,1800); xte=_features(rng,1200)
    wtrue,bias=_truth(seed+10000)
    true_tr=sigmoid(xtr@wtrue+bias)
    true_te=sigmoid(xte@wtrue+bias)
    ytr=np.clip(true_tr+rng.normal(0,.035,true_tr.shape),0,1)

    # E2a: exact orthogonal isomorphism, same information and L2 capacity.
    q=orthogonal(24,seed+20000)
    wp=ridge_fit(xtr,ytr,.02)
    wg=ridge_fit(xtr@q,ytr,.02)
    pp=ridge_predict(xte,wp)
    pg=ridge_predict(xte@q,wg)
    mp=_metrics(pp,true_te); mg=_metrics(pg,true_te)
    e2a=dict(max_prediction_difference=float(np.max(np.abs(pp-pg))),
             return_difference=float(mp['return_mean']-mg['return_mean']),
             action_agreement=float(np.mean(mp['action']==mg['action'])))

    # E2b: identical cardinality and fitting budget, target-agnostic random support.
    generic=[]
    for a,s in enumerate(POLAR_SUPPORTS):
        grng=np.random.default_rng(seed+30000+a)
        generic.append(tuple(sorted(grng.choice(24,len(s),replace=False).tolist())))
    polar_models=_fit_supported(xtr,ytr,POLAR_SUPPORTS)
    generic_models=_fit_supported(xtr,ytr,generic)
    ppol=_predict_supported(xte,polar_models)
    pgen=_predict_supported(xte,generic_models)
    pol=_metrics(ppol,true_te); gen=_metrics(pgen,true_te)
    delta=float(pol['return_mean']-gen['return_mean'])
    life_delta=float(pol['life_fraction']-gen['life_fraction'])
    structural=dict(polar_return=pol['return_mean'],generic_return=gen['return_mean'],
                    return_advantage=delta,polar_life=pol['life_fraction'],
                    generic_life=gen['life_fraction'],life_difference=life_delta,
                    polar_supports=[list(x) for x in POLAR_SUPPORTS],
                    generic_supports=[list(x) for x in generic],
                    active_coefficients_per_output=[len(x) for x in POLAR_SUPPORTS])
    exact_pass=(e2a['max_prediction_difference']<=1e-8 and abs(e2a['return_difference'])<=1e-8)
    specificity_pass=(delta>=.02 and life_delta>=-.01)
    return dict(seed=int(seed),exact_isomorphism=e2a,matched_structural=structural,
                pass_exact=bool(exact_pass),pass_specificity=bool(specificity_pass),
                pass_strong=bool(exact_pass and specificity_pass))
