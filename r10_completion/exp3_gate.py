"""R10 E3 — adaptive-gate necessity in a switching-coupling benchmark."""
from __future__ import annotations
import numpy as np
from .core import sigmoid, fit_logistic, logistic_predict, balanced_accuracy
from .config import STEPS_E3_TRAIN, STEPS_E3_VALID, STEPS_E3_TEST

CONSTANTS=(0.,.25,.5,1.)

def _sequence(seed,n):
    rng=np.random.default_rng(seed)
    mode=np.empty(n,dtype=int)
    t=0; m=int(rng.random()<.5)
    while t<n:
        length=int(rng.integers(35,76)); mode[t:min(n,t+length)]=m
        m=1-m; t+=length
    x=np.zeros((n,3)); prev=rng.normal(0,.3,3)
    for i in range(n):
        prev=.82*prev+rng.normal(0,.38,3)
        x[i]=np.tanh(prev)
    public=(2*mode-1)+rng.normal(0,.72,n)
    # Error-history proxy available from past observations, not the label itself.
    lag=np.r_[0.,(2*mode[:-1]-1)+rng.normal(0,.8,n-1)]
    features=np.c_[public,lag,x[:,2]]
    action_axis=np.linspace(-1,1,4)
    local=np.empty((n,4)); cross=np.empty((n,4))
    for a,z in enumerate(action_axis):
        local[:,a]=.55+.18*x[:,0]*z-.10*(z-x[:,1])**2+.05*x[:,2]
        cross[:,a]=.30*z*(.65+.35*np.abs(x[:,0]))
    true=np.clip(local+(2*mode[:,None]-1)*cross,0,1)
    return dict(mode=mode,x=x,gate_features=features,local=local,cross=cross,true=true)

def _evaluate(seq,gates):
    g=np.asarray(gates,float)
    if g.ndim==0: g=np.full(len(seq['mode']),float(g))
    pred=seq['local']+g[:,None]*seq['cross']
    action=np.argmax(pred,axis=1)
    reward=seq['true'][np.arange(len(action)),action]
    alive=reward>=.25
    return dict(reward=float(np.mean(reward)),alive=float(np.mean(alive)),action=action)

def run(seed):
    train=_sequence(seed+101,STEPS_E3_TRAIN)
    valid=_sequence(seed+202,STEPS_E3_VALID)
    test=_sequence(seed+303,STEPS_E3_TEST)
    # Fit from factual prediction gain on exploratory training actions.
    # The hidden regime/mode is never supplied as a training target or policy input.
    rng=np.random.default_rng(seed+404)
    exploratory=rng.integers(0,4,len(train['mode']))
    idx=np.arange(len(exploratory))
    factual=np.clip(train['true'][idx,exploratory]+rng.normal(0,.025,len(idx)),0,1)
    local=train['local'][idx,exploratory]
    with_cross=local+train['cross'][idx,exploratory]
    factual_gain=(factual-local)**2-(factual-with_cross)**2
    gate_target=(factual_gain>0).astype(float)
    gate_w=fit_logistic(train['gate_features'],gate_target,steps=700,lr=.06,l2=.01)
    p_valid=logistic_predict(valid['gate_features'],gate_w)
    p_test=logistic_predict(test['gate_features'],gate_w)
    learned=(p_test>=.5).astype(float)
    learned_eval=_evaluate(test,learned)
    constant_validation={str(g):_evaluate(valid,g)['reward'] for g in CONSTANTS}
    best=max(CONSTANTS,key=lambda g:(constant_validation[str(g)],-g))
    best_eval=_evaluate(test,best)
    off_eval=_evaluate(test,0.)
    bal=balanced_accuracy(test['mode'],(p_test>=.5).astype(int))
    result=dict(
        gate_balanced_accuracy=float(bal),
        learned_reward=learned_eval['reward'],
        best_constant=float(best),
        best_constant_reward=best_eval['reward'],
        exact_off_reward=off_eval['reward'],
        advantage_best_constant=float(learned_eval['reward']-best_eval['reward']),
        advantage_off=float(learned_eval['reward']-off_eval['reward']),
        learned_alive=learned_eval['alive'],
        best_constant_alive=best_eval['alive'],
        alive_difference=float(learned_eval['alive']-best_eval['alive']),
        gate_active_fraction=float(np.mean(learned)),
        true_cross_useful_fraction=float(np.mean(test['mode'])),
        validation_constant_rewards=constant_validation,
    )
    passed=(bal>=.85 and result['advantage_best_constant']>=.03 and
            result['advantage_off']>=.03 and result['alive_difference']>=-.01)
    return dict(seed=int(seed),confirmatory=result,pass_strong=bool(passed),
                scope="Construct-validity benchmark; hidden cross-useful label is used for training target only, not final policy input.")
