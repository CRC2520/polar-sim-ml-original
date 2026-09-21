#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
def feat_integrated(X):
    X=np.asarray(X)
    feats=[X]
    d=X.shape[1]
    for i in range(d):
        for j in range(i,d):
            feats.append((X[:,i]*X[:,j])[:,None])
    return np.concatenate(feats,axis=1)

def feat_factorized5(X):
    X=np.asarray(X); return np.concatenate([X**p for p in range(1,6)],axis=1)

def gen_family(seed,fam,n=1200,ood=False):
    rng=np.random.default_rng(seed)
    if fam=='sir':
        S=rng.uniform(.15,.95,n); I=rng.uniform(.02,.45,n)
        beta=rng.uniform(.8,1.4,n) if not ood else rng.uniform(1.5,2.0,n)
        gamma=rng.uniform(.15,.35,n)
        X=np.column_stack([S,I,beta,gamma])
        Y=np.column_stack([-beta*S*I, beta*S*I-gamma*I])
    elif fam=='lv':
        x=rng.uniform(.2,1.5,n); y=rng.uniform(.2,1.4,n)
        alpha=rng.uniform(.7,1.2,n); beta=rng.uniform(.5,.9,n)
        delta=rng.uniform(.4,.8,n); gamma=rng.uniform(.6,1.0,n)
        if ood: alpha+=.35; beta+=.2
        X=np.column_stack([x,y,alpha,beta,delta,gamma])
        Y=np.column_stack([alpha*x-beta*x*y, delta*x*y-gamma*y])
    elif fam=='reaction':
        A=rng.uniform(.05,1.5,n); B=rng.uniform(.05,1.5,n)
        k=rng.uniform(.5,1.1,n) if not ood else rng.uniform(1.2,1.6,n)
        X=np.column_stack([A,B,k])
        rate=k*A*B
        Y=np.column_stack([-rate,-rate,rate])
    elif fam=='competition':
        x=rng.uniform(.1,1.3,n); y=rng.uniform(.1,1.3,n)
        r1=rng.uniform(.7,1.1,n); r2=rng.uniform(.6,1.0,n)
        a12=rng.uniform(.35,.65,n); a21=rng.uniform(.3,.6,n)
        if ood: a12+=.15; a21+=.15
        X=np.column_stack([x,y,r1,r2,a12,a21])
        Y=np.column_stack([r1*x*(1-x-a12*y), r2*y*(1-y-a21*x)])
    else: raise ValueError
    return X.astype(np.float64),Y.astype(np.float64)

def fit_eval_models2(seed,fam,ntrain=64,ntest=1000):
    Xtr,Ytr=gen_family(seed,fam,ntrain,False); Xte,Yte=gen_family(seed+1_000_000,fam,ntest,True)
    xs=StandardScaler().fit(Xtr); ys=StandardScaler().fit(Ytr)
    XtrS=xs.transform(Xtr); XteS=xs.transform(Xte); YtrS=ys.transform(Ytr)
    Fi=feat_integrated(XtrS); Fit=feat_integrated(XteS)
    pi=ys.inverse_transform(Ridge(alpha=1e-3).fit(Fi,YtrS).predict(Fit))
    Ff=feat_factorized5(XtrS); Fft=feat_factorized5(XteS)
    pf=ys.inverse_transform(Ridge(alpha=.1).fit(Ff,YtrS).predict(Fft))
    mlp=MLPRegressor(hidden_layer_sizes=(24,24),activation='tanh',solver='lbfgs',alpha=.03,max_iter=1200,random_state=seed)
    mlp.fit(XtrS,YtrS); pm=ys.inverse_transform(mlp.predict(XteS))
    mse=lambda p: float(np.mean((p-Yte)**2))
    return {'integrated':mse(pi),'factorized':mse(pf),'monolithic':mse(pm),
            'features_integrated':Fi.shape[1],'features_factorized':Ff.shape[1]}

def eval_e4_seed(seed):
    fams=['sir','lv','reaction','competition']; rows={}
    for i,f in enumerate(fams):
        rows[f]=fit_eval_models2(seed+100*i,f)
    ratios_f=[rows[f]['integrated']/(rows[f]['factorized']+1e-12) for f in fams]
    ratios_m=[rows[f]['integrated']/(rows[f]['monolithic']+1e-12) for f in fams]
    wins_f=sum(rows[f]['integrated']<rows[f]['factorized'] for f in fams)
    wins_m=sum(rows[f]['integrated']<rows[f]['monolithic'] for f in fams)
    return {'family_rows':rows,'median_factorized_ratio':float(np.median(ratios_f)),
            'max_factorized_ratio':float(np.max(ratios_f)),
            'median_monolithic_ratio':float(np.median(ratios_m)),
            'integrated_wins_factorized':wins_f,'integrated_wins_monolithic':wins_m,
            'factorized_more_features_all':all(rows[f]['features_factorized']>=rows[f]['features_integrated'] for f in fams)}

def summarize(rec):
    keys=['median_factorized_ratio','max_factorized_ratio','median_monolithic_ratio','integrated_wins_factorized','integrated_wins_monolithic']
    med={k:float(np.median([r[k] for r in rec])) for k in keys}
    checks={'median_factorized_ratio':med['median_factorized_ratio']<=.05,'max_factorized_ratio':med['max_factorized_ratio']<=.20,
            'median_monolithic_ratio':med['median_monolithic_ratio']<=.80,'integrated_wins_factorized':med['integrated_wins_factorized']>=4,
            'integrated_wins_monolithic':med['integrated_wins_monolithic']>=3,
            'factorized_more_features':all(r['factorized_more_features_all'] for r in rec)}
    guards=[r['integrated_wins_factorized']==4 and r['median_factorized_ratio']<=.10 and r['median_monolithic_ratio']<=1.0 and r['integrated_wins_monolithic']>=2 and r['factorized_more_features_all'] for r in rec]
    return {'medians':med,'checks':checks,'guard_pass_count':int(sum(guards)),'global_pass':bool(all(checks.values()) and sum(guards)>=9)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    s,e=map(int,a.seeds.split(':')); seeds=list(range(s,e+1)); rec=[]
    for x in seeds:
        r=eval_e4_seed(x); r['seed']=x; rec.append(r)
    out={'experiment':'E4 Naturalistic Organizational Necessity','seeds':seeds,'records':rec,'summary':summarize(rec)}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
