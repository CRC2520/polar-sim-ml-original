"""Exploratory four-factor diagnosis on the original Study 3 task generator."""
from itertools import product,combinations
import numpy as np
from network_tension import NetworkTensionModel
from polar.model import ModelConfig
from study3.core import environment,transition,CELLS,STEPS
from integrated_polar.numerics import CoupledRLS,project

FLAGS=list(product((0,1),repeat=4))
SEEDS=list(range(911001,911013))


class DiagnosticController(NetworkTensionModel):
    def __init__(self,flags,seed):
        from study3.core import templates
        _,K=templates();super().__init__(ModelConfig(agents=1,types=4,seed=seed,step_size=1),tension_coupling=-.2*K)
        self.flags=tuple(flags);self.estimator=CoupledRLS(8,8,initial=np.eye(8))
    def propose_action(self,target,weights,horizon):
        if self.flags==(0,0,0,0):return super().propose_action(target,weights,horizon)
        information,routing,model,priority=self.flags
        p=self.q;v=np.clip(target/self._planning_gain,0,1);e=v-p
        conflict=self.chi*p[...,0]*p[...,1];tau=np.mean(np.abs(e),axis=-1)+conflict
        msg=e.copy() if information else np.repeat(tau[...,None],2,axis=-1)
        sent=msg[:,[3,0,1,2],:].copy()
        if routing:sent[...,1]*=-1
        if model:
            ideal=np.linalg.lstsq(self.estimator.B+.0001*np.eye(8),target.ravel(),rcond=None)[0].reshape(self.shape)
        else:ideal=target/self._planning_gain
        eta=np.clip(self.config.step_size*horizon,0,1)
        proposal=p+eta*(ideal-p-.2*sent)
        self._network_trace={'version':'p0_factorial_diagnostic','flags':list(self.flags),
                              'signed_error':e.tolist(),'coactivation_conflict':conflict.tolist(),
                              'message':msg.tolist(),'delivered':sent.tolist(),'B':self.estimator.B.tolist()}
        return proposal
    def act(self,o):
        u=super().act(o)
        if self.flags[3]:
            tr=self.last_trace
            u=project(np.asarray(tr['proposal']).ravel(),np.asarray(tr['costs']).ravel(),tr['constraints']['budget'],
                      np.asarray(tr['allowed']).ravel(),np.maximum(np.asarray(tr['weights']).ravel(),1e-12)).reshape(self.shape)
            self.q=u;self.last_action=u.copy()
            for key in ('action','q_after'):tr[key]=u.tolist()
            tr['signed']=self.signed.tolist();tr['intensity']=self.intensity.tolist()
            tr['effect_predicted']=(u*self._planning_gain).tolist()
        return u
    def learn(self,feedback):
        if self.flags[2]:
            effect=np.asarray(feedback['effect']).ravel()
            self.estimator.update(self.last_action.ravel(),effect,(effect>0)&(effect<1))
        return super().learn(feedback)


def run_diagnosis():
    records=[]
    for seed in SEEDS:
        for cell in CELLS:
            env=environment(seed,cell)
            for flags in FLAGS+[None]:
                if flags is None:
                    m=NetworkTensionModel(ModelConfig(agents=1,types=4,seed=seed,step_size=1));name='no_K'
                else:m=DiagnosticController(flags,seed);name=''.join(map(str,flags))
                frames=[]
                for t in range(STEPS):
                    o={k:env[k][t] for k in ('target','weights','costs','allowed')}
                    o.update(budget=float(env['budget'][t]),cue=str(t//16),observed=True)
                    m.set_incompatibility(env['chi'][t]);u=m.act(o);y=transition(env,t,u);m.learn({'effect':y})
                    frames.append({'t':t,'action':u.tolist(),'effect':y.tolist(),
                                   'loss':float(np.sum(o['weights']*(y-o['target'])**2)/o['weights'].sum()),
                                   'cost':float(np.sum(u*o['costs'])/np.sum(o['costs']))})
                    m.traces.clear()
                records.append({'seed':seed,'cell':list(cell),'condition':name,'loss':float(np.mean([f['loss'] for f in frames])),
                                'cost':float(np.mean([f['cost'] for f in frames])),'frames':frames})
        print(f'P0 seed {seed}: {len(records)} runs',flush=True)
    rng=np.random.default_rng(911777);idx=rng.integers(0,len(SEEDS),(10000,len(SEEDS)))
    values={name:np.array([np.mean([r['loss'] for r in records if r['seed']==s and r['condition']==name]) for s in SEEDS])
            for name in [''.join(map(str,f)) for f in FLAGS]+['no_K']}
    def describe(x):
        samples=x[idx].mean(axis=1);return {'mean':float(x.mean()),'low95':float(np.quantile(samples,.025)),'high95':float(np.quantile(samples,.975))}
    main={};interactions={}
    for j,label in enumerate(['G02_information','G03_routing','G05_coupled_model','G06_priorities']):
        on=np.mean([values[''.join(map(str,f))] for f in FLAGS if f[j]],axis=0)
        off=np.mean([values[''.join(map(str,f))] for f in FLAGS if not f[j]],axis=0)
        main[label]=describe(on-off)
    for j,k in combinations(range(4),2):
        groups={(a,b):np.mean([values[''.join(map(str,f))] for f in FLAGS if f[j]==a and f[k]==b],axis=0) for a,b in product([0,1],repeat=2)}
        interactions[f'{j}-{k}']=describe(groups[1,1]-groups[1,0]-groups[0,1]+groups[0,0])
    summary={'status':'exploratory_factorial_diagnosis','runs':len(records),'seeds':SEEDS,
             'factor_order':['information','routing','coupled_model','priorities'],
             'condition_means':{k:float(v.mean()) for k,v in values.items()},'main_effects':main,'two_way_interactions':interactions,
             'original_active_minus_noK':describe(values['0000']-values['no_K']),
             'combined_changes_minus_original':describe(values['1111']-values['0000']),
             'scope':'Effects of declared implementation interventions, not a unique psychological explanation. Main effects average interactions; original study outcomes unchanged.'}
    return records,summary
