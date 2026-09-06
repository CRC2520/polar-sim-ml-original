"""Independent mechanical replay of predeclared pilot trials, never final data."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from study2.controllers import ControllerConfig, CoupledController
from study2.evaluation import read_json, sha256, source_hashes, ArtifactReader

pilot = ROOT/'results_study2/coupling/pilot'
manifest = read_json(pilot/'manifest.json')
assert manifest['split'] == 'pilot'
assert manifest['seeds'] == list(range(52001,52007))
assert manifest['source_sha256'] == source_hashes(ROOT)
trials = [x for x in manifest['trials'] if x['seed'] == 52001 and x['controller'] in ('paired','paired_lesion')]
assert len(trials) == 12
results=[]
reader=ArtifactReader(pilot,manifest)

def differences(actual, expected, path=''):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys() == expected.keys(), path+' dictionary keys'
        return sum((differences(actual[k],expected[k],path+'/'+k) for k in expected),[])
    if isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected), path+' list length'
        return sum((differences(a,e,path+'/'+str(i)) for i,(a,e) in enumerate(zip(actual,expected))),[])
    if isinstance(expected,float):
        assert np.isfinite(expected) and np.isfinite(actual),path+' nonfinite'
        return [] if actual==expected else [(path,abs(actual-expected))]
    assert actual==expected,path+' discrete mismatch'
    return []

for item in trials:
    trace=reader.read(item['trace_path'],item['trace_sha256'])
    assert trace['seed']==52001 and trace['controller'] in ('paired','paired_lesion')
    model=CoupledController(ControllerConfig(**trace['config']))
    errors=[]
    count=0
    for record in trace['records']:
        mechanism=record['mechanism']
        obs=dict(mechanism['observation'])
        obs['state']=np.asarray([0. if x is None else x for x in obs['state']])
        obs['observed']=np.asarray(obs['observed'],dtype=bool)
        obs['allowed']=np.asarray(obs['allowed'],dtype=bool)
        for key in ('state_lower','state_upper'):
            if all(x is None for x in obs[key]):
                del obs[key]
        action=model.act(obs)
        errors+=differences(action.tolist(),record['action'],f"step{record['step']}/action")
        saved_fb=mechanism['feedback']
        fb={'state':np.asarray([0. if x is None else x for x in saved_fb['state']]),
            'observed':np.asarray(saved_fb['observed'],dtype=bool),
            'transition_valid':np.asarray(saved_fb['transition_valid'],dtype=bool)}
        model.learn(fb)
        errors+=differences(model.last_trace,mechanism,f"step{record['step']}/mechanism")
        count+=1
    assert count==128
    maximum=max((x[1] for x in errors),default=0.)
    results.append({'family':trace['family'],'regime':trace['regime'],'controller':trace['controller'],
                    'seed':52001,'transitions':count,'exact_numeric_replay':not errors,
                    'nonexact_numeric_entries':len(errors),'maximum_absolute_difference':maximum,
                    'trace_sha256':item['trace_sha256'],
                    'final_reconstructed_covariance_sha256':__import__('hashlib').sha256(model.covariance.tobytes()).hexdigest()})
    assert maximum <= 1e-12, (trace['family'],trace['regime'],trace['controller'],errors[:8])

report={'schema':'polar-study2-pilot-independent-replay-1','seed':52001,'split':'pilot',
        'trials':len(results),'transitions':sum(x['transitions'] for x in results),
        'all_numeric_entries_exact':all(x['exact_numeric_replay'] for x in results),
        'all_discrete_entries_exact':True,'maximum_absolute_difference':max(x['maximum_absolute_difference'] for x in results),
        'input_scope':'Only saved documented observations and own feedback; no true matrices, environment regeneration or final seed data.',
        'covariance_note':'Full covariance is reconstructed through RLS; stored diagonal and subsequent predictions/updates match recorded traces. No claim full original covariance was stored.',
        'pilot_manifest_sha256':sha256(pilot/'manifest.json'),'source_sha256':manifest['source_sha256'],
        'results':results}
out=ROOT/'results_study2/validation/pilot_replay_audit.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('results','source_sha256')},indent=2))
print(str(out))
