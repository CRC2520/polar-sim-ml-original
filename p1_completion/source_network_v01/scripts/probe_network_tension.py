"""Deterministic inter-polar propagation check, not a performance study."""
import hashlib
import json
import pathlib
import sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from polar import ModelConfig
from network_tension import NetworkTensionModel


def run():
    cfg=ModelConfig(agents=1,types=4,seed=2026,use_memory=False,use_self_model=False)
    W=np.zeros((8,8));W[4,2]=.3
    K=np.zeros((8,4));K[2,0]=.4
    chi=np.zeros((1,4));chi[0,0]=1
    full=NetworkTensionModel(cfg,state_coupling=W,tension_coupling=K,incompatibility=chi)
    lesion=NetworkTensionModel(cfg,state_coupling=W,incompatibility=chi)
    initial=np.full((1,4,2),.2);initial[0,0]=[.8,.6]
    full.q=initial;lesion.q=initial
    records=[]
    for t in range(3):
        obs={'target':initial.copy(),'budget':8.}
        uf,ul=full.act(obs),lesion.act(obs)
        full.learn({'effect':uf});lesion.learn({'effect':ul})
        records.append({'transition':t+1,'difference':(uf-ul).tolist(),
                        'full':full.last_trace,'tension_route_lesion':lesion.last_trace})
    assert abs(records[0]['difference'][0][1][0]-.1248)<1e-12
    assert abs(records[1]['difference'][0][2][0]-.024336)<1e-12
    assert all(np.all(np.asarray(r['difference'])[0,3]==0) for r in records)
    root=pathlib.Path(__file__).resolve().parents[1]
    sources=['network_tension.py','polar/model.py','polar/workspace.py','polar/memory.py',
             'polar/polarities.py','scripts/probe_network_tension.py']
    return {'status':'deterministic_engineering_check_not_confirmatory_evidence',
            'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
            'intervention':'K removed; W, initial state, observations and resource budget fixed',
            'traces':records}


if __name__=='__main__':
    out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'docs/network_tension_probe.json')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
    print(out)
