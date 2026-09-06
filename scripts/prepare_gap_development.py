"""Guarded pre-registration development corrections; no final data exists yet.

Three original causal fixtures saturated the first action for both conditions.
Use unsaturated or opposite-demand cases instead; do not lower effect thresholds.
Add an explicit saturated counterexample in the separate evaluation test file.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'docs/P0P2_FREEZE.json').exists():raise RuntimeError('do not modify frozen scientific source')

def replace_once(path,old,new):
    p=ROOT/path;s=p.read_text()
    if new in s:return
    if s.count(old)!=1:raise RuntimeError('unexpected development anchor: '+path+' '+old[:70])
    p.write_text(s.replace(old,new,1))
replace_once('tests/test_integrated_polar.py',
             'ua,ub=a.act(obs(target=.5)),b.act(obs(target=.5))',
             'ua,ub=a.act(obs(target=.36)),b.act(obs(target=.36))')
replace_once('tests/test_integrated_polar.py',
             "a,b=self.agent(),self.agent();b.workspace.cut.add('planner')\n        ua,ub=a.act(obs()),b.act(obs())",
             "a,b=self.agent(),self.agent();b.workspace.cut.add('planner')\n        ua,ub=a.act(obs(target=.2)),b.act(obs(target=.2))")
replace_once('tests/test_integrated_polar.py',
             'a=self.agent();consume(a,obs());b,c=deepcopy(a),deepcopy(a)',
             'a=self.agent();consume(a,obs(target=.2));b,c=deepcopy(a),deepcopy(a)')
replace_once('gap_resolution/tasks.py',
             'def run_trial(seed,cell,mode,monitor=None,*,retain=True):',
             'def run_trial(seed,cell,mode,monitor=None,*,retain=True,review=False):')
replace_once('gap_resolution/tasks.py',
             'AgentConfig(tanks=N,horizon=H,mode=mode),monitor',
             'AgentConfig(tanks=N,horizon=H,mode=mode,meta_threshold=.55 if review else 0.),monitor')
replace_once('gap_resolution/tasks.py',
             "'probability':agent.trace['success_probability_before_feedback'],'success':bool(loss<=.003)",
             "'probability':agent.trace['success_probability_before_feedback'],'success':bool(loss<=.003),'requested_review':agent.trace['requested_review']")
replace_once('gap_resolution/tasks.py',
             "return {'seed':seed,'cell':list(cell),'mode':mode,'calibration':calibration if retain else None,",
             "return {'seed':seed,'cell':list(cell),'mode':mode,'review_policy':bool(review),'calibration':calibration if retain else None,")
replace_once('integrated_polar/agent.py',
             'u=np.zeros(self.m);prediction=rho*y+drift+self.model.predict(u)',
             'u=np.zeros(self.m) if ethics[\'halt\'] else lower.copy();prediction=rho*y+drift+self.model.predict(u)')
print('Applied documented development-only fixture and review-contract corrections')
