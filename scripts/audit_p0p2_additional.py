"""Post-run evidence audit, not a new study or change to frozen inference rules."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gap_resolution.run import iter_records,sha,source_hashes
from gap_resolution.evaluate import capability_probe,coordination_probe,calibration_metrics,CAPABILITY_SEEDS,verify_record
from gap_resolution.p0 import SEEDS,FLAGS
from study3.core import CELLS,environment,transition
from integrated_polar.layers import CalibratedMonitor

root=ROOT/'results_p0p2';freeze=json.loads((ROOT/'docs/P0P2_FREEZE.json').read_text())
if freeze['scientific_source_sha256']!=source_hashes():raise RuntimeError('frozen sources changed')
p0man=json.loads((root/'development/P0_MANIFEST.json').read_text());seen=set();values={}
for r in iter_records(root/'development',p0man['archives'],p0man['records']):
    key=(r['seed'],tuple(r['cell']),r['condition'])
    if key in seen:raise ValueError('duplicate P0 trial')
    seen.add(key);env=environment(r['seed'],tuple(r['cell']));losses=[];costs=[]
    if [f['t'] for f in r['frames']]!=list(range(64)):raise ValueError('P0 step set')
    for f in r['frames']:
        t=f['t'];u=np.asarray(f['action']);effect=transition(env,t,u)
        if not np.allclose(effect,f['effect'],rtol=0,atol=1e-12):raise ValueError('P0 effect replay')
        loss=float(np.sum(env['weights'][t]*(effect-env['target'][t])**2)/env['weights'][t].sum())
        cost=float(np.sum(env['costs'][t]*u)/env['costs'][t].sum())
        if abs(loss-f['loss'])>1e-12 or abs(cost-f['cost'])>1e-12:raise ValueError('P0 per-step outcome')
        losses.append(loss);costs.append(cost)
    if abs(np.mean(losses)-r['loss'])>1e-12 or abs(np.mean(costs)-r['cost'])>1e-12:raise ValueError('P0 run outcome')
    values[key]=float(np.mean(losses))
conditions=[''.join(map(str,f)) for f in FLAGS]+['no_K']
expected={(s,tuple(c),m) for s in SEEDS for c in CELLS for m in conditions}
if seen!=expected:raise ValueError('incomplete P0 factorial')
p0=json.loads((root/'development/P0_DIAGNOSIS.json').read_text())
for condition in conditions:
    average=np.mean([v for (s,c,m),v in values.items() if m==condition])
    if abs(average-p0['condition_means'][condition])>1e-12:raise ValueError('P0 condition mean')
for j,name in enumerate(p0['factor_order']):
    on=np.mean([v for (s,c,m),v in values.items() if m!='no_K' and m[j]=='1'])
    off=np.mean([v for (s,c,m),v in values.items() if m!='no_K' and m[j]=='0'])
    report_key=['G02_information','G03_routing','G05_coupled_model','G06_priorities'][j]
    if abs(on-off-p0['main_effects'][report_key]['mean'])>1e-12:raise ValueError('P0 main effect')

p2=json.loads((root/'final/P2_RESULTS.json').read_text());saved=json.loads((root/'final/P2_CAPABILITY_RECORDS.json').read_text())
if [r['seed'] for r in saved]!=CAPABILITY_SEEDS:raise ValueError('P2 seed set')
for old in saved:
    new=capability_probe(old['seed'])
    for key in ['source_correct','source_trials','ambiguous_abstentions','semantic_max_error','normative_pass']:
        if new[key]!=old[key]:raise ValueError('P2 capability replay: '+key)
    np.testing.assert_allclose(new['learned_fill_drain'],old['learned_fill_drain'],atol=1e-10,rtol=0)
    for a,b in zip(new['normative_records'],old['normative_records']):
        np.testing.assert_allclose(a['action'],b['action'],atol=1e-10,rtol=0)
        if a['rules']!=b['rules']:raise ValueError('normative reason replay')
correct=sum(r['source_correct'] for r in saved);total=sum(r['source_trials'] for r in saved)
if correct/total!=p2['source_attribution_accuracy']:raise ValueError('source accuracy')
coord=json.loads((root/'final/P2_COORDINATION_RECORDS.json').read_text())
for old in coord:
    new=coordination_probe(old['seed'],old['communication'])
    if abs(old['loss']-new['loss'])>1e-10 or old['violations']!=new['violations']:raise ValueError('coordination replay')
coord_delta=np.mean([next(x['loss'] for x in coord if x['seed']==s and x['communication'])-next(x['loss'] for x in coord if x['seed']==s and not x['communication']) for s in CAPABILITY_SEEDS])
if abs(coord_delta-p2['coordination_loss_difference']['mean'])>1e-12:raise ValueError('coordination aggregate')
man=json.loads((root/'final/MANIFEST.json').read_text());monitor=CalibratedMonitor.restore(man['monitor']);baseframes=[];review_frames=[];losses={};seen_p2=set()
for r in iter_records(root/'final/p2',man['p2_archives'],man['p2_records']):
    verify_record(r);key=(r['seed'],tuple(r['cell']),r['review_policy'])
    if key in seen_p2:raise ValueError('duplicate P2 trajectory')
    seen_p2.add(key);losses[key]=r['metrics']['loss']
    for f in r['frames']:
        if f['success']!=(f['loss']<=.003):raise ValueError('incorrect success label')
        if abs(monitor.probability(f['pre_feedback_score'])-f['probability'])>1e-12:raise ValueError('probability recalculation')
        (review_frames if r['review_policy'] else baseframes).append(f)
if len(seen_p2)!=192:raise ValueError('incomplete P2 trajectories')
if calibration_metrics(baseframes,monitor)!=p2['monitor']:raise ValueError('monitor summary')
review_fraction=float(np.mean([f['requested_review'] for f in review_frames]))
if abs(review_fraction-p2['review_fraction'])>1e-12:raise ValueError('review coverage')
review_delta=np.mean([v-losses[(s,c,False)] for (s,c,review),v in losses.items() if review])
if abs(review_delta-p2['review_minus_base_loss'])>1e-12:raise ValueError('review outcome')
report={'status':'verified_post_run_additional_audit','audit_utc':datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256':sha(Path(__file__).read_bytes()),'p0_records_verified':len(seen),'p0_condition_means_and_main_effects_recalculated':True,
        'p2_capability_seed_replays':len(saved),'source_identifications_correct':correct,'source_trials':total,
        'p2_coordination_replays':len(coord),'p2_trajectories_verified':len(seen_p2),'monitor_and_review_outcomes_recalculated':True,
        'original_final_manifest_sha256':sha((root/'final/MANIFEST.json').read_bytes()),
        'scope':'Internal post-run audit of frozen records and mechanism replays; does not change registered statistics or constitute a new confirmatory study.'}
(ROOT/'docs/P0P2_ADDITIONAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
