"""Fail-closed selection, frozen-route interventions and development reporting."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
from .core import ROOT, EDGES, RoutingActor, masks, matrix, degree_class, seed_for, model_digest, write_json
from .instrument import NativeEnv, array_hash, runtime_lock, qa
from .train import registry, plan, load_actor, predict, evaluate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def current_commit():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def preservation():
    base=plan()['base_code_commit'];bad=[];preserved=0
    raw=subprocess.check_output(['git','ls-tree','-r','-z',base])
    for entry in raw.split(b'\0'):
        if not entry:continue
        meta,name=entry.split(b'\t',1);path=name.decode()
        now=subprocess.check_output(['git','ls-tree','HEAD','--',path]).decode().strip()
        if not now or now.split('\t',1)[0]!=meta.decode():bad.append(path)
        else:preserved+=1
    if bad:raise RuntimeError('Historical objects changed: '+repr(bad[:20]))
    return {'status':'PASS','base':base,'preserved_git_objects_and_modes':preserved,'bad':bad}


def additional_tests():
    from .core import log_squashed_gaussian
    samples=[]
    for dose in (0.,.25,.5,.75,1.):
        env=NativeEnv();z,_=env.reset(seed=seed_for('qa','dose',dose))
        a=np.tile(np.array([dose,-dose,dose,-dose],dtype=np.float32),3)
        nxt,r,t,tr,info=env.step(a)
        assert max(info['motor_torque_limit_setpoint'])==80*dose
        samples.append({'dose':dose,'motor_setpoints':info['motor_torque_limit_setpoint'],'reward':r,'terminal':t})
        env.close()
    for aa,bb in ((0.,0.),(1.,0.),(0.,1.),(1.,1.)):
        env=NativeEnv();env.reset(seed=seed_for('qa','max_regime',aa,bb))
        action=np.zeros(12,dtype=np.float32);action[:4]=aa;action[4:8]=bb
        _,_,_,_,info=env.step(action);assert info['admitted']==action.tolist();env.close()
    chain={(0,1),(1,2)};permutations=[]
    for sigma in itertools.permutations(range(3)):
        kept={(sigma[i],sigma[j]) for i,j in chain}==chain
        permutations.append({'sigma':sigma,'keeps_ordered_chain':kept})
    assert sum(x['keeps_ordered_chain'] for x in permutations)==1
    torch.manual_seed(seed_for('qa','ppo_update'))
    actor=RoutingActor('111000');before=model_digest(actor)
    z=torch.randn(32,93);mu,ls=actor(z);pre=(mu+ls.exp()*torch.randn_like(mu)).detach()
    old=log_squashed_gaussian(pre,mu,ls).detach();adv=torch.linspace(-1,1,32)
    opt=torch.optim.Adam(actor.parameters(),lr=3e-4)
    newmu,newls=actor(z);ratio=(log_squashed_gaussian(pre,newmu,newls)-old).exp()
    assert torch.equal(ratio,torch.ones_like(ratio))
    loss=-torch.min(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
    opt.zero_grad();loss.backward();opt.step()
    assert before!=model_digest(actor)
    assert classify_interval(-6.,-5.)=='DEVELOPMENT_PATTERN_IMPROVED'
    assert classify_interval(5.,6.)=='DEVELOPMENT_PATTERN_WORSE'
    assert classify_interval(-.1,.1)=='DEVELOPMENT_PATTERN_WITHIN_MARGIN'
    assert classify_interval(-10.,10.)=='INCONCLUSIVE'
    return {'status':'PASS','dose_cases':samples,'permutations':permutations,
            'maximum_four_regimes':'PASS','ppo_ratio_gradient':'PASS','decision_boundaries':'PASS'}


def freeze_development(out: Path):
    q=qa(out);extra=additional_tests();p=plan()
    assert p['expected_fit_count']==len(registry())*p['independent_training_replicates']
    assert p['expected_training_native_steps']==p['expected_fit_count']*p['training_steps_per_fit']
    write_json(out/'ADDITIONAL_QA.json',extra)
    lock={'status':'FROZEN_FOR_DEVELOPMENT_ONLY','source_commit':current_commit(),'plan':p,'plan_sha256':sha(ROOT/'PLAN.json'),
          'source_hashes':{str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in str(f)},
          'runtime':runtime_lock(),'QA_report_sha256':sha(out/'B1S_QA_REPORT.json'),'preservation':preservation(),
          'utc_before_training':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'registry':registry(),
          'final_seeds_generated':False,'B1E_executed':False}
    write_json(out/'B1S_DEVELOPMENT_FREEZE.json',lock)
    return lock


def validate_fits(fits: Path):
    p=plan();rows=[];problems=[];expected=set();train_seeds=set()
    for cfg in registry():
        for rep in range(p['independent_training_replicates']):
            folder=fits/cfg['id']/f'rep-{rep}';expected.add(str(folder.relative_to(fits)))
            try:
                complete=json.loads((folder/'COMPLETE.json').read_text())
                start=json.loads((folder/'STARTED.json').read_text())
                assert start['source_commit']==current_commit()
                assert start['configuration']==cfg and start['replicate']==rep
                assert start['plan_sha256']==sha(ROOT/'PLAN.json')
                for name,h in complete['files'].items():assert sha(folder/name)==h,(folder,name)
                res=json.loads((folder/'TRAINING_RESOURCES.json').read_text())
                assert res['environment_steps']==p['training_steps_per_fit']
                eps=json.loads((folder/'TRAINING_EPISODES.json').read_text())
                train_seeds.update(x['seed'] for x in eps)
                sel=json.loads((folder/'SELECTION.json').read_text())
                assert len(sel['rows'])==p['selection_episodes_per_replicate']
                assert [x['seed'] for x in sel['rows']]==[seed_for('selection','environment',rep,e) for e in range(p['selection_episodes_per_replicate'])]
                rows.append({'configuration':cfg,'replicate':rep,'mean_selection_loss':sel['mean_loss'],
                             'resources':res,'model_sha256':sha(folder/('model.zip' if cfg['kind']=='sac' else 'model.pt'))})
            except Exception as exc:problems.append({'fit':str(folder),'error':repr(exc)})
    actual={str(p.parent.relative_to(fits)) for p in fits.rglob('STARTED.json')}
    if actual!=expected:problems.append({'unexpected_or_missing_fits':sorted(actual^expected)})
    evaluation_seeds={seed_for(split,'environment',rep,e) for split,count in [('selection',p['selection_episodes_per_replicate']),('heldout',p['heldout_episodes_per_replicate']),('causal',p['causal_episodes_per_replicate'])] for rep in range(3) for e in range(count)}
    if train_seeds & evaluation_seeds:problems.append({'seed_overlap':sorted(train_seeds&evaluation_seeds)})
    return rows,problems


def select(fits: Path,out: Path):
    rows,problems=validate_fits(fits)
    write_json(out/'FIT_AUDIT.json',{'expected':plan()['expected_fit_count'],'verified':len(rows),'problems':problems,'status':'PASS' if not problems else 'FAIL'})
    if problems:raise RuntimeError('Incomplete or invalid discovery data; selection blocked')
    score={}
    for cfg in registry():score[cfg['id']]=float(np.mean([x['mean_selection_loss'] for x in rows if x['configuration']['id']==cfg['id']]))
    support=min(masks(),key=lambda m:(score['graph-'+m],m.count('1'),m))
    generic=min((c for c in registry() if c['kind']=='ppo'),key=lambda c:(score[c['id']],c['id']))['id']
    sac=min((c for c in registry() if c['kind']=='sac'),key=lambda c:(score[c['id']],c['id']))['id']
    reverse=''.join('1' if e in {(0,2),(2,1),(1,0)} else '0' for e in EDGES)
    comparators={'S-M0':'graph-000000','S-M2-fixed':'graph-'+reverse,'S-M3':'graph-'+support,
                 'S-M4':'graph-111111','S-M5':generic,'S-M6':sac}
    for alt in degree_class(support):comparators['S-M2-degree-'+alt]='graph-'+alt
    selected_per_rep=[]
    for rep in range(plan()['independent_training_replicates']):
        per={x['configuration']['mask']:x['mean_selection_loss'] for x in rows if x['replicate']==rep and x['configuration']['kind']=='graph' and x['configuration']['mask']!='111111'}
        selected_per_rep.append(min(per,key=lambda m:(per[m],m.count('1'),m)))
    lock={'status':'SELECTED_ON_DEVELOPMENT_VALIDATION_ONLY','source_commit':current_commit(),
          'selected_support':support,'selected_edges':[list(e) for e,b in zip(EDGES,support) if b=='1'],
          'comparators':comparators,'scores':score,'selected_supports_by_independent_training_rep':selected_per_rep,
          'selected_degree_alternatives':degree_class(support),
          'degree_matching_status':'DEFINED' if degree_class(support) else 'NOT_IDENTIFIABLE_UNDER_EXACT_DEGREE_MATCH',
          'selected_checkpoint_hashes':{f'{k}/rep-{rep}':sha(fits/k/f'rep-{rep}'/('model.zip' if k.startswith('sac-') else 'model.pt')) for k in set(comparators.values()) for rep in range(plan()['independent_training_replicates'])},
          'utc_before_heldout_and_causal':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'heldout_data_accessed':False,'causal_data_accessed':False,'development_only':True,'confirmatory':False,
          'H_CAT':'NOT_EVALUABLE','H_TRANSFER':'NOT_EVALUATED','B1E_executed':False}
    write_json(out/'B1S_STRUCTURE_FREEZE.json',lock)
    write_json(out/'B1S_DISCOVERY_RESULTS.json',{'fits':rows,'selection':lock})
    return lock,rows


def classify_interval(lo,hi):
    d=plan()['practical_margin'];e=plan()['equivalence_margin']
    if hi < -d:return 'DEVELOPMENT_PATTERN_IMPROVED'
    if lo > d:return 'DEVELOPMENT_PATTERN_WORSE'
    if lo >= -e and hi <= e:return 'DEVELOPMENT_PATTERN_WITHIN_MARGIN'
    return 'INCONCLUSIVE'


def descriptive_interval(values):
    a=np.asarray(values,dtype=np.float64)
    if len(a)!=3 or not np.isfinite(a).all():return {'mean':None,'nominal_interval':None,'class':'INCONCLUSIVE','n':len(a)}
    mean=float(a.mean());critical=math.sqrt(2*.95**2/(1-.95**2))
    half=critical*float(a.std(ddof=1))/math.sqrt(len(a));lo,hi=mean-half,mean+half
    return {'mean':mean,'nominal_interval':[lo,hi],'class':classify_interval(lo,hi),'n':3,
            'scope':'Exploratory nominal t interval; only three training replicates; common model selection and multiple contrasts preclude a confirmatory coverage claim'}


def causal(actor: RoutingActor,rep: int,out: Path):
    p=plan();rows=[];active=[i for i,b in enumerate(actor.mask) if b=='1']
    if not active:return {'replicate':rep,'status':'NO_ACTIVE_ROUTES_SELECTED','rows':[],'not_a_technical_failure':True}
    variants={'intact':(),'sham':()};variants.update({f'edge-{e}':(e,) for e in active})
    inactive=[e for e in range(6) if e not in active]
    if inactive:variants['off-support']=(inactive[0],)
    if len(active)>=2:variants['two-edge-factorial']=(active[0],active[1])
    digest=model_digest(actor)
    for ep in range(p['causal_episodes_per_replicate']):
        sd=seed_for('causal','environment',rep,ep);env=NativeEnv();z,_=env.reset(seed=sd)
        prefix=[];prefix_r=[];ended=False
        for t in range(p['causal_prefix_cycles']):
            a=predict(actor,z);prefix.append(a.copy());z,r,term,trunc,info=env.step(a);prefix_r.append(r)
            if term or trunc:ended=True;break
        env.close()
        if ended:
            rows.append({'episode':ep,'seed':sd,'status':'TERMINATED_BEFORE_INTERVENTION','cycles':len(prefix),'no_replacement':True});continue
        reference_z=z.copy();variant_rows={}
        for name,lesion in variants.items():
            env=NativeEnv();z,_=env.reset(seed=sd);ret=0.
            for a,expected_r in zip(prefix,prefix_r):
                z,r,t,tr,_=env.step(a);assert r==expected_r and not(t or tr);ret+=r
            assert np.array_equal(z,reference_z),'Action-prefix native replay failed'
            with torch.no_grad():
                mu,_,tr=actor(torch.from_numpy(z[None]),lesion,True)
                snap={k:v.detach().cpu().numpy().tolist() for k,v in tr.items()}
            events=[];first=True
            while True:
                a=predict(actor,z,lesion if first else ())
                nxt,r,t,trunc,info=env.step(a)
                events.append({'seed':sd,'cycle':env.t,'variant':name,'z':z.tolist(),'action':a.tolist(),'reward':r,'next_z':nxt.tolist(),'audit':info})
                ret+=r;z=nxt;first=False
                if t or trunc:break
            env.close()
            file=out/f'causal-rep{rep}-ep{ep}-{name}.jsonl.gz'
            with gzip.open(file,'wt',encoding='utf-8') as f:
                for event in events:f.write(json.dumps(event,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n')
            variant_rows[name]={'loss':-ret,'snapshot':snap,'terminal_cycle':info['native_cycle'],
                                'package_displacement':info['package_displacement'],'trace_file':file.name,'trace_sha256':sha(file)}
        initial=variant_rows['intact']
        assert initial['loss']==variant_rows['sham']['loss'] and initial['snapshot']==variant_rows['sham']['snapshot']
        if inactive:assert initial['loss']==variant_rows['off-support']['loss']
        effects={}
        for name,row in variant_rows.items():
            assert row['snapshot']['m']==initial['snapshot']['m'];assert row['snapshot']['h']==initial['snapshot']['h']
            effects[name]={'bypass_minus_intact_loss':row['loss']-initial['loss'],
                           'action_max_abs_delta':float(np.max(np.abs(np.array(row['snapshot']['action'])-np.array(initial['snapshot']['action']))))}
        interaction=None
        if len(active)>=2:
            interaction=variant_rows['two-edge-factorial']['loss']-variant_rows[f'edge-{active[0]}']['loss']-variant_rows[f'edge-{active[1]}']['loss']+initial['loss']
        rows.append({'episode':ep,'seed':sd,'status':'COMPLETE','prefix_actions':[a.tolist() for a in prefix],
                     'prefix_rewards':prefix_r,'predecision_observation_sha':array_hash(reference_z),'variants':variant_rows,'effects':effects,'two_edge_interaction':interaction})
        assert model_digest(actor)==digest
    return {'replicate':rep,'status':'COMPLETE','rows':rows,'weights_unchanged':model_digest(actor)==digest,
            'estimand':'one-time selective decision-route use; subsequent policy intact; external contrast is total native policy effect','development_only':True}


def complete(fits: Path,out: Path):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
    lock,rows=select(fits,out)
    freeze_hash=sha(out/'B1S_STRUCTURE_FREEZE.json')
    write_json(out/'EVALUATION_STARTED.json',{'structure_freeze_sha256':freeze_hash,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_commit':current_commit()})
    traces=out/'traces';traces.mkdir(exist_ok=True);held={};causals=[]
    unique=sorted(set(lock['comparators'].values()))
    for ident in unique:
        held[ident]=[]
        for rep in range(plan()['independent_training_replicates']):
            actor=load_actor(fits/ident/f'rep-{rep}')
            result=evaluate(actor,rep,'heldout',plan()['heldout_episodes_per_replicate'],traces/f'heldout-{ident}-rep{rep}.jsonl.gz',ident)
            held[ident].extend(result)
            if ident=='graph-'+lock['selected_support']:causals.append(causal(actor,rep,traces))
            print('heldout complete',ident,rep,flush=True)
    held['no-action-witness']=[]
    for rep in range(3):held['no-action-witness']+=evaluate(None,rep,'heldout',plan()['heldout_episodes_per_replicate'],traces/f'heldout-zero-rep{rep}.jsonl.gz','no-action-witness')
    assert sha(out/'B1S_STRUCTURE_FREEZE.json')==freeze_hash
    candidate=held['graph-'+lock['selected_support']];contrasts={}
    for label,ident in lock['comparators'].items():
        if label=='S-M3':continue
        diffs=[]
        for rep in range(3):
            c=[x for x in candidate if x['replicate']==rep];a=[x for x in held[ident] if x['replicate']==rep]
            assert [x['seed'] for x in c]==[x['seed'] for x in a]
            diffs.append(float(np.mean([x['loss']-y['loss'] for x,y in zip(c,a)])))
        contrasts[label]={'candidate_minus_alternative_by_training_rep':diffs,**descriptive_interval(diffs)}
    summaries={k:{'mean_loss':float(np.mean([r['loss'] for r in v])),
                  'mean_displacement':float(np.mean([r['package_displacement'] for r in v])),
                  'mean_cycles':float(np.mean([r['cycles'] for r in v])),
                  'fall_episode_fraction':float(np.mean([r['fallen_count']>0 for r in v])),
                  'episodes':len(v)} for k,v in held.items()}
    zero=summaries['no-action-witness'];competence={}
    for label in ('S-M5','S-M6'):
        s=summaries[lock['comparators'][label]]
        competence[label]={'improvement_over_zero_loss':zero['mean_loss']-s['mean_loss'],
                          'additional_displacement':s['mean_displacement']-zero['mean_displacement'],
                          'basic_task_diagnostic_pass':bool(zero['mean_loss']-s['mean_loss']>plan()['practical_margin'] and s['mean_displacement']-zero['mean_displacement']>=1.),
                          'optimality_or_convergence_certified':False}
    route_effects=[]
    for cr in causals:
        for ep in cr['rows']:
            for name,eff in ep.get('effects',{}).items():
                if name.startswith('edge-'):route_effects.append(eff)
    route_used=any(e['action_max_abs_delta']>plan()['route_action_numeric_tolerance'] for e in route_effects)
    resource_rows=[{'id':r['configuration']['id'],'kind':r['configuration']['kind'],'replicate':r['replicate'],**r['resources']} for r in rows]
    resource={'status':'AUDITED_WITH_EXPLICIT_DIFFERENCES','fits':resource_rows,
              'total_training_native_steps':sum(r['environment_steps'] for r in resource_rows),
              'total_measured_training_seconds':sum(r['training_seconds'] for r in resource_rows),
              'axes':{'information':'same raw93 for every controller; no privileged audit input',
                      'feasibility':'same native12 actionBox, physics, reward and termination',
                      'learning':'42 search candidates for each selected family,3 fits each,32768 environment steps; algorithms, search spaces and update counts differ',
                      'memory':'no temporal actor memory; SAC replay/targetcritics differ and are measured',
                      'planning':'one message round, all messages computed; nongraph models have different feedforward architectures',
                      'decision_compute':'graph masks use same tensors/operations; nongraph parameter/operation/host differences explicitly not equal-capacity'},
              'same_parameter_count_implies_same_capacity':False,'unmatched_dimensions_preclude_general_superiority':True}
    decision={'status':'B1S_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS','stages':{x:'EXECUTED' for x in ['S1','S2','S3','S4','S5','S6','S7']},
              'selected_support':lock['selected_support'],'route_use_identified_in_inspected_snapshots':route_used,
              'all_generic_competence_diagnostics_pass':all(x['basic_task_diagnostic_pass'] for x in competence.values()),
              'confirmatory_support':False,'H_CAT':'NOT_EVALUABLE','H_TRANSFER':'NOT_EVALUATED',
              'B1E':'ON_HOLD_PENDING_STRUCTURAL_DISCOVERY_REVIEW','ready_for_b1e_freeze':False,'ready_for_b1e_confirmatory_run':False,
              'final_seeds_generated':False,'B1E_executed':False,'no_retuning_after_selection':True,
              'limitations':['Capped 32768-step fits, not demonstrated convergence','Only3 independent training initializations; intervals are exploratory','One third-party domain is not transfer','Common global observations allow graph-route compensation','No catalogue map','Different model classes and update ratios; no broad equal-capacity superiority claim'],
              'next_decision':'Review instrument, development effects and baseline competence before specifying any new confirmatory protocol; a positive result is not automatic authorization'}
    if not any(c['rows'] for c in causals):decision['stages']['S5']='NO_ACTIVE_ROUTES_SELECTED'
    elif not any(ep.get('status')=='COMPLETE' for c in causals for ep in c['rows']):decision['stages']['S5']='NO_REACHABLE_PREFIX_IN_RETAINED_CASES'
    write_json(out/'B1S_COMPETITIVE_RESULTS.json',{'summaries':summaries,'contrasts':contrasts,'competence':competence,'episodes':held,'development_only':True,'confirmatory':False})
    write_json(out/'B1S_CAUSAL_RESULTS.json',{'replicates':causals,'development_only':True,'confirmatory':False})
    write_json(out/'B1S_RESOURCE_AUDIT.json',resource);write_json(out/'B1S_DECISION.json',decision)
    write_json(out/'HISTORICAL_PRESERVATION.json',preservation())
    with (out/'DISCOVERY_TABLE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','kind','replicate','mean_selection_loss','train_steps','train_seconds','actor_parameters'])
        for r in rows:w.writerow([r['configuration']['id'],r['configuration']['kind'],r['replicate'],r['mean_selection_loss'],r['resources']['environment_steps'],r['resources']['training_seconds'],r['resources']['parameters_actor']])
    render_reports(out)
    return decision


def render_reports(out: Path):
    d=json.loads((out/'B1S_DECISION.json').read_text());u=json.loads((out/'B1S_COMPETITIVE_RESULTS.json').read_text());s=json.loads((out/'B1S_STRUCTURE_FREEZE.json').read_text());r=json.loads((out/'B1S_RESOURCE_AUDIT.json').read_text())
    for lang in ('en','es'):
        es=lang=='es';lines=[('# B1-S — Informe de ejecución de desarrollo' if es else '# B1-S — Development execution report'),'','**'+d['status']+'**','',
            ('Se ejecutó una campaña de desarrollo limitada por un presupuesto fijado antes del aprendizaje. No es una confirmación de la arquitectura ni una evaluación B1-E.' if es else 'A budget-capped development campaign was executed under a plan fixed before learning. This is neither architectural confirmation nor a B1-E evaluation.'),'',
            f"Source commit: `{current_commit()}`",f"Selected support: `{s['selected_support']}`",f"Training fits: {plan()['expected_fit_count']}; native training steps: {r['total_training_native_steps']}",'',
            ('## Resultados competitivos descriptivos' if es else '## Descriptive competitive outcomes'),'',
            '| Comparator | Candidate | Mean loss | Mean displacement | Fall fraction |','|---|---|---:|---:|---:|']
        for label,ident in s['comparators'].items():
            x=u['summaries'][ident];lines.append(f"| {label} | {ident} | {x['mean_loss']:.9g} | {x['mean_displacement']:.9g} | {x['fall_episode_fraction']:.9g} |")
        lines+=['','| Contrast | Delta by independent training replicate | Mean delta | Nominal interval | Development class |','|---|---|---:|---|---|']
        for label,x in u['contrasts'].items():lines.append(f"| S-M3 minus {label} | {x['candidate_minus_alternative_by_training_rep']} | {x['mean']:.9g} | {x['nominal_interval']} | {x['class']} |")
        lines+=['',('Los intervalos son nominales y exploratorios, con solo tres inicializaciones, selección común de modelo y múltiples comparaciones. No garantizan cobertura confirmatoria ni demuestran equivalencia poblacional.' if es else 'Intervals are nominal and exploratory, with only three initializations, common model selection and multiple comparisons. They do not guarantee confirmatory coverage or establish population equivalence.'),'',
                ('## Estructura y causalidad' if es else '## Structure and causality'),'',
                f"Selected by replicate: `{s['selected_supports_by_independent_training_rep']}`",f"Exact-degree alternatives: `{s['selected_degree_alternatives']}`",f"Route-use dependence observed: `{d['route_use_identified_in_inspected_snapshots']}`",'',
                ('La prueba causal retira una sola utilización del mensaje después de un prefijo de acciones reproducido. Cambios inmediatos de acción muestran dependencia computacional local; la pérdida posterior es un efecto total de política a través de la física nativa. No es una arista causal física ni evidencia psicológica.' if es else 'The causal assay bypasses one message use after an audited action-prefix replay. Immediate action changes show local computational dependence; subsequent loss is a total policy effect through native physics. This is not a physical causal edge or psychological evidence.'),'',
                ('## Competencia de comparadores y límites' if es else '## Comparator competence and limits'),'',
                '```json',json.dumps(u['competence'],indent=2),'```','',
                ('El presupuesto no demuestra convergencia. Superar al testigo de inacción tampoco acredita optimalidad. Si los comparadores no cumplen el diagnóstico, no se atribuye superioridad general al grafo. Los recursos de entrenamiento, el replay de SAC y las arquitecturas distintas se informan por separado.' if es else 'The budget does not establish convergence. Beating a no-action witness does not establish optimality. If comparators fail the diagnostic, no general graph superiority is inferred. Training resources, SAC replay and architecture differences are reported separately.'),'',
                ('## Estado posterior' if es else '## Subsequent status'),'',
                '`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`; `B1E_executed=false`; `final_seeds_generated=false`.','',
                ('B1-E permanece en pausa. Se requiere revisar estos resultados, la precisión y la competencia de las alternativas antes de diseñar una evaluación final nueva. No se impone un resultado favorable para considerar terminado el trabajo de desarrollo.' if es else 'B1-E remains on hold. These results, precision and alternative competence must be reviewed before designing a new final evaluation. A favorable outcome is not required to finish the development work.'),'',
                ('La taxonomía de ocho pares, B0, B1-D y los estudios históricos no se modifican. La información procede de los JSON de esta campaña; los mensajes no se denominan consciencia, ni la recompensa placer.' if es else 'The eight-pair taxonomy, B0, B1-D and historical studies are not changed. Reported information is rendered from this campaign JSON; messages are not consciousness and reward is not pleasure.')]
        dest=out/('translations/es/B1S_COMPLETION_REPORT.md' if es else 'B1S_COMPLETION_REPORT.md');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines)+'\n')


def archive_and_seal(fits: Path,out: Path):
    archive=out/'EVIDENCE.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        tar.add(fits,arcname='fits');tar.add(out/'traces',arcname='traces')
    digest=sha(archive);parts=[]
    with archive.open('rb') as f:
        i=0
        while True:
            b=f.read(4*1024*1024)
            if not b:break
            name=f'EVIDENCE.tar.gz.part{i:03d}';(out/name).write_bytes(b)
            parts.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});i+=1
    archive.unlink();shutil.rmtree(out/'traces')
    write_json(out/'EVIDENCE_ARCHIVE.json',{'format':'concatenated gzip-compressed tar, 4MiB chunks','archive_sha256':digest,'parts':parts,'contains':'all 381 fits including checkpoints, selection traces, episode/optimizer logs and all heldout/causal traces','reconstruction':'cat EVIDENCE.tar.gz.part* > EVIDENCE.tar.gz; verify archive_sha256 before extracting'})
    freeze={'status':'B1S_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS','source_commit':current_commit(),'workflow_run_id':os.environ.get('GITHUB_RUN_ID'),
            'development_only':True,'confirmatory':False,'historical_preservation':json.loads((out/'HISTORICAL_PRESERVATION.json').read_text()),
            'files':{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='B1S_FREEZE.json'},
            'final_seeds_generated':False,'B1E_executed':False,'ready_for_b1e_confirmatory_run':False}
    write_json(out/'B1S_FREEZE.json',freeze)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['qa','complete','seal']);ap.add_argument('--fits',type=Path,default=Path('fits'));ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.stage=='qa':freeze_development(args.out)
    elif args.stage=='complete':complete(args.fits,args.out)
    else:archive_and_seal(args.fits,args.out)

if __name__=='__main__':main()
