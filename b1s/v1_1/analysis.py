"""Development analyses with source locks, fresh refits and explicit non-claims."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import itertools
import json
import os
import shutil
import tarfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from b1s.execution.core import model_digest
from b1s.execution.instrument import NativeEnv, runtime_lock, qa as legacy_qa, array_hash
from .design import ROOT, BASE, EDGES, DELTA, plan, sha, write, seed, commit, preservation, screening_registry, competition_registry, stability, interval, self_test
from .train import load, predict, evaluate


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def read(p):return json.loads(Path(p).read_text())

def canonical_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_digest(env):
    p=env.physics
    bodies=[dict(position=list(b.position),linear_velocity=list(b.linearVelocity),angle=b.angle,angular_velocity=b.angularVelocity,awake=b.awake,active=b.active) for b in p.world.bodies]
    joints=[[dict(angle=j.angle,speed=j.speed,motor_speed=j.motorSpeed) for j in w.joints] if w.hull is not None else [] for w in p.walkers]
    state=dict(bodies=bodies,joints=joints,fallen=p.fallen_walkers.tolist(),game_over=bool(p.game_over),frames=p.frames,previous_shaping=p.prev_shaping.tolist(),previous_package_shaping=float(p.prev_package_shaping),rng=p.np_random.bit_generator.state,walker_rng=[w.np_random.bit_generator.state for w in p.walkers])
    return canonical_hash(state)


def preflight(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
    checks=self_test();q=legacy_qa(out/'instrument')
    # Analytic vector GAE check: every stream is computed separately; terminals cut return propagation.
    rewards=np.array([[1.,2.],[3.,4.],[5.,6.]]);dones=np.array([[False,False],[True,False],[False,True]])
    values=np.ones((3,2));last=np.array([.5,.5]);ga=np.zeros(2);vector=np.zeros((3,2))
    for t in reversed(range(3)):
        nt=1-dones[t];nv=last if t==2 else values[t+1]
        ga=rewards[t]+.99*nv*nt-values[t]+.99*.95*nt*ga;vector[t]=ga
    for stream in range(2):
        g=0.
        for t in reversed(range(3)):
            nt=1-float(dones[t,stream]);nv=last[stream] if t==2 else values[t+1,stream]
            g=rewards[t,stream]+.99*nv*nt-values[t,stream]+.99*.95*nt*g
            assert abs(g-vector[t,stream])<1e-12
    e1=NativeEnv();e2=NativeEnv()
    try:
        e1.reset(seed=seed('qa','physical-replay'));e2.reset(seed=seed('qa','physical-replay'))
        for _ in range(8):
            assert physical_digest(e1)==physical_digest(e2)
            a=np.zeros(12,dtype=np.float32);x=e1.step(a);y=e2.step(a)
            assert np.array_equal(x[0],y[0]) and x[1:4]==y[1:4]
            if x[2] or x[3]:break
    finally:e1.close();e2.close()
    audit=dict(status='PASS',design=checks,instrument_status=q['status'],vector_gae='PASS',native_physical_prefix='PASS on QA states',history=preservation(),runtime=runtime_lock())
    write(out/'QA.json',audit)
    reg=screening_registry()
    freeze=dict(status='FROZEN_FOR_DEVELOPMENT_ONLY',source_commit=commit(),utc_before_training=now(),plan=plan(),registry=reg,registry_sha256=canonical_hash(reg),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'results' not in p.parts},shared_source_hashes={n:sha(ROOT.parent/'execution'/n) for n in ['core.py','instrument.py','requirements.txt']},runtime=runtime_lock(),QA_sha256=sha(out/'QA.json'),B1E_executed=False,final_seeds_generated=False)
    write(out/'DEVELOPMENT_FREEZE.json',freeze);return audit


def verify_fit(folder,cfg):
    start=read(folder/'STARTED.json');complete=read(folder/'COMPLETE.json')
    assert start['configuration']==cfg and start['source_commit']==commit()
    assert start['plan_sha256']==sha(ROOT/'PLAN.json')
    assert complete['status']=='COMPLETE'
    for rel,h in complete['files'].items():assert sha(folder/rel)==h,(folder,rel)
    resources=read(folder/'TRAINING_RESOURCES.json');assert resources['environment_steps']==cfg['steps']
    assert resources['initial_actor_digest']!=resources['final_actor_digest']
    row=dict(configuration=cfg,resources=resources,checkpoint_sha256=sha(folder/('model.zip' if cfg['kind']=='sac' else 'model.pt')))
    if cfg['phase']!='competitive':
        sel=read(folder/'SELECTION.json');count=plan()['discovery_selection_episodes'] if cfg['phase']=='discovery' else plan()['calibration_selection_episodes']
        assert len(sel['rows'])==count
        assert [r['seed'] for r in sel['rows']]==[seed(cfg['phase']+'-select','environment',cfg['rep'],ep) for ep in range(count)]
        row['selection_mean_loss']=sel['mean_loss'];row['selection_rows']=sel['rows']
    return row


def audit_fits(fits,registry,outfile):
    rows=[];failures=[]
    for cfg in registry:
        try:rows.append(verify_fit(fits/cfg['id'],cfg))
        except Exception:failures.append(dict(fit=cfg['id'],error=traceback.format_exc()))
    actual={p.parent.name for p in fits.rglob('STARTED.json')};expected={x['id'] for x in registry}
    if actual!=expected:failures.append(dict(unexpected_or_missing=sorted(actual^expected)))
    audit=dict(status='FAIL' if failures else 'PASS',expected=len(registry),verified=len(rows),problems=failures,source_commit=commit())
    write(outfile,audit)
    if failures:raise RuntimeError('Fit audit failed; no seed substitution or aggregate selection')
    return rows


def select(fits,out,qa):
    out.mkdir(parents=True,exist_ok=True);rows=audit_fits(fits,screening_registry(),out/'SCREENING_FIT_AUDIT.json')
    frozen=read(qa/'DEVELOPMENT_FREEZE.json');assert frozen['source_commit']==commit()
    scores={}
    for row in rows:
        cfg=row['configuration'];key='graph-'+cfg['mask'] if cfg['kind']=='graph' else cfg['hp']['id']
        scores.setdefault(key,[]).append(row['selection_mean_loss'])
    score={k:float(np.mean(v)) for k,v in scores.items()}
    from .design import sparse_masks
    chosen=min(sparse_masks(),key=lambda m:(score['graph-'+m],m.count('1'),m))
    hp={}
    for kind in ['ppo','sac']:
        candidates=[r['configuration']['hp'] for r in rows if r['configuration']['kind']==kind]
        hp[kind]=min(candidates,key=lambda c:(score[c['id']],c['id']))
    winners=[]
    for rep in range(6):
        subset=[r for r in rows if r['configuration']['kind']=='graph' and r['configuration']['rep']==rep and r['configuration']['mask']!='111111']
        winners.append(min(subset,key=lambda r:(r['selection_mean_loss'],r['configuration']['mask'].count('1'),r['configuration']['mask']))['configuration']['mask'])
    selected=dict(status='SELECTED_ON_DEVELOPMENT_ONLY',source_commit=commit(),utc_before_competitive_training=now(),selected_support=chosen,selected_edges=[list(e) for e,b in zip(EDGES,chosen) if b=='1'],ppo_hp=hp['ppo'],sac_hp=hp['sac'],scores=score,winners_by_discovery_rep=winners,development_freeze_sha256=sha(qa/'DEVELOPMENT_FREEZE.json'),competitive_training_started=False,heldout_accessed=False,causal_accessed=False,B1E_executed=False)
    write(out/'SELECTION_FREEZE.json',selected)
    write(out/'STABILITY.json',stability(winners));write(out/'DISCOVERY_RESULTS.json',dict(rows=rows,scores=score,development_only=True))
    curves={}
    for r in rows:
        cfg=r['configuration']
        if cfg['phase']=='calibration':curves[cfg['id']]=read(fits/cfg['id']/'CHECKPOINT_EVALUATIONS.json')
    write(out/'BASELINE_CALIBRATION.json',dict(terminal_only_selection=True,checkpoint_curves=curves,selected_ppo=hp['ppo'],selected_sac=hp['sac'],selection_independent_of_graph_scores=True,scope='Checkpoint curves are development evidence; terminal checkpoint alone selects. Competitive refits use independent new initializations.'))
    return selected


def causal(actor,rep,traces):
    active=[i for i,b in enumerate(actor.mask) if b=='1'];rows=[]
    if not active:return dict(rep=rep,status='NO_ACTIVE_ROUTES',rows=[],weights_unchanged=True)
    variants={'intact':(),'sham':()};variants.update({f'edge-{e}':(e,) for e in active})
    inactive=[e for e in range(6) if e not in active]
    if inactive:variants['off-support']=(inactive[0],)
    if len(active)>=2:variants['two-edge']=(active[0],active[1])
    digest=model_digest(actor)
    for ep in range(plan()['causal_episodes']):
        sd=seed('causal','environment',rep,ep);env=NativeEnv();z,_=env.reset(seed=sd);prefix=[];rr=[];ended=False
        for _ in range(plan()['causal_prefix_cycles']):
            a=predict(actor,z);prefix.append(a.copy());z,r,t,tr,inf=env.step(a);rr.append(r)
            if t or tr:ended=True;break
        state=physical_digest(env);target=z.copy();env.close()
        if ended:
            rows.append(dict(episode=ep,seed=sd,status='PREFIX_UNAVAILABLE_RETAINED',cycles=len(prefix)));continue
        vr={}
        for label,lesion in variants.items():
            env=NativeEnv();z,_=env.reset(seed=sd);ret=0.
            try:
                for a,expected in zip(prefix,rr):
                    z,r,t,tr,inf=env.step(a);ret+=r;assert r==expected and not(t or tr)
                assert np.array_equal(z,target) and physical_digest(env)==state
                with torch.no_grad():
                    _,_,snap=actor(torch.from_numpy(z[None]),lesion,True)
                snap={k:v.detach().numpy().tolist() for k,v in snap.items()}
                p=traces/f'causal-r{rep}-e{ep}-{label}.jsonl.gz';first=True
                with gzip.open(p,'wt',encoding='utf-8') as f:
                    while True:
                        a=predict(actor,z,lesion if first else ());nxt,r,t,tr,inf=env.step(a);ret+=r
                        f.write(json.dumps(dict(rep=rep,episode=ep,seed=sd,cycle=env.t,z=z.tolist(),action=a.tolist(),reward=r,next_z=nxt.tolist(),audit=inf),sort_keys=True,separators=(',',':'))+'\n')
                        first=False;z=nxt
                        if t or tr:break
                vr[label]=dict(loss=-ret,snapshot=snap,cycles=env.t,displacement=inf['package_displacement'],trace=p.name,trace_sha256=sha(p))
            finally:env.close()
        ref=vr['intact'];assert vr['sham']['loss']==ref['loss'] and vr['sham']['snapshot']==ref['snapshot']
        if inactive:assert vr['off-support']['loss']==ref['loss']
        effects={}
        for label,v in vr.items():
            assert v['snapshot']['h']==ref['snapshot']['h'] and v['snapshot']['m']==ref['snapshot']['m']
            effects[label]=dict(bypass_minus_intact_loss=v['loss']-ref['loss'],max_action_delta=float(np.abs(np.array(v['snapshot']['action'])-np.array(ref['snapshot']['action'])).max()))
        factorial=vr['two-edge']['loss']-vr[f'edge-{active[0]}']['loss']-vr[f'edge-{active[1]}']['loss']+ref['loss'] if len(active)>=2 else None
        rows.append(dict(episode=ep,seed=sd,status='COMPLETE',physical_prefix_sha256=state,prefix_actions=[a.tolist() for a in prefix],prefix_rewards=rr,variants=vr,effects=effects,two_edge_interaction=factorial))
        assert model_digest(actor)==digest
    return dict(rep=rep,status='COMPLETE',rows=rows,weights_unchanged=True,estimand='One-time route-use intervention followed by intact policy; external difference is total policy effect, not a physical causal coefficient.')


def action_agreement(actors,rep,traces):
    panel=[]
    for label in ['no-action','S-M3']:
        env=NativeEnv()
        try:
            for ep in range(4):
                z,_=env.reset(seed=seed('probe',label,rep,ep))
                for t in range(64):
                    panel.append(z.copy());a=np.zeros(12,dtype=np.float32) if label=='no-action' else predict(actors[label],z)
                    z,r,term,trunc,info=env.step(a)
                    if term or trunc:break
        finally:env.close()
    z=np.asarray(panel,dtype=np.float32);np.savez_compressed(traces/f'probe-r{rep}.npz',observations=z)
    actions={role:np.stack([predict(actor,row) for row in z]) for role,actor in actors.items()}
    rows=[]
    for a,b in itertools.combinations(actions,2):
        d=np.abs(actions[a]-actions[b]);rows.append(dict(a=a,b=b,mean_abs=float(d.mean()),rms=float(np.sqrt((d*d).mean())),max_abs=float(d.max()),within_predeclared_probe_margins=bool(d.mean()<=.01 and d.max()<=.05)))
    return dict(rep=rep,observations=len(z),panel_sha256=sha(traces/f'probe-r{rep}.npz'),pairs=rows,scope='Finite common reachable observation panel only. Neither a policy-equivalence theorem nor a population equivalence test.')


def summarize(rows):
    return dict(mean_loss=float(np.mean([x['loss'] for x in rows])),mean_displacement=float(np.mean([x['displacement'] for x in rows])),fall_episode_fraction=float(np.mean([x['fallen_count']>0 for x in rows])),mean_cycles=float(np.mean([x['cycles'] for x in rows])),episodes=len(rows))


def complete(fits,selected_dir,out,qa):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
    for src in selected_dir.iterdir():
        if src.is_file():shutil.copy2(src,out/src.name)
    selected=read(out/'SELECTION_FREEZE.json');rows=audit_fits(fits,competition_registry(selected),out/'COMPETITIVE_FIT_AUDIT.json')
    traces=out/'traces';traces.mkdir(exist_ok=True)
    write(out/'EVALUATION_STARTED.json',dict(source_commit=commit(),utc=now(),selection_freeze_sha256=sha(out/'SELECTION_FREEZE.json'),B1E_executed=False))
    held={role:[] for role in ['S-M0','S-M2-fixed','S-M3','S-M4','S-M5','S-M6']};held['no-action']=[];causals=[];probes=[]
    for rep in range(6):
        actors={role:load(fits/f'competitive-{role}-r{rep}') for role in held if role!='no-action'}
        for role,actor in actors.items():
            held[role]+=evaluate(actor,rep,'competitive-eval',plan()['competitive_episodes'],traces/f'heldout-{role}-r{rep}.jsonl.gz',role)
        held['no-action']+=evaluate(None,rep,'competitive-eval',plan()['competitive_episodes'],traces/f'heldout-no-action-r{rep}.jsonl.gz','no-action')
        causals.append(causal(actors['S-M3'],rep,traces));probes.append(action_agreement(actors,rep,traces))
        print('Competitive, causal and probe evaluation complete for replicate',rep,flush=True)
    summaries={k:summarize(v) for k,v in held.items()};contrasts={}
    for role in held:
        if role in ['S-M3','no-action']:continue
        diffs=[]
        for rep in range(6):
            lhs=[x for x in held['S-M3'] if x['rep']==rep];rhs=[x for x in held[role] if x['rep']==rep]
            assert [x['seed'] for x in lhs]==[x['seed'] for x in rhs]
            diffs.append(float(np.mean([a['loss']-b['loss'] for a,b in zip(lhs,rhs)])))
        contrasts[role]=interval(diffs)
    zero=summaries['no-action'];competence={}
    for role,s in summaries.items():
        if role=='no-action':continue
        di=zero['mean_loss']-s['mean_loss'];dx=s['mean_displacement']-zero['mean_displacement']
        competence[role]=dict(loss_improvement_over_no_action=di,additional_displacement=dx,descriptive_competence_pass=bool(di>DELTA and dx>=1),convergence_or_optimality_certified=False)
    effects=[v for cr in causals for row in cr['rows'] for k,v in row.get('effects',{}).items() if k.startswith('edge-')]
    used=any(x['max_action_delta']>2e-7 for x in effects)
    causal_by_rep=[]
    for cr in causals:
        e=[v['bypass_minus_intact_loss'] for row in cr['rows'] for k,v in row.get('effects',{}).items() if k.startswith('edge-')]
        causal_by_rep.append(float(np.mean(e)) if e else None)
    st=read(out/'STABILITY.json');strong=all(competence[x]['descriptive_competence_pass'] for x in ['S-M5','S-M6'])
    d=dict(status='B1S_V1_1_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS',source_commit=commit(),selected_support=selected['selected_support'],stability_diagnostic_pass=st['engineering_stability_diagnostic'],generic_competence_diagnostics_pass=strong,route_use_identified=used,causal_external_difference_by_rep=causal_by_rep,baseline_scope='Competence is assessed independently; no superiority inference against an incompetent baseline',ready_for_scientific_review=True,ready_for_b1e_freeze=False,ready_for_b1e_confirmatory_run=False,B1E='ON_HOLD_PENDING_REVIEW',B1E_executed=False,final_seeds_generated=False,H_CAT='NOT_EVALUABLE',H_TRANSFER='NOT_EVALUATED',development_only=True,confirmatory=False,no_v1_results_replaced=True,stages={k:'EXECUTED' for k in ['QA','screening','stability','baseline_calibration','selection_freeze','fresh_competition','causal','functional_probes','decision']},limitations=['Six discovery and six fresh competitive initializations: limited precision','Longer finite budgets do not establish convergence','PPO uses eight synchronous native streams; SAC uses one, with update and memory differences disclosed','Different hypothesis search spaces and model sizes; not universal equal capacity','Finite-panel action agreement is not global equivalence','Neither catalogue correspondence nor independent-family transfer is evaluated'])
    if not effects:d['stages']['causal']='NO_ACTIVE_OR_REACHABLE_ROUTE; see retained cases'
    write(out/'COMPETITIVE_RESULTS.json',dict(summaries=summaries,contrasts=contrasts,competence=competence,episodes=held,development_only=True))
    write(out/'CAUSAL_RESULTS.json',dict(replicates=causals,external_difference_by_rep=causal_by_rep,development_only=True));write(out/'FUNCTIONAL_PROBES.json',dict(replicates=probes))
    screen=read(out/'DISCOVERY_RESULTS.json')['rows'];allrows=screen+rows
    write(out/'RESOURCE_AUDIT.json',dict(status='AUDITED_WITH_EXPLICIT_DIFFERENCES',training_native_steps=sum(x['resources']['environment_steps'] for x in allrows),fit_count=len(allrows),sum_training_wall_seconds=float(sum(x['resources']['training_seconds'] for x in allrows)),fits=[dict(id=x['configuration']['id'],**x['resources']) for x in allrows],axes=dict(information='same unnormalized raw93',feasibility='native12 ports and fixed physics',learning='common competitive524288 steps; different updates, streams and searches',memory='no temporal actor memory; SAC replay and target critics differ',planning='one graph-message round vs different nongraph feedforward networks',decision_compute='same all-six-message computation across graph masks; generic/dense comparisons retain disclosed parameter and operator differences'),same_count_is_not_same_capacity=True))
    write(out/'DECISION.json',d);write(out/'HISTORICAL_PRESERVATION.json',preservation());render(out)
    return d


def render(out):
    d=read(out/'DECISION.json');s=read(out/'STABILITY.json');u=read(out/'COMPETITIVE_RESULTS.json');r=read(out/'RESOURCE_AUDIT.json')
    for es in [False,True]:
        title='# B1-S v1.1 — Estabilidad y competencia de comparadores' if es else '# B1-S v1.1 — Structural stability and baseline competence'
        lines=[title,'','**'+d['status']+'**','',('Campaña de desarrollo; no es confirmación ni ejecución B1-E.' if es else 'Development campaign; neither confirmation nor B1-E execution.'),'',('Commit científico: ' if es else 'Scientific commit: ')+f'`{commit()}`',('Soporte seleccionado: ' if es else 'Selected support: ')+f"`{d['selected_support']}`",('Entrenamientos y pasos nativos: ' if es else 'Fits and native training steps: ')+f"{r['fit_count']} / {r['training_native_steps']}",'',('## Estabilidad' if es else '## Stability'),'','```json',json.dumps(s,ensure_ascii=False,indent=2),'```','',('Los umbrales son diagnósticos prospectivos de ingeniería; seis selecciones no establecen estabilidad poblacional ni un catálogo universal.' if es else 'Thresholds are prospective engineering diagnostics; six selections do not establish population stability or a universal catalogue.'),'',('## Comparación externa reservada de desarrollo' if es else '## Held-out external development comparison'),'','| '+('Comparador | Pérdida media | Desplazamiento | Fracción con caída' if es else 'Comparator | Mean loss | Displacement | Fall fraction')+' |','|---|---:|---:|---:|']
        for k,v in u['summaries'].items():lines.append(f"| {k} | {v['mean_loss']:.9g} | {v['mean_displacement']:.9g} | {v['fall_episode_fraction']:.9g} |")
        lines+=['','| '+('Contraste S-M3 menos alternativa | Media | Intervalo nominal | Clase exploratoria' if es else 'S-M3 minus alternative | Mean | Nominal interval | Exploratory class')+' |','|---|---:|---|---|']
        for k,v in u['contrasts'].items():lines.append(f"| {k} | {v['mean']:.9g} | {v['nominal_interval']} | {v['classification']} |")
        lines+=['',('Los intervalos corresponden a seis medias por entrenamiento independiente, no a pasos ni episodios independientes; no están corregidos para una conclusión confirmatoria familiar.' if es else 'Intervals summarize six independent-training means, not independent time steps or episodes; they are not adjusted for confirmatory familywise conclusions.'),'',('## Competencia y causalidad' if es else '## Competence and causality'),'','```json',json.dumps(u['competence'],ensure_ascii=False,indent=2),'```','',('Uso de ruta observado: ' if es else 'Observed route use: ')+str(d['route_use_identified']),('Diferencia externa de lesión por réplica: ' if es else 'External lesion difference by replicate: ')+str(d['causal_external_difference_by_rep']),'',('Un cambio inmediato de acción prueba dependencia computacional en el estado inspeccionado, no utilidad universal, polaridad psicológica o consciencia. La discrepancia posterior de pérdida es un efecto total de política. Se preservan prefijos no disponibles y resultados adversos.' if es else 'An immediate action change identifies computational dependence at the inspected state, not universal utility, psychological polarity or consciousness. Subsequent loss differences are total policy effects. Unavailable prefixes and adverse outcomes are retained.'),'',('## Procedencia, traducción y límites' if es else '## Provenance, translation and limits'),'','`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`; `B1E_executed=false`; `final_seeds_generated=false`.','',('Se conserva B1-S v1. La nueva selección y las inicializaciones competitivas están separadas. No se reajusta el modelo tras abrir los datos reservados. Los JSON y esta traducción comparten cifras, identificadores y decisiones; las claves técnicas permanecen sin traducir para conservar trazabilidad.' if es else 'B1-S v1 is preserved. Selection and competitive initializations are separated. No model is retuned after held-out data are opened. JSON and this translation share numbers, identifiers and decisions; technical keys remain untranslated for traceability.'),'',('Un diagnóstico de competencia fallido o una estructura inestable no se corrigen alterando la tarea. B1-E permanece en pausa hasta una decisión posterior explícita. Los presupuestos finitos, algoritmos y recursos diferentes limitan los claims; no se declara superioridad general.' if es else 'A failed competence diagnostic or unstable support is not repaired by changing the task. B1-E remains on hold pending a separate explicit decision. Finite budgets and differing algorithms/resources limit claims; no general superiority is declared.')]
        p=out/('translations/es/COMPLETION_REPORT.md' if es else 'COMPLETION_REPORT.md');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def seal(out,screen=None,fits=None):
    archive=out/'EVIDENCE.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=5) as tar:
        for path,name in [(screen,'screening'),(fits,'competitive'),(out/'traces','traces')]:
            if path is not None and path.exists():tar.add(path,arcname=name)
    total_sha=sha(archive);parts=[]
    with archive.open('rb') as f:
        i=0
        while True:
            b=f.read(4*1024*1024)
            if not b:break
            p=out/f'EVIDENCE.tar.gz.part{i:03d}';p.write_bytes(b);parts.append(dict(path=p.name,bytes=len(b),sha256=sha(p)));i+=1
    archive.unlink()
    if (out/'traces').exists():shutil.rmtree(out/'traces')
    write(out/'EVIDENCE_ARCHIVE.json',dict(parts=parts,archive_sha256=total_sha,reconstruction='Concatenate parts in lexical order, verify SHA-256, then extract',scope='Retained attempts, checkpoints and full evaluation traces; not all raw training observations or SAC replay'))
    d=read(out/'DECISION.json')
    freeze=dict(status=d['status'],source_commit=commit(),workflow_run_id=os.getenv('GITHUB_RUN_ID'),base_code_commit=BASE,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='FREEZE.json'},historical_preservation=preservation(),development_only=True,confirmatory=False,final_seeds_generated=False,B1E_executed=False,ready_for_b1e_confirmatory_run=False)
    write(out/'FREEZE.json',freeze)


def finish(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True);success=False
    if args.qa and args.qa.exists():shutil.copytree(args.qa,out/'qa',dirs_exist_ok=True)
    try:complete(args.fits,args.selection,out,args.qa);success=True
    except Exception:
        error=traceback.format_exc();write(out/'DECISION.json',dict(status='B1S_V1_1_BLOCKED',source_commit=commit(),incident=error,development_only=True,confirmatory=False,H_CAT='NOT_EVALUABLE',H_TRANSFER='NOT_EVALUATED',final_seeds_generated=False,B1E_executed=False,ready_for_b1e_freeze=False,ready_for_b1e_confirmatory_run=False))
        write(out/'HISTORICAL_PRESERVATION.json',preservation())
        for es in [False,True]:
            p=out/('translations/es/COMPLETION_REPORT.md' if es else 'COMPLETION_REPORT.md');p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(('# B1-S v1.1 — Ejecución bloqueada\n\nSe conservan intentos e incidente. No se sustituyen semillas ni se afirma ejecución completa. B1-E no fue ejecutado.\n\n' if es else '# B1-S v1.1 — Blocked execution\n\nAttempts and incident are retained. No seeds are replaced and completion is not claimed. B1-E was not executed.\n\n')+'```text\n'+error+'\n```\n')
        print(error,flush=True)
    seal(out,args.screen,args.fits)
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('scientific_success='+str(success).lower()+'\n')
    return success


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','select','finish']);p.add_argument('--out',type=Path,required=True);p.add_argument('--fits',type=Path);p.add_argument('--screen',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--qa',type=Path);a=p.parse_args()
    if a.stage=='preflight':print(json.dumps(preflight(a.out),indent=2))
    elif a.stage=='select':print(json.dumps(select(a.fits,a.out,a.qa),indent=2))
    else:finish(a)

if __name__=='__main__':main()
