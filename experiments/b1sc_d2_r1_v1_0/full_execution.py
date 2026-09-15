"""Full prospective D2-R1 analysis pipeline.

This module reuses the frozen D2 estimands/gates but treats D2-R1 as a separate
prospective replication. Scientific entry points require the D2-R1 full
execution guard. QA helpers are pure/synthetic and never produce evidence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from b1s.execution.core import RoutingActor
from experiments.b1sc_d2_v1_0 import implementation as d2
from experiments.b1sc_d2_v1_0 import execution as d2exec
from experiments.b1sc_d2_r1_v1_0 import execution_contract as contract
from experiments.b1sc_d2_r1_v1_0 import scientific_runner as sr


def require(ok, msg):
    if not ok: raise RuntimeError(msg)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,obj): sr.write_json(path,obj)
def sha256_file(path): return sr.sha256_file(path)

def expected_snapshot(block:int)->dict:
    rows=[x for x in contract.manifest()['blocks'] if int(x['block'])==int(block)]
    require(len(rows)==1,'Missing/duplicate D2-R1 snapshot');return rows[0]

def find_fit_dir(root,condition,block):
    target=f'{condition}-b{int(block)}';found=[]
    for marker in Path(root).rglob('COMPLETE.json'):
        try:p=read_json(marker)
        except Exception:continue
        if p.get('configuration',{}).get('id')==target and marker.parent not in found:found.append(marker.parent)
    require(len(found)==1,'Cannot resolve D2-R1 fit '+target);return found[0]

def verify_fit_dir(folder,cfg):
    folder=Path(folder);c=read_json(folder/'COMPLETE.json')
    require(c['status']=='FIT_COMPLETE' and c['configuration']==cfg,'D2-R1 fit mismatch')
    exp=expected_snapshot(cfg['block']);att=c.get('initialization',{})
    require(att.get('validated') is True,'Frozen init not attested')
    require(att.get('snapshot_sha256')==exp['sha256'],'Frozen snapshot SHA mismatch')
    require(att.get('actor_digest')==exp['actor_digest'] and att.get('critic_digest')==exp['critic_digest'],'Frozen actor/critic digest mismatch')
    for n,h in c['files'].items():require((folder/n).is_file() and sha256_file(folder/n)==h,'Fit hash mismatch: '+n)
    s=read_json(folder/'STARTED.json');require(s['configuration']==cfg,'Started config mismatch');require(s['initialization']['snapshot_sha256']==exp['sha256'],'Started snapshot mismatch')
    r=read_json(folder/'TRAINING_RESOURCES.json');require(r['environment_steps']==d2.NATIVE_STEPS and r['split']=='training','Incomplete scientific budget');require(r['initialization']['snapshot_sha256']==exp['sha256'],'Training snapshot mismatch');require(r['training_lesion']==list(d2.training_lesion(cfg['condition'])),'Training lesion mismatch')
    e=read_json(folder/'ENDPOINT.json');require(e['status']=='ENDPOINT_COMPLETE' and len(e['episodes'])==d2.ENDPOINT_EPISODES,'Endpoint mismatch');require(e['initialization']['snapshot_sha256']==exp['sha256'],'Endpoint snapshot mismatch')
    for step in d2.CHECKPOINTS:
        dp=read_json(folder/f'DIAGNOSTIC_{step}.json');require(len(dp['episodes'])==d2.DIAGNOSTIC_EPISODES,'Diagnostic mismatch');require(dp['initialization']['snapshot_sha256']==exp['sha256'],'Diagnostic snapshot mismatch')
        require((folder/f'model-{step}.pt').is_file(),'Checkpoint missing')
    require((folder/'model-final.pt').is_file(),'Final model missing')
    return c

def _competence(model,witness):
    li=witness['loss']-model['loss'];dd=model['displacement']-witness['displacement']
    return {'loss_improvement':li,'additional_displacement':dd,'pass':bool(li>d2.LOSS_MARGIN and dd>=d2.DISPLACEMENT_MARGIN)}

def aggregate_training(fits_root,out):
    sr.scientific_guard();sr.runtime();out=Path(out);out.mkdir(parents=True,exist_ok=False);fits={}
    for c in d2.CONDITIONS:
        for b in range(d2.BLOCKS):
            cfg=sr.config_for(c,b);f=find_fit_dir(fits_root,c,b);verify_fit_dir(f,cfg);fits[(c,b)]=f
    by=[];las=[];das=[];pooled_on=[];pooled_off=[];diagnostics={str(s):[] for s in d2.CHECKPOINTS};witnesses={};competence={c:[] for c in d2.CONDITIONS}
    for b in range(d2.BLOCKS):
        onr=read_json(fits[('S6-ON',b)]/'TRAINING_RESOURCES.json');offr=read_json(fits[('S6-OFF-TRAIN',b)]/'TRAINING_RESOURCES.json')
        require(onr['initialization']['snapshot_sha256']==offr['initialization']['snapshot_sha256'],'Paired frozen snapshot mismatch')
        require(onr['initial_actor_digest']==offr['initial_actor_digest'] and onr['initial_critic_digest']==offr['initial_critic_digest'],'Paired initialization digest mismatch')
        one=read_json(fits[('S6-ON',b)]/'ENDPOINT.json');offe=read_json(fits[('S6-OFF-TRAIN',b)]/'ENDPOINT.json');ons,offs=one['summary'],offe['summary'];pooled_on+=one['episodes'];pooled_off+=offe['episodes']
        la=offs['loss']-ons['loss'];da=ons['displacement']-offs['displacement'];pr=d2.practical_advantage(la,da);las.append(la);das.append(da);by.append(dict(block=b,on=ons,off=offs,loss_advantage=la,displacement_advantage=da,practical_advantage=pr,snapshot_sha256=onr['initialization']['snapshot_sha256']))
        wr=d2exec.evaluate_no_action(b);ws=d2exec.summarize(wr);witnesses[str(b)]=ws;competence['S6-ON'].append(dict(block=b,**_competence(ons,ws)));competence['S6-OFF-TRAIN'].append(dict(block=b,**_competence(offs,ws)))
        for s in d2.CHECKPOINTS:
            od=read_json(fits[('S6-ON',b)]/f'DIAGNOSTIC_{s}.json')['summary'];fd=read_json(fits[('S6-OFF-TRAIN',b)]/f'DIAGNOSTIC_{s}.json')['summary'];diagnostics[str(s)].append(dict(block=b,on=od,off=fd,loss_advantage=fd['loss']-od['loss'],displacement_advantage=od['displacement']-fd['displacement']))
    pon,pof=d2exec.summarize(pooled_on),d2exec.summarize(pooled_off);ala=pof['loss']-pon['loss'];ada=pon['displacement']-pof['displacement'];ap=d2.practical_advantage(ala,ada);gate=d2.training_gate([x['practical_advantage'] for x in by],ap)
    result=dict(status='D2_R1_TRAINING_EFFECT_COMPLETE',H_TRAIN_D2_R1=bool(gate['H_TRAIN_D2']),aggregate=dict(on=pon,off=pof,loss_advantage=ala,displacement_advantage=ada,practical_advantage=ap),gate=gate,by_block=by,loss_advantage_bootstrap=d2exec._bootstrap(las,'d2-r1-training-loss'),displacement_advantage_bootstrap=d2exec._bootstrap(das,'d2-r1-training-displacement'),checkpoint_diagnostics=diagnostics,competence_vs_no_action=competence,witness_by_block=witnesses,fits_verified=d2.FITS,replication_of='B1-SC-D2-v1.0',does_not_rewrite_d2=True,**d2.BOUNDARY)
    write_json(out/'TRAINING_EFFECT_R1.json',result);return result

def load_actor(fit_dir,condition):
    fit_dir=Path(fit_dir);c=read_json(fit_dir/'COMPLETE.json')['configuration'];cfg=sr.config_for(condition,c['block']);verify_fit_dir(fit_dir,cfg);p=torch.load(fit_dir/'model-final.pt',map_location='cpu');require(p['configuration']==cfg,'Final model config mismatch');exp=expected_snapshot(cfg['block']);require(p['initialization']['snapshot_sha256']==exp['sha256'],'Final model initialization mismatch');a=RoutingActor(d2.S6_MASK);a.load_state_dict(p['actor_state']);a.eval();return a

def online_block(block,fits_root,out):
    sr.scientific_guard();sr.runtime();cfg=sr.config_for('S6-ON',block);f=find_fit_dir(fits_root,'S6-ON',block);verify_fit_dir(f,cfg);actor=load_actor(f,'S6-ON');out=Path(out);out.mkdir(parents=True,exist_ok=False)
    modes=('intact','permanent_all_active_lesion','sham','inactive_edge_control','window_9_40_all_active_lesion');rows={m:d2exec.evaluate_actor(actor,'S6-ON',block,'causal',d2.CAUSAL_EPISODES,m) for m in modes};ih=[x['trace_sha256'] for x in rows['intact']];sham=ih==[x['trace_sha256'] for x in rows['sham']];inactive=ih==[x['trace_sha256'] for x in rows['inactive_edge_control']];require(sham and inactive,'Causal control diverged')
    s={k:d2exec.summarize(v) for k,v in rows.items()};lh=s['permanent_all_active_lesion']['loss']-s['intact']['loss'];dh=s['intact']['displacement']-s['permanent_all_active_lesion']['displacement'];wlh=s['window_9_40_all_active_lesion']['loss']-s['intact']['loss'];wdh=s['intact']['displacement']-s['window_9_40_all_active_lesion']['displacement']
    result=dict(status='D2_R1_ONLINE_BLOCK_COMPLETE',block=int(block),snapshot_sha256=expected_snapshot(block)['sha256'],summaries=s,permanent=dict(loss_harm=lh,displacement_harm=dh,practical_harm=d2.practical_harm(lh,dh)),window_9_40=dict(loss_harm=wlh,displacement_harm=wdh,practical_harm=d2.practical_harm(wlh,wdh),primary_gate=False),controls=dict(sham_exact=sham,inactive_edge_exact=inactive),episodes=rows,does_not_rewrite_d2=True,**d2.BOUNDARY);write_json(out/'ONLINE_BLOCK_R1.json',result);return result

def find_online_dir(root,block):
    found=[]
    for m in Path(root).rglob('ONLINE_BLOCK_R1.json'):
        try:p=read_json(m)
        except Exception:continue
        if p.get('block')==int(block) and m.parent not in found:found.append(m.parent)
    require(len(found)==1,'Cannot resolve D2-R1 online block');return found[0]

def aggregate_online(online_root,out):
    sr.scientific_guard();out=Path(out);out.mkdir(parents=True,exist_ok=False);blocks=[]
    for b in range(d2.BLOCKS):
        r=read_json(find_online_dir(online_root,b)/'ONLINE_BLOCK_R1.json');require(r['status']=='D2_R1_ONLINE_BLOCK_COMPLETE','Incomplete online block');require(r['snapshot_sha256']==expected_snapshot(b)['sha256'],'Online snapshot provenance mismatch');blocks.append(r)
    controls=all(x['controls']['sham_exact'] and x['controls']['inactive_edge_exact'] for x in blocks);intact=[e for b in blocks for e in b['episodes']['intact']];perm=[e for b in blocks for e in b['episodes']['permanent_all_active_lesion']];win=[e for b in blocks for e in b['episodes']['window_9_40_all_active_lesion']]
    si,sp,sw=d2exec.summarize(intact),d2exec.summarize(perm),d2exec.summarize(win);lh=sp['loss']-si['loss'];dh=si['displacement']-sp['displacement'];pr=d2.practical_harm(lh,dh);gate=d2.online_gate([b['permanent']['practical_harm'] for b in blocks],pr,controls);wlh=sw['loss']-si['loss'];wdh=si['displacement']-sw['displacement']
    result=dict(status='D2_R1_ONLINE_EFFECT_COMPLETE',H_ONLINE_D2_R1=bool(gate['H_ONLINE_D2']),aggregate=dict(intact=si,permanent_lesion=sp,loss_harm=lh,displacement_harm=dh,practical_harm=pr),gate=gate,by_block=[dict(block=b['block'],permanent=b['permanent'],controls=b['controls'],snapshot_sha256=b['snapshot_sha256']) for b in blocks],loss_harm_bootstrap=d2exec._bootstrap([b['permanent']['loss_harm'] for b in blocks],'d2-r1-online-loss'),displacement_harm_bootstrap=d2exec._bootstrap([b['permanent']['displacement_harm'] for b in blocks],'d2-r1-online-disp'),window_9_40_replication=dict(intact=si,lesion=sw,loss_harm=wlh,displacement_harm=wdh,practical_harm=d2.practical_harm(wlh,wdh),blocks_with_practical_harm=sum(bool(b['window_9_40']['practical_harm']) for b in blocks),primary_gate=False),controls_pass=controls,replication_of='B1-SC-D2-v1.0',does_not_rewrite_d2=True,**d2.BOUNDARY);write_json(out/'ONLINE_EFFECT_R1.json',result);return result

def final_aggregate(training_file,online_file,out):
    sr.scientific_guard();out=Path(out);out.mkdir(parents=True,exist_ok=False);t=read_json(training_file);o=read_json(online_file);integrity=t.get('status')=='D2_R1_TRAINING_EFFECT_COMPLETE' and t.get('fits_verified')==d2.FITS and o.get('status')=='D2_R1_ONLINE_EFFECT_COMPLETE' and o.get('controls_pass') is True;adjud=d2.adjudicate(bool(t.get('H_TRAIN_D2_R1')),bool(o.get('H_ONLINE_D2_R1')),integrity)
    result=dict(status='D2_R1_DEVELOPMENT_COMPLETE_FOR_REVIEW',integrity_pass=bool(integrity),H_TRAIN_D2_R1=bool(t.get('H_TRAIN_D2_R1')),H_ONLINE_D2_R1=bool(o.get('H_ONLINE_D2_R1')),mechanistic_adjudication=adjud,training_effect=t,online_effect=o,replication_of='B1-SC-D2-v1.0',does_not_rewrite_d2=True,automatic_b1e_progression=False,**d2.BOUNDARY);write_json(out/'SUMMARY_R1.json',result);return result

# QA-only pure fixtures.  They exercise frozen gate/adjudication logic without
# reading scientific artifacts or running environments.
def qa_training_gate(block_flags,aggregate):
    return d2.training_gate([bool(x) for x in block_flags],bool(aggregate))
def qa_online_gate(block_flags,aggregate,controls):
    return d2.online_gate([bool(x) for x in block_flags],bool(aggregate),bool(controls))
def qa_adjudication_matrix():
    return {(a,b):d2.adjudicate(a,b,True) for a in (False,True) for b in (False,True)}
