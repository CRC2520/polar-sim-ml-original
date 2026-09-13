"""Publication finalizer. No retraining, seed replacement or favorable-result filter."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import traceback
from pathlib import Path
from .analysis import complete, render, seal, read
from .design import write, sha, commit, preservation


def finalize(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    for source,name in [(args.qa,'qa'),(args.selection,'selection_attempt')]:
        if source is not None and source.exists():shutil.copytree(source,out/name,dirs_exist_ok=True)
    success=False
    try:
        complete(args.fits,args.selection,out,args.qa)
        selection=read(out/'SELECTION_FREEZE.json');results=read(out/'COMPETITIVE_RESULTS.json');identity=[]
        for role,mask in [('S-M0','000000'),('S-M2-fixed','011001'),('S-M4','111111')]:
            if selection['selected_support']==mask:
                assert all(abs(x)<1e-12 for x in results['contrasts'][role]['by_training_rep'])
                results['contrasts'][role]['classification']='IDENTITY_CONTROL_NOT_INDEPENDENT_EVIDENCE'
                results['contrasts'][role]['scope']='Identical actor/support/training configuration with common initialization and episode identities; not independent evidence of equivalence or usefulness.'
                identity.append(role)
        write(out/'COMPETITIVE_RESULTS.json',results)
        decision=read(out/'DECISION.json');decision['identity_control_roles']=identity
        decision['unfavorable_findings_are_not_technical_failures']=True
        write(out/'DECISION.json',decision)
        resources=read(out/'RESOURCE_AUDIT.json')
        assert resources['fit_count']==312 and resources['training_native_steps']==62128128
        render(out)
        write(out/'TRANSLATION_AUDIT.json',dict(status='RENDERED_FROM_COMMON_JSON',english_sha256=sha(out/'COMPLETION_REPORT.md'),spanish_sha256=sha(out/'translations/es/COMPLETION_REPORT.md'),numbers_and_identifiers='Common source dictionaries; same formatting expressions; technical JSON keys unchanged',scope='Renderer provenance, not independent automated semantic certification'))
        success=True
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        write(out/'DECISION.json',dict(status='B1S_V1_1_BLOCKED',source_commit=commit(),incident=error,development_only=True,confirmatory=False,H_CAT='NOT_EVALUABLE',H_TRANSFER='NOT_EVALUATED',B1E_executed=False,final_seeds_generated=False,ready_for_b1e_freeze=False,ready_for_b1e_confirmatory_run=False,missing_seeds_replaced=False))
        write(out/'HISTORICAL_PRESERVATION.json',preservation())
        for es in [False,True]:
            p=out/('translations/es/COMPLETION_REPORT.md' if es else 'COMPLETION_REPORT.md');p.parent.mkdir(parents=True,exist_ok=True)
            text=('# B1-S v1.1 — Ejecución bloqueada\n\n**B1S_V1_1_BLOCKED**\n\nNo se completaron todas las etapas. Se conservan los entrenamientos e incidentes, sin reemplazar semillas ni afirmar resultados inexistentes. B1-E no se ejecutó.\n\n' if es else '# B1-S v1.1 — Blocked execution\n\n**B1S_V1_1_BLOCKED**\n\nNot all stages completed. Fits and incidents are retained, without replacing seeds or claiming absent results. B1-E was not executed.\n\n')
            p.write_text(text+'```text\n'+error+'\n```\n',encoding='utf-8')
    seal(out,args.screen,args.fits)
    write(out/'PUBLICATION_RECEIPT.json',dict(source_commit=commit(),workflow_run_id=os.getenv('GITHUB_RUN_ID'),scientific_pipeline_finished=success,freeze_sha256=sha(out/'FREEZE.json'),scientific_status=read(out/'DECISION.json')['status'],development_only=True,B1E_executed=False))
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('scientific_success='+str(success).lower()+'\n')
    return success

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--qa',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--screen',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);a=p.parse_args();finalize(a)
