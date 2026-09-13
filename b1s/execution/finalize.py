"""Finalize one authorized development attempt; never substitute missing fits."""
import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from .analysis import complete, archive_and_seal, current_commit, preservation, sha
from .core import write_json


def run(fits: Path,out: Path,qa: Path):
    out.mkdir(parents=True,exist_ok=True)
    (out/'traces').mkdir(exist_ok=True)
    if qa.exists():shutil.copytree(qa,out/'qa',dirs_exist_ok=True)
    success=False
    try:
        complete(fits,out)
        success=True
    except Exception:
        error=traceback.format_exc()
        incident={'status':'B1S_BLOCKED','source_commit':current_commit(),'workflow_run_id':os.getenv('GITHUB_RUN_ID'),
                  'incident':error,'failure_time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  'successful_fits_retained':True,'missing_seeds_replaced':False,'development_only':True,
                  'confirmatory':False,'B1E_executed':False,'final_seeds_generated':False,
                  'stages':{'S1':'IMPLEMENTED','S2':'SEE_QA_REPORT','S3':'SEE_DEVELOPMENT_FREEZE','S4':'SEE_FIT_AUDIT','S5':'NOT_CERTIFIED_COMPLETE','S6':'NOT_CERTIFIED_COMPLETE','S7':'BLOCKED'},
                  'ready_for_b1e_freeze':False,'ready_for_b1e_confirmatory_run':False,'H_CAT':'NOT_EVALUABLE','H_TRANSFER':'NOT_EVALUATED'}
        write_json(out/'B1S_DECISION.json',incident)
        (out/'B1S_COMPLETION_REPORT.md').write_text('# B1-S — blocked development attempt\n\n**B1S_BLOCKED**\n\nThe campaign did not pass every required stage. Successful fits and incident evidence are retained; missing observations are not replaced. No confirmatory conclusion, final seeds or B1-E execution.\n\n```text\n'+error+'\n```\n')
        es=out/'translations/es';es.mkdir(parents=True,exist_ok=True)
        (es/'B1S_COMPLETION_REPORT.md').write_text('# B1-S — ejecución de desarrollo bloqueada\n\n**B1S_BLOCKED**\n\nLa campaña no superó todas las etapas exigidas. Se conservan las ejecuciones completadas y la evidencia del incidente, sin sustituir observaciones faltantes. No hay conclusión confirmatoria, semillas finales ni ejecución B1-E.\n\n```text\n'+error+'\n```\n')
        write_json(out/'HISTORICAL_PRESERVATION.json',preservation())
        print(error,flush=True)
    archive_and_seal(fits,out)
    frozen=json.loads((out/'B1S_FREEZE.json').read_text())
    frozen['status']=json.loads((out/'B1S_DECISION.json').read_text())['status']
    write_json(out/'B1S_FREEZE.json',frozen)
    write_json(out/'PIPELINE_COMPLETION.json',{'all_scientific_stages_executed_without_exception':success,
              'source_commit':current_commit(),'freeze_sha256':sha(out/'B1S_FREEZE.json'),
              'scope':'A successful publication can contain B1S_BLOCKED; inspect B1S_DECISION, not only CI color',
              'B1E_executed':False})
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('scientific_success='+str(success).lower()+'\n')
    return success

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--qa',type=Path,required=True);a=p.parse_args();run(a.fits,a.out,a.qa)
