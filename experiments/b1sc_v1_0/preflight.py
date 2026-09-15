"""B1-SC v1.0 QA/preflight closure. No training entry point exists here."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.b1sc_v1_0 import implementation as sc
from b1s.execution.instrument import runtime_lock

WORKFLOW=".github/workflows/b1sc-v1-0-qa.yml"
ALLOWED_ADDITIONS={
    "experiments/b1sc_v1_0/PROTOCOL_ES.md",
    "experiments/b1sc_v1_0/DESIGN_FREEZE.json",
    "experiments/b1sc_v1_0/implementation.py",
    "experiments/b1sc_v1_0/tests.py",
    "experiments/b1sc_v1_0/preflight.py",
    WORKFLOW,
}

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

def git(*args):
    return subprocess.check_output(["git",*args],text=True).strip()

def preservation():
    diff=git("diff","--name-status",sc.BASE_COMMIT+"..HEAD").splitlines()
    observed={}
    for line in diff:
        if not line: continue
        status,path=line.split("\t",1)
        observed[path]=status
    require(set(observed)==ALLOWED_ADDITIONS, "Unexpected changed paths: "+repr(observed))
    require(all(v=="A" for v in observed.values()), "Only new B1-SC implementation files are allowed")
    require(git("status","--porcelain")=="", "Working tree is not clean")
    return {"status":"PASS","base_commit":sc.BASE_COMMIT,"added_paths":sorted(observed)}

def workflow_contract():
    text=(sc.REPO_ROOT/WORKFLOW).read_text(encoding="utf-8")
    lower=text.lower()
    require("workflow_dispatch" not in lower, "Manual workflow dispatch is not part of QA-only workflow")
    require("matrix:" not in lower, "Training matrix is forbidden in QA-only workflow")
    require("runner.py" not in lower, "Scientific runner must not be invoked by QA-only workflow")
    require("start_request.json" in lower, "Workflow must explicitly verify START_REQUEST absence")
    require("permissions:\n  contents: read" in text, "QA-only workflow must have read-only repository permission")
    require("python -m experiments.b1sc_v1_0.tests" in text, "QA suite is not invoked")
    require("python -m experiments.b1sc_v1_0.preflight" in text, "Preflight is not invoked")
    return {"status":"PASS","workflow_sha256":sc.sha256_file(sc.REPO_ROOT/WORKFLOW),
            "training_job_present":False,"repository_write_permission":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--qa",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)

    require(os.environ.get("GITHUB_REPOSITORY")==sc.REPOSITORY,"Wrong repository")
    require(os.environ.get("GITHUB_REF")=="refs/heads/"+sc.BRANCH,"Wrong branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT")=="1","QA/preflight reruns are not accepted as qualification")
    require(git("rev-parse","HEAD")==os.environ.get("GITHUB_SHA"),"Unexpected checkout")
    require(not (sc.ROOT/"START_REQUEST.json").exists(),"START_REQUEST exists: training boundary violated")

    validation=json.loads((a.qa/"VALIDATION.json").read_text())
    require(validation["status"]=="QA_PASS_NOT_SCIENTIFIC_EXECUTION","QA did not pass")
    require(validation["training_performed"] is False and validation["evaluation_performed"] is False,"QA unexpectedly executed science")
    require(validation["start_request_present"] is False,"QA saw a start request")

    design=sc.validate_design_contract()
    preserve=preservation()
    wf=workflow_contract()
    lock=runtime_lock()
    sources=[
        "experiments/b1sc_v1_0/PROTOCOL_ES.md",
        "experiments/b1sc_v1_0/DESIGN_FREEZE.json",
        "experiments/b1sc_v1_0/implementation.py",
        "experiments/b1sc_v1_0/tests.py",
        "experiments/b1sc_v1_0/preflight.py",
        WORKFLOW,
        "b1s/execution/core.py",
        "b1s/execution/instrument.py",
        "b1s/execution/requirements.txt",
    ]
    hashes={p:sc.sha256_file(sc.REPO_ROOT/p) for p in sources}
    qa_hashes={p.name:sc.sha256_file(p) for p in sorted(a.qa.iterdir()) if p.is_file()}
    reg=sc.registry()
    freeze={
        "status":"IMPLEMENTATION_QUALIFIED_NOT_EXECUTED",
        "schema":sc.SCHEMA,
        "repository":sc.REPOSITORY,
        "branch":sc.BRANCH,
        "design_commit":sc.DESIGN_COMMIT,
        "design_freeze_sha256":sc.sha256_file(sc.DESIGN_PATH),
        "protocol_sha256":sc.sha256_file(sc.PROTOCOL_PATH),
        "implementation_source_commit":os.environ["GITHUB_SHA"],
        "workflow_run_id":int(os.environ["GITHUB_RUN_ID"]),
        "workflow_run_attempt":int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "source_hashes":hashes,
        "qa_hashes":qa_hashes,
        "registry_entries":len(reg),
        "registry_sha256":sc.sha256_bytes(sc.json_bytes(reg)),
        "runtime":lock,
        "historical_preservation":preserve,
        "workflow_contract":wf,
        "design_contract":design,
        "execution_boundary":{
            "execution_authorized":False,
            "start_request_exists":False,
            "training_started":False,
            "evaluation_started":False,
            "results_exist":False,
            "training_job_present_in_qa_workflow":False,
        },
        "next_step_requires_explicit_second_consent":True,
        **sc.BOUNDARY,
    }
    write(a.out/"IMPLEMENTATION_FREEZE.json",freeze)
    write(a.out/"PREFLIGHT_STATUS.json",{
        "status":"IMPLEMENTATION_QUALIFIED_NOT_EXECUTED",
        "implementation_freeze_sha256":sc.sha256_file(a.out/"IMPLEMENTATION_FREEZE.json"),
        "training_performed":False,"evaluation_performed":False,
        "start_request_present":False,
        "next_step":"AUTHOR_REVIEW_BEFORE_ANY_EXECUTION",
        **sc.BOUNDARY,
    })
    print(json.dumps(json.loads((a.out/"PREFLIGHT_STATUS.json").read_text()),indent=2,sort_keys=True))

if __name__=="__main__":
    main()
