"""B1-SC v1.0 execution-readiness qualification.

This preflight proves the execution code and workflow are frozen and guarded.
It does not authorize or perform scientific training/evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.b1sc_v1_0 import implementation as sc
from experiments.b1sc_v1_0 import execution as ex
from b1s.execution.instrument import runtime_lock

QA_WORKFLOW = ".github/workflows/b1sc-v1-0-execution-qa.yml"
EXEC_WORKFLOW = ".github/workflows/b1sc-v1-0-execution.yml"
PARENT_IMPL_FREEZE = "experiments/b1sc_v1_0/run/IMPLEMENTATION_FREEZE.json"
ALLOWED_ADDITIONS = {
    "experiments/b1sc_v1_0/execution.py",
    "experiments/b1sc_v1_0/runner.py",
    "experiments/b1sc_v1_0/execution_tests.py",
    "experiments/b1sc_v1_0/execution_preflight.py",
    QA_WORKFLOW,
    EXEC_WORKFLOW,
}

def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)

def git(*args):
    return subprocess.check_output(["git",*args],text=True).strip()

def preservation():
    rows=git("diff","--name-status",ex.QUALIFIED_BASE_COMMIT+"..HEAD").splitlines()
    observed={}
    for row in rows:
        if row:
            status,path=row.split("\t",1)
            observed[path]=status
    require(set(observed)==ALLOWED_ADDITIONS, "Unexpected execution-readiness changes: "+repr(observed))
    require(all(x=="A" for x in observed.values()), "Execution readiness may add files only")
    require(git("status","--porcelain")=="", "Working tree not clean")

    parent=ex.read_json(sc.REPO_ROOT/PARENT_IMPL_FREEZE)
    require(parent["status"]=="IMPLEMENTATION_QUALIFIED_NOT_EXECUTED", "Parent implementation freeze status changed")
    require(parent["implementation_source_commit"]=="0ae43e798c20353aebbcade6d9285876dff354df",
            "Unexpected parent implementation source")
    for rel,digest in parent["source_hashes"].items():
        require((sc.REPO_ROOT/rel).is_file(), "Parent frozen source absent: "+rel)
        require(ex.sha256_file(sc.REPO_ROOT/rel)==digest, "Parent frozen source changed: "+rel)
    return dict(
        status="PASS",
        qualified_base_commit=ex.QUALIFIED_BASE_COMMIT,
        added_paths=sorted(observed),
        parent_implementation_freeze_sha256=ex.sha256_file(sc.REPO_ROOT/PARENT_IMPL_FREEZE),
    )

def qa_workflow_contract():
    text=(sc.REPO_ROOT/QA_WORKFLOW).read_text(encoding="utf-8")
    lower=text.lower()
    require(ex.EXEC_BRANCH in text, "QA workflow wrong branch")
    require("workflow_dispatch" not in lower, "QA manual dispatch forbidden")
    require("permissions:\n  contents: read" in text, "QA workflow not read-only")
    require("start_request.json" in lower, "QA must assert absence of start request")
    require("execution_tests" in text and "execution_preflight" in text, "QA suite/preflight missing")
    require(" runner fit " not in lower and "stage fit" not in lower, "QA workflow invokes scientific fit")
    return dict(status="PASS",sha256=ex.sha256_file(sc.REPO_ROOT/QA_WORKFLOW),scientific_training_job=False)

def execution_workflow_contract():
    text=(sc.REPO_ROOT/EXEC_WORKFLOW).read_text(encoding="utf-8")
    lower=text.lower()
    require(ex.EXEC_BRANCH in text, "Execution workflow wrong branch")
    require("workflow_dispatch" not in lower, "Manual dispatch forbidden")
    require("permissions:\n  contents: read" in text, "Execution workflow must be repository read-only")
    require("experiments/b1sc_v1_0/start_request.json" in lower, "Execution trigger is not START_REQUEST")
    trigger_block=lower.split("permissions:",1)[0]
    require("execution.py" not in trigger_block and "runner.py" not in trigger_block,
            "Execution workflow can trigger from code changes")
    require("fail-fast: false" in lower, "Fit matrix must retain other evidence after one failure")
    require("max-parallel: 4" in lower, "Concurrency cap changed")
    for role in sc.ALL_ROLES:
        require(role.lower() in lower, "Workflow matrix missing role "+role)
    for stage in ("authorize","fit","admission","primary","primary-aggregate","secondary","secondary-aggregate","final"):
        require((" "+stage+" ") in lower or (" "+stage+"\n") in lower,
                "Workflow missing runner stage "+stage)
    require("rerun" not in lower and "re-run" not in lower, "Workflow contains automatic retry language")
    require("create-pull-request" not in lower and "git push" not in lower,
            "Scientific workflow must not publish to repository")
    return dict(
        status="PASS",
        sha256=ex.sha256_file(sc.REPO_ROOT/EXEC_WORKFLOW),
        trigger_only_start_request=True,
        repository_write_permission=False,
        max_parallel_fits=4,
        automatic_retry=False,
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--qa",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)

    require(os.environ.get("GITHUB_REPOSITORY")==sc.REPOSITORY,"Wrong repository")
    require(os.environ.get("GITHUB_REF")=="refs/heads/"+ex.EXEC_BRANCH,"Wrong execution-readiness branch")
    require(os.environ.get("GITHUB_RUN_ATTEMPT")=="1","Qualification rerun not accepted")
    require(git("rev-parse","HEAD")==os.environ.get("GITHUB_SHA"),"Unexpected checkout")
    require(not ex.START_REQUEST.exists(),"START_REQUEST exists during qualification")
    require(not ex.EXECUTION_FREEZE.exists(),"Execution freeze unexpectedly pre-exists")

    validation=ex.read_json(a.qa/"VALIDATION.json")
    require(validation["status"]=="EXECUTION_QA_PASS_NOT_AUTHORIZED","Execution QA did not pass")
    require(validation["start_request_present"] is False,"QA saw a start request")
    require(validation["scientific_training_performed"] is False,"Scientific training occurred in QA")
    require(validation["scientific_evaluation_performed"] is False,"Scientific evaluation occurred in QA")

    preserve=preservation()
    qaw=qa_workflow_contract()
    ew=execution_workflow_contract()
    lock=runtime_lock()

    sources=[
        "experiments/b1sc_v1_0/PROTOCOL_ES.md",
        "experiments/b1sc_v1_0/DESIGN_FREEZE.json",
        "experiments/b1sc_v1_0/implementation.py",
        "experiments/b1sc_v1_0/tests.py",
        "experiments/b1sc_v1_0/preflight.py",
        PARENT_IMPL_FREEZE,
        "experiments/b1sc_v1_0/run/PREFLIGHT_STATUS.json",
        "experiments/b1sc_v1_0/execution.py",
        "experiments/b1sc_v1_0/runner.py",
        "experiments/b1sc_v1_0/execution_tests.py",
        "experiments/b1sc_v1_0/execution_preflight.py",
        QA_WORKFLOW,EXEC_WORKFLOW,
        "b1s/execution/core.py","b1s/execution/instrument.py","b1s/execution/requirements.txt",
    ]
    source_hashes={rel:ex.sha256_file(sc.REPO_ROOT/rel) for rel in sources}
    reg=sc.registry()
    qa_hashes={p.name:ex.sha256_file(p) for p in sorted(a.qa.iterdir()) if p.is_file()}

    freeze=dict(
        status="EXECUTION_PIPELINE_QUALIFIED_NOT_AUTHORIZED",
        schema=sc.SCHEMA,
        repository=sc.REPOSITORY,
        branch=ex.EXEC_BRANCH,
        qualified_base_commit=ex.QUALIFIED_BASE_COMMIT,
        qualified_source_commit=os.environ["GITHUB_SHA"],
        design_freeze_sha256=ex.sha256_file(sc.DESIGN_PATH),
        implementation_freeze_sha256=ex.sha256_file(sc.REPO_ROOT/PARENT_IMPL_FREEZE),
        protocol_sha256=ex.sha256_file(sc.PROTOCOL_PATH),
        registry_entries=len(reg),
        registry_sha256=sc.sha256_bytes(sc.json_bytes(reg)),
        source_hashes=source_hashes,
        qa_hashes=qa_hashes,
        runtime=lock,
        historical_preservation=preserve,
        qa_workflow_contract=qaw,
        execution_workflow_contract=ew,
        pipeline=dict(
            roles=list(sc.ALL_ROLES), blocks=sc.BLOCKS, fits=72,
            native_steps=sc.NATIVE_STEPS, input_mode="raw",
            admission_episodes_per_block=sc.EPISODES_PER_BLOCK,
            admission_minimum_blocks=sc.MIN_BLOCKS,
            causal_window=list(sc.CAUSAL_WINDOW),
            primary_causal="all active routes",
            secondary_causal="individual active edges after primary gate",
            primary_utility_reference="G0",
            stages=["authorize","fit","admission","primary","primary-aggregate",
                    "secondary","secondary-aggregate","final"],
        ),
        authorization_request_required_fields=[
            "schema","authorize_b1sc_v1_0","authorized_run_attempt",
            "authorization_base_commit","qualified_source_commit",
            "execution_freeze_sha256","registry_sha256","design_freeze_sha256",
            "implementation_freeze_sha256","workflow_sha256",
            "B1E_executed","final_seeds_generated",
        ],
        training_performed=False,
        scientific_evaluation_performed=False,
        qa_microfit_performed=True,
        qa_microfit_scientific_evidence=False,
        execution_authorized=False,
        start_request_exists=False,
        results_exist=False,
        next_step_requires_explicit_author_consent=True,
        **sc.BOUNDARY,
    )
    ex.write_json(a.out/"EXECUTION_IMPLEMENTATION_FREEZE.json",freeze)
    ex.write_json(a.out/"EXECUTION_PREFLIGHT_STATUS.json",dict(
        status="EXECUTION_PIPELINE_QUALIFIED_NOT_AUTHORIZED",
        execution_freeze_sha256=ex.sha256_file(a.out/"EXECUTION_IMPLEMENTATION_FREEZE.json"),
        qualified_source_commit=os.environ["GITHUB_SHA"],
        training_performed=False,
        scientific_evaluation_performed=False,
        start_request_present=False,
        next_step="AUTHOR_REVIEW_AND_EXPLICIT_GO_NO_GO",
        **sc.BOUNDARY,
    ))
    print(json.dumps(ex.read_json(a.out/"EXECUTION_PREFLIGHT_STATUS.json"),indent=2,sort_keys=True))

if __name__=="__main__":
    main()
