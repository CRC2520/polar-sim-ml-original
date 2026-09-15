"""CLI for the guarded B1-SC v1.0 execution pipeline."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

from experiments.b1sc_v1_0 import execution as ex
from experiments.b1sc_v1_0 import implementation as sc

def blocked(out: Path, stage: str, err: BaseException) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ex.write_json(out / "BLOCKED.json", dict(
        status="PIPELINE_BLOCKED_WITH_RETAINED_EVIDENCE",
        stage=stage, error="".join(traceback.format_exception(err)),
        automatic_retry=False, replacement_seed=False, **sc.BOUNDARY,
    ))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=[
        "authorize", "fit", "admission", "primary", "primary-aggregate",
        "secondary", "secondary-aggregate", "final"
    ])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--role")
    p.add_argument("--block", type=int)
    p.add_argument("--fits", type=Path)
    p.add_argument("--admission", type=Path)
    p.add_argument("--primary-root", type=Path)
    p.add_argument("--primary", type=Path)
    p.add_argument("--secondary-root", type=Path)
    p.add_argument("--secondary", type=Path)
    args = p.parse_args()

    try:
        if args.stage == "authorize":
            req = ex.execution_guard()
            args.out.mkdir(parents=True, exist_ok=False)
            ex.write_json(args.out / "AUTHORIZED.json", dict(
                status="AUTHORIZED_FOR_SINGLE_B1SC_RUN_ATTEMPT_1",
                workflow_run_id=int(os.environ["GITHUB_RUN_ID"]),
                run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
                authorization_request_sha256=ex.sha256_file(ex.START_REQUEST),
                authorization=req, **sc.BOUNDARY,
            ))
        elif args.stage == "fit":
            ex.require(args.role is not None and args.block is not None, "fit requires role/block")
            ex.fit(args.role, args.block, args.out)
        elif args.stage == "admission":
            ex.require(args.fits is not None, "admission requires fits")
            ex.aggregate_admission(args.fits, args.out)
        elif args.stage == "primary":
            ex.require(args.role is not None and args.block is not None, "primary requires role/block")
            ex.require(args.fits is not None and args.admission is not None, "primary requires fits/admission")
            ex.primary_causal(args.role, args.block, args.fits, args.admission, args.out)
        elif args.stage == "primary-aggregate":
            ex.require(args.primary_root is not None and args.admission is not None,
                       "primary aggregate requires primary root/admission")
            ex.aggregate_primary(args.primary_root, args.admission, args.out)
        elif args.stage == "secondary":
            ex.require(args.role is not None and args.block is not None, "secondary requires role/block")
            ex.require(args.fits is not None and args.primary is not None, "secondary requires fits/primary")
            ex.secondary_causal(args.role, args.block, args.fits, args.primary, args.out)
        elif args.stage == "secondary-aggregate":
            ex.require(args.secondary_root is not None and args.primary is not None,
                       "secondary aggregate requires secondary root/primary")
            ex.aggregate_secondary(args.secondary_root, args.primary, args.out)
        else:
            ex.require(args.admission is not None and args.primary is not None and args.secondary is not None,
                       "final requires admission/primary/secondary")
            ex.final_aggregate(args.admission, args.primary, args.secondary, args.out)
    except Exception as e:
        blocked(args.out, args.stage, e)
        print(json.dumps({"status":"PIPELINE_BLOCKED_WITH_RETAINED_EVIDENCE","stage":args.stage}, indent=2))
        raise

if __name__ == "__main__":
    main()
