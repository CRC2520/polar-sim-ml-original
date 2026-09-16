"""Generate the predesignated canonical D4 trainable initialization bytes.

This is initialization-only: no environment, optimizer, training, evaluation or B1-E.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from b1s.execution.core import Value
from experiments.b1sc_d4_v1_0 import implementation as d4
from experiments.b1sc_d4_v1_0.init_format import sha256_file, trainable_digest, write_snapshot

BOUNDARY = dict(
    scientific_training_performed=False,
    scientific_evaluation_performed=False,
    scientific_results_exist=False,
    execution_authorized=False,
    B1E_disposition="ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION",
    B1E_executed=False,
    final_seeds_generated=False,
    H_CAT="NOT_EVALUABLE",
    H_TRANSFER="NOT_EVALUATED",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != "2.2.2+cpu":
        raise RuntimeError("D4 snapshot generation requires torch 2.2.2+cpu")
    if np.__version__ != "1.26.4":
        raise RuntimeError("D4 snapshot generation requires numpy 1.26.4")
    d4.validate_design()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    rows = []
    for block in range(d4.BLOCKS):
        seed = d4.snapshot_seed(block)
        torch.manual_seed(seed)
        actor = d4.D4RoutingActor()
        critic = Value()
        path = a.out / f"block-{block}.bin"
        file_sha = write_snapshot(path, block, seed, actor, critic)
        rows.append(dict(
            block=block,
            source_seed=seed,
            file=path.name,
            sha256=file_sha,
            actor_trainable_digest=trainable_digest(actor),
            critic_trainable_digest=trainable_digest(critic),
            bytes=path.stat().st_size,
            shared_conditions=list(d4.CONDITIONS),
            support_buffer_included=False,
            temporal_gate_state_included=False,
            condition_included=False,
        ))

    manifest = dict(
        status="D4_CANONICAL_INIT_GENERATED_NOT_SCIENTIFIC",
        schema="B1-SC-D4-INIT-CANDIDATE-1.0.0-20260916",
        design_commit=d4.DESIGN_COMMIT,
        source_branch=d4.INIT_BRANCH,
        runtime=dict(
            python=sys.version.split()[0],
            torch=torch.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
            torch_threads=torch.get_num_threads(),
        ),
        snapshot_contract=dict(
            trainable_only=True,
            one_per_block=True,
            count=d4.BLOCKS,
            shared_byte_for_byte_across_three_conditions=True,
            support_buffer_excluded=True,
            temporal_gate_state_excluded=True,
            condition_excluded=True,
            d3_snapshot_reuse=False,
            seed_reconstruction_at_scientific_runtime_forbidden=True,
        ),
        blocks=rows,
        **BOUNDARY,
    )
    (a.out / "CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in rows:
        if sha256_file(a.out / row["file"]) != row["sha256"]:
            raise RuntimeError("D4 candidate snapshot changed before publication")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
