# R9 reproducibility

Run these commands from the `POLAR_reconciled` repository root. R9 is a new
bounded realization; it does not replace or reinterpret R8. Historical R8
source remains protected by `r8_completion/FREEZE_R8.json`; its scientific
source commit is `8823bda542f8a6c7c8e827a494d3acffb9b88a97`, and the published
post-run repository base is `6743addb639fe77e0e4439ffc89cd4a58e8e07be`.

The definitive R9 source/configuration/runtime identity is the prospective
`r9_completion/FREEZE_R9.json` and its source closure, verified before and
after every final seed. This guide is included in that closure. A working
freeze is not an external preregistration. Pilot source is retained separately
in `r9_results/pilot_v1/SOURCE_SNAPSHOT` with `SOURCE_RECORD.json`; preserve it
when comparing pilot and final implementations. Code revisions and snapshots
belong in Git; external data archives restore paths relative to this repository
root. Keep every original adverse/null result and its manifest.

Pilot v1 preserves plain `workspace_events.jsonl`. Subsequent persistence may
use `workspace_events.jsonl.gz` with deterministic gzip (`mtime=0`); decompressed
event lines and experimental computations are unchanged. The auditor accepts
either format, validates gzip integrity while reading, and rejects duplicate
plain/compressed streams for the same rollout. Pilot source comparison permits
only this runner persistence function/import difference; other scientific
definitions must match the retained snapshot.

## Runtime and contracts

The audited development runtime is Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
Linux 6.18.44 x86_64/glibc 2.39, OpenBLAS 0.3.30 (SkylakeX). Set BLAS threads
before Python starts; the default process may otherwise use multiple threads.
The freeze records the runtime actually used. Numerical closeness and exact
hash identity are distinct checks; neither substitutes for the other.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m unittest r9_completion.test_network r9_completion.test_workspace r9_completion.test_environments r9_completion.test_statistics r9_completion.test_agent r9_completion.test_audit -v
python -m r8_completion.provenance
```

Tests use mathematical, synthetic-actor and physics fixtures; they are not
confirmatory seeds. `test_audit` never instantiates a physical environment.

## Regenerate a campaign

Each output root must be fresh: existing `seed_*` directories and evidence
files are never overwritten. Do not rerun into the original campaign.

```sh
python -m r9_completion.runner --phase pilot --seeds 950101 950102 950103 --output r9_results/reproduction_pilot/data
```

Pilots evaluate only `ecology_train` and `ecology_delay9`. They do not generate
the transfer outcomes or supply six-claim confirmatory inference. Every seed
trains four actors from its own experiences: 8 training episodes, 2 separate
calibration episodes, and 7 gate-selection rollouts per actor. Every rollout
has 320 steps. Calibration episode 200 determines the residual margin;
episode 201 fits the gate using the already calibrated margin. Constant and
contextual selector sets each contain four options and share `off`.

The original final execution additionally requires the matching immutable
`r9_results/FINAL_EXECUTION_AUTHORIZATION.json`. Restore this record with the
data before reproducing the archived final; do not fabricate an authorization
to bypass a failed source check. The source verifier must pass first.

```sh
python -m r9_completion.provenance
python -m r9_completion.runner --phase final --seeds $(seq 952001 952080) --output r9_results/reproduction_final/data
```

This regenerates all 80 exact seeds; each final seed evaluates all four
domains and all 15 declared variants from saved pre-evaluation checkpoints.
One seed may be run separately for debugging, but a final campaign cannot be
closed with a missing seed. Do not tune parameters from reproduced final
outcomes or silently replace seeds. Q/predictor/gate/calibration parameters
remain frozen in evaluation; the specified factual episodic memory and goal
updates remain active except under their named lesions.

## Audit restored data without running experiments

```sh
python -m r9_completion.audit --phase pilot --input r9_results/pilot_v1/data --replay-seed 950101 --output r9_results/reproduction_pilot_audit.json
python -m r9_completion.audit --phase final --input r9_results/reproduction_final/data --replay-seed 952001 --output r9_results/reproduction_final_audit.json
```

For an archived final, replace `--input` with its restored data directory. The
auditor reads every `COMPLETE.json`, verifies file SHA256 and NPZ CRC, rejects
missing/unlisted artifacts and reconstructs observations, factual targets,
actions, scores, both gated predictor passes, feasibility, fixed-horizon
endpoints and delayed-memory event equations. It checks training-only
normalization, support/capacity, direct residual losses, acquisition hashes,
saved Q arithmetic, separate calibration and the prespecified selector.

`--replay-seed` replays the actor against saved observations for `full`,
`noCross`, `permuted_content`, `observation_shock` and `dense_shock`; it never
advances an environment or optimizes a new model. Without this option, raw
algebraic/integrity checks still cover the full campaign. The report names the
selected replay seed explicitly.

For finals, six claim vectors are rebuilt independently from raw reward/alive
traces in seed order 952001–952080. Each win requires every declared domain
and comparator, its reward margin, full alive fraction at least .80, and alive
loss no greater than .005. An independent SciPy exact binomial calculation
and Holm adjustment are compared with `statistics.compile_from_metrics`.
All six hypotheses remain in the family, including null/adverse outcomes.
Allocated coefficients, active support and statistical capacity are reported
separately. Successful consistency checks establish neither consciousness nor
external replication.
