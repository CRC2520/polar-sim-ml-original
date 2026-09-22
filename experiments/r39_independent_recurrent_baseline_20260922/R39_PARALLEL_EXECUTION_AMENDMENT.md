# R39 execution amendment — parallel training only

The sequential R39 workflow proved impractical because four independent
100,000-transition RecurrentPPO trainings were executed serially before any
development evaluation.

Scientific status before this amendment:
- no development evaluation record was produced;
- no confirmatory evaluation was authorized or opened;
- training seeds, environments and hyperparameters were already frozen.

The amended workflow changes only orchestration:
- the same four models are trained in four parallel jobs;
- each job uses the same environment-specific training seed;
- each job uses the same 100,000-transition budget and identical hyperparameters;
- model artifacts are merged;
- one evaluation job then runs the unchanged development gate and conditional
  confirmation logic.

Unchanged:
- baseline implementation/version;
- frozen CORE implementation;
- training/evaluation seeds;
- all thresholds and adjudication rules;
- evaluation episode count.

This is an infrastructure amendment, not a model or hypothesis change.
