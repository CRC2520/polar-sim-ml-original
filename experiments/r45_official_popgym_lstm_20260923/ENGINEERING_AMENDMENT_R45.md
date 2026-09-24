# R45 engineering amendment — failed pre-adjudication execution

Date: 23 September 2026, America/Lima.

## Scope

This note records the engineering repair made after the first R45 workflow attempt failed before any development-gate adjudication.

Affected workflow run: `35906599914`.

The preregistered scientific design in `PREREG_R45.md` is unchanged.

## What happened

The engineering smoke completed successfully and all three Easy development training jobs opened the preregistered seeds:

- 2166001
- 2166002
- 2166003

The official-code LSTM training process itself ran. Two jobs reached the frozen 1,048,576-transition budget before failing during held-out evaluation; the third job was cancelled by the original 90-minute CI timeout after reaching 788,242 transitions.

The failed evaluation did **not** produce an R45 development result or scientific gate decision. The workflow therefore skipped `development_gate`, `confirmatory`, and `confirmatory_gate`.

## Engineering defects

### 1. Wrapped observation parsing

The historical POPGym launcher trains on:

`Antialias(PreviousAction(env))`.

Under Gym 0.24 this wrapper returns a tuple-structured **observation**. The initial evaluation helper incorrectly treated every Python tuple returned by `reset()` as the newer `(obs, info)` API and selected element zero. RLlib then received a raw ndarray where the policy observation space required the full wrapped tuple.

Observed exception:

`ValueError: The two structures don't have the same nested structure`.

Repair:

Only a 2-tuple whose second element is a dictionary is interpreted as `(obs, info)`; all other tuple structures are retained as the policy observation.

This changes no environment, model, action, reward, seed, training budget, threshold, or scientific endpoint.

### 2. GitHub Actions matrix interpolation

The workflow accidentally retained a literal backslash before GitHub matrix expressions. This produced artifact names such as:

`r45-dev-\\2166002`

which GitHub rejects.

Repair:

Remove the literal backslash. This affects artifact naming/path interpolation only.

### 3. CI timeout

The original per-development-job timeout was 90 minutes. Seed 2166001 was still training normally when GitHub cancelled it at 788,242 transitions.

Repair:

Increase the CI timeout to 240 minutes. The frozen training budget remains exactly 1,048,576 environment timesteps.

## Information observed before repair

Training progress logs were visible for the failed attempt. They are treated as execution diagnostics only and are **not** used to change R45 parameters, thresholds, seeds, evaluation episodes, or the confirmation decision rule.

No held-out R45 development evaluation summary was successfully produced, and no checkpoint satisfied or failed the preregistered competence gate through a completed adjudication.

## Rerun governance

The repaired run reuses the original preregistered development seeds 2166001–2166003 because the prior attempt yielded no valid development adjudication.

Frozen quantities remain unchanged:

- three development checkpoints;
- 1,048,576 timesteps per checkpoint;
- Easy development environment;
- held-out evaluation seeds 2166101–2166112;
- four episodes per evaluation seed;
- competence score threshold 0.90;
- intact-minus-step-reset history benefit threshold +0.15;
- confirmation requires at least 2/3 eligible checkpoints;
- confirmatory seeds remain unopened unless that gate passes.

The repair does not authorize result-dependent retuning.

## Relevant repair commits

- `ea7087f76fe476c5e539a7b0883b17f392ac456e` — preserve historical wrapped tuple observations during evaluation.
- `e1052cf9e766d5e6f8a3fcfd6f40a472a093cea1` — correct matrix interpolation and extend CI timeout.

R45 remains a reproduced official-code comparator study, not an author-provided paper-checkpoint comparison and not an E6b independent replication.
