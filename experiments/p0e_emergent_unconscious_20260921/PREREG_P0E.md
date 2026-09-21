# PREREG — POLAR P0-E Emergent Latent Unconscious

Freeze date: 2026-09-21 (America/Lima)

Development seeds excluded: 1201001–1201004.
Stress seeds excluded: 1201101–1201102.
Untouched confirmatory seeds: 1202001–1202012.
Training steps per confirmatory agent: 500.

No architecture, task generator, optimizer, threshold, endpoint, seed, workspace capacity/noise or causal-lesion procedure may change after confirmatory execution begins.

## Central question

Can a lightly constrained neural agent with a single shared recurrent state learn a functional separation between causally active historical memory and workspace-accessible content without being given an explicit latent-memory channel or a memory-specific broadcast gate?

## Architecture

- GRU hidden size 72.
- No explicit latent store/gate/salience/reconsolidation variables.
- Action head from shared recurrent state + current observation.
- Workspace bottleneck dimension 2, generic Gaussian access noise SD 0.58.
- Generic report head from workspace.
- Generic factual-outcome predictor.
- Workspace L2 capacity penalty; no memory-specific adversarial or masking loss.

## P0-E1 — representational separation and implicit behavioral use

Confirmatory median criteria:
- hidden-memory AUC >= 0.95;
- hidden-minus-workspace pre-trigger AUC >= +0.20;
- current-observation-only AUC <= 0.60;
- near-trigger action accuracy >= 0.95.

## P0-E2 — context-dependent promotion to workspace

Confirmatory median criteria:
- workspace query AUC >= 0.95;
- workspace promotion gain (query AUC - pre-trigger workspace AUC) >= +0.20;
- query report accuracy >= 0.95.

## P0-E3 — causal necessity and selectivity of the emergent memory direction

A linear memory direction is fitted post hoc on one half of held-out evaluation episodes and intervened on only in the other half.

Confirmatory median criteria:
- action-accuracy drop after memory-direction projection >= 0.35;
- action accuracy after memory-direction lesion <= 0.65;
- matched random-direction lesion drop <= 0.05;
- current-evidence action accuracy >= 0.95;
- current-evidence drop under the same memory lesion <= 0.05.

## Per-seed guardrail

At least 9/12 confirmatory seeds must jointly satisfy:
- hidden-memory AUC >=0.90;
- hidden-minus-workspace AUC >=0.15;
- pre-trigger workspace AUC <=0.82;
- workspace query AUC >=0.90;
- workspace promotion gain >=0.15;
- current-only AUC <=0.65;
- near action accuracy >=0.90;
- memory-direction lesion drop >=0.25;
- random lesion drop <=0.08;
- current-evidence lesion drop <=0.08.

## Secondary, non-confirmatory online-update metrics

Post-correction action/report accuracy, recon-like update accuracy, no-reactivation preservation, reactivation-without-contradiction preservation, and post-update hidden-memory AUC are reported descriptively only. They do not determine the P0-E verdict because development showed that rapid reconsolidation-like updating is not yet stable across seeds.

## Global verdict

`EMERGENT_FUNCTIONAL_SEPARATION_PASS` iff all median P0-E1–E3 criteria pass and the per-seed guardrail is satisfied in >=9/12 seeds.

A PASS supports emergence of a functional distinction between recurrent causal memory and workspace access in this synthetic multitask agent. It does not establish phenomenal unconsciousness/consciousness, biological trauma/reconsolidation, human repression, or independent external replication.
