# POLAR R10 — five discriminating experiments after R9

**Status:** prospective protocol written before any R10 pilot or confirmatory result is inspected.
**Scientific unit:** independent seed. Pilot seeds are excluded from confirmation.
**Base:** merged R9 source at `a0696148e568c94365a26311953430b082a3725b`.
**Scope:** bounded synthetic mechanistic tests. Passing any block is not evidence of phenomenal consciousness.

## Seeds and discipline

Pilot/debug only: 970101–970104.
Confirmatory: 972001–972016 (n=16).
No final threshold, seed, comparator, or decision rule may change after the confirmatory run starts.
All adverse outcomes remain in the report.

## E1 — Causal-State Discovery

Question: can a lower-dimensional state be fed back recurrently while preserving **action, memory, confidence and routing** rather than only reconstructing recorded outputs?

Candidate PCA bottlenecks: d = {2,4,6,8,10,12}. Dimension is chosen on validation only as the smallest d satisfying:
- offline action agreement >= 0.98;
- routing probability MAE <= 0.05;
- confidence MAE <= 0.05;
- memory-state MAE <= 0.05.

Confirmatory strong criterion, per seed:
- selected d <= 8;
- closed-loop action agreement >= 0.95;
- routing accuracy loss <= 0.03;
- confidence Brier increase <= 0.01;
- memory MAE <= 0.08;
- normalized reward loss <= 0.03.

Global E1 PASS: >=12/16 seeds satisfy the complete conjunction.
A decoder rescue from the recorded full state is reported separately and does not count as closed-loop sufficiency.

## E2 — POLAR-vs-GENERIC Isomorphic Control

Two controls are required.

**E2a exact isomorphism.** A fixed orthogonal transform of the same feature vector is given to an equal-size ridge predictor. Training samples, outputs, parameter count, regularization, actions and budget are identical. Predictions/actions should be numerically equivalent after basis transformation.

E2a criterion: median max prediction difference <=1e-8 and median absolute return difference <=1e-8.

**E2b matched structural comparator.** POLAR and GENERIC use the same features, number of active coefficients, training rows, ridge/lasso budget and action selector. POLAR uses the declared pair-local support; GENERIC uses a seed-fixed target-agnostic random support of identical cardinality.

Per-seed specificity criterion:
- POLAR normalized return - GENERIC >= 0.02;
- POLAR life is not lower by >0.01.

Global E2-specificity PASS: >=12/16 seeds. E2a equivalence must also pass; otherwise E2 is invalid rather than positive.

## E3 — Adaptive-Gate Necessity

A dedicated switching-coupling benchmark is used because R9 often selected gate OFF. The hidden relation alternates between a regime where the cross term improves prediction/control and a regime where that same term is misleading. The regime itself is not given to the policy; only noisy public context and past factual error are available.

Per seed:
- gate balanced accuracy against the evaluator-only “cross useful” label >=0.85;
- reward advantage over the **best constant gate selected on validation** >=0.03;
- reward advantage over exact OFF >=0.03;
- alive/feasible fraction is not lower than the best constant by >0.01.

Global E3 PASS: >=12/16.
This is a construct-validity benchmark for adaptive routing, not evidence that natural environments contain the designed relation.

## E4 — Cross-Domain Transfer

Train on two source families only. Test with frozen learned parameters on two separately coded held-out families whose transition equations were not used for training or model selection. Episodic state may update; model parameters may not.

Per seed, on **each** held-out family:
- normalized return >=0.65;
- alive/feasible fraction >=0.80;
- return advantage over the matched generic controller >=0.02.

Global E4 PASS: >=12/16 seeds satisfy all six held-out conditions.
Results are reported for each held-out family; averaging cannot rescue a failed family.
The families are internally designed by the same research program and are not independent external replication.

## E5 — Integrated Consciousness Precursors

A single recurrent agent is trained and then evaluated on one uninterrupted 4096-step trajectory per seed. The same agent/state must simultaneously support:
1. primary task;
2. self/world attribution;
3. source attribution;
4. temporal age;
5. workspace routing;
6. metacognitive confidence;
7. episodic-memory retrieval;
8. counterfactual action evaluation;
9. context-dependent polar control.

Confirmatory intact thresholds:
- primary balanced accuracy >=0.75;
- self/world accuracy >=0.80;
- source accuracy >=0.80;
- age-bin accuracy >=0.70;
- routing accuracy >=0.75;
- meta Brier improvement over a primary-margin baseline >=0.01;
- memory retrieval accuracy >=0.80;
- counterfactual regret improvement over action-agnostic model >=0.02;
- polar-control reward advantage over fixed-extreme policy >=0.03.

Selective lesions are applied to the **same frozen agent and identical exogenous trajectory**:
- WORKSPACE: routing drop >=0.15;
- SELF: self/world drop >=0.15 and primary drop <=0.03;
- MEMORY: source or age drop >=0.15 and primary drop <=0.03;
- COUNTERFACTUAL: regret advantage loss >=0.02 and primary drop <=0.03;
- POLAR: polar-control advantage loss >=0.03 and primary drop <=0.03.

Matched sham interventions of the same scalar/dimensional budget are reported.

Per-seed E5 strong PASS: all nine intact thresholds and at least four of five selective lesion signatures.
Global E5 PASS: >=12/16 seeds.

## Multiple-block interpretation

No omnibus “consciousness PASS” is defined. Each block has its own verdict.
For the five-block research program to be described as jointly resolved, E1–E5 must all pass their own frozen rules. Failure of one block is preserved.
