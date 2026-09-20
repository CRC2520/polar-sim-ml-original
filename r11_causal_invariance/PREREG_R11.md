# POLAR R11 — Causal invariance correction of the four remaining R10 gaps

Base: R10 branch head 5e522e3c255ace57bf7be68166013becea1feee2, which itself preserves merged R9 unchanged.
R10 results remain immutable. R11 is a new architecture candidate and new experiment family.

## Design correction
R11 retains the R9 native agent but adds one bounded action-conditioned online adapter driven by the R10 causal-state result:
- recent factual prediction error,
- persistent goal state,
- associative memory/workspace,
plus the current public observation.

The adapter learns only from the agent's own executed action -> observed consequence.
It predicts next public observation and reward for each candidate action.
Its reliability is estimated from past factual residuals and controls whether the adapter can modify the native R9 candidate evaluation.

A relational adapter receives an invertible 24-dimensional orientation/intensity+tension basis.
The generic isomorphic adapter receives a fixed orthogonal rotation of exactly the same 24 values.
Both have the same coefficient count, update equations, action budget, observations and target data.
L1 shrinkage makes basis alignment an explicit, testable inductive bias rather than hidden extra information.

R11 also records a native operational source estimate from action-predicted change versus unexplained residual,
a world-event age estimate, and counterfactual candidate values. These are functional variables, not claims of subjective experience.

## Development / confirmation separation
Pilot seeds: 971001-971004.
Pilots may tune only numerical adapter hyperparameters listed in PILOT_CHOICES.md.
No confirmatory seed may be executed before FREEZE_R11.json is committed.
Confirmatory seeds, once frozen: 973001-973012.
Seed is the inferential unit. Global criterion: >=9/12 seeds unless specified otherwise.

## R11-E2 — Relational specificity under isomorphic control
Training: each relational and generic-isomorphic controller acquires its own experience on the same two internally designed relational families.
Held-out test: two new exogenous tapes and a delayed regime switch.
Same information, dimensions, parameter count, target labels, update count and action set.

Per seed PASS:
- relational - generic return >= +0.02 on both relational test families;
- alive difference >= -0.01 on both;
- ecology_delay9 relational - generic return >= -0.01 (anti-overfitting guard).
Global PASS >=9/12.
A failure does not imply universal equivalence.

## R11-E3 — Utility-aligned adaptive gate
The gate target is no longer an investigator's semantic regime label.
Training target = whether the cross correction reduces factual squared prediction error for that row.
The same context vector must predict that utility on held-out samples.

Per seed PASS:
- balanced accuracy for cross-utility label >= .85;
- adaptive MSE improves over better constant gate by >= .02;
- permuting context worsens adaptive MSE by >= .02.
Global PASS >=9/12.

## R11-E4 — Fast causal transfer
Core R9/predictor/Q parameters are trained only on ecology_train.
No target-family trajectory is used before test.
During each held-out target episode R11 may update only its small online adapter from factual own-action transitions.
First 64 steps are an explicitly declared adaptation window; metrics are computed on steps 64-319.
Targets are the unchanged R10 cyclic_buffer and repair_queue environments.

Per seed PASS on BOTH:
- alive fraction after adaptation >= .80;
- mean reward after adaptation >= .35;
- reward improvement versus frozen R9 baseline >= .05;
- alive improvement versus frozen R9 baseline >= .10.
Global PASS >=9/12.

## R11-E5 — Integrated functional precursors in one agent
One 640-step viable attribution/transfer trajectory with repeated contexts.
The agent must expose native, action-dependent estimates before outcome observation.
No evaluator source/time/counterfactual label enters the policy or learning update.

Per seed validity:
- alive fraction >= .80;
- all three factual source classes represented in the second half.

Base conjunction:
- native self/world balanced accuracy >= .70;
- native 3-way source balanced accuracy >= .55;
- native time-since-world-event balanced accuracy >= .45;
- metacognitive confidence-vs-error AUC >= .65;
- native counterfactual best-action accuracy >= .45.

Selective lesion conjunction on matched exogenous tapes:
- noMemory return drop >= .01;
- permuted_content return drop >= .01;
- noCross/tension return drop >= .01;
- each lesion alive-fraction damage <= .10.

Global PASS requires validity + base + lesion conjunction in >=9/12.

E5 remains a functional precursor benchmark. It does not measure phenomenal consciousness.
