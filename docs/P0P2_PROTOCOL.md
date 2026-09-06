# Prioritized gap resolution: operational specification and prospective evaluation

Status at initial commit: implementation specification BEFORE new experiments.
This is one integrated research prototype accompanying ONE unpublished manuscript.
The original layered proposal is preserved. No claim of phenomenal consciousness,
general ethical competence, emotional grounding, or unique polar superiority follows.
Frozen source/data for Studies 1–3 MUST remain byte-identical. Changes live under
`integrated_polar/`, `gap_resolution/` and new tests/docs/results only.

## Operational mapping

G01: distinct internal p, unconstrained intention v, constrained executed u. p is
updated toward v, not overwritten by u; p participates in the next planning cost.
G02: retain signed output residual, positive/negative deficits and coactivation
separately. Do not add them into an unsigned message that conceals correction direction.
G03: directed action-channel messages are the learned prediction Jacobian transpose
weighted by actual priorities and signed residuals; opposite valves can receive
opposite messages. C selects content and modulates priorities, not just total budget.
G04: RLS learns each free action->output coefficient. The induced K=J^T Q and
W=-J^T Q J are local predictive-control operators, not arbitrarily named synapses.
This is learning directed physical consequences and deriving regulatory routes;
it is NOT discovering a psychological polarity taxonomy or arbitrary neural topology.
G05: dense learned effect matrix and explicit persistent physical state; a planning
lesion removes off-pair entries while retaining the same estimator and observations.
G06: priorities enter the actual quadratic planning objective. Finite resource
allocation uses constrained optimization, not a single radial scale.
G07: ContentWorkspace selectively copies an identifiable goal/priority message to
planner, memory and reporter consumers. A selective planner cut leaves reporter
and memory delivery intact; content effects are not budget effects.
G08: cue memory persists after stimulus withdrawal, has context/privacy gates,
erasure and authorized versioned reconsolidation. No trauma/clinical theory is claimed.
G09: an explicit institutional policy implements permissions, protected-resource
prohibitions, lower-priority requests, reasons and infeasibility/human-review states.
It enforces specified action rules, NOT general morality or physical safety under
arbitrary model error. Provenance fields are consistency checks, not authentication.
G10: persistent goal commitments and multi-step forecasts; MPC anticipates future
states. Goals are supplied/retained, not freely invented intrinsic human-like motives.
G11: action/effect token matching plus separate likelihood-based candidate-source
inference, ambiguous-source abstention, and empirical decision-quality calibration.
Monitor calibration must be measured on held-out data; class name is not proof.
G12: independent local controller objects, private own observations/goals/learning,
scalar offers, bounded allocations and independently generated actions. Scalar
cooperation is limited; rich negotiation or collective consciousness is not claimed.
G13: typed sensors/actuators and strict command grammar for fill/drain tank levels.
Meanings are operational and physically signed; the original philosophical labels
are not validated. No general vision/language module or pretrained LLM is claimed.
G14: single lifecycle C->U/goals->predictive tension->intention->E/action->own feedback,
full snapshots, per-layer traces, synchronous timing and contract violations rejected.
Bounded actions/empirical perturbation behavior are NOT a global stability theorem.

## P0 diagnostic: exploratory causal interventions, not a replacement Study 3

Use unchanged Study 3 task generation, 12 diagnostic seeds 911001–911012, 12 cells,
16 factorial intervention combinations plus original no-K: 2448 runs, 64 steps.
These are new diagnostic seeds in a KNOWN task family, not confirmation of novelty.
Four binary changes to the active legacy configuration eta=1,a=0,b=-.2:
I: replace the duplicated scalar tau message by two signed per-pole residuals;
R: reverse the receiving second-pole sign rather than duplicate receiver signs;
M: replace local gain-only inverse planning by a full learned 8x8 response inverse;
P: replace radial clipping with priority-weighted capped-simplex projection.
I/R are simple controlled readouts, not the final integrated gradient implementation.
M changes the estimator AND its use, so its factorial effect is not attributed to
one independently isolated RLS detail. All disabled must reproduce original full.
Report every factorial cell, paired seed contrasts, all two-way interactions and
full-vs-no-K differences. Use descriptive 95% bootstrap (10000 draws); no selected
subgroup may rescue the original conclusion. All prior conclusions remain unchanged.

## P1 external dynamical evaluation: registered AFTER development, BEFORE final

Four tanks, two independent valve actions per tank, one primary controller; physical
state y_next=clip(.9*y + .03 + B*u + disturbance,0,1). Own fill/drain coefficients
are +[.09,.14]/-[.09,.14]; directed cross-effects up to .025 per valve. A second
family changes coefficients halfway; tasks have genuine persistence, unlike Study 3.
No true B is passed to a controller. A fixed 48-step calibration sequence with random
own actions is supplied identically to every condition and learned via own feedback.
Identification skips clipped physical outputs. Evaluation is 48 steps with four
12-step goal episodes A/B/C/A, target and priority forecasts of horizon 3. Context
A is withheld on its return: only its cue identifies the stored goal. Forecasts
are refreshed through content messages when available; future goals are contractual
intentions, not hidden physical truth. Priorities alternate; budget .6 or 2.0.
A protected drain rule and an action block are activated during a predeclared phase.

Modes: full, diagonal_plan, myopic, no_priorities, no_memory, no_broadcast,
generic_equivalent, no_internal, no_learning, reactive, zero. Each receives the
same available observations and effect feedback; lesions remove named operations.
Generic equivalent is an independently computed matrix form of the same MPC,
not a competitor expected to be worse. Cross-model superiority must not be
misattributed to coordinate labels. Sparse/diagonal use does not match numerical
parameter utilization although the learned dense estimator is retained.

Development seeds 921001–921006. Used for correctness diagnostics and monitor
calibration only; no automatic architecture selection. All changes motivated by
development are recorded as development, with source freeze only AFTER final tests.
Final seeds 931001–931024; 2 dynamic families x 2 budgets x 11 conditions =1056
runs, 50688 evaluated transitions plus matched calibration. This fixed finite
sample is not a power guarantee. Final seed exposure requires a public Git commit
containing all source hashes, configuration, development reports and calibration.

Primary measures: externally weighted tracking MSE, action cost, failures/violations,
post-change MSE and return-cue MSE. Report separately internal-state/action separation,
content delivery, learned off-pair weights and normative reasons. A primary success
requires full-minus-diagonal_plan MSE one-sided 98.75% bootstrap upper < -0.0002;
full-minus-no_priorities upper < -0.0002; return MSE full-minus-no_memory upper <
-0.0002; and full-minus-myopic post-change MSE upper < -0.0002. Four directional
claims use Bonferroni .05/4 and 20000 seed-block resamples RNG 941777.
Zero hard action violations, generic-equivalent max action discrepancy <=1e-8 and
zero action comparator MSE worse than full are validity checks. Runtime descriptive.
If tests pass but primary comparisons fail, report engineering closure only for
those functions, not empirical superiority. Never change final thresholds to pass.

## P2 specific capability evaluation and limitations

Deterministic regression interventions cover each of 14 gaps and integration.
A separate held-out probe set uses 24 seeds 951001–951024, not the P1 seeds:
- two autonomous tank controllers compare bid-based coordination with equal shares;
  report weighted external error, budget violations and distinct local actions.
- candidate-source attribution evaluates distinguishable own/other effects plus
  identical-effect ambiguity, no label given to inference; report accuracy/abstention.
- grammar parses counterbalanced targets/priorities and rejects unknown/duplicate
  channels; signs of learned fill/drain effects are tested, not emotional semantics.
- normative cases permute protected channels/requests/approval and check reasons,
  explicit precedence, budget impossibility and logged review state.
- memory reconsolidation, gate cuts and content lesions use complete matched states.
- probability monitor is calibrated solely on development pre-feedback scores and
  tracking-success labels (per-step weighted MSE <=.003). Report held-out Brier,
  base-rate Brier, ECE with fixed 10 probability bins, coverage and action changes
  under review threshold .55. Engineering acceptance of the monitor is separate
  from calibration acceptance: ECE<=.15 and Brier<=constant-development-base+.02.

These capability tests are narrow interventions in specified synthetic functions.
They are not proof of generalized autonomy, ethical reasoning, self-awareness or
consciousness. No hidden truth, present/future answer key, or hardcoded output is
allowed in the controller. Tests are not scored from class names or flags alone.

## Evidence and reporting

Store complete input/action/effect and state/control traces as compressed JSON,
per-file SHA-256, source hashes, run environment and public registration timestamp.
Regenerate metrics from traces and replay a fixed seed/family/budget before reporting.
Publish a gap matrix with implementation path, tests, measured outcomes, status
(engineering_verified / empirical_supported / partial / not_supported) and remaining
scope. Never mark full philosophical gaps closed merely because a unit test passes.
Keep the original full theory and architecture in the single LaTeX manuscript.

Methodological bases: Rawlings, Mayne & Diehl, Model Predictive Control (2nd ed.,
2017), https://sites.engineering.ucsb.edu/~jbraw/mpc/; Guo et al. (2017),
https://proceedings.mlr.press/v70/guo17a.html; Alshiekh et al. (2018),
https://ojs.aaai.org/index.php/AAAI/article/view/11797; Nosek et al. (2018),
doi:10.1073/pnas.1708274114. These motivate methods, not validation of this model.
