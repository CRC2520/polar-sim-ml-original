# Iteration 2: causal regulation and usefulness of integrated functions

Status at first commit: prospective design, BEFORE development and final outcomes.
This is a continuation of the ONE unpublished Polar Dynamics manuscript. It does
not merge main or alter any earlier experiment, source, criterion or result.
Parent code: 05caf7abef6fdcb769604539a22cdcd1cfd6b397.

## Research object and boundaries

Keep the original Tension Engine, C/E/U, motivation, IACL and hybrid interfaces as
the architectural proposal. The revised agent reuses C/U/E software contracts and
adds an explicit actuator state and masked-sensor belief, separate from the polar
continuity state p, a box-limited unconstrained one-step intention v and executed u.
A benefit of the belief estimator is NOT a benefit attributable to p. We separately
retain the no_intention lesion. Supplying future goals is not intrinsic motivation.
Conventional RLS and MPC are not new inventions, nor do polar coordinates have an
advantage over an information-equivalent generic calculation by construction.

## A. Exploratory mechanism diagnosis

Eight seeds 961001..961008, two domains, weak/strong physical couplings, four fixed
state/target/action fixtures per seed/cell and 16 combinations: 2048 one-transition
interventions. Calibrate the same directed effect matrix on 64 observed transitions,
then FREEZE it across the interventions. Do not learn from intervention outcomes.
Independent flags: signed versus absolute residual (coactivation term unchanged);
coactivation-gradient present/absent (residual and receiving matrix unchanged);
directed receiver sensitivities versus within-pair row averaging (message unchanged);
and full versus own-pair-only effect matrix for the readout (same learned estimator).
All coefficients, state fixtures, resource projection and environmental noise remain
fixed within a paired fixture. This disaggregates the earlier compound interventions.
The readout is one declared synchronous step, not the complete multistep controller.
Report all cell means, main effects and two-way interactions, 10000 seed-block 95%
percentile intervals, RNG 961777. Descriptive exploratory evidence, not a unique
cause of all historical deterioration or a replacement of Study 3.

## B. Registered external closed-loop evaluation

Three operational polarities, six nonnegative channels, independent positive/negative
actuators. The two synthetic domains are NOT mere renamings:
- water: F=.91 I, constant drift=.0405, actuator retention=.5, negative cross-effects
  of filling another tank; normalized level in [0,1];
- thermal: F=.90 I + .0175*(ones-I), ambient-dependent drift, actuator retention=.7,
  positive heat diffusion from other zones; physical temperature T=16+16*x degC.
The environmental equations are h_next=a*h+(1-a)*u and
x_next=clip(F*x+drift+B*h_next+noise,0,1). Noise s.d.=.0015.
Public F, actuator retention and exogenous forecasts are identical for all policies.
True B, future noise and masked physical values are evaluator-only.
Own fill/heat gains in [.10,.14], own drain/cool gains in [-.14,-.10]; cross coefficient
magnitudes .006 or .055 times [.7,1.3]. Fixed versus per-channel gain change at t=24
is crossed with domain and coupling: EIGHT cells. Same seed supplies targets,
priorities, calibration commands and noise across those factorial manipulations.

There are 64 shared identification commands, then a physically reset 48-step episode.
Every controller gets its own observed transitions from the same calibration command
sequence; cross-coefficients are learned, not handed to the agent. Saturated outputs
are omitted from identification. Only valid observed rows with known preceding F
inputs update the online model. The prior estimator forgetting is .99, ridge .005.

Targets change over four 12-step episodes A/B/C/A. During the first three, the next
FIVE contractual targets and budgets are supplied to everyone; the forecast lesion
repeats only the current target while keeping the same horizon. On return to A,
target/forecast messages are withdrawn; the cue and retrieved content remain.
The regular target consumer competes with a lower-salience trusted maintenance
proposal and a high-salience unverified-source distractor. C copies selected content
to planner, memory and reporter. This is bounded content competition, not perception
or reasoning specialists trained independently. Memory, priorities and permissions
are preserved except in the named lesions.

Sensors for selected coordinates drop out at t mod12 in 4..8. Masked sensor entries
are zero placeholders both before actions and in feedback, never hidden true values.
Belief prediction uses only past measured state, public dynamics, learned B and
executed actions. observer_hold keeps the last measured value instead. This tests
state inference, not a subjective internal experience or the utility of p.
Budgets alternate 1.5/0.7 within each period. An action block and a protected negative
actuator rule apply in specified windows. Current normative permissions are used
throughout the planning horizon; future rule changes are not revealed in advance.

Conditions (12): full, diagonal, myopic, no_forecast, observer_hold, no_intention,
no_memory, no_priorities, utility_meta, legacy_review, generic_equivalent, zero.
The dense estimator is retained in the diagonal planning lesion; after different
actions its online coefficients can diverge. All use the same physical-information
interface. generic_equivalent computes the SAME algebra via an explicit Q matrix;
it is an equivalence check, not a competitor expected to lose. zero honors any
mandatory minimum; the current external benchmark has no nonzero mandatory requests.

Planning is accelerated projected quadratic optimization, H=5 (myopic H=1), 40
iterations, regularization=.0002 and polar continuity=.0002. Projection enforces
bounds, permissions, feasible lower obligations and cost-weighted budget, not global
physical safety. Every action records a projection residual and source parameters.
Clipping in the environment is NOT counted as a proof of controller stability.

## C. Utility-based metacognitive policy, not blind abstention

The base planner remains the fallback. Alternatives are a diagonal-model plan and
a feasible hold-last-action sequence. Each alternative is evaluated by the SAME
learned dense predictor. Score=predicted weighted tracking MSE + .0002 mean(action^2)
+ beta*estimated one-step predictive variance*(1+mean sensor age).
Switch only if predicted improvement exceeds margin, at most 12 switches per
48-step trajectory and a minimum two-step separation. No action is removed merely
because a success probability falls below a threshold. Actions stay subject to E.
This is an engineered expected-utility rule; its variance term is not claimed to be
a calibrated confidence interval or a robust-control theorem.

Development grid is exactly beta in {0,.25,1} with margin=.00001. Evaluate complete
closed loops for each on six development seeds. Select minimum mean MSE + .0001*
intervention fraction, then smaller beta. No retuning from final observations.
The forecast monitor is fitted first on BASE pre-feedback scores and the base
tracking-success event MSE<=.003, using the existing empirical-bin method. Legacy
review is the old RULE: use minimum feasible action when probability<.55, now with
a calibrator fitted to this development domain. It is not the identical historical
controller nor reuse of an out-of-domain old calibrator. Only base-policy held-out
forecasts are used to assess calibration; do not label base-action probabilities
as calibrated probabilities of a changed counterfactual policy.

## Development, public freeze and untouched final sets

Development: six seeds 971001..971006. 48 full runs fit the monitor; all twelve
conditions then produce 576 comparison runs; three meta settings produce 144 runs.
Development may expose defects; corrections must be reported and scientific sources
re-frozen before final. No outcome-based architecture change is silently relabeled
prospective. Mechanical fixtures use seeds below 800000.
Final: 24 seeds 981001..981024, 8 cells, 12 conditions =2304 complete runs and
110592 evaluated transitions, plus 64 calibration transitions per run.
The separate capability panel uses 24 seeds 991001..991024. None overlap prior
P0P2 seeds. Publish source hashes, development files, selected meta parameters and
monitor in a Git commit before final generation. The runner verifies public commit,
ancestry, registered freeze bytes, protected historical tree and file hashes. It
refuses to overwrite final evidence. A defect discovered after final exposure
invalidates that run; no repair and reuse may be called untouched confirmation.

## Primary measures and acceptance

Raw priority-weighted tracking MSE over all 48 steps. Secondary: switch-window MSE
(t in [s-3,s+2] for s=12,24,36), masked-input MSE, return-cue MSE, normalized action
cost, violations, interventions, physical clipping, and act+learn wall time.
The inferential unit is a complete seed averaged EQUALLY over all eight cells.
No step/channel/domain subgroup is a new independent seed. Runtime is descriptive,
machine-dependent; mode order is rotated by seed/cell, and meta computes more
candidates, so no claim of equal compute or latency superiority is assumed.

Four directional claims, 20000 paired seed bootstrap samples, RNG 981777,
one-sided 98.75% percentile upper bounds (Bonferroni .05/4):
H1 full minus diagonal, total MSE < -.0002.
H2 full minus myopic, switch-window MSE < -.0002.
H3 full minus observer_hold, masked-input MSE < -.0002.
H4 utility_meta minus legacy_review, total MSE < -.0002.
Additional H4 gates: utility_meta minus full upper98.75 <=+.0003; zero hard action
violations; intervention fraction between .01 and .25 inclusive. If the selector
never acts, it is a fallback identity result, not proof of useful active metacognition.
Cost/coverage and all failures are reported, not used to choose favorable tasks.
These finite engineering relevance margins are declared here, not universal cutoffs.

Validity: exact run set and 48-step records; replayed masked observations and physical
effects; checksum identity; independently recomputed outcomes; full/generic maximum
action difference <=1e-8; zero-policy MSE worse than full with descriptive 95% lower>0;
full and meta hard violations=0. Failure => invalid_evaluation. Otherwise report each
claim separately; all four plus meta gates => bounded_iteration2_support, otherwise
partial_iteration2_support. Failed criteria are not equivalence or impossibility.
Priorities, memory, polar continuity and forecast-only lesions are descriptive
replication/scope diagnostics, not additional confirmatory successes.

## D. Specific extension probes (not inferred from primary MSE)

Two-domain typed unit conversion/unknown-symbol rejection, competing content with
selective consumer cuts, and institutional protected-channel approval/precedence
checks. Three autonomous local agents receive only their own state, goal and model;
scalar offers travel with fixed message losses, duplicates and delays. Compare
freshness-aware offers (TTL2), indefinite retention of the same accepted offers,
and equal allocation. 24 seeds x2 domains x3 coordination modes=144 runs of 24 steps.
Report joint weighted error, resource violations and rejected packets; descriptive
seed-block 95% intervals only. Identity fields are consistency checks, NOT sender
authentication, truthful-preference guarantees or a Byzantine-agreement solution.
No extra independent-agent superiority claim is included in the four primary tests.
Semantic grounding remains operational; original philosophical labels and general
language, intrinsic motivation and consciousness are not established.

## Audit and manuscript

Archive every evaluated input, action, effect, masked feedback, learned model,
selected content, internal variables, candidate scores and intervention decision.
Recompute all outcomes, compare checksums, replay preselected complete controllers,
recalculate calibration and run the prior regression suite unchanged. Preserve prior
results and the original theoretical/architectural sources. Add the new realization
and outcomes to the same LaTeX draft, with a per-gap status matrix. No all-gaps-closed
claim is licensed by test counts or a favorable synthetic task alone.

Method bases: Rawlings/Mayne/Diehl, Model Predictive Control (2nd ed.);
Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html;
Dehaene/Kerszberg/Changeux (1998), doi:10.1073/pnas.95.24.14529.
These motivate conventional prediction, measured calibration and content access,
not validation of a unique polar theory. Public registration is not peer review.
