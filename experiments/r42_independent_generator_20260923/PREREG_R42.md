# R42 — independent-generator / independent-implementation D+C+R transport

Status: prospective design before development and confirmatory seeds are opened.
Date: 23 September 2026 (America/Lima).
Parent evidence: R41 P1_SCALE_NORMALIZED_CORE_TRANSPORT_PASS.

## Scientific question

Does the current reduced organizational core candidate D+C+R retain its causal signatures in a fresh task generator and a new controller implementation that does not import or call the R26–R41 scientific runners?

R42 is still executed by the same research program. It is therefore NOT E6b independent replication even if code and task generation are independent of the historical implementation.

## Independence constraints

The scientific runner in this directory:
- imports only NumPy and Python standard-library modules;
- does not import R36/R36-R1/R41 controllers, environments, thresholds or helper functions;
- uses a 4D state, five latent regimes, nine discrete control actions and a ten-block lifetime;
- implements its own Bayesian recurrent context filter, current-only matched shadow filter, online ridge system identification and two-step MPC;
- uses fresh generator families and seed ranges.

No source from historical experimental runners is called at runtime.

## Task generator

State dimension N=4. Five hidden regimes. Ten blocks of 200 steps. Block targets are generated independently from hidden-regime identities.

Transition:
x_{t+1} = A_r x_t + B u_t + b_r + 0.025*tanh(H_r x_t) + process_noise + sparse_shock.

The controller is intentionally a generic online LINEAR identifier; the small nonlinear term is unmodelled structure, preventing exact simulator matching.

Context is signaled only by a noisy 3D cue generated from regime-specific centers. The true regime is unavailable to the adaptive controller.

Development generator family:
- sparse_mixed.

Confirmatory held-out generator families:
- ring_signed;
- lowrank_skew.

The source code for all families is frozen before any confirmatory seed is opened.

## Controller

R42Adaptive:
- recurrent Bayesian belief with fixed stay probability 0.97;
- current-only matched shadow belief using the same cue likelihood without historical prior;
- separate online ridge models per inferred context;
- fixed broad-excitation phase per context;
- two-step finite-action MPC after model fitting.

The current-only comparator has the same feature space, update budget and model capacity, differing only in removal of recurrent belief.

## D/C/R interventions

All lesion effects are evaluated on the same realized transitions and chosen action after the corresponding online model is fitted.

D:
- replace the four differentiated state coordinates by their scalar mean repeated across dimensions before prediction.

R:
- retain self terms, action map and bias but set all learned off-diagonal state relations to zero.

C:
- compare the recurrent-context model prediction with the matched current-cue-only shadow model prediction.

For each component:
delta = MSE(lesion_prediction, observed_next_state) - MSE(intact_prediction, observed_next_state)
and the primary effect is within-seed dz = mean(delta)/sd(delta).

## Control capability

Matched evaluator controls are:
- RANDOM: random action indices from a separate predeclared RNG stream;
- ORACLE: two-step MPC using the true current regime dynamics, including the fixed nonlinear term.

Normalized capability:
score = (cost_RANDOM - cost_AGENT)/(cost_RANDOM - cost_ORACLE).

No clipping is applied.

## Coordinate / implementation non-privilege control

A second adaptive controller receives an orthogonal rotation P of state and target and uses rotated action vectors P*u, while the physical environment remains unchanged. It uses the same generic algorithm and cue observations.

Primary non-privilege diagnostics:
- native/rotated action-index agreement;
- absolute normalized-score gap.

The rotation is an evaluator-chosen coordinate transformation, not a separate scientific mechanism.

## Development and confirmation

Development:
- seeds 2136001–2136004;
- family sparse_mixed only.

Development is instrument validation only. It may reveal infrastructure/design defects before confirmatory seeds are opened. Any amendment must be committed before confirmation and retained.

Frozen confirmatory seeds:
- 2137001–2137016.

Each confirmatory seed is evaluated on BOTH held-out families ring_signed and lowrank_skew, giving 16 paired cases per family.

## Confirmatory criteria

Per native seed/family:
- normalized score >= 0.50;
- D dz >= 0.20;
- C dz >= 0.20;
- R dz >= 0.20;
- stable fraction >= 0.99.

Family-level support:
- >=12/16 native conjunction passes in ring_signed;
- >=12/16 native conjunction passes in lowrank_skew.

Coordinate non-privilege:
- median native/rotated action agreement >=0.98 in each family;
- median absolute normalized-score gap <=0.02 in each family.

R42 PASS requires all family and coordinate criteria.

A negative result remains negative. Thresholds cannot be changed after confirmatory seeds are opened.

## Interpretation boundaries

A PASS supports same-program transfer of the D+C+R organizational signatures to a separately coded generator/controller family. It does NOT establish:
- E6b independent replication;
- global minimality or universal sufficiency;
- implementation privilege;
- universal practical superiority of recurrence;
- biological correspondence;
- phenomenal consciousness or AGI/ASI.

POLAR Core v1.1 membership remains unchanged unless a separate versioned theory revision is justified.
