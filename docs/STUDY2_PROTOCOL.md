# Study 2 — prospective functional test of contextual pair coupling

This protocol is designed before its six-seed development pilot and is publicly
committed, together with source hashes and the pilot-derived sample-size freeze,
before any final seed is generated. It is a repository registration, not an
independently hosted preregistration or external replication.

## Hypothesis and causal estimand

Using learned within-pair effects in a contextual action planner lowers dynamic
tracking loss without unacceptable changes in failures, costs, reacquisition or
inventory consequences. The primary intervention disables only off-diagonal
effects in the planner while retaining the same paired estimator architecture,
memory, constraints, learning rule and parameter count. Subsequent actions change
feedback, so fitted estimates may diverge: the primary contrast estimates the
total closed-loop effect, including its mediation through later learning. The
controller unit test separately isolates an immediate intervention at fixed state.

The controls are paired, paired planning lesion, diagonal-only estimation,
shifted equal-size pairing, dense estimation, exact signed/intensity coordinates,
zero action and fixed 0.3 action. Every control uses the same public observations
and action constraints. The dense estimator has more coefficients, explicitly
reported; no false claim of identical total capacity is made. The exact
coordinate change is a consistency control, not evidence for a distinctive
ontology. `STUDY2_CONTROLLER_SPEC.md` maps all equations to code.

## Two internally designed task families

Each trial has two agents, two generic action types per agent and two independent
nonnegative poles per type: eight action channels. They are generic channels,
not validated psychological categories or all eight historical labels.

Tracking has persistent affine dynamics
`y_next = rho*y + (1-rho)*B*u + drift + noise`, with rho=0.55.
Finite-capacity inventory uses rho=0.93 and clips this transition to known
nonnegative stock bounds. Exogenous demand is public. Stockout/net-withdrawal
deficit and overflow are physical quantities calculated from the unclipped
transition, not alternate names for tracking loss. This saturation makes the
inventory task nonlinear. Controllers exclude censored rows from affine system
identification. Their common horizon planner remains affine, an explicit shared
model mismatch; state memory respects the known bounds.

The transition matrix and latent reference actions are hidden. Missing state
components are masked; zeros in those positions are placeholders accompanied by
an explicit boolean mask. Feedback observations and physical transition-validity
flags are separate. All controllers receive the same information. A periodic
observation schedule prevents a channel from remaining unseen indefinitely.

Three regimes test the structural prior: within-pair cross effects, diagonal
dynamics where pair coupling is unnecessary, and misaligned cyclic cross effects
with the same number of physical off-diagonal terms. The paired regime explicitly
aligns the model prior with the plant; superiority there alone is conditional
evidence, not universal or unique polar advantage. Feasible targets are generated
from hidden reference actions; this makes each target relative to its plant.
Different regimes therefore do not share exactly the same physical target vector.

Four 32-step phases use contexts 0,1,2,0. Context changes alter priorities,
available budget, plant coefficients and demand as declared; the horizon is fixed
within each family. The final
phase returns to the first target, plant and constraints, with fresh noise and
observation masks. Tail-loss change measures reacquisition in this task; it is
not by itself evidence of general memory retention or catastrophic forgetting.

## Fixed measurement, inference and decision

`study2_protocol.json` is the machine-readable authority for every threshold,
seed allocation and decision operand. The primary loss is weighted squared
error of the next physical state against the public target, without subtraction
of a constructed oracle. The inferential unit is a seed equally averaging all
six family-by-regime cells. The primary effect is paired minus paired lesion;
negative values favor use of coupling. Secondary contrasts and family/regime
intervals are descriptive, unadjusted and cannot replace the primary endpoint.

The minimum relevant effect is 0.002 squared state units, selected before the
pilot as an engineering threshold (sqrt(0.002)=0.0447), without biological or
clinical meaning. Five thousand paired-seed bootstrap resamples give percentile
95% intervals. Pilot SD determines sample size for an approximate 0.0015
halfwidth using the fixed formula, bounded to 40–60 fresh final seeds. Only
variance enters this formula. A cap below the unbounded requirement is reported
as a precision limitation; this is precision planning, not a power guarantee.

Tracking passes require loss <=0.06, zero hard action violations and recovery
after at least two of three switches. Recovery is the first three consecutive
steps with loss <=0.025, censored at the next phase. No recovery is null and
counts against success. Inventory stockouts/overflow remain separate consequence
guardrails, because perfect tracking of an empty-stock target does not fulfill
withdrawals. Invalid or missing records raise errors, never silently contribute
zero values.

A broader continuation decision requires an upper primary interval below -0.002,
all prespecified guardrails, improvement over shifted pairing and noninferiority
to dense estimation outside the aligned regime. Otherwise a benefit confined to
the aligned regime supports a conditional structural prior, or the exclusive
polar-advantage claim is suspended. Either result is scientifically reportable.
No outcome in this experiment establishes consciousness, ASI or a cosmological
principle. The separate consciousness protocol defines future functional tests.

## Execution and evidence

Use the pinned environment from `requirements-repro.txt`, with one CPU thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_study2.py --split pilot
python scripts/run_study2.py --make-freeze docs/STUDY2_FREEZE.json --pilot-manifest results_study2/coupling/pilot/manifest.json --reviewed-by "root and independent reviewer"
# Publish this source and freeze as a public Git commit, then use that exact SHA.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_study2.py --split final --registration-sha COMMIT_SHA
python scripts/regenerate_study2.py results_study2/coupling/final/manifest.json
```

The final command requires the reviewed freeze and exact hashes. Output paths
cannot overwrite existing trials. Every environment and controller record is
saved at float64 JSON precision with deterministic gzip as individual logical
trial files, transported in deterministic tar chunks smaller than 6 MB. The
manifest indexes every member and archive and declares exact expected cells and SHA256 hashes;
regeneration verifies completeness, identity, hashes, state continuity and physical
transitions before calculating outcomes. Complete observed information, actions,
estimates, planner matrices, memory, uncertainty, censoring and learning updates
are logged. Covariance diagonals are logged; full covariances are reproducible
from the initial ridge, observation/action sequence and RLS rule. Measured act
and learn walltimes are implementation-dependent and include instrumentation;
they are not hardware FLOP measurements. No model tuning, changed thresholds or
optional stopping is allowed after final data generation begins.
