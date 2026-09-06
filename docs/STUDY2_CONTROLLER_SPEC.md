# Study 2: structured transition identification and contextual planning

Status: prospective implementation specification, established before the second
study's final evaluation. Entry point: `study2/controllers.py`; configuration is
`ControllerConfig`; execution is `CoupledController.act` followed exactly once by
`learn`. This is a computational control hypothesis, not a consciousness measure
or a claim that physical existence requires polarity. Study-one source is unchanged.

## Hypothesis and its limits

The distinctive operation is the use of **learned cross-effects between nominated
action/output pairs in a forward planner**. Two coordinate names are insufficient.
The primary comparison keeps the same within-pair estimator, covariance,
observation channel, state memory, planner iterations, action constraints and
initial values, but removes the estimated cross-effects from the planning
operator. The coefficients remain learned and remain available to state memory.
This isolates their use in planning, including subsequent effects of changed
actions on future learning; it does not isolate all uses of coupling in the system.

An advantage in aligned environments would support a finite-data structural prior
for this controller and these tasks. A penalty when relationships do not align is
a predicted possibility and must be reported. Dense and equally sparse shuffled
controls help distinguish useful prior structure from mere parameter reduction.
Nothing establishes a universal polar organization or a uniquely conscious process.

## Variables, access, ranges and timing

| Symbol | Meaning and range | Code / update |
|---|---|---|
| `n` | Even number of action and output channels, at least four; study default eight | `ControllerConfig.channels` |
| `q_t` | Own nonnegative action, each channel in `[0,1]` | `act`, `project_action` |
| `y_t` | Observed plant state; any finite real value, subject to optional known stock bounds | Required `observation.state` |
| `o_t` | Boolean observation mask | Required `observation.observed` |
| `r_t` | Supplied desired state, finite real vector | Required `observation.target` |
| `w_t,c_t` | Strictly positive priority weights and resource costs | Required `weights,costs` |
| `a_t,b_t` | Boolean allowed-action mask and finite nonnegative total budget | Required `allowed,budget` |
| `rho_t` | Known persistence, `0 <= rho < 1` | Required `persistence` |
| `H_t` | Positive integer planning horizon | Required `horizon` |
| `d_t` | Known exogenous per-step drift, finite vector; zero if explicitly omitted | Optional `drift` |
| `l_t,h_t` | Known finite lower/upper state bounds when supplied; absent sides are unbounded | Optional `state_lower,state_upper` |
| `B` | Actual action-effect matrix; inaccessible to controller | Environment only |
| `Bhat_t` | Learned coefficients, finite real values; signed cross-effects allowed | `B`, `learn` |
| `M` | Boolean structural edge mask | `edge_mask`, set by mode |
| `P_i` | RLS covariance on active features for output `i` | `covariance`, `learn` |
| `v_i` | Exponential residual second moment, initially `0.05`; descriptive uncertainty | `error_variance`, `learn` |
| `m_t` | Persistent predicted/observed state, finite vector after initialization | `memory_state` |
| `s,I` | Action sign difference `s=q_0-q_1` and intensity `I=q_0+q_1` for each pair | `_signed,_intensity` |

The initial state, and the first state after an explicit memory erasure, must be
fully observed. Missing state keys and nonfinite vectors produce errors. Masked
channels must use finite placeholders; their values never affect actions or
learning, and traces store them as `null`. Supplying an empty/zero state is never
an implicit substitute for a missing state. `act` refuses to overwrite an
unconsumed action; `learn` refuses feedback without a pending own action.

The controller reads only its documented input keys. Undocumented truth matrices,
answer keys and evaluator metrics do not enter the calculation. Context is a
trace label: priorities, targets, horizons, constraints and observed transitions
carry functional contextual information; a string label does not reveal plant
parameters or select a hidden pre-trained model.

## Dynamic state and memory

The affine transition model is

`y_(t+1) = rho_t y_t + (1-rho_t) B_t q_t + d_t + noise_t`.

The inventory family additionally clips this transition to its known stock bounds.
The controller's prior state prediction supplies only unobserved channels:

`yhat_t = where(o_t, y_t, m_t)`.

Before feedback, the model records its one-step prediction using its current
estimated matrix, own action, known drift and persistence. This prediction is
clipped to supplied stock bounds. After feedback,

`m_(t+1) = where(o_(t+1), y_(t+1), predicted_y_(t+1))`.

Memory is therefore an internal state estimate, not repeated external stimulation.
It is not episodic or autobiographical memory. `erase_state_memory` is a logged
intervention; initialization requirements apply again after erasure. The first
study contains the separately tested episodic memory mechanism.

Memory uncertainty propagates as

`U_(t+1) = where(o_(t+1), 0, rho_t^2 U_t + (1-rho_t)^2 predictive_variance)`.

Prediction variance is the explicit RLS diagnostic

`predictive_variance_i = v_i (1 + q_t^T P_i q_t)`.

It is not a calibrated confidence interval; it does not independently select an
action or certify reliable self-knowledge. The learned operator itself participates
in planning and is the limited functional capacity model tested here.

## Identification from own transitions

For each row `i`, update only if both consecutive state samples are observed and
the environment's sensor reports an uncensored affine transition. A censored stock
value remains valid for state memory; it is excluded only from identification.
Let `J_i` be the active columns under `M`, `x=q_t[J_i]`, and

`z_i = (y_(t+1,i) - rho_t y_(t,i) - d_(t,i)) / (1-rho_t)`.

With forgetting `lambda=0.98`, ridge `r=1`, initial `Bhat=0.5 I` and
`P_i=I/r` on active columns:

1. `epsilon_i = z_i - Bhat_i[J_i] x`.
2. `k_i = P_i x / (lambda + x^T P_i x)`.
3. `Bhat_i[J_i] <- Bhat_i[J_i] + k_i epsilon_i`.
4. `P_i <- (P_i - k_i x^T P_i)/lambda`, then arithmetic symmetrization.
5. `v_i <- 0.95 v_i + 0.05 epsilon_i^2`.

The covariance/coefficients of skipped rows are unchanged; skipped updates are
logged, not counted as zero-error samples. Observability, excitation, noise,
saturation selection and structural misspecification limit identification. The
procedure does not guarantee unbiased estimation. No true `B`, realized noise, or
latent reference action is provided to `learn`.

## Contextual constrained planner

Define the planning operator `C=Bhat`, except that a planning lesion uses
`C=diag(diag(Bhat))`. The learned matrix and state-memory operator are unchanged
by this intervention at the moment it is applied.

Assuming the current action, context, drift and persistence remain constant over
the supplied horizon:

`f_H(q) = rho^H yhat + (1-rho^H)/(1-rho) d + (1-rho^H) C q`.

The shared finite-horizon objective is

`J(q) = sum_i [w_i/sum(w)] [f_H(q)_i-r_i]^2`

`       + 0.002 (c^T q)/n + 0.002 ||q-q_previous||^2/n`.

All modes perform 24 projected-gradient iterations from the projected previous
action. With `A=(1-rho^H)C`, the Hessian is

`Q=2 A^T diag(w/sum(w)) A + (2*movement_penalty/n) I`.

The gradient step uses the largest eigenvalue of `Q`, lower bounded by `1e-12`.
The action projection minimizes Euclidean distance to the proposal subject to

`0 <= q_i <= 1`, `q_i=0 if not allowed_i`, and `sum_i c_i q_i <= budget`.

`project_action` solves the weighted capped-simplex threshold using exact
piecewise-linear breakpoints. Numerical stability, resource feasibility and task
performance are separate mechanisms and separate evaluation quantities. No graph
homogeneity is used as an ethical constraint or success criterion.

For bounded stocks the horizon planner intentionally remains an affine
approximation shared by all modes; only one-step state memory respects output
clipping. This explicit model mismatch must be considered in inventory results.
The finite iteration budget also does not guarantee convergence to the exact
constrained optimum.

## Conditions and capacity accounting

| Mode | Estimated edges | Edges used in planning | Interpretation |
|---|---|---|---|
| `paired` | Diagonal plus both directed cross-edges within adjacent pairs | All estimated edges | Proposed structural prior |
| `paired_lesion` | Exactly the same structure and estimator as paired | Diagonal only | Primary intervention on cross-effect use in planning |
| `diagonal` | Diagonal only | Diagonal | Reduced structural estimator control |
| `shuffled` | Diagonal plus a shifted, disjoint pairing, e.g. `(7,0),(1,2),(3,4),(5,6)` | All estimated edges | Equally sparse alternative prior |
| `dense` | Every directed edge | All estimated edges | Generic dense identified transition model |
| `signed_intensity` | Same as paired | Same as paired | Invertible coordinate-consistency control |

The primary conditions have identical initial coefficients, covariances and
active parameter counts. After their actions diverge, their learned coefficients
can diverge; claiming identical *realized learned weights* would be incorrect.
The shuffled condition has matched sparsity, while dense and diagonal explicitly
have different parameter and identification operation counts. `profile` reports
coefficient counts, independent covariance entries, state memory, previous action,
uncertainty entries and multiplication-term estimates. These estimates exclude
some overhead, eigendecomposition and memory access; actual elapsed time belongs
in the run data. They are not measured hardware FLOPs.

For eight channels, coefficient counts are 16 for paired/lesion/shuffled/signed,
8 for diagonal and 64 for dense; independent covariance entries are 24, 8 and
288, respectively. All modes store dense arrays and use the same-size dense
planner implementation, so mathematical sparsity does not imply proportional
wall-clock savings.

## Pair meanings and equivalence

Pairs denote nominated interacting actuator/output channels in these synthetic
tasks. Their meanings are specified by the observable demands, costs and action
effects, not by assigning validated emotional or metaphysical attributes. Positive
and negative learned cross-effects can express cooperation or interference.
Independent nonnegative actions permit inactivity `(0,0)`, either predominance and
coactivation. The model contains no force requiring all channels or agents to agree.

The inverse coordinate map is `q_0=(I+s)/2`, `q_1=(I-s)/2` with feasible bounds
`0<=I<=2` and `|s|<=min(I,2-I)`. `signed_intensity` stores previous action through
this map and otherwise executes the same equations; equivalent behavior is
expected and is not evidence of a new advantage.

For a channel permutation `P`, `relabel` transforms the learned operator as
`Bhat'=P Bhat P^T`, its edge mask likewise, and each covariance on output and both
feature axes. State memory, own actions and uncertainty are also permuted. The
caller must transform all future state, target, cost, priority, constraint and
feedback channels. A pole-name reversal is a special case and preserves equivalent
behavior. Merely switching sign labels without the corresponding relationships
would be a different system.

## Trace contract and verification

`last_trace` is JSON serializable with `allow_nan=False` after feedback. It records
documented observation values/masks, internal state-memory sources, action,
planning/estimated matrices, predictions made before feedback, uncertainty,
constraints, RLS residuals and valid/skipped rows, updated coefficients and memory,
configuration and intervention events. Off-diagonal covariance entries are not
duplicated in every record: the initial covariance plus all actions, masks,
transition samples and the exact update above reconstruct the full covariance.
Diagonal covariance diagnostics are logged before and after updates.

Meaningful regression tests in `tests/test_study2_controller.py` cover:

- Invalid/missing state and uninitialized hidden-state errors.
- Budget, blocked actions and projection optimality conditions.
- Own-action/feedback timing and preservation of pre-feedback predictions.
- Hidden truth-key and masked-value isolation.
- Immediate causal planning lesion with unchanged estimator state/capacity.
- Signed-intensity equivalence through an adaptive closed loop.
- Pole relabeling with transformed learned relations through continued learning.
- Censored stock samples excluded from identification but retained in memory.
- Identification of interacting effects and adaptation after a coupling sign shift.

The tests validate specified behavior, not consciousness or general intelligence.
