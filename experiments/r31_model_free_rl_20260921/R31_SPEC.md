# R31 — matched model-free policy-bandit baseline

R31 addresses the remaining internal comparison gap after R30: a controller
that adapts **without fitting or querying an action-to-state transition model**.

## Protocol

For each seed and task:

- 3,200 training interactions;
- 3,200 frozen-policy evaluation steps;
- continuous CartPole and Pendulum;
- partial observations only;
- the same recurrent velocity/state estimator;
- the same nominal LQR safety anchor and action bounds;
- evaluation includes repeated and held-out physical parameter regimes.

Controllers:

- **CORE** — sparse online model identification + adaptive LQR residual around
  the nominal safe controller.
- **ADAPTIVE_LQR** — dense matched model-based baseline from R30.
- **MODEL_FREE_RL** — a model-free policy-bandit over seven fixed gain-scaled
  residual policies around the same nominal controller. Training uses only
  realized scalar costs from policy rollouts; no transition model is fit or
  queried.
- **FROZEN** — nominal fixed LQR.
- **ORACLE_LQR** — privileged descriptive reference with true state/current
  physics; not a matched competitor.

The model-free training budget is exactly 3,200 interactions per task:
two paired 200-step evaluations of all seven policy arms (2*7*200 = 2,800),
followed by one paired 200-step refinement of the best two arms (2*200 = 400).
The selected policy is frozen before evaluation.

## Development history

Development seeds are 1546001--1546004 only.

The first residual online-Q implementation was stable but invalid as a strong
baseline: it incurred ~4.92x FROZEN cost. A conservative fitted-Q revision
collapsed to the nominal policy (nonzero-action fraction 0). Both outcomes were
development-only. Before any confirmatory seed was opened, the learner was
reformulated as the current policy-bandit, which is model-free and uses the
same interaction budget.

Confirmatory seeds 1547001--1547012 remain reserved until the final source,
hyperparameters, criteria and source hash are frozen.

## Scope

R31 can test whether CORE remains competitive with **this specific matched
model-free learner**. It cannot establish superiority over RL as a field, cannot
close E6b, and cannot close E7.
