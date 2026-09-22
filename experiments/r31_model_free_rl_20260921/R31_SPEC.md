# R31 — matched online model-free residual RL baseline

R31 addresses the remaining internal comparison gap after R30: a controller that
adapts **without fitting an action-to-state transition model**.

## Protocol

For each seed and task:

- 3,200 training/adaptation steps;
- 3,200 frozen-policy evaluation steps;
- continuous CartPole and Pendulum;
- partial observations only;
- the same recurrent velocity/state estimator;
- the same nominal LQR safety anchor and action bounds;
- identical exogenous seed/noise construction within a task;
- training physical regimes use nominal and shifted parameter sets;
- evaluation contains repeated and held-out parameter combinations.

Controllers:

- **CORE** — sparse online model identification + adaptive LQR residual around
  the nominal safe controller.
- **ADAPTIVE_LQR** — dense matched model-based baseline from R30.
- **MODEL_FREE_RL** — residual linear Q-learning over a discrete set of bounded
  residual actions around the same nominal safe controller. It receives state
  estimates, actions and realized scalar costs only; it never fits or queries a
  transition model.
- **FROZEN** — nominal fixed LQR.
- **ORACLE_LQR** — privileged descriptive reference using true state/current
  physics; not a matched competitor.

All adaptive matched controllers receive exactly 3,200 training interactions
per task before the evaluation weights/model are frozen.

## Development and confirmation

Development seeds: 1546001--1546004.

Confirmatory seeds 1547001--1547012 are reserved and MUST NOT be executed until
the source, hyperparameters, validity criteria and decision rules are frozen.

R31 can internally test whether CORE remains competitive with one specific
matched model-free learner. It cannot establish superiority over RL as a field,
cannot close E6b, and cannot close E7.
