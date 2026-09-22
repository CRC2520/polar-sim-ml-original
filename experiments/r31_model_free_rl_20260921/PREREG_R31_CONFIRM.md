# R31 confirmatory protocol

R31 tests the remaining internal **model-free comparator** gap after R30.

The current policy-bandit source, candidate policies, interaction budget and all
decision thresholds are frozen before confirmatory seeds 1547001--1547012 are
opened.

## What the baseline is

MODEL_FREE_RL is a model-free policy-bandit over seven gain-scaled residual
policies around the same nominal safe controller. It receives the same partial
observations and recurrent state estimator used by the matched controllers. It
selects a frozen policy using exactly 3,200 realized-cost interactions per task
and does not fit or query an action-to-state transition model.

## Frozen questions

1. **Baseline validity.** The model-free baseline must itself be stable, improve
   materially over FROZEN, remain reasonably competitive with Adaptive-LQR, and
   select a non-nominal policy. If this fails, no CORE-vs-RL scientific
   inference is accepted.
2. **CORE non-inferiority.** With a valid baseline, CORE must stay within the
   frozen global and post-shift cost margins while preserving stability and the
   already-established CORE≈Adaptive-LQR relation.
3. **Practical equivalence.** Tested separately with a tighter symmetric band.
4. **Exclusive CORE advantage.** Requires >=10% median benefit plus the per-seed
   guardrail.
5. **Exclusive model-free advantage.** Symmetric >=10% median benefit in the
   opposite direction plus the per-seed guardrail.

A scientific FAIL is preserved as a result; the workflow should fail only for
integrity/execution violations.

## Fixed scope boundaries

R31 does not establish superiority over RL generally, cannot close E6b or E7,
and does not alter the R26 strong relational-transfer FAIL or R30 classical
comparison.
