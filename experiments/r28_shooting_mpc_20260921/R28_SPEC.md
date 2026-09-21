# R28 — stabilized strong classical-control comparison

R28 corrects the invalid R27 finite-horizon comparator while preserving R27 as an adverse engineering record.

The comparison is prospectively fixed before PR execution. The same external CartPole/Pendulum tasks, partial observations, recurrent velocity/state estimator, online identification cadence, action limits and 3,200-step multi-shift schedule are retained.

Controllers:
- CORE: sparse adaptive relation/state model + adaptive LQR, anchored to the nominal safe policy.
- GENERIC: dense adaptive model with the same information/history budget.
- SHOOTING_MPC: dense online model + receding-horizon first-action shooting over an LQR tail policy. It uses the same estimated state and identification data as CORE/GENERIC.
- FROZEN: nominal fixed LQR.
- ORACLE_MPC: privileged descriptive reference using true state and current physical parameters.

The MPC comparison is considered scientifically valid only if the comparator itself satisfies frozen stability and baseline-competitiveness checks.

Primary questions:
1. Is CORE non-inferior to a valid strong classical controller?
2. Is there an exclusive CORE advantage of at least 10%?
3. Does CORE remain equivalent to GENERIC when both reconstruct the same effective organization?

Failure of exclusive advantage is retained and expected to be scientifically meaningful. E6b and E7 remain external/open.
