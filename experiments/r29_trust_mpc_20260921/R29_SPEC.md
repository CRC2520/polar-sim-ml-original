# R29 — trust-region MPC strong-comparator campaign

R29 is the replacement for the invalid R27 MPC and the cancelled R28 shooting-MPC attempt.

The campaign is prospectively fixed before PR execution and uses a new seed family.

## External tasks
- continuous CartPole;
- continuous Pendulum;
- 3,200-step trajectories;
- eight regime blocks with repeated and novel physical parameter settings;
- partial observations only; no regime ID or true parameters for matched controllers.

## Matched controllers
- CORE: sparse adaptive effective model + recurrent state estimate + nominal safety anchor;
- GENERIC: dense adaptive effective model with the same information/history budget;
- TRUST_MPC: dense adaptive model, same recurrent state estimate and identification budget, receding-horizon evaluation of a small trust-region action set around the nominal/adaptive actions, with nominal LQR tail;
- FROZEN: nominal fixed LQR;
- ORACLE_MPC: privileged descriptive reference with true state/current physics.

The nominal safe action is always included in the TRUST_MPC candidate set. This makes the comparator testable as a strong controller rather than allowing an unstable optimizer to masquerade as evidence for CORE.

## Frozen decisions
1. Comparator validity must pass before any CORE-vs-MPC inference.
2. CORE classical-control non-inferiority is tested separately from exclusive CORE advantage.
3. CORE≈GENERIC is interpreted as implementation non-privilege.
4. Failure of exclusive advantage is retained.
5. E6b independent replication and E7 prospective biological validation cannot be closed internally.
