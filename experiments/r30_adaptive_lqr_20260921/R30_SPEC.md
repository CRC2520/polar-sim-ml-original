# R30 — qualified dense Adaptive-LQR baseline

R30 closes the internal strong-classical-baseline question after the invalid R27/R29 MPC
comparators and the cancelled R28 shooting-MPC attempt.

The baseline is not called MPC. It is a **dense adaptive LQR controller** with the same:
- partial observations;
- recurrent velocity/state estimator;
- own-action history;
- online identification window and cadence;
- action limits;
- nominal safety anchor;
- execution horizon.

This is a matched classical adaptive-control baseline and is scientifically valid only if it
is itself stable and at least non-inferior to the frozen nominal controller.

## Tasks
Continuous CartPole and Pendulum, 6,400 steps, 16 regime blocks with repeated and novel
physical parameter settings. No controller receives regime identity or true physical parameters.

## Controllers
- CORE: sparse/adaptive effective state model + nominal/adaptive LQR blend.
- ADAPTIVE_LQR: dense adaptive state model + the same nominal/adaptive LQR blend.
- FROZEN: nominal fixed LQR.
- ORACLE_LQR: privileged descriptive reference with true state and current physics.

## Frozen decisions
1. Baseline validity is a prerequisite.
2. CORE-vs-ADAPTIVE_LQR non-inferiority is distinct from exclusive CORE advantage.
3. CORE≈ADAPTIVE_LQR supports implementation non-privilege.
4. Exclusive CORE advantage requires >=10% median benefit and seed guard.
5. E6b and E7 remain OPEN.
