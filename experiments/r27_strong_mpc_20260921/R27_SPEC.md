# R27 — strong external controller comparison

Status: prospective fixed-criteria campaign, 2026-09-21.

R27 addresses the remaining internal engineering question after R26:

> Does the currently supported POLAR organizational core remain competitive with
> a strong classical adaptive controller when both receive the same partial
> observations, recurrent state estimate, online identification budget and action limits?

This is **not** an independent replication and cannot close E6b.

## Tasks

Two canonical continuous-control systems are used without POLAR semantic pairs:

- CartPole continuous-force dynamics.
- Pendulum continuous-torque dynamics.

Each trajectory lasts 3,200 steps and contains seven physical-regime changes,
including returns to previously encountered regimes. The controller does not
receive the regime ID or true physical parameters.

## Controllers

- **CORE** — R26-style adaptive recurrent state estimate + online action-effect
  identification + sparse effective relation model + adaptive LQR action.
- **GENERIC** — same state/history/information budget with dense adaptive model.
- **MPC** — dense adaptive model with finite-horizon Riccati/MPC control,
  same observations/history/identification cadence and action bounds.
- **FROZEN** — nominal fixed controller.
- **ORACLE** — descriptive privileged reference with true state and current
  physics; not a matched competitor.

## Questions

1. **External classical-control non-inferiority**:
   CORE should remain stable and no worse than 10% relative mean cost versus
   matched adaptive MPC, with a separate post-shift non-inferiority bound.
2. **Exclusive CORE advantage**:
   tested separately at a 10% improvement margin. Failure does not negate
   non-inferiority; it means no privileged POLAR advantage is established.
3. **Generic equivalence**:
   CORE and GENERIC should remain close when they reconstruct the same effective
   organization.

## Fixed boundaries

- R26 strong H_TRANSFER remains FAIL unless a genuinely new frozen criterion says otherwise.
- A CORE≈MPC or CORE≈GENERIC result supports implementation non-privilege, not superiority.
- ORACLE is descriptive only.
- E6b independent replication and E7 prospective biological validation remain OPEN.
