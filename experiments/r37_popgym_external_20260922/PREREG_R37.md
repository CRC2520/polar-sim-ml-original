# R37 — external POPGym persistent-state validation

Status: prospective external-task campaign, development gate first.
Date: 22 September 2026.

## Motivation

R36-R2 ended same-program tuning of the unified higher-order conjunction.
R37 therefore moves to an environment library developed outside POLAR.

External dependency:
- package: popgym
- version: 1.0.7
- upstream: proroklab/popgym
- PyPI wheel SHA-256:
  23be7f754f3167f58c4ac844426f3e0f7230035bedfcd5d85b004524eef21bb6

R37 does NOT claim independent replication. The POLAR team still implements and
runs the controller. E6b remains OPEN.

## External environments

Unmodified POPGym registered implementations:
- PositionOnlyCartPoleMedium
- PositionOnlyCartPoleHard
- NoisyPositionOnlyCartPoleMedium
- NoisyPositionOnlyCartPoleHard

POPGym removes velocity information in PositionOnlyCartPole; this is a third-party
partially observable control problem intended to require memory.

## Frozen controllers

All matched non-oracle controllers use the same nominal CartPole LQR gain and
receive only the POPGym observation.

- CORE_C: recurrent finite-difference state estimator with the pre-existing
  R30/R31 EMA convention; no environment hidden state.
- GENERIC_ISO: an invertible signed-coordinate recoding of the exact same
  recurrent state and correspondingly transformed feedback gain.
- NO_C_CURRENT: current visible positions only; hidden velocities fixed to zero.
- ORACLE_STATE: descriptive reference using POPGym get_state(); evaluator-only,
  not a matched competitor.

No training, reward shaping, environment parameter modification, or task-specific
threshold tuning is allowed.

## Outcome

Per episode:
  normalized duration = executed steps / environment max_episode_length.

Per seed:
  equal average across the four external environment classes.

Primary frozen criteria:
- median CORE_C score >= 0.70;
- median CORE_C - NO_C_CURRENT >= +0.05;
- median |CORE_C - GENERIC_ISO| <= 1e-12;
- median ORACLE_STATE score >= 0.95.

Per-seed guardrail (required 9/12 confirmatory):
- CORE_C score >= 0.60;
- CORE_C - NO_C_CURRENT >= +0.03;
- |CORE_C - GENERIC_ISO| <= 1e-12.

Development seeds: 2036001–2036004.
Development is eligible only if at least 3/4 satisfy the per-seed guardrail and
all four median criteria pass.

Confirmatory seeds: 2037001–2037012.
They MUST remain unopened unless development is eligible and the exact source
hash is frozen. Confirmatory thresholds are identical to those above; no
post-development relaxation is permitted.

## Interpretation

A PASS would support external-task transfer of the C/persistent-state contract
and coordinate non-privilege. It would NOT establish:
- D or R necessity in POPGym;
- the full R36 higher-order conjunction;
- E6b independent replication;
- E7;
- consciousness or AGI.

A FAIL constrains external transfer and does not authorize tuning on confirmatory
seeds.
