# R38 — external POPGym robust recurrent observer

Status: prospective development-selection campaign.
Date: 22 September 2026.
Parent: R37_DEVELOPMENT_FAIL_NO_CONFIRM.

R38 does not modify POLAR Core v1.1 and does not reopen the prohibited R36-R2 tuning loop.
It targets the external observer-robustness failure localized by R37.

## External benchmark

Exactly the same unmodified third-party POPGym 1.0.7 environments as R37:
- PositionOnlyCartPoleMedium
- PositionOnlyCartPoleHard
- NoisyPositionOnlyCartPoleMedium
- NoisyPositionOnlyCartPoleHard

The package version, upstream commit and outcome thresholds are unchanged.

## Fixed controller structure

All recurrent variants use:
- only the public 2D POPGym observation;
- previous executed action;
- nominal CartPole physics as a generic state observer;
- the same nominal LQR action policy;
- no environment get_state(), noise_sigma, regime label, hidden velocity or reward shaping.

NO_C_CURRENT and ORACLE_STATE retain the R37 definitions.
GENERIC_ISO remains an exact coordinate recoding of the selected recurrent state.

## Development observer variants

- KF_R01: fixed Kalman-style observation variance 0.01.
- KF_R04: fixed variance 0.04.
- KF_R09: fixed variance 0.09.
- KF_ADAPT: innovation-variance adaptation initialized at 0.04, clipped to [0.01, 0.16].

All use the same process covariance and nominal dynamics.

Development seeds: 2046001–2046008.
Confirmatory seeds: 2047001–2047012 remain unopened until selection and source freeze.

## Unchanged scientific gate

The R37 numerical outcome thresholds are reused without relaxation:
- median CORE score >= 0.70;
- median CORE - NO_C >= +0.05;
- median isomorphic gap <= 1e-12;
- median oracle score >= 0.95;
- per-seed guard: CORE >=0.60, gain >=0.03, iso gap <=1e-12.

Development eligibility requires >=6/8 full seed guards and all median criteria.
Among eligible variants:
1. maximize minimum environment CORE score;
2. maximize overall CORE score;
3. maximize CORE-NO_C gain;
4. tie-break by simplicity: KF_R04, KF_R09, KF_R01, KF_ADAPT.

If no variant is eligible, R38 development FAILS and no confirmation is authorized.

## Boundaries

- R38 tests external robustness of C/persistent-state estimation only.
- It does not test D/R necessity or the full higher-order conjunction.
- A generic model-based observer is intentionally admissible; algorithmic POLAR privilege is not predicted.
- R37 confirmatory seeds 2037001–2037012 remain unopened.
- R36-R2 confirmatory seeds 2027001–2027012 remain unopened.
- E6b and E7 remain OPEN.
