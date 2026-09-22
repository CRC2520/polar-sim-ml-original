# R39 — independently developed recurrent baseline on external POPGym tasks

Status: prospective baseline-comparison campaign, 2026-09-22.

R39 follows the R37/R38 decision to stop tuning POLAR observers on the POPGym
panel. The POLAR-side controller is frozen exactly as the best R38 development
variant (`KF_R09`). The only new learner is **RecurrentPPO** from the third-party
`sb3-contrib` package.

This is an independently developed *baseline implementation*, not an independent
replication of POLAR. E6b therefore remains OPEN.

## Environments

The benchmark remains the unmodified POPGym 1.0.7 control panel used in R37/R38:

- PositionOnlyCartPoleMedium
- PositionOnlyCartPoleHard
- NoisyPositionOnlyCartPoleMedium
- NoisyPositionOnlyCartPoleHard

R37/R38 confirmatory seeds remain unopened. R39 uses new evaluation seeds.

## Frozen POLAR-side controller

- R38 `KF_R09` recurrent observer;
- same nominal LQR action rule;
- no parameter change, retraining, threshold adjustment or environment-specific tuning;
- GENERIC_ISO remains the exact coordinate-isomorphic control.

## Independent recurrent baseline

- package: `sb3-contrib==2.7.1`;
- algorithm: `RecurrentPPO`;
- policy: `MlpLstmPolicy`;
- one separately trained model per environment;
- fixed hyperparameters across all four environments;
- 100,000 training transitions per environment;
- one fixed predeclared training seed per environment;
- deterministic evaluation.

No hyperparameter sweep is allowed on the R39 environment panel.

## Decision structure

1. **Baseline validity** is evaluated first.
2. Confirmatory evaluation is authorized only if the independently developed
   recurrent baseline clears the frozen development validity gate.
3. If valid, CORE competitiveness is judged separately from baseline validity.
4. No result from R39 can establish POLAR algorithmic privilege.
5. A valid independently developed baseline outperforming frozen CORE is retained
   as adverse external evidence, not tuned away.
6. E6b and E7 remain OPEN.

R39 does not reuse or open the R37/R38 confirmatory seed panels.
