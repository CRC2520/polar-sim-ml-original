# R45 result — official POPGym LSTM reproduced paired comparator

Frozen final resolution:

`R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_PASS`

## Canonical execution

Scientific workflow run: `35924763722`.

Only this run is adjudicative. The duplicate pull-request-triggered run `35924770044` is excluded from scientific interpretation by the prospectively recorded canonical-run rule.

Scientific execution commit:

`e1052cf9e766d5e6f8a3fcfd6f40a472a093cea1`

Historical external implementation:

`proroklab/popgym@e397e5e`

R45 trained newly reproduced checkpoints using the official POPGym LSTM model and PPO launcher configuration at the frozen bounded budget of 1,048,576 environment timesteps per checkpoint. These are not author-provided paper checkpoints and R45 does not reproduce the paper's 15M-timestep endpoint.

## Development — Easy

Training seeds: 2166001–2166003.

All three checkpoints are eligible under the preregistered gate:

| seed | median intact score | median step-reset score | history benefit |
|---|---:|---:|---:|
| 2166001 | 1.000000 | 0.233125 | +0.766875 |
| 2166002 | 1.000000 | 0.182500 | +0.817500 |
| 2166003 | 1.000000 | 0.182500 | +0.817500 |

Each checkpoint is evaluated on 12 held-out seeds. The frozen gate requires intact score >=0.90 and history benefit >=+0.15 in at least 2/3 checkpoints.

Observed: **3/3 eligible**.

Development resolution:

`R45_DEVELOPMENT_AUTHORIZE_CONFIRM`

Development-gate artifact: `10782023347`.

## Confirmatory — Medium

Training seeds: 2167001–2167003.

All three reproduced LSTM checkpoints are competent:

| seed | intact | step reset | history benefit |
|---|---:|---:|---:|
| 2167001 | 1.000000 | 0.113750 | +0.886250 |
| 2167002 | 1.000000 | 0.088750 | +0.911250 |
| 2167003 | 1.000000 | 0.088750 | +0.911250 |

Aggregate frozen endpoints:

- eligible checkpoints: **3/3**;
- median CORE_C score: **1.000000**;
- median CORE_C − LSTM: **0.000000**;
- median CORE_C − GENERIC_ISO absolute gap: **0.000000**;
- frozen compatibility margin: **−0.05**.

Medium: **PASS**.

## Confirmatory — Hard

All three reproduced LSTM checkpoints are competent:

| seed | intact | step reset | history benefit |
|---|---:|---:|---:|
| 2167001 | 1.000000 | 0.0591667 | +0.9408333 |
| 2167002 | 1.000000 | 0.0591667 | +0.9408333 |
| 2167003 | 1.000000 | 0.0591667 | +0.9408333 |

Aggregate frozen endpoints:

- eligible checkpoints: **3/3**;
- median CORE_C score: **1.000000**;
- median CORE_C − LSTM: **0.000000**;
- median CORE_C − GENERIC_ISO absolute gap: **0.000000**;
- frozen compatibility margin: **−0.05**.

Hard: **PASS**.

Confirmatory-gate artifact: `10785928926`.

## Interpretation

R45 closes the narrower direct reproduced-checkpoint comparator boundary left open by R44.

At the frozen R45 budget, newly trained checkpoints using the official historical POPGym LSTM implementation are competent recurrent comparators on Easy, Medium and Hard, and their recurrent state is strongly policy-causal under the step-reset lesion.

The frozen R44 CORE_C controller is compatible with those reproduced official-code LSTM checkpoints on the same held-out episode seeds in Medium and Hard. The median paired score gap is exactly zero in both levels.

This is **compatibility/noninferiority-style evidence, not superiority evidence**.

## Boundaries

R45 does not establish:

- POLAR superiority;
- comparison against an author-provided paper checkpoint;
- reproduction of the published 15M-timestep endpoint;
- external D or R;
- global mathematical minimality;
- E6b independent replication;
- E7 prospective biological correspondence;
- phenomenal consciousness.

POLAR Core v1.1 remains:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]
