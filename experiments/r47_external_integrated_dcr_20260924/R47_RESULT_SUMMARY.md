# R47 result — external integrated D+C+R on one POPGym Concentration agent

Frozen final resolution:

`R47_EXTERNAL_INTEGRATED_DCR_PASS_SAME_PROGRAM`

## Canonical execution

Scientific workflow run:

`36043862460`

Scientific execution commit:

`a0ed7720f0b0a03f39e6d6ff10720f30e7c3b624`

External benchmark:

- repository: `proroklab/popgym`;
- exact historical source commit: `e397e5eac9965f9963d18c9f455cd1983bca14fb`;
- environment: `ConcentrationHard`;
- environment source unchanged.

Frozen source integrity:

- protocol SHA-256: `db4441710958efef2e0e7092f2ef71ca9d53ca8fba241fdd3aa149c27ae84dad`;
- runner SHA-256: `1c15169b7e829ab5e0c0c3bf0726b5f5e408d0304087aa8d5ec58652b56f4c0e`.

## Engineering smoke

Seed `2185000` is explicitly non-scientific.

- FULL completion: 1.0000;
- FULL return: 0.6971;
- D effect: +0.8558;
- C effect: +0.9231;
- R effect: +0.9808;
- GENERIC_ISO completion gap: 0;
- GENERIC_ISO return gap: 0;
- ORACLE completion: 1.0.

Smoke artifact: `10827237948`.

The smoke was used only to verify execution/API/isomorphism invariants. Scientific thresholds were already frozen.

## Development

Scientific seeds:

`2186001–2186016`.

Four paired external deck shuffles per statistical seed.

Medians:

- FULL completion: **1.000000**;
- FULL native return: **0.697115**;
- D lesion effect: **+0.841346**;
- C persistent-memory lesion effect: **+0.932692**;
- R wrong-binding lesion effect: **+0.980769**;
- GENERIC_ISO completion gap: **0**;
- GENERIC_ISO return gap: **0**;
- ORACLE completion: **1.000000**;
- same-agent seed guard: **16/16** (required 13/16).

Development resolution:

`R47_DEVELOPMENT_AUTHORIZE_CONFIRM`.

Development artifact: `10827882138`.

## Confirmatory panel

Reserved seeds:

`2187001–2187032`.

Four paired external deck shuffles per statistical seed.

Medians:

- FULL completion: **1.000000**;
- FULL native return: **0.699519**;
- D lesion effect: **+0.836538**;
- C persistent-memory lesion effect: **+0.942308**;
- R wrong-binding lesion effect: **+0.990385**;
- GENERIC_ISO completion gap: **0**;
- GENERIC_ISO return gap: **0**;
- ORACLE completion: **1.000000**;
- same-agent seed guard: **32/32** (required 28/32).

Confirmatory resolution:

`R47_EXTERNAL_INTEGRATED_DCR_PASS_SAME_PROGRAM`.

Confirmatory artifact: `10827068508`.

## Integrity

Integrity artifact:

`10827103407`.

The verifier recomputes and passes:

- exact development seeds;
- exact confirmatory seeds;
- four paired episodes per seed;
- exact external POPGym source;
- all seed-level same-agent guards;
- both isomorphic gaps;
- development authorization;
- final confirmatory resolution;
- bounded single-task D+C+R support;
- bounded online relational acquisition support;
- all retained epistemic boundaries.

## Scientific interpretation

R47 closes the specific boundary left after R45/R46:

> D, C and R are no longer supported only in separate external tasks. In unmodified POPGym ConcentrationHard, one persistent functional agent requires all three contracts on the same statistical seeds.

The intact agent:
- preserves distinct rank identities (D);
- retains hidden card/location information across pair attempts (C);
- acquires and queries the position↔identity relation online inside each externally shuffled episode (R).

Removing each contract independently causes a large preregistered completion loss while the other machinery remains present. The same agent therefore shows a same-task external D+C+R conjunction.

The exact GENERIC_ISO equivalence shows that the result does not depend on the privileged native rank labels.

## What R47 newly supports

- external same-task functional D+C+R conjunction: **SUPPORTED_SAME_PROGRAM**;
- bounded external online R acquisition/binding from observations: **SUPPORTED_BOUNDED_TASK**;
- same-seed causal necessity of D, C and R in ConcentrationHard: **SUPPORTED**.

## Boundaries retained

R47 does **not** establish:

- neural R learnability from reward optimization;
- global mathematical minimality of D+C+R;
- universal sufficiency across adaptive systems;
- POLAR implementation or algorithmic superiority;
- E6b independent-team replication;
- E7 prospective biological correspondence;
- AGI or ASI;
- phenomenal consciousness.

POLAR Core v1.1 remains:

[
\mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]
