# R46 result — external D/R causal transport on MPE2

Frozen final resolution:

`R46_EXTERNAL_D_R_TRANSPORT_PASS_SAME_PROGRAM`

## Canonical execution

Scientific workflow run:

`36029741503`

Scientific execution commit:

`bd0189bb1853dcc0715d19d2ffa0241bdf62dc82`

External benchmark:

- Farama Foundation MPE2 `v1.1.0`;
- exact source commit `7590d9d52791e321974d4fda6090fb18f34dbf49`;
- environments: `simple_reference_v3` and `simple_crypto_v3`.

Frozen source integrity:

- protocol SHA-256: `86be95d2794e57af050a8c3b4f3fbcb21bd1bf9e5f58d4c6c240cd48aa2fd0a7`;
- runner SHA-256: `bd7751d574c540eeb314fe11a645de9b4bf80d39061cc93137d6782b00dff1ae`.

## Development — Simple Reference

Seeds: `2176001–2176016`.

All 16 seeds pass the frozen seed guard.

Medians:

- FULL normalized capability: **0.94605462**;
- D lesion effect: **+0.51645202**;
- R wrong-binding effect: **+0.91362697**;
- R previous-episode reidentification lesion: **+0.53045950**;
- GENERIC_ISO absolute gap: **0.0**;
- externally generated relation-change fraction: **0.60**;
- seed guard: **16/16** (required 13/16).

Development resolution:

`R46_DEVELOPMENT_AUTHORIZE_CONFIRM`.

Artifact: `10821335019`.

## Confirmatory B1 — fresh Simple Reference

Seeds: `2177001–2177032`.

Medians:

- FULL normalized capability: **0.94532471**;
- D lesion effect: **+0.48454553**;
- R wrong-binding effect: **+0.82547898**;
- R previous-episode reidentification lesion: **+0.55617430**;
- GENERIC_ISO absolute gap: **0.0**;
- relation-change fraction: **0.70**;
- seed guard: **32/32** (required 28/32).

B1: **PASS**.

## Confirmatory B2 — Simple Crypto cross-task transport

Seeds: `2177101–2177132`.

Medians:

- FULL good-team native reward: **0.97090909**;
- D target/key-collapse effect: **+0.69818182**;
- R direct/no-key-relation effect: **+0.97090909**;
- R stale previous-key reidentification effect: **+1.04727273**;
- GENERIC_ISO absolute gap: **0.0**;
- externally generated key-change fraction: **0.45454545**;
- seed guard: **32/32** (required 26/32).

B2: **PASS**.

Confirmatory artifact: `10820554701`.

## Integrity

Artifact `10821285854` reruns the frozen adjudication logic from the exported development and confirmatory JSON.

All checks pass, including:

- exact development seeds;
- exact confirmatory seed sets;
- development authorization recomputation;
- both confirmatory family decision rules;
- final R46 resolution;
- E6b/E7 boundaries;
- no POLAR-superiority inference;
- no consciousness inference;
- no single-task external D+C+R overclaim.

## Scientific interpretation

R46 provides external same-program evidence that, in the tested MPE2 tasks:

1. differentiated causal channels **D** perform causal work;
2. relation-specific binding **R** performs causal work;
3. current episode relations must be reidentified when the external environment changes targets/keys;
4. these effects transport from spatial cooperative communication (`Simple Reference`) to a distinct key-conditioned communication task (`Simple Crypto`);
5. a bijective internal coordinate recoding remains behaviorally equivalent (`GENERIC_ISO gap = 0`).

Together with R45, the program now has external benchmark evidence for:

- **C** — competent recurrent-state comparison on POPGym;
- **D** — external functional lesion evidence on MPE2;
- **R binding/reidentification** — external functional lesion evidence on MPE2.

## Boundaries retained

R46 does **not** establish:

- external learnability of R;
- one single external agent/task demonstrating the full D+C+R conjunction;
- global minimality or universal sufficiency of D+C+R;
- implementation privilege or POLAR superiority;
- E6b independent-team replication;
- E7 prospective biological correspondence;
- AGI or ASI;
- phenomenal consciousness.

POLAR Core v1.1 remains:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]
